# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
import torch.nn.functional as F

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors,
rotate_to_local
)
from scipy.optimize import linear_sum_assignment
from torch.nn.utils.rnn import pad_sequence
from .attr_tokenizer import Attr_Tokenizer

class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
        pred_entry=False,
        pred_init=False,
        learn_init=False,
        learn_autoencoder=False
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.pred_entry=pred_entry
        self.autoregressive_entry=True
        self.use_smart=False
        self.use_bird=False
        self.noise=False
        self.use_token=True
        self.use_time=False
        self.use_goal=False
        self.pred_exit = False
        self.token_initial = False
        self.pred_map_token = False
        self.match_all=False
        self.token_offset=False
        self.invalid_state=0
        self.valid_state= 1
        self.enter_state= 2
        self.exit_state= 3
        self.pred_init=pred_init
        self.pl2seed_radius=81
        #
        # self.attr_tokenizer= Attr_Tokenizer(grid_range=self.pl2seed_radius*2,
        #                                      grid_interval=3,
        #                                      radius=self.pl2seed_radius,
        #                                      angle_interval=3)

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]
        self.n_token_map = self.map_token_traj_src.shape[0]

        if  self.token_initial:
            self.pl2seed_radius = 80
            self.grid_interval = 2.5
            self.offset_interval = 0.05
            self.angle_interval = 0.25
            self.attr_tokenizer = Attr_Tokenizer(grid_range=self.pl2seed_radius * 2,
                                                 grid_interval=self.grid_interval,
                                                 radius=self.pl2seed_radius,
                                                 angle_interval=self.angle_interval)

            self.offset_tokenizer = Attr_Tokenizer(grid_range=self.grid_interval,
                                                   grid_interval=self.offset_interval,
                                                   radius=100,
                                                   angle_interval=3)

            res = 0.25

            res1 = 0.25

            length, width = 12.25, 3.25

            # number of tokens
            nx = int(length / res)
            ny = int(width / res1)

            # grid coordinates (centered)
            x = torch.arange(1, nx + 1) * res
            y = torch.arange(1, ny + 1) * res1

            # meshgrid
            yy, xx = torch.meshgrid(y, x, indexing="ij")

            self.shape_grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

            self.n_token_entry_head = self.attr_tokenizer.angle_size
            self.n_token_entry_head2 = self.n_token_entry_head // 2
            self.n_token_entry=self.attr_tokenizer.grid_size


        if self.pred_exit:
            self.n_token_agent+=1

        self.use_infgen=False

        self.learn_init=learn_init #True

        self.pred_vel=True

        self.learn_autoencoder = learn_autoencoder

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        if not self.training:
            tokenized_map = self.tokenize_map(data)

            if self.learn_init:
                batch = data["agent"]["batch"]

                type = data["agent"]["type"]

                ego_mask = torch.ones_like(batch)
                ego_mask[:-1] = batch[:-1] != batch[1:]

                sort_rank = batch.to(torch.float64)  * 30+ego_mask.to(torch.float64)*3 + type.to(torch.float64)

                sort_idx = sort_rank.argsort()

                for key in ['valid_mask', 'role', 'id', 'type', 'position', 'heading', 'velocity', 'shape']:
                    data["agent"][key] = data["agent"][key][sort_idx]

            tokenized_agent = self.tokenize_agent(data)

            if self.learn_init:
                if self.learn_autoencoder:
                    idx=10
                else:
                    idx=0

                agent=data["agent"]
                #valid = agent["valid_mask"][:, idx]  # [n_agent, n_step]
                heading = agent["heading"][:, idx]  ## [n_agent, n_step]
                pos = agent["position"][..., :2].contiguous()[:, idx]  # # [n_agent, n_step, 2]
                vel = agent["velocity"][:, idx]  ## [n_agent, n_step, 2]
                shape = agent["shape"]
                type = agent["type"]


                tokenized_agent["initial_heading"] = heading  # [n_agent, n_step]
                tokenized_agent["initial_pos"] = pos  # [n_agent, n_step, 2]
                tokenized_agent["initial_vel"] = vel  # [n_agent, n_step, 2]
                tokenized_agent["initial_shape"] = shape
                tokenized_agent["initial_type"] = type.long()
                tokenized_agent["ego_traj"] = agent["position"][:, idx+1:idx+11, :2][tokenized_agent["ego_mask"]]

            tokenized_agent["type"] = tokenized_agent["type"].long().clone()
        else:
            tokenized_map, tokenized_agent=self.process_data(data)

        batch = tokenized_agent["batch"]

        if "ego_mask" not in tokenized_agent:
            ego_mask = torch.ones_like(batch)
            ego_mask[:-1] = batch[:-1] != batch[1:]
            tokenized_agent["ego_mask"] = ego_mask.bool()

        if self.use_goal and self.training:
            self.compute_goal(tokenized_agent)
        else:
            tokenized_agent["goal_pos"]=None
            tokenized_agent["goal_mask"]=None

        if not self.training and self.pred_entry:
            batch = tokenized_agent["batch"].clone()

            for key, value in tokenized_agent.items():
                if type(value) is torch.Tensor and len(value)==len(batch):
                    new_tensor=[]
                    for b in range(data.num_graphs):
                        valueb=value[batch==b]
                        if 'valid_mask' in key:
                            value_repeat=torch.zeros_like(valueb[:1]).repeat_interleave(100,dim=0)
                        elif key=='id':
                            value_repeat=-torch.arange(100,device=valueb.device)
                        else:
                            value_repeat=valueb[:1].repeat_interleave(100,dim=0)
                        new_tensor.append(torch.cat([value_repeat,valueb]))
                    tokenized_agent[key]=torch.cat(new_tensor)

        if self.pred_exit:
            valid_mask=tokenized_agent["valid_mask"]
            exit_mask=valid_mask[:,:-1] & ~valid_mask[:,1:]
            exit_mask=torch.cat([torch.zeros_like(exit_mask[:,:1]),exit_mask], dim=1)
            tokenized_agent["sampled_idx"][exit_mask]=self.n_token_agent-1


        if self.pred_init and not self.learn_init:
            if self.training:
                for key in ["sampled_idx", "token_mask", "valid_mask", "sampled_pos", "sampled_heading"]:
                    if self.token_initial:
                        tokenized_agent[key] = torch.cat([tokenized_agent[key][:, :1], tokenized_agent[key]], dim=1)
                    else:
                        tokenized_agent[key] = tokenized_agent[key][:, 1:]
            else:
                tokenized_agent["gt_initial_pos"] = tokenized_agent["sampled_pos"][:, 1][:, None]
                tokenized_agent["gt_initial_heading"] = tokenized_agent["sampled_heading"][:, 1][:, None]
                tokenized_agent["gt_initial_idx"] = tokenized_agent["sampled_idx"][:, 1][:, None]

        return tokenized_map, tokenized_agent


    def init_map_token(self, map_token_traj_path, argmin_sample_len=3) -> None:
        map_token_traj = pickle.load(open(map_token_traj_path, "rb"))["traj_src"]
        indices = torch.linspace(
            0, map_token_traj.shape[1] - 1, steps=argmin_sample_len
        ).long()

        self.register_buffer(
            "map_token_traj_src",
            torch.tensor(map_token_traj, dtype=torch.float32).flatten(1, 2),
            persistent=False,
        )  # [n_token, 11*2]

        self.register_buffer(
            "map_token_sample_pt",
            torch.tensor(map_token_traj[:, indices], dtype=torch.float32).unsqueeze(0),
            persistent=False,
        )  # [1, n_token, 3, 2]

    def init_agent_token(self, agent_token_path) -> None:
        agent_token_data = pickle.load(open(agent_token_path, "rb"))

        all_token_local_traj = []
        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32)[:, 1:self.shift + 1]
            # [n_token, 6, 4, 2], countour, 10 hz
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)

            pred_pos = v.mean(2)
            diff_xy = v[:, :, 0] - v[:, :, 3]
            pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
            token_local_traj = torch.cat([pred_pos, pred_head[:, :, None]], dim=-1)
            all_token_local_traj.append(token_local_traj)

        all_token_local_traj = torch.stack(all_token_local_traj)
        self.register_buffer(f"all_token_local_traj", all_token_local_traj, persistent=False)
        self.register_buffer(f"trajectory_token_veh", self.agent_token_all_veh[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_ped", self.agent_token_all_ped[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_cyc", self.agent_token_all_cyc[:, -1].flatten(1, 2), persistent=False)

        module_dir = os.path.dirname(__file__)

        if self.autoregressive_entry:
            entry_token = os.path.join(module_dir, 'entry_prev_global512.pkl')
            entry_pos_token = pickle.load(open(entry_token, "rb"))
            self.register_buffer(f"entry_pos_token", entry_pos_token, persistent=False)
            self.n_token_entry = self.entry_pos_token.shape[0]
            if self.token_offset:
                module_dir = os.path.dirname(__file__)
                offset_token = os.path.join(module_dir, 'offset512.pkl')

                offset_token = pickle.load(open(offset_token, "rb"))
                self.register_buffer(f"offset_token", offset_token, persistent=False)
                self.n_token_offset = self.offset_token.shape[0]
            else:
                self.n_token_offset = 4
        else:
            entry_token = os.path.join(module_dir, 'entry512.pkl')

            entry_pos_token = pickle.load(open(entry_token, "rb"))
            self.register_buffer(f"entry_pos_token", entry_pos_token, persistent=False)
            self.n_token_entry = self.entry_pos_token.shape[0]
            self.n_token_entry = 3

    def decode_head(self, entry_head_idx):
        return (entry_head_idx - self.n_token_entry_head2) / (self.n_token_entry_head2) * torch.pi

    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:

        traj_pos = data["map_save"]["traj_pos"] # [n_pl, 3, 2]
        traj_theta = data["map_save"]["traj_theta"] # [n_pl]
        type = data["pt_token"]["type"]  # [n_pl]
        #pl_type = data["pt_token"]["pl_type"]  # [n_pl]

        # if self.training:
        #     traj_pos=traj_pos+torch.randn_like(traj_pos)*0.05
        #
        # traj_theta = torch.atan2(
        #     traj_pos[:,1, 1] - traj_pos[:,0, 1],
        #     traj_pos[:,1, 0] - traj_pos[:,0, 0]
        # )

        traj_pos_local, _ = transform_to_local(
            pos_global=traj_pos,  # [n_pl, 3, 2]
            head_global=None,  # [n_pl, 1]
            pos_now=traj_pos[:, 0],  # [n_pl, 2]
            head_now=traj_theta,  # [n_pl]
        )
        # [1, n_token, 3, 2] - [n_pl, 1, 3, 2]
        dist = torch.sum(
            (self.map_token_sample_pt - traj_pos_local.unsqueeze(1)) ** 2,
            dim=(-2, -1),
        )  # [n_pl, n_token]

        gt_idx = torch.argmin(dist, dim=-1)

        # if  self.training and self.noise:
        #     topk_indices = torch.argsort(dist, dim=1)[:, :8]
        #     sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
        #     token_idx = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)
        # else:
        token_idx = gt_idx

        position=traj_pos[:, 0].contiguous()

        tokenized_map = {
            "position": position,  # [n_pl, 2]
            "orientation": traj_theta,  # [n_pl]
            "token_idx": token_idx,  # [n_pl]
           "traj_pos_local":traj_pos_local[:,1:],
            #"token_traj_src": self.map_token_traj_src,  # [n_token, 11*2]
            "type": type,  # [n_pl]
            #"pl_type": pl_type.long(),  # [n_pl]
           # "light_type": light_type#.long(),  # [n_pl]
           # "batch": batch,  # [n_pl]
        }
        if "batch" in data["pt_token"].keys():
            tokenized_map["batch"] = data["pt_token"]["batch"]


        if "light_type" in data["pt_token"].keys():
            tokenized_map["light_type"] = data["pt_token"]["light_type"]

        #if self.training and self.pred_map_token:
            # pt_valid_mask=torch.rand_like(traj_theta)< 0.5
        #if self.training:
        if 'pl_idx_list' in data.keys():
            pl_idx = data['map_save']['pl_idx_list']  # shape [T]

            T = pl_idx.numel()
            idx = torch.arange(T, device=pl_idx.device)

            # --- per-lane local index (0,1,2,...) without loops ---
            # lane boundary flag
            lane_change = torch.ones_like(pl_idx, dtype=torch.bool)
            lane_change[1:] = pl_idx[1:] != pl_idx[:-1]

            # start index of current lane for each position (forward-filled)
            starts = torch.where(lane_change, idx, torch.zeros((), dtype=idx.dtype, device=idx.device))
            lane_start_idx = torch.cummax(starts, dim=0).values

            # local index within its lane
            local_idx = idx - lane_start_idx  # 0,1,2,... within each lane

            # --- keep every 2nd point per lane (even local index) ---
            keep_mask =(local_idx % 2 == 0)  # True => keep, False => drop

            # keep_mask= torch.rand_like(traj_theta)<0.5
            #
            # keep_mask[-1]=False

            for key in tokenized_map.keys():
                tokenized_map[key] = tokenized_map[key][keep_mask]

        if self.pred_map_token:

            kept_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)  # [K]
            next_idx = kept_idx + 1  # [K] 安全：上面已保证最后一个不保留

            same_mask = (pl_idx[next_idx] == pl_idx[kept_idx])  # [K] 布尔

            tokenized_map['pt_pred_mask'] = same_mask  # [K]，与过滤后的 token 对齐
            tokenized_map['pt_target'] = gt_idx[next_idx][same_mask]  # 目标是“下一帧”的 token

        return tokenized_map


    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:
        # ! collate width/length, traj tokens for current batch
        agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(
            data["agent"]["type"]
        )

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"].clone()  # [n_agent, n_step]
        heading = data["agent"]["heading"].clone()   # [n_agent, n_step]
        pos = data["agent"]["position"][..., :2].contiguous().clone()   # [n_agent, n_step, 2]
        vel = data["agent"]["velocity"].clone()   # [n_agent, n_step, 2]

        # # if not (self.pred_last_res and self.pred_all_res):
        heading = self._clean_heading(valid, heading)

        valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
            valid, pos, heading, vel
        )
        #valid,pos,heading,vel=extrapolate_agent_to_first_step_vectorized(valid, pos, heading, vel)

        role_mask = data["agent"]["role"]

        pred_mask = role_mask[:, 0] | role_mask[:, 2]

        ego_mask=data["agent"]["role"][:, 0]

        # ! prepare output dict
        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"].clone(),
            "ego_mask": ego_mask,  # [n_agent]
            "token_agent_shape":  agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            "token_traj_all": token_traj_all,  # [n_agent, n_token, 6, 4, 2]
            "token_traj": token_traj,  # [n_agent, n_token, 4, 2]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "gt_traj_10hz": pos,
            "gt_head_10hz": heading,
            "gt_valid": valid,
            "pred_mask":pred_mask,
        }
        # [n_token, 8]
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] =getattr(self,f"trajectory_token_{k}")

        # ! match token for each agent
        if not self.training:
            # [n_agent]
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][:, 10, 2]

        batch = data["agent"]["batch"]

        if  self.token_initial:
            valid =valid[:,10:]
            pos=pos[:,10:]
            heading=heading[:,10:]

            token_initial_pos, token_initial_heading,pos_token_idx,heading_token_idx,offset_idx,initial_pos,initial_heading=self.tokenize_initial(pos[:,0],heading[:,0],ego_mask,batch)

            pos[:, :1]=initial_pos
            heading[:, :1]=initial_heading

            tokenized_agent["initial_pos"]=pos[:,0]
            tokenized_agent["initial_heading"]=heading[:,0]

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
            batch=batch,#[:,None],
            num_graphs=data.num_graphs,
            ego_mask=ego_mask,
            shape=data["agent"]["shape"],
            type=data["agent"]["type"].long(),
            data=data
        )
        if "route_map_index" in data["agent"].keys():
            tokenized_agent['route_map_index']=data["agent"]["route_map_index"]
        if "id" in data["agent"].keys():
            tokenized_agent['id']=data["agent"]["id"]

        tokenized_agent.update(token_dict)

        return tokenized_agent

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
        batch,
        num_graphs=None,
        ego_mask=None,
        shape=None,
        type=None,
        data=None,
        shift=5,
        error_dist=0.3
    ) -> Dict[str, Tensor]:
        #num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]
        #prev_pos_sample, prev_head_sample = pos[:, 0], heading[:, 0]

        out_dict = {
            "valid_mask": [],
            "sampled_idx": [],
            "sampled_pos": [],#pos[:, 0]
            "sampled_heading": [],
            'gt_idx':[],
            'token_mask':[],
            "token_contour":[]
           # 'token_valid':[]
        }
        entry_token_invalid_mask = []
        entry_idx_list = []
        entry_pos_list=[]

        entry_head_idx_list = []
        entry_pos_offset_list = []
        entry_type_list=[]
        entry_shape_list=[]

        # if self.learn_init:
        #     n_step=11+10+5

        n_step = 11

        if self.pred_entry and not self.autoregressive_entry:
            out_dict["entry_idx"] = []

            if self.match_all:
                entry_pos_token = self.entry_pos_token[None].repeat(len(pos), 1, 1)  # .to(torch.float16)

                entry_token_xy = entry_pos_token[:, :, :2]
                entry_token_z = entry_pos_token[:, :, 2]

        for i in range(shift, n_step, shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - shift] & valid[:, i]  # [n_agent]

            out_dict["token_mask"].append(_valid_mask.clone())

            #! gt_contour: [n_agent, 4, 2] in global coord
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            # ! tokenize without sampling
            token_world_gt = transform_to_global(
                pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos,  # [n_agent, 2]
                head_now=prev_head,  # [n_agent]
            )[0].view(*token_traj.shape)
            all_dist=torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1)
            min_dist, token_idx_gt = torch.min(all_dist, dim=-1)  # [n_agent]

            out_dict["gt_idx"].append(token_idx_gt)

            # if self.training and self.noise:
            #     topk_indices = torch.argsort( all_dist,dim=-1)[:, :5]
            #     sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
            #     token_idx_gt = topk_indices[np.arange(topk_indices.shape[0]), sample_topk]

            # [n_agent, 4, 2]
            token_contour_gt = token_world_gt[range_a, token_idx_gt]

            out_dict["token_contour"].append(token_contour_gt)

            token_valid=min_dist<error_dist
            _valid_mask[~token_valid]=False

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            prev_pos = pos[:, i].clone()

            if self.pred_entry:
                if not self.autoregressive_entry:
                    entry_idx = torch.zeros_like(token_idx_gt) + self.n_token_entry

                if i > self.shift + self.shift:
                    entry_agent = ~valid[:, i - self.shift] & valid[:, i]

                    if self.autoregressive_entry:
                        self.auto_enter(entry_agent,prev_pos,prev_head,batch,num_graphs,ego_mask,shape,type,out_dict, entry_idx_list)
                    else:
                        present_agent = valid[:, i - self.shift]

                        self.entry_agent(entry_idx, entry_agent,present_agent,prev_pos,prev_head,batch,num_graphs,shape,type,out_dict,
                                         entry_head_idx_list,entry_pos_offset_list,entry_pos_list,entry_shape_list)



                if not self.autoregressive_entry:
                    out_dict["entry_idx"].append(entry_idx)

            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
            next_head=torch.arctan2(dxy[:, 1], dxy[:, 0])
            prev_head[_valid_mask] = next_head[_valid_mask]


            next_pos = token_contour_gt.mean(1)
            prev_pos[_valid_mask] = next_pos[_valid_mask]

            _valid_mask=valid[:, i]
            _invalid_mask = ~valid[:, i]

            # add to output dict
            out_dict["sampled_idx"].append(token_idx_gt)
            out_dict["sampled_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )
            out_dict["sampled_heading"].append(prev_head.masked_fill(_invalid_mask, 0))
            out_dict["valid_mask"].append(_valid_mask)

        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}

        if self.training and self.pred_entry:
            if len(entry_token_invalid_mask)>0:
                out_dict["entry_token_invalid_mask"]=torch.cat(entry_token_invalid_mask, dim=0)


            if len(entry_pos_list) :
                out_dict["entry_pos"]=torch.cat(entry_pos_list)

            if len(entry_head_idx_list) :
                out_dict["entry_head_idx"]=torch.cat(entry_head_idx_list)
                #out_dict["entry_head_idx_num"]=torch.tensor([len(entry_head_idx) for entry_head_idx in entry_head_idx_list])
            else:
                out_dict["entry_head_idx"] =torch.zeros([0])
                # out_dict["entry_head_idx_num"]=torch.zeros([out_dict["valid_mask"].shape[1]-1],device=out_dict["sampled_idx"].device)

            if len(entry_pos_offset_list) :
                out_dict["entry_pos_offset"]=torch.cat(entry_pos_offset_list)
            else:
                out_dict["entry_pos_offset"] =torch.zeros([0,4])


            if len(entry_shape_list) :
                out_dict["entry_shape"]=torch.cat(entry_shape_list)

            if len(entry_type_list) :
                out_dict["entry_type"]=torch.cat(entry_type_list)


            if len(entry_idx_list) :
                # entry_length=out_dict['sampled_idx'].shape[1]-1
                entry_idx=pad_sequence(entry_idx_list, batch_first=True, padding_value=self.n_token_entry)#.reshape(entry_length,batch_num,-1)

                out_dict["pos_idx"]=entry_idx[:,:,0].long()

                out_dict["head_idx"] = torch.clamp(
                    entry_idx[:, :, 1],
                    max=self.n_token_entry_head - 1
                ).long()

                if self.token_offset:
                    offset=entry_idx[:,:,2]
                else:
                    offset=entry_idx[:,:,2:5]

                offset[offset==self.n_token_entry]=0

                out_dict["offset"]=offset

                out_dict["entry_type"]= torch.clamp(
                    entry_idx[:, :, 5],
                    max=2
                ).long()
                out_dict["entry_shape"]=entry_idx[:,:,6:]

        if self.use_infgen:
            self.process_state(data, pos, out_dict["sampled_idx"].clone(), out_dict["token_contour"], [], valid)

        return out_dict

    def auto_enter(self,entry_agent,prev_pos,prev_head,batch,num_graphs,ego_mask,shape,type,out_dict,  entry_idx_list):

        entry_pos = prev_pos[entry_agent]
        entry_heading = prev_head[entry_agent]
        entry_batch = batch[entry_agent]

        entry_pos_list = []
        entry_heading_list = []
        entry_type_list = []
        entry_shape_list = []

        for b in range(num_graphs):
            ego_pos = out_dict["sampled_pos"][-1][ego_mask][b]
            ego_heading = out_dict["sampled_heading"][-1][ego_mask][b]

            entry_pos_b, entry_heading_b = transform_to_local(
                entry_pos[entry_batch == b][None,],  # [n_agent, n_step, 2]
                entry_heading[entry_batch == b][None],  # [n_agent, n_step]
                ego_pos[None],  # [n_agent, 2]
                ego_heading[None]  # [n_agent]
            )

            entry_pos_b = entry_pos_b[0]
            entry_heading_b = entry_heading_b[0]

            #
            # C = 10000
            # sort_key = entry_batch.float() * C + entry_pos[:, 0]
            sort_idx = torch.argsort(entry_pos_b[:, 0])

            entry_pos_b = entry_pos_b[sort_idx]
            entry_heading_b = entry_heading_b[sort_idx]

            entry_pos_list.append(entry_pos_b)
            entry_heading_list.append(entry_heading_b)

            entry_type = type[entry_agent][entry_batch == b][sort_idx]
            entry_shape = shape[entry_agent][entry_batch == b][sort_idx]

            entry_type_list.append(entry_type)
            entry_shape_list.append(entry_shape)

        entry_pos = torch.cat(entry_pos_list)  # 1
        entry_heading = torch.cat(entry_heading_list)  # 1
        entry_type = torch.cat(entry_type_list)  # 1
        entry_shape = torch.cat(entry_shape_list)  # 3

        # entry_idx= self.tokenizer(entry_pos,entry_heading)

        pos_entry_idx = torch.linalg.norm(entry_pos[:, None] - self.entry_pos_token[None], dim=-1).argmin(1)

        entry_head_idx = (wrap_angle(
            entry_heading) / np.pi * self.n_token_entry_head2).round().long() + self.n_token_entry_head2

        entry_head_idx[entry_head_idx == self.n_token_entry_head] = 0

        tokenized_pos = self.entry_pos_token[pos_entry_idx]

        tokenized_heading = self.decode_head(entry_head_idx)

        offset_pos = entry_pos - tokenized_pos

        if self.token_offset:
            offset_local = torch.linalg.norm(offset_pos[:, None] - self.offset_token[None], dim=-1).argmin(1)[:,
                           None]
        else:
            offset_local = torch.cat((offset_pos, wrap_angle(entry_heading - tokenized_heading)[:, None]),
                                     dim=-1)

        entry_idx = torch.cat(
            [pos_entry_idx[:, None], entry_head_idx[:, None], offset_local, entry_type[:, None], entry_shape],
            dim=-1)

        entry_length = torch.bincount(entry_batch, minlength=num_graphs).tolist()

        entry_idx_list.extend(torch.split(entry_idx, entry_length))

    def entry_agent(self,entry_idx,entry_agent,present_agent,prev_pos,prev_head,batch,num_graphs,shape,type,out_dict,entry_head_idx_list,entry_pos_offset_list,entry_pos_list,entry_shape_list):

        entry_pos = prev_pos[entry_agent]

        if entry_agent.any():

            present_pos = out_dict["sampled_pos"][-1][present_agent]

            present_heading = out_dict["sampled_heading"][-1][present_agent]

            if self.match_all:
                global_token_xy = \
                    transform_to_global(entry_token_xy[:len(present_pos)], None, present_pos[:, :2], present_heading)[0]

                global_token_z = entry_token_z[:len(present_pos)] + present_pos[:, None, 2]

                global_token_pos = torch.cat((global_token_xy, global_token_z[:, :, None]), dim=-1)

            present_batch = batch[present_agent]
            entry_batch = batch[entry_agent]

            row_ind = []
            col_ind = []
            entry_idx_gt = []

            for b in range(num_graphs):
                entry_pos_i = entry_pos[entry_batch == b]
                if len(entry_pos_i):

                    if self.match_all:
                        global_token_pos_i = global_token_pos[present_batch == b]
                        diff = global_token_pos_i[:, None] - entry_pos_i[None, :, None]  # (Np, Ne,token, D)
                        cost, min_idx = torch.linalg.norm(diff, dim=-1).min(-1)
                    else:
                        global_token_pos_i = present_pos[present_batch == b]
                        diff = global_token_pos_i[:, None] - entry_pos_i[None]  # (Np, Ne,token, D)
                        cost = torch.linalg.norm(diff, dim=-1)

                    row_ind_i, col_ind_i = linear_sum_assignment(cost.cpu().numpy())

                    row_ind.append(row_ind_i + torch.sum(present_batch < b).item())
                    col_ind.append(col_ind_i + torch.sum(entry_batch < b).item())

                    if self.match_all:
                        entry_idx_gt_i = min_idx[row_ind_i, col_ind_i]
                        entry_idx_gt.append(entry_idx_gt_i)

            row_ind = np.concatenate(row_ind)
            col_ind = np.concatenate(col_ind)

            gt_entry_id = torch.nonzero(entry_agent, as_tuple=False).squeeze(1)
            gt_present_id = torch.nonzero(present_agent, as_tuple=False).squeeze(1)

            entry_id = gt_entry_id[col_ind]

            entry_pos1 = prev_pos[entry_id]
            entry_heading = prev_head[entry_id]

            select_pos = present_pos[row_ind]
            select_heading = present_heading[row_ind]
            present_id = gt_present_id[row_ind]

            local_xy, local_heading = transform_to_local(
                entry_pos1[:, None, :2],
                entry_heading[:, None],
                select_pos[:, :2],
                select_heading
            )

            entry_head_idx = (wrap_angle(
                local_heading[:, 0]) / np.pi * self.n_token_entry_head2).round().long() + self.n_token_entry_head2

            entry_head_idx[entry_head_idx == self.n_token_entry_head] = 0

            tokenized_heading = self.decode_head(entry_head_idx)

            head_offset = wrap_angle(local_heading[:, 0] - tokenized_heading)  # for selecting heading, not for

            entry_head_idx_list.append(entry_head_idx)

            local_z = entry_pos1[:, 2:] - select_pos[:, 2:]

            local_pos = torch.cat([local_xy[:, 0], local_z], dim=-1)

            if self.match_all:
                entry_idx_gt = torch.cat(entry_idx_gt)
            else:
                dist, entry_idx_gt = torch.linalg.norm(local_pos[:, None] - self.entry_pos_token[None], dim=-1).min(
                    -1)  # .mean(-1)

            entry_pos_list.append(entry_idx_gt)

            entry_type = type[entry_id]
            entry_shape = shape[entry_id]

            # entry_type_list.append(entry_type)
            entry_shape_list.append(entry_shape)

            entry_idx[present_id] = entry_type

            local_tok = self.entry_pos_token[entry_idx_gt]

            offset_local = torch.cat((local_pos - local_tok, head_offset[:, None]), dim=-1)
            entry_pos_offset_list.append(offset_local)

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]#sequential
        return heading

    def _extrapolate_agent_to_first_step(
            self,
            valid: Tensor,  # [n_agent, n_step]
            pos: Tensor,  # [n_agent, n_step, 2]
            heading: Tensor,  # [n_agent, n_step]
            vel: Tensor,  # [n_agent, n_step, 2]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # [n_agent], max will give the first True step
        first_valid_step = torch.max(valid, dim=1).indices  # [n_agent]

        for i, t in enumerate(first_valid_step):
            if t > 0:
                # Fill from step 0 up to (but not including) step t
                vel[i, :t] = vel[i, t]
                valid[i, :t] = True
                heading[i, :t] = heading[i, t]

                for j in range(t - 1, -1, -1):
                    pos[i, j] = pos[i, j + 1] - vel[i, t] * 0.1  # 0.1 is the time delta

        return valid, pos, heading, vel

    def _extrapolate_agent_to_prev_token_step(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        vel: Tensor,  # [n_agent, n_step, 2]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # [n_agent], max will give the first True step
        first_valid_step = torch.max(valid, dim=1).indices

        for i, t in enumerate(first_valid_step):  # extrapolate to previous 5th step.
            n_step_to_extrapolate = t % self.shift
            if (t == 10) and (not valid[i, 10 - self.shift]):
                # such that at least one token is valid in the history.
                n_step_to_extrapolate = self.shift

            if n_step_to_extrapolate > 0:
                vel[i, t - n_step_to_extrapolate : t] = vel[i, t]
                valid[i, t - n_step_to_extrapolate : t] = True
                heading[i, t - n_step_to_extrapolate : t] = heading[i, t]

                for j in range(n_step_to_extrapolate):
                    pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1

        return valid, pos, heading, vel

    def _get_agent_shape_and_token_traj(
        self, agent_type: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        agent_shape: [n_agent, 2]
        token_traj_all: [n_agent, n_token, 6, 4, 2]
        token_traj: [n_agent, n_token, 4, 2]
        """
        agent_type_masks = {
            "veh": agent_type == 0,
            "ped": agent_type == 1,
            "cyc": agent_type == 2,
        }
        agent_shape = 0.0
        token_traj_all = 0.0
        for k, mask in agent_type_masks.items():
            if k == "veh":
                width = 2.0
                length = 4.8
            elif k == "ped":
                width = 1.0
                length = 1.0
            else:
                width = 1.0
                length = 2.0
            agent_shape += torch.stack([width * mask, length * mask], dim=-1)

            token_traj_all += mask[:, None, None, None, None] * (
                getattr(self, f"agent_token_all_{k}").unsqueeze(0)
            )

        self.token_traj_all = token_traj_all.reshape(-1, 2048, 5 * 4 * 2)

        token_traj = token_traj_all[:, :, -1, :, :].contiguous()

        return agent_shape, token_traj_all, token_traj

    def process_data(self,data):

        tokenized_agent = {}
        tokenized_map = {}
        tokenized_agent['num_graphs'] = data.num_graphs

        map = data["tokenized_map"]
        agent = data["tokenized_agent"]

        if len(map) == 0:
            tokenized_map=self.tokenize_map(data)
        else:
            if "token_idx"in map.keys():
                tokenized_map["token_idx"] = map["token_idx"]
            else:
                # [1, n_token, 3, 2] - [n_pl, 1, 3, 2]
                dist = torch.sum(
                    (self.map_token_sample_pt[:,:,1:] - map["traj_pos_local"].unsqueeze(1)) ** 2,
                    dim=(-2, -1),
                )  # [n_pl, n_token]
                if   self.noise:
                    topk_indices = torch.argsort(dist, dim=1)[:, :8]
                    sample_topk = torch.randint(0, topk_indices.shape[-1], size=(topk_indices.shape[0], 1), device=topk_indices.device)
                    token_idx = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)
                else:
                    token_idx = torch.argmin(dist, dim=-1)

                tokenized_map["token_idx"] = token_idx

                tokenized_map["traj_pos_local"] = map["traj_pos_local"]

            for key in ["position", "orientation", "batch", "type"]:#, "pl_type", "light_type"
                tokenized_map[key] = map[key]

            if "light_type" in data.keys():
                tokenized_map["light_type"] = map["light_type"]


        if len(agent) == 0:
            tokenized_agent=self.tokenize_agent(data)
        else:
            if self.learn_init:
                for key in ["initial_heading", "initial_pos", "initial_shape", "initial_type","batch","ego_traj"]:
                    tokenized_agent[key] = agent[key]  # [agent_mask]

                if "initial_vel" in agent.keys():
                    tokenized_agent["initial_vel"] = agent["initial_vel"]
                else:
                    tokenized_agent["initial_vel"]=(agent["initial_pos"]-agent["prev_pos"])/0.5

                tokenized_agent['initial_type'] = tokenized_agent['initial_type'].long()

                agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(
                    tokenized_agent['initial_type']
                )

                tokenized_agent["token_traj"]=token_traj

            else:
                agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(  agent['type'] )

                tokenized_agent["token_agent_shape"] = agent_shape  # [n_token, 2]
                tokenized_agent["token_traj"] = token_traj  # [n_token, 2]
                tokenized_agent["token_traj_all"] = token_traj_all  # [n_token, 6, 4, 2]

                if "col_mask" in agent.keys():
                    tokenized_agent["col_mask"] = agent["col_mask"]

                if "pred_mask" in agent.keys():
                    tokenized_agent["pred_mask"] = agent["pred_mask"]

                if "initial_pos" in agent.keys():
                    tokenized_agent["initial_pos"] = agent["initial_pos"]
                    tokenized_agent["initial_heading"] = agent["initial_heading"]


                if "gt_valid_raw" in data.keys():
                    for key in ["type", "batch", "shape"]:
                        tokenized_agent[key] = agent[key]

                    if "gt_speed_raw" in agent.keys():
                        speed=agent["gt_speed_raw"]
                    else:
                        speed=None

                    token_dict = self._match_agent_token(agent["gt_valid_raw"], agent["gt_pos_raw"],
                                                                agent["gt_head_raw"],
                                                                agent_shape, token_traj,
                                                                 speed,
                                                                    )
                    tokenized_agent.update(token_dict)

                    tokenized_agent["gt_pos_raw"]= agent["gt_pos_raw"][:,5::5]
                    tokenized_agent["gt_head_raw"]= agent["gt_head_raw"][:,5::5]
                    tokenized_agent["gt_valid_raw"]= agent["gt_valid_raw"][:,5::5]

                else:

                    for key in ["sampled_pos", "sampled_heading", "type", "batch", "shape", "valid_mask"]:
                        tokenized_agent[key] = agent[key]#[agent_mask]

                    tokenized_agent["sampled_idx"]=agent["sampled_idx"].long()#[agent_mask]
                    tokenized_agent['type'] = tokenized_agent['type'].long()

                    if 'token_mask' in agent.keys():
                        tokenized_agent['token_mask'] = agent['token_mask']#[agent_mask]
                    else:
                        tokenized_agent["token_mask"]=torch.cat([agent["valid_mask"][:,:1], agent["valid_mask"][:,:-1]], dim=-1)


                    if "gt_pos_raw" in agent.keys():

                        for key in ["gt_pos_raw", "gt_head_raw"]:
                            tokenized_agent[key] = agent[key]
                        tokenized_agent['gt_valid_raw'] = agent["valid_mask"]
                        tokenized_agent['train_mask_ce'] = agent["train_mask"]


        tokenized_map["token_traj_src"]= self.map_token_traj_src

        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(
                self, f"agent_token_all_{k}"
            )[:, -1].flatten(1, 2)

        # tokenized_agent["gt_idx"]=tokenized_agent["sampled_idx"]
        # tokenized_agent["gt_pos"]=tokenized_agent["sampled_pos"]
        # tokenized_agent["gt_heading"]=tokenized_agent["sampled_heading"]

        return tokenized_map, tokenized_agent

