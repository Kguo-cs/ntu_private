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
    angle_between_2d_vectors
)
from src.smart.loss.iq_loss import padding
from scipy.optimize import linear_sum_assignment
from torch.nn.utils.rnn import pad_sequence

class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
        pred_entry=False
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.pred_entry=pred_entry
        self.autoregressive_entry=False
        self.use_smart=False
        self.use_bird=False
        self.noise=False
        self.use_token=True
        self.use_time=False
        self.use_goal=False
        self.pred_exit=False
        self.pred_map_token = False
        self.match_all=False
        self.token_offset=False

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]
        self.n_token_map = self.map_token_traj_src.shape[0]

        if self.pred_exit:
            self.n_token_agent+=1

    @torch.no_grad()
    def forward(self, data: HeteroData,extrapolate=True) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        if not self.training:
            tokenized_map = self.tokenize_map(data)

            tokenized_agent = self.tokenize_agent(data,extrapolate)
        else:
            tokenized_map, tokenized_agent=self.process_data(data)

        tokenized_agent["abs_time"]=torch.zeros([0,18])

        if self.use_goal and self.training:

            sampled_pos=tokenized_agent["sampled_pos"]
            sampled_heading=tokenized_agent["sampled_heading"]
            valid_mask=tokenized_agent["valid_mask"]

            A, T, _ = sampled_pos.shape

            # Convert heading → unit direction (XY)
            dir_xy = torch.stack([torch.cos(sampled_heading), torch.sin(sampled_heading)], dim=-1)  # (A,T,2)

            # Find index of last valid step for each agent
            valid_mask = valid_mask.bool()
            # We want *last* valid, not first
            last_idx = (valid_mask.float() * torch.arange(T, device=sampled_pos.device).float()).max(dim=1).indices

            # Gather last valid pos and heading
            idx = last_idx.view(-1, 1, 1).expand(-1, 1, 2)  # shape (A,1,3)
            last_pos = sampled_pos.gather(1, idx).squeeze(1)  # (A,3)
            last_dir = dir_xy.gather(1, idx).squeeze(1)  # (A,2)

            # Sample random extrapolation distances [0, max_extend)
            goal_dist = torch.rand((A,), device=sampled_pos.device) * 50

            goal_dist[np.random.random(A)<0.5]=0

            # Compute goal position = last_pos + dist * direction
            goal_pos = last_pos + goal_dist[:, None] * last_dir

            tokenized_agent["goal_pos"]=goal_pos

            batch_idx=tokenized_agent["batch"]

            rand_idx = torch.randint(low=0, high=2, size=(max(batch_idx) + 1, 1), device=batch_idx.device)

            goal_mask = rand_idx[batch_idx] < 1

            goal_mask[np.random.random(len(goal_mask)) < 0.5] = True

            tokenized_agent["goal_mask"]=goal_mask[:,0]
        else:
            tokenized_agent["goal_pos"]=None
            tokenized_agent["goal_mask"]=None

        tokenized_agent['type']=tokenized_agent['type'].long()


        if not self.training and self.pred_entry:
            batch = tokenized_agent["batch"].clone()

            for key, value in tokenized_agent.items():
                if type(value) is torch.Tensor and len(value)==len(batch):
                    new_tensor=[]
                    for b in range(data.num_graphs):
                        valueb=value[batch==b]
                        if 'valid_mask' in key:
                            value_repeat=torch.zeros_like(valueb[:1]).repeat_interleave(100,dim=0)
                        else:
                            value_repeat=valueb[:1].repeat_interleave(100,dim=0)
                        new_tensor.append(torch.cat([value_repeat,valueb]))
                    tokenized_agent[key]=torch.cat(new_tensor)
            batch = tokenized_agent["batch"]
            av_mask = torch.ones_like(batch)
            av_mask[:-1] = batch[:-1] != batch[1:]

            tokenized_agent["av_mask"]=av_mask

        # if self.training:
        #     current_valid = data['agent']["current_valid"]
        #
        #     for key, value in tokenized_agent.items():
        #         if type(value) is torch.Tensor and len(value) == len(current_valid):
        #             # print(key,current_valid.device,value.device)
        #
        #             tokenized_agent[key]=value[current_valid]

        if self.pred_exit:
            valid_mask=tokenized_agent["valid_mask"]
            exit_mask=valid_mask[:,:-1] & ~valid_mask[:,1:]
            exit_mask=torch.cat([torch.zeros_like(exit_mask[:,:1]),exit_mask], dim=1)
            tokenized_agent["sampled_idx"][exit_mask]=self.n_token_agent-1

        # print(len(tokenized_agent['batch']))
        #54,21,20,1538,16319
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

        all_token_local_traj=[]
        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32)[:,1:self.shift+1]
            # [n_token, 6, 4, 2], countour, 10 hz
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)

            pred_pos = v.mean(2)
            diff_xy = v[:, :, 0] - v[:, :,  3]
            pred_head = torch.arctan2(diff_xy[:, :,1], diff_xy[:, :,0])
            token_local_traj = torch.cat([pred_pos, pred_head[:, :,  None]], dim=-1)
            all_token_local_traj.append(token_local_traj)

        all_token_local_traj=torch.stack(all_token_local_traj)
        self.register_buffer(f"all_token_local_traj", all_token_local_traj, persistent=False)

        if "max_diff" in agent_token_data.keys():
            self.register_buffer(f"max_diff", 0.01*agent_token_data["max_diff"], persistent=False)
        else:
            self.max_diff=None

        self.register_buffer(f"trajectory_token_veh", self.agent_token_all_veh[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_ped", self.agent_token_all_ped[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_cyc", self.agent_token_all_cyc[:, -1].flatten(1, 2), persistent=False)

        module_dir = os.path.dirname(__file__)

        if self.autoregressive_entry:
            entry_token = os.path.join(module_dir, 'entry_global512.pkl')

            entry_pos_token = pickle.load(open(entry_token, "rb"))
            self.register_buffer(f"entry_pos_token", entry_pos_token, persistent=False)
            self.n_token_entry = self.entry_pos_token.shape[0]

            if self.token_offset:
                module_dir = os.path.dirname(__file__)
                offset_token=os.path.join(module_dir, 'offset512.pkl')

                offset_token = pickle.load(open(offset_token, "rb"))
                self.register_buffer(f"offset_token", offset_token, persistent=False)
                self.n_token_offset = self.offset_token.shape[0]
            else:

                self.n_token_offset=4

        else:
            entry_token = os.path.join(module_dir, 'entry512.pkl')

            entry_pos_token = pickle.load(open(entry_token, "rb"))
            self.register_buffer(f"entry_pos_token", entry_pos_token, persistent=False)
            self.n_token_entry = self.entry_pos_token.shape[0]

        self.n_token_entry_head=128
        self.n_token_entry_head2=self.n_token_entry_head//2

    def decode_head(self,entry_head_idx):
        return (entry_head_idx - self.n_token_entry_head2) / (self.n_token_entry_head2) * torch.pi

    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:

        traj_pos = data["map_save"]["traj_pos"] # [n_pl, 3, 2]
        traj_theta = data["map_save"]["traj_theta"] # [n_pl]
        type = data["pt_token"]["type"]  # [n_pl]
        #pl_type = data["pt_token"]["pl_type"]  # [n_pl]
        #light_type= data["pt_token"]["light_type"]   # [n_pl]

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
            #"light_type": light_type.long(),  # [n_pl]
           # "batch": batch,  # [n_pl]
        }
        if "batch" in data["pt_token"].keys():
            tokenized_map["batch"] = data["pt_token"]["batch"]
        if "light_type" in data["pt_token"].keys():
            tokenized_map["light_type"] = data["pt_token"]["light_type"].long()

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

    def tokenize_agent(self, data: HeteroData,extrapolate=True) -> Dict[str, Tensor]:
        """
        Args: data["agent"]: Dict
            "valid_mask": [n_agent, n_step], bool
            "role": [n_agent, 3], bool
            "id": [n_agent], int64
            "type": [n_agent], uint8
            "position": [n_agent, n_step, 3], float32
            "heading": [n_agent, n_step], float32
            "velocity": [n_agent, n_step, 2], float32
            "shape": [n_agent, 3], float32
        """
        # ! collate width/length, traj tokens for current batch
        agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(
            data["agent"]["type"]
        )

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"]  # [n_agent, n_step]
        heading = data["agent"]["heading"]  # [n_agent, n_step]
        pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, n_step, 2]
        vel = data["agent"]["velocity"]  # [n_agent, n_step, 2]

        # # ! agent, specifically vehicle's heading can be 180 degree off. We fix it here.

        # # if not (self.pred_last_res and self.pred_all_res):
        heading = self._clean_heading(valid, heading)

        if extrapolate:
        # ! extrapolate to previous 5th step.
            valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
                valid, pos, heading, vel
            )

        role_mask = data["agent"]["role"]
        av_mask =data["agent"]["role"][:, 0]


        pred_mask = role_mask[:, 0] | role_mask[:, 2]

        # ! prepare output dict
        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent]
            "token_agent_shape":  agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            "token_traj_all": token_traj_all,  # [n_agent, n_token, 6, 4, 2]
            "token_traj": token_traj,  # [n_agent, n_token, 4, 2]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "pred_traj_10hz": pos,
            "pred_head_10hz": heading,
            "all_valid": valid,
            "pred_mask":pred_mask,
        }
        # [n_token, 8]
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] =getattr(self,f"trajectory_token_{k}")
            #     getattr(
            #     self, f"agent_token_all_{k}"
            # )[:, -1].flatten(1, 2)

        # ! match token for each agent
        if not self.training:
            # [n_agent]
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][:, 10, 2]

        batch=data["agent"]["batch"]


        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
          #  batch=batch,#[:,None],
           # num_graphs=data.num_graphs,
           # av_mask=av_mask
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
            speed=None,
            shift=5,
            error_dist=0.3
    ) -> Dict[str, Tensor]:
        """n_step_token=n_step//5
        n_step_token=18 for train with BC.
        n_step_token=2 for val/test and train with closed-loop rollout.
        Returns: Dict
            # ! action that goes from [(0->5), (5->10), ..., (85->90)]
            "valid_mask": [n_agent, n_step_token]
            "gt_idx": [n_agent, n_step_token]
            # ! at step [5, 10, 15, ..., 90]
            "gt_pos": [n_agent, n_step_token, 2]
            "gt_heading": [n_agent, n_step_token]
            # ! noisy sampling for training data augmentation
            "sampled_idx": [n_agent, n_step_token]
            "sampled_pos": [n_agent, n_step_token, 2]
            "sampled_heading": [n_agent, n_step_token]
        """

        # num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]
        # prev_pos_sample, prev_head_sample = pos[:, 0], heading[:, 0]

        out_dict = {
            "valid_mask": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
            'gt_idx': [],
            'token_mask': []
            # 'token_valid':[]
        }

        for i in range(shift, n_step, shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - shift] & valid[:, i]  # [n_agent]

            out_dict["token_mask"].append(_valid_mask.clone())

            # ! gt_contour: [n_agent, 4, 2] in global coord
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            # ! tokenize without sampling
            token_world_gt = transform_to_global(
                pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos,  # [n_agent, 2]
                head_now=prev_head,  # [n_agent]
            )[0].view(*token_traj.shape)
            all_dist = torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1)
            min_dist, token_idx_gt = torch.min(all_dist, dim=-1)  # [n_agent]

            out_dict["gt_idx"].append(token_idx_gt)

            # if self.training and self.noise:
            #     topk_indices = torch.argsort( all_dist,dim=-1)[:, :5]
            #     sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
            #     token_idx_gt = topk_indices[np.arange(topk_indices.shape[0]), sample_topk]

            # [n_agent, 4, 2]
            token_contour_gt = token_world_gt[range_a, token_idx_gt]

            # if  self.pred_last_res:
            token_valid = min_dist < error_dist
            # token_idx_gt[~token_valid]=self.agent_token_all_veh.shape[0]
            # _valid_mask=token_valid & _valid_mask
            _valid_mask[~token_valid] = False

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
            next_head = torch.arctan2(dxy[:, 1], dxy[:, 0])

            # if self.training and self.pred_all_res and self.max_diff is None:
            #
            #     head_diff=wrap_angle(next_head-prev_head).abs().clamp_max(0.05)/4
            #
            #     next_head = prev_head + head_diff * torch.randn_like(next_head)  # *( torch.rand_like(pos)-0.5)
            #
            #     next_head=wrap_angle(next_head)
            #
            #     _valid_mask = valid[:, i]

            prev_head[_valid_mask] = next_head[_valid_mask]

            prev_pos = pos[:, i].clone()
            next_pos = token_contour_gt.mean(1)

            # if self.training and self.pred_all_res and self.max_diff is None:
            #     pos_diff=(next_pos-prev_pos).abs().clamp_max(0.1)/4
            #
            #     next_pos = prev_pos + pos_diff * torch.randn_like(pos_diff) # *( torch.rand_like(pos)-0.5)

            prev_pos[_valid_mask] = next_pos[_valid_mask]

            # if self.pred_last_res:
            _valid_mask = valid[:, i]

            _invalid_mask = ~valid[:, i]

            # out_dict["token_valid"].append(token_valid &_valid_mask)

            # add to output dict
            out_dict["sampled_idx"].append(token_idx_gt)
            out_dict["sampled_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )
            out_dict["sampled_heading"].append(prev_head.masked_fill(_invalid_mask, 0))
            out_dict["valid_mask"].append(_valid_mask)

            # ! tokenize from sampled rollout state
            # if num_k == 1:  # K=1 means no sampling
            #     out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
            #     out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
            #     out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])
            # else:
            #     # contour: [n_agent, n_token, 4, 2], 2HZ, global coord
            #     token_world_sample = transform_to_global(
            #         pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
            #         head_local=None,
            #         pos_now=prev_pos_sample,  # [n_agent, 2]
            #         head_now=prev_head_sample,  # [n_agent]
            #     )[0].view(*token_traj.shape)
            #
            #     # dist: [n_agent, n_token]
            #     dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
            #     topk_dists, topk_indices = torch.topk(
            #         dist, num_k, dim=-1, largest=False, sorted=False
            #     )  # [n_agent, K]
            #
            #     topk_logits = (-1.0 * topk_dists) / self.agent_token_sampling.temp
            #     _samples = Categorical(logits=topk_logits).sample()  # [n_agent] in K
            #     token_idx_sample = topk_indices[range_a, _samples]
            #     token_contour_sample = token_world_sample[range_a, token_idx_sample]
            #
            #     # udpate prev_pos_sample, prev_head_sample
            #     prev_head_sample = heading[:, i].clone()
            #     dxy = token_contour_sample[:, 0] - token_contour_sample[:, 3]
            #     prev_head_sample[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[
            #         _valid_mask
            #     ]
            #     prev_pos_sample = pos[:, i].clone()
            #     prev_pos_sample[_valid_mask] = token_contour_sample.mean(1)[_valid_mask]
            #     # add to output dict
            #     out_dict["sampled_idx"].append(token_idx_sample)
            #     out_dict["sampled_pos"].append(
            #         prev_pos_sample.masked_fill(_invalid_mask.unsqueeze(1), 0.0)
            #     )
            #     out_dict["sampled_heading"].append(
            #         prev_head_sample.masked_fill(_invalid_mask, 0.0)
            #     )

        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}


        return out_dict

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
            agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(  agent['type'] )

            tokenized_agent["token_agent_shape"] = agent_shape  # [n_token, 2]
            tokenized_agent["token_traj"] = token_traj  # [n_token, 2]
            tokenized_agent["token_traj_all"] = token_traj_all  # [n_token, 6, 4, 2]

            if "col_mask" in agent.keys():
                tokenized_agent["col_mask"] = agent["col_mask"]

            if "pred_mask" in agent.keys():
                tokenized_agent["pred_mask"] = agent["pred_mask"]

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

        tokenized_agent["gt_idx"]=tokenized_agent["sampled_idx"]
        tokenized_agent["gt_pos"]=tokenized_agent["sampled_pos"]
        tokenized_agent["gt_heading"]=tokenized_agent["sampled_heading"]

        return tokenized_map, tokenized_agent

    def traj_to_idx(self, sampled_traj, token_agent_shape, token_traj):

        contour_local = cal_polygon_contour(sampled_traj[..., :2], sampled_traj[...,  2], token_agent_shape[:, None, None])

        sampled_idx =  torch.norm(contour_local - token_traj.unsqueeze(1), dim=-1).mean(-1).argmin(-1)
        #sampled_idx = contour_local.reshape(len(sampled_traj), -1, 8)  # [n_agent, n_step, 3]

        return sampled_idx