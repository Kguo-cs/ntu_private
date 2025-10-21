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
from src.route_process import process_route

class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.use_dynamic=False

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]
        self.n_token_map = self.map_token_traj_src.shape[0]

        self.light_type=5

        self.use_light=False

        self.pred_proposal=False

        if self.pred_proposal:
            self.n_token_agent=16

        self.interval_t=self.shift /10

        self.pred_last_res= False
        if self.pred_last_res:
            self.n_token_agent+=1
            
        self.pred_all_res = False

        if self.pred_all_res:
            self.n_token_agent=self.agent_token_all_veh.shape[0]
            self.pred_last_res=False

        self.pred_goal=False

        self.use_smart=False

        self.use_route=False

        self.noise=True

        self.pred_map_token=False

    @torch.no_grad()
    def forward(self, data: HeteroData,extrapolate=True) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        if not self.training:
            tokenized_map = self.tokenize_map(data)

            tokenized_agent = self.tokenize_agent(data,extrapolate)

            if self.use_light:
                light=data["light"]
                light_idx=light["light_idx"].long()
                tokenized_agent["light_idx"]=light_idx
                tokenized_agent["batch_lg"]=light["batch"]
                tokenized_agent["pos_lg"] = light["light_pos"]
                tokenized_agent["orient_lg"] = light["light_orient"]
            else:
                tokenized_agent["light_idx"]=torch.zeros([0,18])
        else:
            tokenized_map, tokenized_agent=self.process_data(data)

            if self.use_light:

                lengths_lg = torch.bincount(tokenized_agent["batch_lg"], minlength=tokenized_agent["num_graphs"]).tolist()

                tokenized_agent["lengths_lg"] = lengths_lg
                tokenized_agent["pad_pos_lg"] = padding(tokenized_agent["pos_lg"], lengths_lg)
                tokenized_agent["pad_orient_lg"] = padding(tokenized_agent["orient_lg"], lengths_lg)

            if self.pred_goal:
                sampled_pos=tokenized_agent["sampled_pos"]
                valid_mask=tokenized_agent["valid_mask"]

                goal_pos=sampled_pos[:,-1]
                goal_valid=valid_mask[:,-1:] & valid_mask

                goal_valid[:,:2]=False

                goal_vector=goal_pos[:,None]-sampled_pos
                head_a=tokenized_agent["sampled_heading"]

                head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

                goal_heading=angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=goal_vector
                )

                goal_idx=(goal_heading+np.pi)//(np.pi/5) #10 directory

                goal_idx[~goal_valid]=10

                tokenized_agent["goal_idx"]=goal_idx.to(torch.long)
            else:
                tokenized_agent["goal_idx"]=torch.zeros([0,18])

        # if self.training:
        #     batch_idx=tokenized_agent['batch']
        #
        #     token_mask=tokenized_agent['token_mask']
        #
        #     rand_idx = torch.randint(low=0, high=2, size=(max(batch_idx) + 1,1), device=batch_idx.device)
        #
        #     rand_mask=rand_idx[batch_idx]<1
        #
        #     token_mask[rand_mask[:,0],:2]=False
        #
        #     tokenized_agent['token_mask']=token_mask

        if self.use_route and self.training:
            batch=tokenized_agent['batch']
            keep_mask = torch.rand(len(batch), device=batch.device)<0.5

            tokenized_agent['route_map_index'][keep_mask]=-2

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

        if self.use_dynamic:
            module_dir = os.path.dirname(__file__)
            codebook=torch.load(os.path.join(module_dir, "codebook.pt"))

            self.register_buffer(f"agent_token_all_veh", codebook[0,:,None,None], persistent=False)
            self.register_buffer(f"agent_token_all_ped", codebook[1,:,None,None], persistent=False)
            self.register_buffer(f"agent_token_all_cyc", codebook[2,:,None,None], persistent=False)

        self.register_buffer(f"trajectory_token_veh", self.agent_token_all_veh[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_ped", self.agent_token_all_ped[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_cyc", self.agent_token_all_cyc[:, -1].flatten(1, 2), persistent=False)

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
            "id": data["agent"]["id"],
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

        speed=torch.norm(vel,dim=-1)

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
            speed=speed
        )
        if "route_map_index" in data["agent"].keys():
            tokenized_agent['route_map_index']=data["agent"]["route_map_index"]

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

        if self.pred_proposal:
            return self.my_match_agent_token(valid, pos, heading,agent_shape, token_traj)

        if self.use_dynamic:
            return self.dynamic_match(valid, pos, speed, heading,agent_shape, token_traj)

        #num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]
        #prev_pos_sample, prev_head_sample = pos[:, 0], heading[:, 0]

        out_dict = {
            "valid_mask": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
            'gt_idx':[],
            'token_mask':[]
           # 'token_valid':[]
        }

        for i in range(shift, n_step, shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - shift] & valid[:, i]  # [n_agent]

            out_dict["token_mask"].append(_valid_mask)

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

            #if  self.pred_last_res:
            token_valid=min_dist<error_dist
            # token_idx_gt[~token_valid]=self.agent_token_all_veh.shape[0]
            # _valid_mask=token_valid & _valid_mask
            _valid_mask[~token_valid]=False

            if self.pred_all_res and self.max_diff is not None:
                token_local_traj= self.token_local_traj[torch.arange(n_agent), token_idx_gt][:,-1:]  # [n_agent, 5,3]

                diff=self.token_diff[torch.arange(n_agent), token_idx_gt][:,-1:]  # [n_agent, 5, 3]

                local_pos,local_heading=transform_to_local(
                    pos_global=pos[:, i:i+1],  # [n_agent, 5, 2],
                    head_global=heading[:, i:i+1],  # [n_agent, 5]
                    pos_now=prev_pos,  # [n_agent, 2]
                    head_now=prev_head,  # [n_agent]
                )

                local_traj=torch.cat([local_pos, local_heading[:, :, None]], dim=-1)  # [n_agent, 5, 3]

                real_diff=local_traj-token_local_traj  # [n_agent, 5, 3]

                token_diff=torch.minimum(real_diff,diff)
                token_diff=torch.maximum(token_diff,-diff)

                local_token_traj=token_local_traj + token_diff

                global_pos,global_heading = transform_to_global(
                        local_token_traj[:,:,:2],
                        local_token_traj[:,:,2],
                        prev_pos,  # [n_agent, 2]
                        prev_head,  # [n_agent]
                )

                token_contour_gt = cal_polygon_contour(global_pos[:,0], global_heading[:,0], agent_shape)  # [n_agent, 4, 2]

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
            next_head=torch.arctan2(dxy[:, 1], dxy[:, 0])

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

            #if self.pred_last_res:
            _valid_mask=valid[:, i]

            _invalid_mask = ~valid[:, i]

            #out_dict["token_valid"].append(token_valid &_valid_mask)

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


        def get_future_30_every_5th_step_with_padding(tensor, pad_value=0.0):
            B, T, D = tensor.shape
            max_future = 30

            # Pad extra steps to the right to safely index up to t+30
            padded_tensor = F.pad(tensor, (0, 0, 0, max_future), value=pad_value)  # shape: (B, T+30, D)

            # Start indices every 5 steps
            starts = torch.arange(0, T, 5, device=tensor.device)  # (T//5,)

            # For each start t, get future steps t+1 to t+30 (exclude t)
            offsets = torch.tensor([1,2,3,4,5],device=tensor.device)#,10,15,20,25,30torch.arange(1, max_future + 1, device=tensor.device)  # (30,)
            indices = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (T//5, 30)
            gathered = padded_tensor[:, indices]  # (B, T//5, 30, D)

            return gathered

        if self.pred_last_res or self.pred_all_res:

            gt_traj = torch.cat([pos, heading[:, :, None]], dim=-1)

            gt_traj[~valid] = 0

            valid_mask = out_dict["valid_mask"]  #valid[:,::5]# current position, heading valid

            target_global_traj = get_future_30_every_5th_step_with_padding(gt_traj)  # shape: (B, T//5, 30, 2)
            out_dict["target_global_traj"] =target_global_traj[:,1:]
            target_mask = target_global_traj.any(-1) != 0
            out_dict["target_mask"] = target_mask[:, 1:]  & valid_mask[:,:,None] #.all(-2,keepdim=True)  # [n_agent, n_step=18, 30]

            if self.pred_last_res:
                token_mask=out_dict["sampled_idx"]==self.agent_token_all_veh.shape[0]
                out_dict["target_mask"] = out_dict["target_mask"] & token_mask[:, :, None]

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

        token_traj = token_traj_all[:, :, -1, :, :].contiguous()

        if self.pred_all_res:
            self.token_local_traj = torch.index_select(self.all_token_local_traj, dim=0,
                                                       index=agent_type.long())

            if self.max_diff is not None:
                self.token_diff = torch.index_select(self.max_diff, dim=0, index=agent_type.long())

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

        #agent_type = agent["type"]

        #agent_mask = agent_type < 3

        agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(
            agent['type']#[agent_mask]
        )

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

            # token_dict = self._match_agent_token(agent["gt_valid_raw"], agent["gt_pos_raw"],
            #                                     agent["gt_head_raw"],
            #                                     agent_shape, token_traj,
            #                                     speed,
            #                                     error_dist=0.3
            #                                         )
            #
            # tokenized_agent["expert_sampled_pos"]=token_dict["sampled_pos"]
            # tokenized_agent["expert_sampled_heading"]=token_dict["sampled_heading"]

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

            if self.pred_last_res:
                for key in ["target_global_traj","target_mask"]:
                    tokenized_agent[key] = agent[key]

            # valid_mask = agent["valid_mask"]
            #
            # valid_mask[:,1:]=valid_mask[:,:-1] & valid_mask[:,1:]
            #
            # tokenized_agent['valid_mask']=valid_mask
        if "route_map_index" in agent.keys():
            tokenized_agent['route_map_index']=agent["route_map_index"]

        # pos = tokenized_agent["sampled_pos"]  # [N,T,2]
        # valid = tokenized_agent["valid_mask"]  # [N,T]
        #
        # # consecutive displacements
        # disp = pos[:, 1:] - pos[:, :-1]  # [N,T-1,2]
        # dist = torch.norm(disp, dim=-1)  # [N,T-1]
        #
        # # compute speeds per step
        # speed = dist / 0.5  # [N,T-1]
        #
        # # valid timesteps = both ends must be valid
        # valid_step = valid[:, 1:] & valid[:, :-1]
        #
        # # mask invalid speeds
        # speed = speed * valid_step
        #
        # # mean per agent
        # sum_speed = speed.sum(dim=1)
        # count = valid_step.sum(dim=1).clamp(min=1)  # avoid div by 0
        # mean_speed = sum_speed / count
        #
        # tokenized_agent["mean_speed"]=mean_speed

        if self.use_light:

            tokenized_light = data["tokenized_light"]

            light_idx = tokenized_light["light_idx"]
            tokenized_agent["light_idx"] = light_idx.long()#[shuffle_Id]
            tokenized_agent["batch_lg"] = tokenized_light["batch"]
            tokenized_agent["pos_lg"] = tokenized_light["pos_lg"]#[shuffle_Id]
            tokenized_agent["orient_lg"] = tokenized_light["orient_lg"]#[shuffle_Id]

        else:
            tokenized_agent["light_idx"] = torch.zeros([0, 18])

        tokenized_map["token_traj_src"]= self.map_token_traj_src
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
