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
rotate_to_local,
infer_prev_pose
)
from src.smart.utils.edge_utils import build_batch
import math

class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
        pred_init=False,
        learn_init=False,
        learn_autoencoder=False
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.pred_init=pred_init
        self.learn_init=learn_init
        self.learn_autoencoder = learn_autoencoder

        self.use_bird=False
        self.use_token=True
        self.use_goal=False
        self.init_map_range=100

        self.use_all_pos=False
        self.pred_all_pos=False

        self.pred_2step=False

        self.traj_diffusion=False
        self.use_gradient_penalty = True

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]
        self.n_token_map = self.map_token_traj_src.shape[0]

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        if not self.training:
            tokenized_map = self.tokenize_map(data)
            tokenized_agent = self.tokenize_agent(data,tokenized_map)

            if self.pred_init:
                self.get_init(tokenized_agent)
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

        return tokenized_map, tokenized_agent

    def get_init(self,tokenized_agent):

        if "ego_mask" not in tokenized_agent:
            batch = tokenized_agent["batch"]
            ego_mask = torch.ones_like(batch)
            ego_mask[:-1] = batch[:-1] != batch[1:]
            tokenized_agent["ego_mask"] = ego_mask.bool()

        ego_mask = tokenized_agent["ego_mask"]
        tokenized_agent["initial_shape"] = tokenized_agent["shape"].clone()

        init_idx = 0

        tokenized_agent["initial_pos"] = tokenized_agent["sampled_pos"][:, init_idx]
        tokenized_agent["initial_heading"] = tokenized_agent["sampled_heading"][:, init_idx]

        token_traj_all=tokenized_agent["token_traj_all"]

        # tokenized_agent["initial_pos"] = tokenized_agent["gt_traj_10hz"][:, init_idx]
        # tokenized_agent["initial_heading"] = tokenized_agent["gt_head_10hz"][:, init_idx]
        # tokenized_agent["ego_traj"] = tokenized_agent["gt_traj_10hz"][:, 1:11, :2][ego_mask]

        ego_idx = tokenized_agent["sampled_idx"][ego_mask]
        ego_token_traj_all = token_traj_all[ego_mask] # .mean(-2)
        #
        # local_ego_traj = ego_token_traj_all[torch.arange(len(ego_idx))[:, None].repeat(1, 2), ego_idx, -1].reshape(
        #     -1, 16)  # ego later 10 steps
        #
        # tokenized_agent["local_ego_traj"] = local_ego_traj

        pos0, head0 = infer_prev_pose(tokenized_agent["sampled_pos"][ego_mask,:1], tokenized_agent["sampled_heading"][ego_mask,:1], ego_idx[:,:1], ego_token_traj_all)

        tokenized_agent["ego_pos2"]=torch.cat([pos0,tokenized_agent["sampled_pos"][ego_mask, :2]],dim=1)
        tokenized_agent["ego_heading2"]=torch.cat([head0,tokenized_agent["sampled_heading"][ego_mask, :2]],dim=1)

        cur_idx = tokenized_agent["sampled_idx"][:, 1].clone()

        prev_idx= tokenized_agent["sampled_idx"][:, 0].clone()

        invalid_mask = ~tokenized_agent["token_mask"][:, 0]

        prev_idx[invalid_mask] = cur_idx[invalid_mask]

        cur_traj = token_traj_all[torch.arange(len(cur_idx), device=cur_idx.device), cur_idx]
        prev_traj = token_traj_all[torch.arange(len(prev_idx), device=prev_idx.device), prev_idx]

        if init_idx==1:
            tokenized_agent["local_vel"] = cur_traj[:, -1].mean(-2) / 0.5

            if self.pred_2step:
                prev_traj = token_traj_all[torch.arange(len(prev_idx), device=prev_idx.device), prev_idx]

                tokenized_agent["prev_vel"] = prev_traj[:, -1].mean(-2) / 0.5

        else:
            tokenized_agent["local_vel"] = prev_traj[:, -1].mean(-2) / 0.5

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

    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:

        traj_pos = data["map_save"]["traj_pos"] # [n_pl, 3, 2]
        traj_theta = data["map_save"]["traj_theta"] # [n_pl]
        type = data["pt_token"]["type"]  # [n_pl]
        #pl_type = data["pt_token"]["pl_type"]  # [n_pl]

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

        return tokenized_map


    def tokenize_agent(self, data: HeteroData,tokenized_map) -> Dict[str, Tensor]:
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

        shape=data["agent"]["shape"].clone()

        # ! prepare output dict
        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"].long(),
            "shape": shape,
            #"ego_mask": ego_mask,  # [n_agent]
            "token_agent_shape":  agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            "token_traj_all": token_traj_all,  # [n_agent, n_token, 6, 4, 2]
            "token_traj": token_traj,  # [n_agent, n_token, 4, 2]
            # for step {5, 10, ..., 90}
           # "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
           # "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
           # "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "gt_traj_10hz": pos,
            "gt_head_10hz": heading,
            "gt_valid_10hz": valid,
            #"pred_mask":pred_mask,
        }


        if self.pred_init and not self.learn_init and self.training and not self.traj_diffusion:
            # std=0.05
            #
            # pd=torch.randn_like(pos[:,5]).clamp(min=-3,max=3)*std*2
            # hd=torch.randn_like(heading[:,5]).clamp(min=-3,max=3)*std
            #
            # pos[:,5]=pos[:,5]+pd
            # heading[:,5]=heading[:,5]+hd
            # shape=shape+torch.randn_like(shape).clamp(min=-3,max=3)*0.1
            #
            # pos[:,0]=pos[:,0]+pd+torch.randn_like(pos[:,0]).clamp(min=-3,max=3)*std
            # heading[:,0]=heading[:,0]+hd+torch.randn_like(heading[:,0]).clamp(min=-3,max=3)*std/2
            #
            # error_dist=10
            # ego_mask = data["agent"]["role"][:, 0].bool()
            #
            # pos, heading, vel, shape, perturb_info = (
            #     self._perturb_initial_context(
            #         pos=pos,
            #         heading=heading,
            #         vel=vel,
            #         shape=shape,
            #         valid=valid,
            #         agent_type=data["agent"]["type"].long(),
            #         ego_mask=ego_mask,
            #         recovery_steps=self.shift * 2,
            #     )
            # )
            #
            # # The previous value 10 is excessively permissive.
            # # The perturbation is temporally coherent, so 1.0–2.0 is sufficient.
            # error_dist = 0.8
            # tokenized_agent["shape"]=shape

            error_dist=0.3
        else:
            error_dist=0.3

        # role_mask = data["agent"]["role"]

        # pred_mask = role_mask[:, 0] | role_mask[:, 2]

        # ego_mask=data["agent"]["role"][:, 0]

        # [n_token, 8]
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] =getattr(self,f"trajectory_token_{k}")

        # ! match token for each agent
        if not self.training:
            # [n_agent]
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][:, 10, 2]

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
            error_dist=error_dist
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
            #'gt_idx':[],
            'token_mask':[],
            #"token_contour":[]
           # 'token_valid':[]
        }


        if not self.training:
            n_step = 11

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

            #out_dict["gt_idx"].append(token_idx_gt)

            # [n_agent, 4, 2]
            token_contour_gt = token_world_gt[range_a, token_idx_gt]

           # out_dict["token_contour"].append(token_contour_gt)
            if self.traj_diffusion:
                error_dist=1
            token_valid=min_dist<error_dist
            _valid_mask[~token_valid]=False

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            prev_pos = pos[:, i].clone()

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

        return out_dict

    def _match_agent_token_reverse(
            self,
            valid: Tensor,  # [n_agent, n_step]
            pos: Tensor,  # [n_agent, n_step, 2]
            heading: Tensor,  # [n_agent, n_step]
            agent_shape: Tensor,  # [n_agent, 2]
            token_traj: Tensor,  # [n_agent, n_token, 4, 2], FORWARD token library
            shift=5,
            error_dist=0.3,
    ) -> Dict[str, Tensor]:

        n_agent, n_step = valid.shape
        device = valid.device
        range_a = torch.arange(n_agent, device=device)

        if not self.training:
            n_step = 11

        last_idx = n_step - 1

        # Current reverse anchor starts from the last GT point.
        cur_pos = pos[:, last_idx].clone()
        cur_head = heading[:, last_idx].clone()

        out_dict = {
            "valid_mask": [],
            "sampled_idx": [],
            "sampled_pos": [cur_pos],
            "sampled_heading": [cur_head],
            "token_mask": [],
        }

        for i in range(last_idx - shift, -1, -shift):
            next_i = i + shift

            # Segment validity: t_i -> t_{i+shift}
            _valid_mask = (valid[:, i] & valid[:, next_i]).clone()

            out_dict["token_mask"].append(_valid_mask.clone())

            # Current anchor contour at t_{i+shift}.
            # This may be GT for the first step, and quantized for later steps.
            cur_contour = cal_polygon_contour(
                cur_pos,
                cur_head,
                agent_shape,
            ).unsqueeze(1)  # [n_agent, 1, 4, 2]

            # Put the FORWARD token library at the candidate previous GT pose t_i.
            # Each token predicts a future contour at t_{i+shift}.
            token_world = transform_to_global(
                pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token * 4, 2]
                head_local=None,
                pos_now=pos[:, i],  # previous pose
                head_now=heading[:, i],
            )[0].view(*token_traj.shape)  # [n_agent, n_token, 4, 2]

            # Match predicted future contour to current reverse anchor contour.
            all_dist = torch.norm(
                token_world - cur_contour,
                dim=-1,
            ).sum(-1)  # [n_agent, n_token]

            min_dist, token_idx_gt = torch.min(all_dist, dim=-1)  # [n_agent]

            if self.traj_diffusion:
                error_dist = 1

            token_valid = min_dist < error_dist
            _valid_mask = _valid_mask & token_valid

            # Selected local forward token contour.
            # This is expressed in the local frame of t_i.
            token_local_gt = token_traj[range_a, token_idx_gt]  # [n_agent, 4, 2]

            # Recover local forward displacement encoded by the token.
            # token center in previous-frame coordinates
            local_next_pos = token_local_gt.mean(1)  # [n_agent, 2]

            # token endpoint heading relative to previous heading
            dxy_local = token_local_gt[:, 0] - token_local_gt[:, 3]
            local_next_head = torch.atan2(dxy_local[:, 1], dxy_local[:, 0])  # [n_agent]

            # Invert the forward token:
            #
            # cur_head = prev_head + local_next_head
            # cur_pos  = prev_pos + R(prev_head) @ local_next_pos
            #
            # therefore:
            # prev_head = cur_head - local_next_head
            # prev_pos  = cur_pos - R(prev_head) @ local_next_pos
            recovered_prev_head = cur_head - local_next_head

            cos_h = torch.cos(recovered_prev_head)
            sin_h = torch.sin(recovered_prev_head)

            rot_local_x = cos_h * local_next_pos[:, 0] - sin_h * local_next_pos[:, 1]
            rot_local_y = sin_h * local_next_pos[:, 0] + cos_h * local_next_pos[:, 1]

            recovered_prev_pos = cur_pos - torch.stack(
                [rot_local_x, rot_local_y],
                dim=-1,
            )

            # Default fallback uses GT previous pose.
            next_cur_pos = pos[:, i].clone()
            next_cur_head = heading[:, i].clone()

            # For valid matched agents, use quantized recovered previous pose.
            next_cur_pos[_valid_mask] = recovered_prev_pos[_valid_mask]
            next_cur_head[_valid_mask] = recovered_prev_head[_valid_mask]

            cur_pos = next_cur_pos
            cur_head = next_cur_head

            _invalid_mask = ~valid[:, i]

            out_dict["sampled_idx"].append(token_idx_gt)

            out_dict["sampled_pos"].append(
                cur_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )

            out_dict["sampled_heading"].append(
                cur_head.masked_fill(_invalid_mask, 0)
            )

            out_dict["valid_mask"].append(valid[:, i])

        out_dict = {k: torch.stack(v, dim=1).flip(1)  for k, v in out_dict.items()}

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
                token_idx = torch.argmin(dist, dim=-1)
                tokenized_map["token_idx"] = token_idx

                tokenized_map["traj_pos_local"] = map["traj_pos_local"]

            for key in ["position", "orientation", "batch", "type"]:#, "pl_type", "light_type"
                tokenized_map[key] = map[key]

            if "light_type" in data.keys():
                tokenized_map["light_type"] = map["light_type"]


        if len(agent) == 0:
            if self.learn_init:
                agent = data["agent"]

                valid_mask = agent["valid_mask"]  # [n_agent, n_step]
                heading = agent["heading"]  ## [n_agent, n_step]
                pos = agent["position"][..., :2].contiguous()  # # [n_agent, n_step, 2]
                vel = agent["velocity"]  ## [n_agent, n_step, 2]
                shape = agent["shape"]
                type = agent["type"]

                batch = agent["batch"]

                ego_mask = torch.ones_like(batch)
                ego_mask[:-1] = batch[:-1] != batch[1:]
                ego_mask=ego_mask.bool()

                tokenized_agent["batch_a"]=batch[~ego_mask]

                if self.use_all_pos or self.pred_all_pos:
                    init_idx = 10

                    tokenized_agent["initial_heading"] = heading[:, init_idx]#[valid]
                    tokenized_agent["initial_pos"] = pos[:, init_idx]#[valid]  # [valid]
                    tokenized_agent["initial_shape"] = shape#[valid]
                    tokenized_agent["initial_vel"]= vel[:, init_idx]
                    tokenized_agent["type"] = type.long()#[valid]
                    tokenized_agent["batch"] = batch#[valid]

                    tokenized_agent["all_pos"] = pos[:,:10]#torch.cat((pos,pos[:,:4]),dim=1)#[valid]
                    tokenized_agent["all_heading"] = heading[:,:10]#torch.cat((heading,heading[:,:4]),dim=1)#[valid]
                    tokenized_agent["valid_mask"] = valid_mask[:,:10] #torch.cat((valid_mask,torch.zeros_like(valid_mask)[:,:4]),dim=1)#[valid]
                    ego_traj = pos[ego_mask, :11].contiguous()
                    ego_head = heading[ego_mask, :11].contiguous()

                    tokenized_agent["ego_pos2"] = ego_traj[:,::5]
                    tokenized_agent["ego_heading2"] = ego_head[:,::5]

                    #tokenized_agent["ego_traj"] = ego_traj
                    tokenized_agent["all_pos"][~tokenized_agent["valid_mask"]] = torch.nan
                    tokenized_agent["all_heading"][~tokenized_agent["valid_mask"]] = torch.nan

                else:

                    batch = torch.stack(
                        [
                            batch + tokenized_agent["num_graphs"] * t
                            for t in range(81)
                        ],
                        dim=1,
                    ).transpose(0,1)  # [n_agent*n_step]

                    valid=valid_mask[:, :-10].transpose(0,1)

                    tokenized_agent["initial_heading"] = heading[:, :-10].transpose(0,1)[valid]
                    tokenized_agent["initial_pos"] = pos[:, :-10].transpose(0,1)[valid]
                    tokenized_agent["initial_shape"] = shape[:,None].repeat(1,81,1).transpose(0,1)[valid]
                    tokenized_agent["initial_type"] = type[:,None].repeat(1,81).transpose(0,1)[valid].long()
                    tokenized_agent["batch"] = batch[valid]
                    tokenized_agent["initial_vel"] = vel[:, :-10].transpose(0,1)[valid]
                    tokenized_agent["ego_mask"] = ego_mask[:,None].repeat(1,81).transpose(0,1)[valid]
                    tokenized_agent["ego_traj"] = pos[ego_mask][:,1:].unfold(dimension=1, size=10, step=1).transpose(0,1)

                    tokenized_agent["num_graphs"]=data.num_graphs*81
                    tokenized_agent["non_ego_valid"] =valid[:,~ego_mask]
            else:
               # self.eval()
                tokenized_agent=self.tokenize_agent(data,tokenized_map)

                # if self.pred_init:
                #     self.get_init(tokenized_agent)

              #  self.train()

        else:
            if  "initial_pos" in agent.keys():

                for key in ["initial_heading", "initial_pos", "initial_shape", "batch","type"]:
                    tokenized_agent[key] = agent[key]  # [agent_mask]

                if "initial_vel" in agent.keys():
                    tokenized_agent["initial_vel"] = agent["initial_vel"]
                else:
                    # tokenized_agent["initial_vel"] = (agent["initial_pos"] - agent["prev_pos"]) / 0.5
                    #
                    # tokenized_agent["prev_heading"] = agent["prev_heading"]
                    batch=tokenized_agent["batch"]
                    ego_mask = torch.ones_like(batch)
                    ego_mask[:-1] = batch[:-1] != batch[1:]
                    ego_mask = ego_mask.bool()

                    tokenized_agent["local_vel"] = agent["local_vel"]
                    tokenized_agent["ego_pos2"] = agent["ego_pos2"][ego_mask]
                    tokenized_agent["ego_heading2"] = agent["ego_heading2"][ego_mask]
                    tokenized_agent['type'] = agent['type'].long()


                # tokenized_agent['type'] = agent['initial_type'].long()
               # tokenized_agent['shape'] = tokenized_agent['initial_shape']

                if "sampled_pos" in agent.keys():
                    for key in ["sampled_pos", "sampled_heading"]:
                        tokenized_agent[key] = agent[key][:,None]
                    tokenized_agent["sampled_idx"] =agent["sampled_idx"].long()
                    agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(tokenized_agent['type'])

                    tokenized_agent["token_traj_all"] = token_traj_all  # [n_token, 6, 4, 2]

                    self.get_init(tokenized_agent)
            else:
                agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(  agent['type'] )

                tokenized_agent["token_agent_shape"] = agent_shape  # [n_token, 2]
                tokenized_agent["token_traj"] = token_traj  # [n_token, 2]
                tokenized_agent["token_traj_all"] = token_traj_all  # [n_token, 6, 4, 2]

                if "col_mask" in agent.keys():
                    tokenized_agent["col_mask"] = agent["col_mask"]


                if not self.pred_init:
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

                else:
                    for key in ["sampled_pos", "sampled_heading", "type", "batch", "shape", "valid_mask","token_mask"]:
                        tokenized_agent[key] = agent[key]#[agent_mask]

                    tokenized_agent["sampled_idx"]=agent["sampled_idx"].long()#[agent_mask]
                    tokenized_agent['type'] = tokenized_agent['type'].long()

                    if self.pred_init:
                        self.get_init(tokenized_agent)

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

        return tokenized_map, tokenized_agent#1030



    @torch.no_grad()
    def _perturb_initial_context(
        self,
        pos: Tensor,
        heading: Tensor,
        vel: Tensor,
        shape: Tensor,
        valid: Tensor,
        agent_type: Tensor,
        ego_mask,
        recovery_steps,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Apply physically coherent initial-state perturbation.

        Args:
            pos:
                Global positions with shape [N, T, 2].
            heading:
                Global headings with shape [N, T].
            vel:
                Global velocities with shape [N, T, 2].
            shape:
                Agent shape, typically [N, 3] or [N, T, 3].
            valid:
                Valid mask with shape [N, T].
            agent_type:
                Agent type with shape [N]. Expected:
                    0: vehicle
                    1: pedestrian
                    2: cyclist
            ego_mask:
                Optional ego mask with shape [N]. Ego is kept unchanged.
            recovery_steps:
                Number of future frames over which the perturbation smoothly
                decays to zero. Defaults to two token intervals.

        Returns:
            Perturbed pos, heading, vel, shape, and perturbation diagnostics.
        """
        if pos.ndim != 3 or pos.shape[-1] != 2:
            raise ValueError(f"Expected pos [N,T,2], got {tuple(pos.shape)}")
        if heading.shape != pos.shape[:2]:
            raise ValueError(
                f"heading must match pos[:2], got heading={tuple(heading.shape)}, "
                f"pos={tuple(pos.shape)}"
            )
        if vel.shape != pos.shape:
            raise ValueError(
                f"vel must have the same shape as pos, got {tuple(vel.shape)}"
            )
        if valid.shape != pos.shape[:2]:
            raise ValueError(
                f"valid must match pos[:2], got {tuple(valid.shape)}"
            )

        num_agents, num_steps, _ = pos.shape
        device = pos.device
        dtype = pos.dtype

        # In the original code, frame self.shift is the perturbed current state.
        anchor_idx = min(int(self.shift), num_steps - 1)

        if recovery_steps is None:
            # For shift=5 at 10 Hz, this gives about 1 second of recovery.
            recovery_steps = max(int(self.shift) * 2, 1)

        # ------------------------------------------------------------------
        # 1. Sample augmentation severity.
        #
        # Keep 20% clean samples to avoid creating a train-test mismatch.
        # ------------------------------------------------------------------
        # severity_prob = torch.tensor(
        #     [0.30, 0.50, 0.18, 0.02],
        #     device=device,
        #     dtype=dtype,
        # )
        # severity_values = torch.tensor(
        #     [0.0, 0.5, 1.0, 1.5],
        #     device=device,
        #     dtype=dtype,
        # )
        severity_prob = torch.tensor(
            [0.20, 0.45, 0.30, 0.05],
            device=device,
            dtype=dtype,
        )
        severity_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.8],
            device=device,
            dtype=dtype,
        )

        severity_idx = torch.multinomial(
            severity_prob.expand(num_agents, -1),
            num_samples=1,
        ).squeeze(-1)

        severity = severity_values[severity_idx]

        anchor_valid = valid[:, anchor_idx]
        severity = severity * anchor_valid.to(dtype)

        # Unknown/padded types are not perturbed.
        known_type = (agent_type >= 0) & (agent_type < 3)
        severity = severity * known_type.to(dtype)

        if ego_mask is not None:
            severity = severity.masked_fill(ego_mask.bool(), 0.0)

        safe_type = agent_type.clamp(min=0, max=2).long()

        # ------------------------------------------------------------------
        # 2. Type-specific perturbation scales.
        #
        # Columns:
        #   longitudinal position [m]
        #   lateral position [m]
        #   heading [rad]
        #   relative speed scale
        #   log shape scale
        # ------------------------------------------------------------------
        sigma_table = pos.new_tensor(
            [
                # longitudinal, lateral, heading, speed, shape
                [0.35, 0.18, math.radians(3.0), 0.05, 0.020],  # vehicle
                [0.18, 0.18, math.radians(8.0), 0.10, 0.015],  # pedestrian
                [0.25, 0.18, math.radians(5.0), 0.08, 0.020],  # cyclist
            ]
        )
        # sigma_table = pos.new_tensor(
        #     [
        #         # longitudinal [m], lateral [m], heading [rad],
        #         # relative speed scale, log shape scale
        #
        #         [0.15, 0.08, math.radians(1.5), 0.025, 0.008],  # vehicle
        #         [0.10, 0.10, math.radians(4.0), 0.050, 0.008],  # pedestrian
        #         [0.13, 0.10, math.radians(3.0), 0.040, 0.008],  # cyclist
        #     ],
        #     dtype=pos.dtype,
        #     device=pos.device,
        # )
        sigma = sigma_table[safe_type] * severity[:, None]

        random_noise = torch.randn(
            num_agents,
            5,
            device=device,
            dtype=dtype,
        )

        delta_long = (random_noise[:, 0] * sigma[:, 0]).clamp(
            min=-1.5,
            max=1.5,
        )
        delta_lat = (random_noise[:, 1] * sigma[:, 1]).clamp(
            min=-0.8,
            max=0.8,
        )
        delta_heading = (random_noise[:, 2] * sigma[:, 2]).clamp(
            min=-math.radians(15.0),
            max=math.radians(15.0),
        )

        speed_scale = (
            1.0 + random_noise[:, 3] * sigma[:, 3]
        ).clamp(min=0.75, max=1.25)

        # ------------------------------------------------------------------
        # 3. Convert longitudinal/lateral noise to global coordinates.
        # ------------------------------------------------------------------
        anchor_heading = heading[:, anchor_idx]

        forward = torch.stack(
            [anchor_heading.cos(), anchor_heading.sin()],
            dim=-1,
        )
        left = torch.stack(
            [-anchor_heading.sin(), anchor_heading.cos()],
            dim=-1,
        )

        delta_xy = (
            delta_long[:, None] * forward
            + delta_lat[:, None] * left
        )

        # ------------------------------------------------------------------
        # 4. Build temporal envelope.
        #
        # History:
        #   full coherent perturbation.
        #
        # Future:
        #   smooth decay to the original trajectory.
        #
        # This creates a plausible recovery target instead of an abrupt jump.
        # ------------------------------------------------------------------
        envelope = torch.ones(num_steps, device=device, dtype=dtype)

        recovery_end = min(
            anchor_idx + int(recovery_steps),
            num_steps - 1,
        )

        if recovery_end > anchor_idx:
            num_recovery = recovery_end - anchor_idx

            u = torch.linspace(
                0.0,
                1.0,
                num_recovery + 1,
                device=device,
                dtype=dtype,
            )[1:]

            # SmoothStep from 1 to 0.
            smooth = u.square() * (3.0 - 2.0 * u)
            envelope[anchor_idx + 1: recovery_end + 1] = 1.0 - smooth

        if recovery_end + 1 < num_steps:
            envelope[recovery_end + 1:] = 0.0

        envelope_xy = envelope[None, :, None]
        envelope_angle = envelope[None, :]

        # ------------------------------------------------------------------
        # 5. Coherently transform position.
        #
        # Rotate and scale the trajectory around the original anchor position,
        # then translate the anchor.
        # ------------------------------------------------------------------
        anchor_pos = pos[:, anchor_idx: anchor_idx + 1]
        relative_pos = pos - anchor_pos

        time_angle = delta_heading[:, None] * envelope_angle
        cos_angle = time_angle.cos()
        sin_angle = time_angle.sin()

        relative_x = relative_pos[..., 0]
        relative_y = relative_pos[..., 1]

        rotated_relative = torch.stack(
            [
                cos_angle * relative_x - sin_angle * relative_y,
                sin_angle * relative_x + cos_angle * relative_y,
            ],
            dim=-1,
        )

        time_speed_scale = (
            1.0
            + (speed_scale[:, None] - 1.0) * envelope_angle
        )

        candidate_pos = (
            anchor_pos
            + delta_xy[:, None] * envelope_xy
            + rotated_relative * time_speed_scale[..., None]
        )

        # ------------------------------------------------------------------
        # 6. Heading and velocity use the same rotation and speed scale.
        # ------------------------------------------------------------------
        candidate_heading = heading + time_angle
        candidate_heading = torch.atan2(
            candidate_heading.sin(),
            candidate_heading.cos(),
        )

        vel_x = vel[..., 0]
        vel_y = vel[..., 1]

        candidate_vel = torch.stack(
            [
                cos_angle * vel_x - sin_angle * vel_y,
                sin_angle * vel_x + cos_angle * vel_y,
            ],
            dim=-1,
        )
        candidate_vel = candidate_vel * time_speed_scale[..., None]

        # Do not modify invalid frames.
        valid_xy = valid[..., None]

        pos = torch.where(valid_xy, candidate_pos, pos)
        heading = torch.where(valid, candidate_heading, heading)
        vel = torch.where(valid_xy, candidate_vel, vel)

        # ------------------------------------------------------------------
        # 7. Shape uses small multiplicative noise.
        #
        # Multiplicative perturbation guarantees positive dimensions.
        # ------------------------------------------------------------------
        shape_noise = torch.randn(
            num_agents,
            2,
            device=device,
            dtype=shape.dtype,
        )

        shape_sigma = sigma[:, 4:5].to(shape.dtype)

        shape_scale = torch.exp(shape_noise * shape_sigma).clamp(
            min=0.90,
            max=1.10,
        )

        if shape.ndim == 2:
            shape[:, :2] = (
                shape[:, :2] * shape_scale
            ).clamp_min(0.1)
        else:
            scale_shape = [
                num_agents,
                *([1] * (shape.ndim - 2)),
                2,
            ]
            shape[..., :2] = (
                shape[..., :2] * shape_scale.view(*scale_shape)
            ).clamp_min(0.1)

        perturb_info = {
            "severity": severity,
            "delta_xy": delta_xy,
            "delta_heading": delta_heading,
            "speed_scale": speed_scale,
            "temporal_envelope": envelope,
        }

        return pos, heading, vel, shape, perturb_info