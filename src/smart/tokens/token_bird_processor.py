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

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
import numpy as np

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
        self.use_dynamic = False

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.n_token_agent = self.agent_token_all.shape[0]

        self.light_type = 5

        self.use_light = False

        self.pred_proposal = False

        if self.pred_proposal:
            self.n_token_agent = 16

        self.interval_t = self.shift / 10

        self.pred_last_res = False
        if self.pred_last_res:
            self.n_token_agent += 1

        self.pred_all_res = False

        if self.pred_all_res:
            self.n_token_agent = self.agent_token_all_veh.shape[0]
            self.pred_last_res = False

        self.pred_goal = False

        self.use_smart = False

        self.use_route = False

        self.noise = True

        self.pred_map_token = False

        self.use_bird=True

        self.use_goal=True

        self.pred_exit=True

        self.use_token=True

        if self.pred_exit:
            self.n_token_agent+=1

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        tokenized_agent = self.tokenize_agent(data)
        # batch_number=torch.amax(tokenized_agent['batch']).item()+1
        #
        # position=torch.zeros([batch_number,3],device=tokenized_agent['batch'].device)
        #
        # position[:,0]=0.6
        # position[:,1]= 14.5
        # position[:,2]=2.6
        #
        # orientation=torch.zeros_like(position[:,0])
        # batch=torch.arange(batch_number,device=tokenized_agent['batch'].device)
        #
        # tokenized_map = {
        #     #"pt_token": pt_token,
        #     "position": position,
        #     "orientation": orientation,
        #     "batch": batch,
        # }
        goal_pos=torch.zeros_like(tokenized_agent["sampled_pos"][:,0])

        goal_pos[:,0]=0.6
        goal_pos[:,1]= 14.5
        goal_pos[:,2]=2.6

        tokenized_agent["goal_pos"] = goal_pos

        tokenized_map={}

        tokenized_agent["light_idx"] = torch.zeros([0, 18])

        # if self.training:
        #     tokenized_agent["goal_mask"] =torch.rand_like(tokenized_agent["type"])<0.5
        # else:
        tokenized_agent["goal_mask"]=torch.ones_like(tokenized_agent["type"]).to(torch.bool)

        if self.pred_exit:
            valid_mask=tokenized_agent["valid_mask"]
            exit_mask=valid_mask[:,:-1] & ~valid_mask[:,1:]
            exit_mask=torch.cat([torch.zeros_like(exit_mask[:,:1]),exit_mask], dim=1)
            tokenized_agent["sampled_idx"][exit_mask]=self.n_token_agent-1


        return tokenized_map, tokenized_agent

    def init_agent_token(self, agent_token_path) -> None:

        agent_token = pickle.load(open(agent_token_path, "rb"))

        self.register_buffer(f"agent_token_all", agent_token, persistent=False)

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"]  # [n_agent, n_step]
        pos = data["agent"]["position"] # [n_agent, n_step, 2]

        vel=pos[:,1:]-pos[:,:-1]

        heading= wrap_angle(torch.arctan2(vel[:,:,1], vel[:,:,0]))

        pos=pos[:,1:]
        valid=valid[:,1:] & valid[:,:-1]

        fut_valid=valid[:,self.shift:: self.shift].any(dim=-1)

        pos=pos[fut_valid]
        valid=valid[fut_valid]
        heading=heading[fut_valid]
        token_traj_all=self.agent_token_all[None,:,:].repeat(len(pos),1,1,1)
        token_traj=token_traj_all[:,:,-2:]
        batch=data["agent"]["batch"][fut_valid]

        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": torch.zeros_like(pos[:,0,0]),
            "shape": None,
            "token_agent_shape":None,
            "batch": batch,
            "token_traj_all": token_traj_all,  # [n_agent, n_token, 6, 4, 2]
            "token_traj": token_traj,  # [n_agent, n_token, 4, 2]
            # for step {5, 10, ..., 90}
             "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
            # "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
             "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
             # "gt_traj_10hz": pos,
             # "gt_head_10hz": heading,
             #  "gt_valid_10hz": valid
        }

        data["agent"]["position"]=pos.to(torch.float16)
        data["agent"]["valid_mask"]=valid

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            token_traj=token_traj,
        )
        tokenized_agent.update(token_dict)


        return tokenized_agent

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    ) -> Dict[str, Tensor]:
        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]

        out_dict = {
            "valid_mask": [],
            "gt_idx": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
            'token_mask': []
        }

        token_xy=token_traj[:,:,:,:2]
        token_z=token_traj[:,:,:,2:]

        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid_mask = ~_valid_mask

            out_dict["token_mask"].append(_valid_mask)

            # out_dict["valid_mask"].append(_valid_mask)

            # #! gt_contour: [n_agent, 4, 2] in global coord
            # gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            # gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            gt_contour=pos[:, i].unsqueeze(1)

            # ! tokenize without sampling
            token_world_xy = transform_to_global(
                pos_local=token_xy.flatten(1, 2),  # [n_agent, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos[:,:2],  # [n_agent, 2]
                head_now=prev_head,  # [n_agent]
            )[0].view(*token_xy.shape)

            token_world_gt_z=prev_pos[:,None,None,2:]+token_z

            token_world_gt=torch.cat((token_world_xy, token_world_gt_z), dim=-1)

            all_dist=torch.norm(token_world_gt[:,:,-1] - gt_contour, dim=-1)

            min_dist, token_idx_gt = torch.min(all_dist , dim=-1)  # [n_agent]

            out_dict["gt_idx"].append(token_idx_gt)

            if self.training and self.noise:
                topk_indices = torch.argsort( all_dist,dim=-1)[:, :self.n_token_agent//400]
                sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
                token_idx_gt = topk_indices[np.arange(topk_indices.shape[0]), sample_topk]
                min_dist = all_dist[np.arange(topk_indices.shape[0]), token_idx_gt]

            # [n_agent, 4, 2]
            token_contour_gt = token_world_gt[range_a, token_idx_gt]#next_pos

            token_valid=min_dist<2
            _valid_mask[~token_valid]=False

            #out_dict["token_mask"].append(_valid_mask)

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:,-1] - token_contour_gt[:,-2]
            prev_head[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid_mask]
            prev_pos = pos[:, i].clone()
            prev_pos[_valid_mask] = token_contour_gt[:,-1][_valid_mask]

            _valid_mask=valid[:, i]

            _invalid_mask = ~valid[:, i]

            out_dict["valid_mask"].append(_valid_mask)

            # add to output dict
            out_dict["sampled_idx"].append(token_idx_gt)
            out_dict["sampled_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )
            out_dict["sampled_heading"].append(prev_head.masked_fill(_invalid_mask, 0))

        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}
        return out_dict