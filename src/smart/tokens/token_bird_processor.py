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
from click.core import batch
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
from scipy.optimize import linear_sum_assignment

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

        self.use_time=True

        if self.pred_exit:
            self.n_token_agent+=1

        self.pred_entry=False


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

        tokenized_agent["pred_mask"]=torch.ones_like(tokenized_agent["type"]).to(torch.bool)


        return tokenized_map, tokenized_agent

    def init_agent_token(self, agent_token_path) -> None:

        agent_token = pickle.load(open(agent_token_path, "rb"))

        self.register_buffer(f"agent_token_all", agent_token, persistent=False)

        entry_pos_token = pickle.load(open('./smart/tokens/first1024.pkl', "rb"))

        self.register_buffer(f"entry_pos_token", entry_pos_token, persistent=False)


    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"]  # [n_agent, n_step]
        pos = data["agent"]["position"] # [n_agent, n_step, 2]

        vel=pos[:,1:]-pos[:,:-1]

        heading= wrap_angle(torch.arctan2(vel[:,:,1], vel[:,:,0]))

        pos=pos[:,1:]
        valid=valid[:,1:] & valid[:,:-1]


        token_traj_all=self.agent_token_all[None,:,:].repeat(len(pos),1,1,1)
        token_traj=token_traj_all[:,:,-2:]
        batch=data["agent"]["batch"]

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
        }

        data["agent"]["position"] = pos.to(torch.float16)
        data["agent"]["valid_mask"] = valid

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            token_traj=token_traj,
            batch=batch,#[:,None],
            num_graphs=data.num_graphs
        )
        tokenized_agent.update(token_dict)

        if self.use_time:
            t_list=[]
            for t in data["agent"]["time"]:
                t_list.append(t[0][self.shift :: self.shift])

            t_list=torch.from_numpy(np.stack(t_list)).to(torch.float32).to(batch.device)

            tokenized_agent["abs_time"]=t_list[batch]
        else:
            tokenized_agent["abs_time"]=torch.zeros_like(pos[:0,:,0])

        return tokenized_agent

    def _match_agent_token(
            self,
            valid: Tensor,  # [n_agent, n_step]
            pos: Tensor,  # [n_agent, n_step, 2]
            heading: Tensor,  # [n_agent, n_step]
            token_traj: Tensor,  # [n_agent, n_token, 4, 2]
            batch,
            num_graphs
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
            'token_mask': [],
            'reset_mask':[],
            "entry_idx":[]
            #"entry_mask": [],
            #"entry_idx": [],
        }

        token_xy=token_traj[:,:,:,:2]
        token_z=token_traj[:,:,:,2:]

        entry_pos_token=self.entry_pos_token[None].repeat(len(pos),1,1)#.to(torch.float16)

        entry_token_xy=entry_pos_token[:,:,:2]
        entry_token_z=entry_pos_token[:,:,2]

        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid_mask = ~_valid_mask

            out_dict["token_mask"].append(_valid_mask.clone())

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

            token_contour_gt = token_world_gt[range_a, token_idx_gt]#next_pos
            token_in_valid=min_dist>1
            out_dict["reset_mask"].append(token_in_valid)

            _valid_mask[token_in_valid]=False

            entry_idx = torch.zeros_like(token_idx_gt) + self.n_token_agent - 1


            if self.pred_entry and i > self.shift:
                entry_agent = ~valid[:, i - self.shift] & valid[:, i]
                present_agent = valid[:, i - self.shift]

                # entry_num = torch.bincount(batch[entry_agent, 0], minlength=num_graphs)
                # present_num = torch.bincount(batch[present_agent, 0], minlength=num_graphs)
                #
                # entry_mask = entry_num <= present_num

                # if entry_agent.any() and not entry_mask.all():
                #     modify_batch=torch.arange(num_graphs)[~entry_mask]
                #     batch=
                #     entry_agent[]
                #
                #     print(entry_num,present_num)

                if entry_agent.any() :#and entry_mask.all()

                    entry_pos =pos[:, i][entry_agent]#.to(torch.float16)

                    present_pos =prev_pos[present_agent]#.to(torch.float16)

                    present_heading=prev_head[present_agent]#.to(torch.float16)

                    global_token_xy=transform_to_global(entry_token_xy[:len(present_pos)],None,present_pos[:, :2],present_heading)[0]

                    global_token_z=entry_token_z[:len(present_pos)]+present_pos[:,None,2]

                    global_token_pos=torch.cat((global_token_xy, global_token_z[:,:,None]), dim=-1)#.to(torch.float16)

                    present_batch=batch[present_agent]
                    entry_batch=batch[entry_agent]

                    row_ind=[]
                    col_ind=[]
                    entry_idx_gt=[]

                    for b in range(num_graphs):
                        global_token_pos_i=global_token_pos[present_batch==b]
                        entry_pos_i=entry_pos[entry_batch==b]

                        diff = global_token_pos_i[:, None] - entry_pos_i[None, :, None]  # (Np, Ne, D)

                        cost, min_idx = torch.linalg.norm(diff, dim=-1).min(-1)

                        row_ind_i, col_ind_i = linear_sum_assignment(cost.cpu().numpy())

                        entry_idx_gt_i = min_idx[row_ind_i, col_ind_i]

                        row_ind.append(row_ind_i+torch.sum(present_batch<b).item())
                        col_ind.append(col_ind_i+torch.sum(entry_batch<b).item())
                        entry_idx_gt.append(entry_idx_gt_i)

                    row_ind=np.concatenate(row_ind)
                    col_ind=np.concatenate(col_ind)
                    entry_idx_gt=torch.cat(entry_idx_gt)

                    # entry_idx_gt2=entry_idx_gt[np.argsort(col_ind)]
                    # row_ind2=row_ind[np.argsort(col_ind)]

                    # diff = (global_token_pos+ batch[:,None][present_agent][:,None]*1000)[:,None]- (entry_pos+batch[:,None][entry_agent]*1000 )[None,:,None] # (Np, Ne, D)
                    #
                    # cost,min_idx=torch.linalg.norm(diff, dim=-1).min(-1)
                    #
                    # row_ind1,col_ind1 = linear_sum_assignment(cost.cpu().numpy())
                    #
                    # row_ind=row_ind1[np.argsort(col_ind1)]
                    # col_ind=col_ind1[np.argsort(col_ind1)]
                    #
                    # entry_idx_gt=min_idx[row_ind,col_ind]

                    # present_agent_pos = present_pos[row_ind]  # (M, 3)
                    # present_agent_heading = present_heading[row_ind]  # (M,)

                    # entry_agent_pos = entry_pos[col_ind]  # (M, 3)
                    #
                    # # IMPORTANT: index headings of entries by col_ind (was missing)
                    # #entry_agent_heading = heading[:, i][entry_agent][col_ind]  # (M,)
                    #
                    # # Transform entry positions to present-local frame (one per matched pair)
                    # # transform_to_local expects shapes like ([n, 1, 2], [n,1], [n,2], [n])
                    # local_pos = transform_to_local(
                    #     entry_agent_pos[:, None, :2],  # (M, 1, 2)
                    #     None,  # (M, 1)
                    #     present_agent_pos[:, :2],  # (M, 2)
                    #     present_agent_heading  # (M,)
                    # )[0]
                    #
                    # local_z = entry_agent_pos[:, 2] - present_agent_pos[:, 2]  # (M,)
                    #
                    # local_poses = torch.cat((local_pos[:, 0, :], local_z[:, None]), dim=-1)  # (M, 3)
                    #
                    # # compute distances between these M local_poses and the token bank (T,3)
                    # # self.entry_pos_token shape assumed (T,3)
                    # pose_dist = torch.linalg.norm(local_poses[:, None, :] - self.entry_pos_token[None, :, :],
                    #                               dim=-1)  # (M, T)
                    # entry_idx_gt = torch.argmin(pose_dist, dim=-1)  # (M,) token idx per matched pair

                    # global_token_xy=transform_to_global(entry_token_xy[:len(present_agent_pos)],None,present_agent_pos[:, :2], present_agent_heading)[0]
                    #
                    # global_token_z=entry_token_z[:len(present_agent_pos)]+present_agent_pos[:,None,2]
                    # global_token_pos=torch.cat((global_token_xy, global_token_z[:,:,None]), dim=-1)
                    #
                    #
                    # pose_dist1 = torch.linalg.norm(global_token_pos-entry_agent_pos[:,None], dim=-1)#.mean(-1) #49,1024


                    present_id = torch.nonzero(present_agent, as_tuple=False).squeeze(1)

                    entry_idx[present_id[row_ind]] = entry_idx_gt

                    entry_id = torch.nonzero(entry_agent, as_tuple=False).squeeze(1)[col_ind]

                    prev_head = heading[:, i].clone()
                    prev_pos = pos[:, i].clone()

                    #local_traj=self.entry_pos_token[entry_idx_gt]

                    global_traj=global_token_pos[row_ind][torch.arange(len(entry_idx_gt)),entry_idx_gt]

                    prev_pos[entry_id]=global_traj#.to(torch.float32)

                    # global_xy=transform_to_global(pos_local=local_traj[:,None,:2], head_local=None,pos_now=present_agent_pos[:, :2],head_now=present_agent_heading)[0]
                    #
                    # global_z=local_traj[:,2]+present_agent_pos[:, 2]

                    #prev_pos[entry_id,:2] = global_xy.mean(-2)
                    #prev_pos[entry_id,2] = global_z

                    # dxy = global_xy[:, 0] - global_xy[:, 3]
                    # prev_head[entry_id] = torch.arctan2(dxy[:, 1], dxy[:, 0])

                    # dist=torch.linalg.norm(pos[:,i][entry_id]-prev_pos[entry_id], dim=-1)
                    #
                    # print(dist.mean()) #0.8

                    #real_id=entry_id[dist>1]

                   # prev_pos[real_id]=pos[real_id,i]
                    #prev_head[real_id]=heading[real_id,i]
                    # dist1=torch.linalg.norm(pos[:,i][entry_id]-prev_pos[entry_id], dim=-1)
                    # print(1)

                    # print(torch.linalg.norm(pos[:,i][entry_agent]-prev_pos[entry_agent], dim=-1).mean())

                else:
                    prev_head = heading[:, i].clone()
                    prev_pos = pos[:, i].clone()

            else:
                prev_head = heading[:, i].clone()
                prev_pos = pos[:, i].clone()

            out_dict["entry_idx"].append(entry_idx)

            # udpate prev_pos, prev_head
            dxy = token_contour_gt[:,-1] - token_contour_gt[:,-2]
            prev_head[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid_mask]
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