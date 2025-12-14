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
from torch_geometric.data import HeteroData

from torch.nn.utils.rnn import pad_sequence
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
import numpy as np
from scipy.optimize import linear_sum_assignment
from .HierarchicalStateTokenizer import HierarchicalStateTokenizer

class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
        use_noise=False,
        pred_entry=False
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.autoregressive_entry=True

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file),os.path.join(module_dir, map_token_file))
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

        self.noise = use_noise

        self.pred_map_token = False

        self.use_bird=True

        self.use_goal=True

        self.pred_exit=True

        self.use_token=True

        self.use_time=True

        if self.pred_exit:
            self.n_token_agent+=1

        self.pred_entry=pred_entry

        self.match_all=False

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:


        if "sampled_idx" in data.keys():

            agent=data["tokenized_agent"]

            tokenized_agent = {
                "num_graphs": data.num_graphs,
                "type": torch.zeros_like(agent["sampled_pos"][:, 0, 0]),
                "shape": None,
                "token_agent_shape": None}

            for key in ["sampled_pos", "sampled_heading", "batch", "valid_mask",'token_mask']:
                tokenized_agent[key] = agent[key]#[agent_mask]

            tokenized_agent["sampled_idx"]=agent["sampled_idx"].long()

            if self.pred_entry:
                if self.autoregressive_entry:
                    entry_idx=[torch.from_numpy(entry_idx).long().permute(1, 0, 2) for entry_idx in agent["entry_idx"]]
                  #  entry_state=[torch.from_numpy(entry_state).permute(1, 0, 2) for entry_state in agent["entry_state"]]
                    tokenized_agent["entry_idx"] =  pad_sequence(entry_idx,batch_first=True,padding_value=self.n_token_entry).permute(2 ,0, 1, 3).flatten(0,1).to(self.agent_token_all.device)
                   # tokenized_agent["entry_state"] =pad_sequence(entry_state,batch_first=True).permute(2 ,0, 1, 3).flatten(0,1).to(self.agent_token_all.device)
                else:
                    tokenized_agent["entry_idx"] = agent["entry_idx"].long()

                    tokenized_agent["entry_head_idx_num"] = agent["entry_head_idx_num"].long()

                    #tokenized_agent["entry_pos_offset"] = agent["entry_pos_offset"]

                    # tokenized_agent["entry_head_idx"]=agent["entry_head_idx"].long()

                    s=torch.split(agent["entry_head_idx"], tokenized_agent["entry_head_idx_num"].tolist())

                    all_t=tokenized_agent["entry_idx"].shape[1]-1
                    b_num=len(s)//all_t
                    new_s=[]

                    for t in range(all_t):
                        for b in range(b_num):
                            new_s.append(s[b*all_t+t])       #t=0 ,b= k

                    tokenized_agent["entry_head_idx"] =torch.cat(new_s).long()

                    s=torch.split(agent["entry_pos_offset"], tokenized_agent["entry_head_idx_num"].tolist())

                    all_t=tokenized_agent["entry_idx"].shape[1]-1
                    b_num=len(s)//all_t
                    new_s=[]

                    for t in range(all_t):
                        for b in range(b_num):
                            new_s.append(s[b*all_t+t])       #t=0 ,b= k

                    tokenized_agent["entry_pos_offset"] =torch.cat(new_s)


            tokenized_agent["token_traj_all"] = self.agent_token_all[:,:,None]#[None, :, :]#.repeat(len(agent["sampled_idx"]), 1, 1, 1)[:,:,:,None]

            fut=torch.arange(0, self.shift*agent["sampled_idx"].shape[1],self.shift,device=agent["sampled_idx"].device)

            tokenized_agent["abs_time"]=agent["abs_time"][:,None]+fut[None,:]

        else:
            tokenized_agent = self.tokenize_agent(data)

        goal_pos=torch.zeros_like(tokenized_agent["sampled_pos"][:,0])

        goal_pos[:,0]=0.6
        goal_pos[:,1]= 14.5
        goal_pos[:,2]=2.6

        tokenized_agent["goal_pos"] = goal_pos

        tokenized_map={}

        tokenized_agent["goal_mask"]=torch.ones_like(tokenized_agent["type"]).to(torch.bool)

        if self.pred_exit:
            valid_mask=tokenized_agent["valid_mask"]
            exit_mask=valid_mask[:,:-1] & ~valid_mask[:,1:]
            exit_mask=torch.cat([torch.zeros_like(exit_mask[:,:1]),exit_mask], dim=1)
            tokenized_agent["sampled_idx"][exit_mask]=self.n_token_agent-1

        tokenized_agent["pred_mask"]=None

        batch=tokenized_agent["batch"].clone()


        if not self.training and self.pred_entry:
            tokenized_agent["pos"] = data["agent"]["position"]
            tokenized_agent["gt_valid_mask"]=data["agent"]["valid_mask"]

            for key, value in tokenized_agent.items():
                if type(value) is torch.Tensor:
                    new_tensor=[]
                    for b in range(data.num_graphs):
                        valueb=value[batch==b]
                        if 'valid_mask' in key:
                            value_repeat=torch.zeros_like(valueb[:1]).repeat_interleave(1000,dim=0)
                        else:
                            value_repeat=valueb[:1].repeat_interleave(1000,dim=0)
                        new_tensor.append(torch.cat([valueb,value_repeat]))
                    tokenized_agent[key]=torch.cat(new_tensor)

            data["agent"]["position"] = tokenized_agent["pos"]
            data["agent"]["valid_mask"] = tokenized_agent["gt_valid_mask"]

        return tokenized_map, tokenized_agent

    def init_agent_token(self, agent_token_path,map_token_path) -> None:

        agent_token = pickle.load(open(agent_token_path, "rb"))

        agent_shape = torch.ones_like(agent_token[:, -1, :2])

        last_vel=agent_token[:,-1]-agent_token[:,-2]

        token_heading =torch.arctan2(last_vel[:,1], last_vel[:,0])

        agent_token_box=cal_polygon_contour(agent_token[:,-1],token_heading,agent_shape)

        agent_token_box=torch.cat([agent_token_box, agent_token[:,-1,None,2:].repeat(1,4,1)],dim=-1)

        self.register_buffer(f"agent_token_all", agent_token, persistent=False)

        self.register_buffer(f"agent_token_box", agent_token_box, persistent=False)

        if self.autoregressive_entry:
            # self.position_only=False
            # self.tokenizer=HierarchicalStateTokenizer(position_only=self.position_only)
            # if self.position_only:
            #     self.n_token_entry = self.tokenizer.base ** 3
            # else:
            #     self.n_token_entry = self.tokenizer.base ** 4
            entry_pos_token = pickle.load(open(map_token_path, "rb"))
            self.register_buffer(f"entry_pos_token", entry_pos_token, persistent=False)
            self.n_token_entry = self.entry_pos_token.shape[0]

        else:
            entry_pos_token = pickle.load(open(map_token_path, "rb"))
            self.register_buffer(f"entry_pos_token", entry_pos_token, persistent=False)
            self.n_token_entry = self.entry_pos_token.shape[0]

        self.n_token_entry_head=64
        self.n_token_entry_head2=self.n_token_entry_head//2

    def decode_head(self,entry_head_idx):
        return (entry_head_idx - self.n_token_entry_head2) / (self.n_token_entry_head2) * torch.pi

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"]  # [n_agent, n_step]
        pos = data["agent"]["position"]  # [n_agent, n_step, 2]

        vel=pos[:,1:]-pos[:,:-1]

        heading= wrap_angle(torch.arctan2(vel[:,:,1], vel[:,:,0]))

        pos=pos[:,1:]
        valid=valid[:,1:] & valid[:,:-1]

        token_traj_all=self.agent_token_all[None,:,:].repeat(len(pos),1,1,1)
        token_traj=self.agent_token_box[None,:,:].repeat(len(pos),1,1,1)
        batch=data["agent"]["batch"]

        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": torch.zeros_like(pos[:,0,0]),
            "shape": None,
            "token_agent_shape":None,
            "batch": batch,
            "token_traj_all": token_traj_all[:,:,:,None],  # [n_agent, n_token, 5, 1, 2]
            #"token_traj": token_traj,  # [n_agent, n_token, 4, 2]
            # for step {5, 10, ..., 90}
            # "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
            # "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
           #  "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
        }

        data["agent"]["position"] = pos#.to(torch.float16)
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

        # sampled_pos = tokenized_agent["sampled_pos"]
        # gt_pos_raw = pos[:, self.shift :: self.shift]
        # valid_mask =tokenized_agent["valid_mask"][:,1:] &  ~tokenized_agent["valid_mask"][:,:-1]
        #
        # max_dist = torch.linalg.norm(sampled_pos - gt_pos_raw, dim=-1)[:,1:][valid_mask]
        #
        # print(max_dist.mean(),max_dist.max())
        #
        # tokenized_agent["max_dist"]=max_dist

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
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]

        out_dict = {
            "valid_mask": [],
           # "gt_idx": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
            'token_mask': [],
        }

        entry_token_invalid_mask=[]
        entry_idx_list=[]
        entry_head_idx_list=[]
        entry_pos_offset_list=[]

        if self.pred_entry and not self.autoregressive_entry :
            out_dict["entry_idx"]=[]
           # out_dict["entry_head_idx"]=[]

        token_xy=token_traj[:,:,:,:2]
        token_z=token_traj[:,:,:,2:]
        agent_shape = torch.ones_like(pos[:, 0, :2])
        batch_num=batch.max()+1

        if self.pred_entry and not self.autoregressive_entry and self.match_all:
            entry_pos_token=self.entry_pos_token[None].repeat(len(pos),1,1)#.to(torch.float16)

            entry_token_xy=entry_pos_token[:,:,:2]
            entry_token_z=entry_pos_token[:,:,2]

        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid_mask = ~_valid_mask

            out_dict["token_mask"].append(_valid_mask.clone())

            gt_contour = cal_polygon_contour(pos[:, i,:2], heading[:, i], agent_shape)

            gt_contour=torch.cat([gt_contour, pos[:, i,None,2:].repeat(1,4,1)], dim=-1).unsqueeze(1)

            token_world_xy = transform_to_global(
                pos_local=token_xy.flatten(1, 2),  # [n_agent, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos[:,:2],  # [n_agent, 2]
                head_now=prev_head,  # [n_agent]
            )[0].view(*token_xy.shape)

            token_world_gt_z=prev_pos[:,None,None,2:]+token_z

            token_world_gt=torch.cat((token_world_xy, token_world_gt_z), dim=-1)

            all_dist=torch.norm(token_world_gt - gt_contour, dim=-1).mean(-1)

            min_dist, token_idx_gt = torch.min(all_dist , dim=-1)  # [n_agent]

           # out_dict["gt_idx"].append(token_idx_gt)

            if self.training and self.noise:
                topk_indices = torch.argsort( all_dist,dim=-1)[:, :self.n_token_agent//400]
                sample_topk = np.random.choice(range(0, topk_indices.shape[1]), topk_indices.shape[0])
                token_idx_gt = topk_indices[np.arange(topk_indices.shape[0]), sample_topk]
                min_dist = all_dist[np.arange(topk_indices.shape[0]), token_idx_gt]

            token_contour_gt = token_world_gt[range_a, token_idx_gt]#next_pos
            token_in_valid=min_dist>1
            #out_dict["reset_mask"].append(token_in_valid)

            _valid_mask[token_in_valid]=False

            if self.pred_entry and not self.autoregressive_entry :
                entry_idx = torch.zeros_like(token_idx_gt) + self.n_token_entry

                #entry_head_idx = torch.zeros_like(token_idx_gt)

            prev_head = heading[:, i].clone()
            prev_pos = pos[:, i].clone()

            if self.pred_entry and i > self.shift :
                entry_agent = ~valid[:, i - self.shift] & valid[:, i]
                present_agent = valid[:, i - self.shift]
                entry_pos = pos[:, i][entry_agent]

                if self.autoregressive_entry:
                    entry_heading = heading[:, i][entry_agent]
                    entry_batch=batch[entry_agent]

                    # sort_idx=torch.argsort(torch.linalg.norm(entry_pos,dim=-1))
                    # large constant to separate batches
                    C = 10000

                    sort_key = entry_batch.float() * C + torch.linalg.norm(entry_pos,dim=-1)#[:, 0]
                    sort_idx = torch.argsort(sort_key)

                    entry_pos = entry_pos[sort_idx]
                    entry_heading=entry_heading[sort_idx]

                    # entry_idx= self.tokenizer(entry_pos,entry_heading)

                    pos_entry_idx=torch.linalg.norm(entry_pos[:,None] - self.entry_pos_token[None], dim=-1).argmin(1)

                    entry_head_idx= (wrap_angle(entry_heading)/np.pi*self.n_token_entry_head2).round().long()+self.n_token_entry_head2

                    entry_head_idx[entry_head_idx==self.n_token_entry_head]=0

                    tokenized_pos = self.entry_pos_token[pos_entry_idx]

                    tokenized_heading =self.decode_head(entry_head_idx)

                    offset_local=torch.cat((entry_pos-tokenized_pos,wrap_angle(entry_heading-tokenized_heading)[:,None]), dim=-1)

                    entry_idx=torch.cat([pos_entry_idx[:,None], entry_head_idx[:,None],offset_local], dim=-1)

                    entry_length = torch.bincount(entry_batch,minlength=batch_num).tolist()

                    entry_idx_list.extend(torch.split(entry_idx,entry_length))


                    # pos_rec, heading_rec = self.tokenizer.decode_tokens_to_state(entry_idx)
                    #
                    # res_pos=entry_pos-pos_rec
                    #
                    # entry_pos_offset_list.append(res_pos)
                    #
                    # prev_pos[entry_id]=pos_rec
                    # prev_head[entry_id]=heading_rec
                    #
                    # dist=torch.linalg.norm(pos[:,i][entry_id]-prev_pos[entry_id], dim=-1)
                    #
                    # entry_token_invalid_mask.append(dist)
                    #
                    # real_id=entry_id[dist>1]
                    #
                    # prev_pos[real_id]=pos[real_id,i]

                elif entry_agent.any():#and entry_mask.all()

                    present_pos = out_dict["sampled_pos"][-1][present_agent]

                    present_heading=out_dict["sampled_heading"][-1][present_agent]

                    if self.match_all:

                        global_token_xy=transform_to_global(entry_token_xy[:len(present_pos)],None,present_pos[:, :2],present_heading)[0]

                        global_token_z=entry_token_z[:len(present_pos)]+present_pos[:,None,2]

                        global_token_pos=torch.cat((global_token_xy, global_token_z[:,:,None]), dim=-1)

                    present_batch=batch[present_agent]
                    entry_batch=batch[entry_agent]

                    row_ind=[]
                    col_ind=[]
                    entry_idx_gt=[]

                    for b in range(num_graphs):
                        entry_pos_i=entry_pos[entry_batch==b]
                        if len(entry_pos_i):

                            if self.match_all:
                                global_token_pos_i=global_token_pos[present_batch==b]
                                diff = global_token_pos_i[:, None] - entry_pos_i[None, :, None]  # (Np, Ne,token, D)
                                cost, min_idx = torch.linalg.norm(diff, dim=-1).min(-1)
                            else:
                                global_token_pos_i=present_pos[present_batch==b]
                                diff = global_token_pos_i[:, None] - entry_pos_i[None]  # (Np, Ne,token, D)
                                cost= torch.linalg.norm(diff, dim=-1)

                            row_ind_i, col_ind_i = linear_sum_assignment(cost.cpu().numpy())

                            row_ind.append(row_ind_i+torch.sum(present_batch<b).item())
                            col_ind.append(col_ind_i+torch.sum(entry_batch<b).item())

                            if self.match_all:

                                entry_idx_gt_i = min_idx[row_ind_i, col_ind_i]
                                entry_idx_gt.append(entry_idx_gt_i)

                    row_ind=np.concatenate(row_ind)
                    col_ind=np.concatenate(col_ind)

                    gt_entry_id=torch.nonzero(entry_agent, as_tuple=False).squeeze(1)

                    entry_id = gt_entry_id[col_ind]
                    entry_pos1=prev_pos[entry_id]

                    select_heading=present_heading[row_ind]
                    select_pos = present_pos[row_ind]

                    entry_heading=prev_head[entry_id]

                    local_xy,local_heading = transform_to_local(
                        entry_pos1[:, None, :2],
                        entry_heading[:, None],
                        select_pos[:, :2],
                        select_heading
                    )

                    entry_head_idx= (wrap_angle(local_heading[:,0])/np.pi*self.n_token_entry_head2).round().long()+self.n_token_entry_head2

                    entry_head_idx[entry_head_idx==self.n_token_entry_head]=0

                    tokenized_heading =self.decode_head(entry_head_idx)

                    head_offset=wrap_angle(entry_heading-tokenized_heading-select_heading)         #for selecting heading, not for

                    #entry_head_idx = entry_head_idx[row_ind.argsort()]
                    # if not np.all((row_ind[1:]-row_ind[:-1])>0):
                    #     print(np.all((row_ind[1:]-row_ind[:-1])>0))

                    entry_head_idx_list.append(entry_head_idx)

                    local_z=entry_pos1[:,2:]-select_pos[:,2:]

                    local_pos=torch.cat([local_xy[:, 0] , local_z], dim=-1)

                    if self.match_all:
                        entry_idx_gt=torch.cat(entry_idx_gt)
                    else:
                        dist,entry_idx_gt=torch.linalg.norm(local_pos[:,None]-self.entry_pos_token[None], dim=-1).min(-1)#.mean(-1)

                    # non_entry_agent= ~torch.isin( gt_entry_id,entry_id)
                    #
                    # non_entry_id=gt_entry_id[non_entry_agent]
                    #
                    # valid[non_entry_id, i]=False
                    #
                    # prev_pos[entry_id]=global_token_pos[row_ind][torch.arange(len(entry_idx_gt)),entry_idx_gt] #set to new entry position

                    present_id = torch.nonzero(present_agent, as_tuple=False).squeeze(1)[row_ind]

                    entry_idx[present_id] = entry_idx_gt

                    local_tok=self.entry_pos_token[entry_idx_gt]

                    offset_local=torch.cat((local_pos-local_tok,head_offset[:,None]), dim=-1)
                    entry_pos_offset_list.append(offset_local)


                    # dist=torch.linalg.norm(pos[:,i][entry_id]-prev_pos[entry_id], dim=-1)
                    #
                    # entry_token_invalid_mask.append(dist>2)
                    #
                    # real_id=entry_id[dist>2]
                    #
                    # prev_pos[real_id]=pos[real_id,i]

                    # offset_pos=pos[:,i][entry_id]-prev_pos[entry_id]
                    #
                    # entry_pos_offset_list.append(offset_pos)

                    # heading_diff=wrap_angle(heading[:,i][entry_id]-prev_head[entry_id])

                    #print(heading_diff.max(), heading_diff.mean())


                    #prev_head[real_id]=heading[real_id,i]
                    #
                    # dist1=torch.linalg.norm(pos[:,i][entry_id]-prev_pos[entry_id], dim=-1)
                    #
                    # print(dist1.mean(),dist1.max()) #0.8
                    #
                    # print(1)
                    # print(torch.linalg.norm(pos[:,i][entry_agent]-prev_pos[entry_agent], dim=-1).mean())

            if self.pred_entry and not self.autoregressive_entry :

                out_dict["entry_idx"].append(entry_idx)
                #out_dict["entry_head_idx"].append(entry_head_idx)

            # udpate prev_pos, prev_head
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
            prev_head[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid_mask]
            prev_pos[_valid_mask] = token_contour_gt.mean(1)[_valid_mask]

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

        if self.training:
            if len(entry_token_invalid_mask)>0:
                out_dict["entry_token_invalid_mask"]=torch.cat(entry_token_invalid_mask, dim=0)

            if len(entry_pos_offset_list) :
                out_dict["entry_pos_offset"]=torch.cat(entry_pos_offset_list)
            else:
                out_dict["entry_pos_offset"] =torch.zeros([0,4])

            if len(entry_head_idx_list) :
                out_dict["entry_head_idx"]=torch.cat(entry_head_idx_list)
                out_dict["entry_head_idx_num"]=torch.tensor([len(entry_head_idx) for entry_head_idx in entry_head_idx_list])
            else:
                out_dict["entry_head_idx"] =torch.zeros([0])
                out_dict["entry_head_idx_num"]=torch.zeros([out_dict["valid_mask"].shape[1]-1])

            if len(entry_idx_list) :
                # entry_length=out_dict['sampled_idx'].shape[1]-1
                entry_idx=pad_sequence(entry_idx_list, batch_first=True, padding_value=self.n_token_entry)#.reshape(entry_length,batch_num,-1)

                out_dict["pos_idx"]=entry_idx[:,:,0].long()

                out_dict["head_idx"] = torch.clamp(
                    entry_idx[:, :, 1],
                    max=self.n_token_entry_head - 1
                ).long()

                offset=entry_idx[:,:,2:]

                offset[offset==self.n_token_entry]=0

                out_dict["offset"]=offset



            # if len(entry_head_idx_list):
            #     out_dict["entry_head_idx"]=pad_sequence(entry_head_idx_list, batch_first=True, padding_value=self.n_token_entry_head)#.reshape(entry_length,batch_num,-1)

            #out_dict["entry_state"]=pad_sequence(entry_state_list, batch_first=True, padding_value=0)#.reshape(entry_length,batch_num,-1)
            #out_dict["entry_batch"]=pad_sequence(entry_batch_list, batch_first=True, padding_value=-1)


            # print(out_dict["entry_idx"].shape[1])

            #out_dict["entry_state"] = torch.cat([torch.zeros_like(out_dict["entry_state"][:, :1]), out_dict["entry_state"]], dim=1)

            # out_dict["entry_idx"] = torch.cat(
            #     [out_dict["entry_idx"], torch.zeros_like(out_dict["entry_idx"][:, :1]) + self.n_token_entry - 1], dim=1)

        return out_dict