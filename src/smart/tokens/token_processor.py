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

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from torch_scatter import scatter_mean,scatter_max
from ..layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock, general_rope
from torch.nn.utils.rnn import pad_sequence


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

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]

        self.use_lane=False

        light_token_all=torch.IntTensor(np.load(os.path.join(module_dir, "light_cluster.npy") ))#261

        self.register_buffer(f"light_token_all", light_token_all, persistent=False)

        light_token_last=light_token_all[:,-1].long()

        map_tensor=torch.tensor([3,4,0,1,2])

        light_token_last=map_tensor[light_token_last]

        self.register_buffer(f"light_token_last", light_token_last, persistent=False)

        self.use_my=False

    @torch.no_grad()
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map = self.tokenize_map(data)

        tokenized_agent = self.tokenize_agent(data)

        if "light_polyline" in data.keys():
            light=data["light"]

            light_all=light["type"]

            light_match=torch.all(light_all[None]==self.light_token_all[:,None,None],dim=-1)

            light_idx=torch.argmax(light_match.to(torch.int),dim=0)

            light_idx=self.light_token_last[light_idx]

            tokenized_agent["light_idx"]=light_idx
            pos_lg=light["pos"]
            orient_lg=torch.atan2(light["light_polyline"][:,-1],light["light_polyline"][:,-2])
            batch_lg=light["batch"]
            lengths_lg = torch.bincount(batch_lg, minlength=data.num_graphs).tolist()

            sinusoidal_lg = general_rope(pos_lg, 16, orient_lg)    

            sinusoidal_lg = pad_sequence(list(torch.split(sinusoidal_lg, lengths_lg)), batch_first=True, padding_value=0)

            tokenized_agent["lengths_lg"] = lengths_lg
            tokenized_agent["batch_lg"]=batch_lg
            tokenized_agent["sinusoidal_lg"] = sinusoidal_lg

        
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
        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32)
            # [n_token, 6, 4, 2], countour, 10 hz
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)

        self.register_buffer(f"trajectory_token_veh", self.agent_token_all_veh[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_ped", self.agent_token_all_ped[:, -1].flatten(1, 2), persistent=False)
        self.register_buffer(f"trajectory_token_cyc", self.agent_token_all_cyc[:, -1].flatten(1, 2), persistent=False)

    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:

        traj_pos = data["map_save"]["traj_pos"] # [n_pl, 3, 2]
        traj_theta = data["map_save"]["traj_theta"] # [n_pl]
        type = data["pt_token"]["type"]  # [n_pl]
        pl_type = data["pt_token"]["pl_type"]  # [n_pl]
        light_type= data["pt_token"]["light_type"]   # [n_pl]
        batch = data["pt_token"]["batch"]

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

        token_idx = torch.argmin(dist, dim=-1)

        position=traj_pos[:, 0].contiguous()

        tokenized_map = {
            "position": position,  # [n_pl, 2]
            "orientation": traj_theta,  # [n_pl]
            "token_idx": token_idx,  # [n_pl]
            "token_traj_src": self.map_token_traj_src,  # [n_token, 11*2]
            "type": type.long(),  # [n_pl]
            "pl_type": pl_type.long(),  # [n_pl]
            "light_type": light_type.long(),  # [n_pl]
            "batch": batch,  # [n_pl]
        }

        return tokenized_map

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:
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

        # ! agent, specifically vehicle's heading can be 180 degree off. We fix it here.
        heading = self._clean_heading(valid, heading)
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
            # "next_route": data["agent"]["next_route"],
            # "light":data["agent"]["light"]
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

        # av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
        #
        # initial_token_dict =self.tokenized_initial_pos(
        #     valid=valid,
        #     pos=pos,
        #     heading=heading,
        #     role=data["agent"]["role"],
        # )
        # tokenized_agent.update(initial_token_dict)



        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
        )
        # token_dict = self.my_match_agent_token(
        #     valid=valid,
        #     pos=pos,
        #     heading=heading,
        #     agent_shape=agent_shape,
        #     token_traj=token_traj,
        # )

        tokenized_agent.update(token_dict)
        return tokenized_agent

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
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

        if self.use_my:
            return self.my_match_agent_token(valid, pos, heading,agent_shape, token_traj)

        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]
        prev_pos_sample, prev_head_sample = pos[:, 0], heading[:, 0]

        out_dict = {
            "valid_mask": [],
            "gt_idx": [],
            "gt_pos": [],
            "gt_heading": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
        }

        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid_mask = ~_valid_mask
            out_dict["valid_mask"].append(_valid_mask)

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
            token_idx_gt = torch.argmin(
                torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1), dim=-1
            )  # [n_agent]

            # if self.training:
            #     token_world_gt = transform_to_local(
            #         pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
            #         head_local=None,
            #         pos_now=prev_pos,  # [n_agent, 2]
            #         head_now=prev_head,  # [n_agent]
            #     )[0].view(*token_traj.shape)

                # token_contour_local = token_traj[range_a, token_idx_gt]
                #
                # token_contour_local[:,:,0]+=0.1*torch.randn_like(token_contour_local[:,:,0])
                # token_contour_local[:,:,1]+=0.01*torch.randn_like(token_contour_local[:,:,1])
                # token_contour_gt = transform_to_global(
                #     pos_local=token_contour_local,  # [n_agent, n_token*4, 2]
                #     head_local=None,
                #     pos_now=prev_pos,  # [n_agent, 2]
                #     head_now=prev_head,  # [n_agent]
                # )[0]
            # [n_agent, 4, 2]
            token_contour_gt = token_world_gt[range_a, token_idx_gt]


            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]

            next_head=torch.arctan2(dxy[:, 1], dxy[:, 0])

            if self.training:
                prev_head=wrap_angle(prev_head+torch.randn_like(prev_head)*0.001)

            #prev_head[_valid_mask] = next_head[_valid_mask]
            prev_pos = pos[:, i].clone()

            #next_pos = token_contour_gt.mean(1)

            if self.training:
                prev_pos =prev_pos+torch.randn_like(prev_pos)*0.01

            #prev_pos[_valid_mask] = next_pos[_valid_mask]
            # add to output dict
            out_dict["gt_idx"].append(token_idx_gt)
            out_dict["gt_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )
            out_dict["gt_heading"].append(prev_head.masked_fill(_invalid_mask, 0))

            # ! tokenize from sampled rollout state
            if num_k == 1:  # K=1 means no sampling
                out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
                out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
                out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])
            else:
                # contour: [n_agent, n_token, 4, 2], 2HZ, global coord
                token_world_sample = transform_to_global(
                    pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
                    head_local=None,
                    pos_now=prev_pos_sample,  # [n_agent, 2]
                    head_now=prev_head_sample,  # [n_agent]
                )[0].view(*token_traj.shape)

                # dist: [n_agent, n_token]
                dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
                topk_dists, topk_indices = torch.topk(
                    dist, num_k, dim=-1, largest=False, sorted=False
                )  # [n_agent, K]

                topk_logits = (-1.0 * topk_dists) / self.agent_token_sampling.temp
                _samples = Categorical(logits=topk_logits).sample()  # [n_agent] in K
                token_idx_sample = topk_indices[range_a, _samples]
                token_contour_sample = token_world_sample[range_a, token_idx_sample]

                # udpate prev_pos_sample, prev_head_sample
                prev_head_sample = heading[:, i].clone()
                dxy = token_contour_sample[:, 0] - token_contour_sample[:, 3]
                prev_head_sample[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[
                    _valid_mask
                ]
                prev_pos_sample = pos[:, i].clone()
                prev_pos_sample[_valid_mask] = token_contour_sample.mean(1)[_valid_mask]
                # add to output dict
                out_dict["sampled_idx"].append(token_idx_sample)
                out_dict["sampled_pos"].append(
                    prev_pos_sample.masked_fill(_invalid_mask.unsqueeze(1), 0.0)
                )
                out_dict["sampled_heading"].append(
                    prev_head_sample.masked_fill(_invalid_mask, 0.0)
                )
        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}
        return out_dict


    def my_match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
        noise=None
    ) -> Dict[str, Tensor]:

        # pos_2hz= pos[:, ::self.shift]
        #
        # heading_2hz= heading[:, ::self.shift]
        # valid_2hz = valid[:, ::self.shift] # [n_agent]

        pos_now, head_now, valid_now = pos[:, self.shift::self.shift], heading[:, self.shift::self.shift], valid[:,
                                                                                                           self.shift::self.shift]

        if self.training:
            # token_dict = self._match_agent_token(
            #     valid=valid,
            #     pos=pos,
            #     heading=heading,
            #     agent_shape=agent_shape,
            #     token_traj=token_traj,
            # )
            #
            # noise=token_dict["sampled_pos"]- pos_now
            # heading_noise=wrap_angle(token_dict["sampled_heading"]- head_now)
            #
            # pos_now= pos_now+noise.abs()*torch.randn_like(noise)*
            # head_now= wrap_angle(head_now+heading_noise.abs()*torch.randn_like(heading_noise)*10)
            #
            # valid_now=token_dict["valid_mask"]
            # diff_xy = token_traj[:, :, 0] - token_traj[:, :, 3]
            # pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
            #
            # token_pos=token_traj.abs().mean(1).mean(1)[:,None]*0.1
            #
            # token_head=pred_head.abs().mean(1)[:,None]*0.1

            #noise=noise.clamp(min=-1, max=1)


            # sampled_pos= pos[:, self.shift:-self.shift:self.shift]+token_pos*torch.randn_like(pos_now)
            # sampled_head= wrap_angle(heading[:, self.shift:-self.shift:self.shift]+noise[...,2])

            # pos_now=torch.cat([pos_now[:,:1], sampled_pos], dim=1)
            # head_now=torch.cat([head_now[:,:1], sampled_head], dim=1)

            #valid_now[:,1:]=valid_now[:,:-1]
            pos_now=pos_now+torch.randn_like(pos_now)*0.1
            head_now=head_now+torch.randn_like(head_now)*0.01

        pos_2hz=torch.cat([pos[:,:1], pos_now], dim=1)
        heading_2hz=torch.cat([heading[:,:1], head_now], dim=1)
        valid_2hz=torch.cat([valid[:,:1], valid_now], dim=1)

        prev_pos, prev_head = pos_2hz[:, :-1], heading_2hz[:, :-1] # [n_agent, 2], [n_agent]

        # prev_pos+=0.1*torch.randn_like(prev_pos)
        # prev_head+=0.1*torch.randn_like(prev_head)

        target_pos, target_head = transform_to_local(
            pos_global=pos_now.flatten(0, 1).unsqueeze(1),  # [n_agent*18, 1, 2]
            head_global=head_now.flatten(0, 1).unsqueeze(1),  # [n_agent*18, 1]
            pos_now=prev_pos.flatten(0, 1),  # [n_agent*18, 2]
            head_now=prev_head.flatten(0, 1),  # [n_agent*18]
        )
        target_pos = target_pos.view(pos_now.shape)  # n_agent, 18, 2]
        target_head = wrap_angle(target_head)  # [n_agent, 18]
        target_head = target_head.view(head_now.shape)

        contour_local = cal_polygon_contour(target_pos[:,:,None],target_head[:,:,None], agent_shape[:,None,None])

        dist = torch.norm(contour_local - token_traj.unsqueeze(1), dim=-1).mean(-1)  # [n_batch, n_token]

        token_idx_gt = dist.argmin(-1)

        _valid_mask = valid_2hz[:,1:] & valid_2hz[:,:-1]
        _invalid_mask = ~_valid_mask

        sampled_pos=pos_now.masked_fill(_invalid_mask.unsqueeze(-1), 0)
        sampled_heading=head_now.masked_fill(_invalid_mask, 0)

        out_dict = {
            "valid_mask":_valid_mask,
            "sampled_idx": token_idx_gt,
            "sampled_pos": sampled_pos,
            "sampled_heading": sampled_heading,
        }

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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            elif k == "cyc":
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0
            agent_shape += torch.stack([width * mask, length * mask], dim=-1)

            token_traj_all += mask[:, None, None, None, None] * (
                getattr(self, f"agent_token_all_{k}").unsqueeze(0)
            )

        token_traj = token_traj_all[:, :, -1, :, :].contiguous()
        return agent_shape, token_traj_all, token_traj
