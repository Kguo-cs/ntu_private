from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from src.smart.layers import MLPLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle,project_to_local_frame

class AgentTokenEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands:int,
            token_processor,
            discriminator,
            traj_diffusion=False
    ) -> None:
        super(AgentTokenEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.token_processor=token_processor

        input_dim_token = 8

        self.use_mean_speed = False

        if self.use_mean_speed:
            self.speed_embed = nn.Embedding(5, hidden_dim)

        self.use_type=True

        if self.token_processor.use_bird:
            self.use_type=False
            input_dim_x_a=3
        else:
            self.shape_dim = 2

            self.type_a_emb = nn.Embedding(3, hidden_dim)
            self.shape_emb = MLPLayer(self.shape_dim, hidden_dim, hidden_dim)
            input_dim_x_a=2

            if token_processor.use_gradient_penalty:
                self.differentiable_edge=True
            else:
                self.differentiable_edge=not discriminator

        self.use_goal = self.token_processor.use_goal & ((not discriminator) | self.token_processor.use_bird)
        self.use_bird=token_processor.use_bird

        if self.use_goal:
            input_dim_x_a*=2

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands
        )

        self.discriminator=discriminator
        self.use_state_action=False

        self.traj_diffusion=traj_diffusion

        if self.traj_diffusion:
            input_dim_token=4*3+1

        if self.discriminator:
            if self.use_state_action:
                self.token_emb_veh = MLPEmbedding(
                    input_dim=15, hidden_dim=hidden_dim
                )
        else:
            if self.use_type:
                self.token_emb_veh = MLPEmbedding(
                    input_dim=input_dim_token, hidden_dim=hidden_dim
                )
                self.token_emb_ped = MLPEmbedding(
                    input_dim=input_dim_token, hidden_dim=hidden_dim
                )
                self.token_emb_cyc = MLPEmbedding(
                    input_dim=input_dim_token, hidden_dim=hidden_dim
                )

                # self.invalid_token_emb=nn.Embedding(1,hidden_dim)
               # self.invalid_feat_emb=nn.Embedding(1,input_dim_x_a)

            else:
                self.embedding = nn.Embedding(token_processor.n_token_agent, hidden_dim)
            self.fusion_emb = MLPEmbedding(
                input_dim=hidden_dim * 2, hidden_dim=self.hidden_dim
            )

        self.apply(weight_init)


    def get_embedding(self,agent_token_index,agent_type,token_mask):
        if  not self.discriminator:
            n_agent, n_step = agent_token_index.shape[0], agent_token_index.shape[1]
            _device = agent_token_index.device

            agent_token_emb = torch.zeros(
                (n_agent, n_step, self.hidden_dim), device=_device
            )#previous invalid

            if self.use_type:
                veh_mask =agent_type == 0
                ped_mask = agent_type == 1
                cyc_mask = agent_type == 2
                if token_mask is not None:
                    veh_mask = veh_mask[:, None] & token_mask
                    ped_mask = ped_mask[:, None] & token_mask
                    cyc_mask = cyc_mask[:, None] & token_mask

                if self.traj_diffusion:
                    agent_token_emb[veh_mask] = self.token_emb_veh(agent_token_index[veh_mask])
                    agent_token_emb[ped_mask] =  self.token_emb_ped(agent_token_index[ped_mask])
                    agent_token_emb[cyc_mask] = self.token_emb_cyc(agent_token_index[cyc_mask])
                else:

                    agent_token_emb_veh = self.token_emb_veh(self.token_processor.trajectory_token_veh)
                    agent_token_emb_ped = self.token_emb_ped(self.token_processor.trajectory_token_ped)
                    agent_token_emb_cyc = self.token_emb_cyc(self.token_processor.trajectory_token_cyc)
                    agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
                    agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
                    agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]
            else:
                if token_mask is None:
                    agent_token_emb = self.embedding(agent_token_index)
                else:
                    agent_token_emb[token_mask] = self.embedding(agent_token_index[token_mask])

        else:
            if self.use_state_action:
                n_agent, n_step = agent_token_index.shape[0], agent_token_index.shape[1]
                _device = agent_token_index.device

                agent_token_emb = torch.zeros(
                    (n_agent, n_step - 1, self.hidden_dim),
                    device=_device,
                    dtype=next(self.token_emb_veh.parameters()).dtype,
                )

                veh_mask =agent_type == 0
                veh_mask=veh_mask[:,None] & token_mask[:,1:]

                agent_token_all=self.token_processor.agent_token_all
                agent_token_emb_veh = self.token_emb_veh(agent_token_all.reshape(agent_token_all.shape[0], -1))
                agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[:,1:][veh_mask]]

                agent_token_emb=agent_token_emb.transpose(0, 1).flatten(0,1)

            else:
                agent_token_emb = None

        return agent_token_emb

    def forward(
            self,
            agent_token_index,  # [n_agent, n_step]
            pos_a,  # [n_agent, n_step, 2]
            head_vector_a,  # [n_agent, n_step, 2]
            mask_a,
            agent_type,  # [n_agent]
            agent_shape,  # [n_agent, 3]
            token_mask=None,
            goal_pos=None,
            goal_mask=None,
    ):
        n_agent, n_step = head_vector_a.shape[0], head_vector_a.shape[1]
        _device = pos_a.device

        agent_token_emb=self.get_embedding(agent_token_index,agent_type,token_mask)

        if pos_a.shape[1]==n_step:
            motion_vector_a = torch.cat(
                [
                    pos_a.new_zeros(n_agent, 1, pos_a.shape[-1]),
                    pos_a[:, 1:] - pos_a[:, :-1],
                ],
                dim=1,
            )[:,-n_step:]
        else:
           motion_vector_a=pos_a[:, 1:] - pos_a[:, :-1]
        
        feature_a=project_to_local_frame(motion_vector_a, head_vector_a,self.differentiable_edge)

        if token_mask is not None:
            feature_a[~token_mask]= -10#self.invalid_feat_emb.weight

        if self.use_goal:
            if goal_pos is not None:
                goal_vector_a = goal_pos[:, None] - pos_a[:, -n_step:]

                feature_goal=torch.stack(
                    [
                        torch.norm(goal_vector_a, p=2, dim=-1),
                        angle_between_2d_vectors(
                            ctr_vector=head_vector_a, nbr_vector=goal_vector_a[:, :, :2]
                        ),
                    ],
                    dim=-1,
                )  # [n_agent, n_step, 2]

                if self.use_bird:
                    feature_goal = torch.cat([feature_goal, goal_vector_a[:, :, 2:]], dim=-1)

                if goal_mask is None:
                    goal_mask = torch.ones(
                        feature_goal.shape[:-1],
                        dtype=torch.bool,
                        device=feature_goal.device,
                    )
                elif goal_mask.ndim == 1:
                    goal_mask = goal_mask[:, None].expand(feature_goal.shape[:-1])
                feature_goal[~goal_mask.bool()] = 0
            else:
                feature_goal = torch.zeros_like(feature_a)

            feature_a = torch.cat([feature_a, feature_goal], dim=-1)

        if self.use_type and agent_shape is not None:
            categorical_embs = self.type_a_emb(agent_type) + self.shape_emb(
                agent_shape[..., :self.shape_dim]
            )
            categorical_embs = categorical_embs[None].repeat(n_step, 1, 1)
        else:
            categorical_embs = None

        if mask_a is not None:
            mask_s=mask_a.transpose(0, 1)

            feature_a=feature_a.transpose(0, 1)[mask_s]
            if agent_token_emb is not None:
                agent_token_emb=agent_token_emb.transpose(0, 1)[mask_s]
            if categorical_embs is not None:
                categorical_embs = categorical_embs[mask_s]
        else:
            feature_a=feature_a.view(-1, feature_a.size(-1))

        feat_a = self.x_a_emb(
            continuous_inputs=feature_a,
            categorical_embs=categorical_embs,
        )  # [n_agent*n_step, hidden_dim]
        #x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        counter_feat_a=None

        if not self.discriminator:
            feat_a = torch.cat((agent_token_emb, feat_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)

        if not self.use_state_action:
            agent_token_emb=None

        return feat_a, agent_token_emb  # [n_agent, n_step, hidden_dim] #1258

