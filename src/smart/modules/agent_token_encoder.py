from typing import Dict, Optional

import torch
import torch.nn as nn
from src.smart.layers import MLPLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
)
from src.smart.utils.rollout import cal_polygon_contour

class AgentTokenEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands:int,
            token_processor,
            discriminator=False
    ) -> None:
        super(AgentTokenEncoder, self).__init__()
        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.token_processor=token_processor

        input_dim_token = 8
        input_dim_x_a = 2

        self.use_mean_speed = True

        if self.use_mean_speed:
            self.speed_embed = nn.Embedding(5, hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )

        self.use_type=True

        self.discriminator=discriminator

        if not self.discriminator:
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

            else:
                self.embedding = nn.Embedding(token_processor.n_token_agent, hidden_dim)
            self.fusion_emb = MLPEmbedding(
                input_dim=hidden_dim * 2, hidden_dim=self.hidden_dim
            )


    def forward(
            self,
            agent_token_index,  # [n_agent, n_step]
            # trajectory_token_veh,  # [n_token, 8]
            # trajectory_token_ped,  # [n_token, 8]
            # trajectory_token_cyc,  # [n_token, 8]
            mean_speed,
            pos_a,  # [n_agent, n_step, 2]
            head_vector_a,  # [n_agent, n_step, 2]
            agent_type,  # [n_agent]
            agent_shape,  # [n_agent, 3]
            inference=False,
    ):
        n_agent, n_step = agent_token_index.shape[0], agent_token_index.shape[1]
        _device = pos_a.device

        if not self.discriminator:
            if self.use_type:
                veh_mask = agent_type == 0
                ped_mask = agent_type == 1
                cyc_mask = agent_type == 2
                #  [n_token, hidden_dim]
                agent_token_emb_veh = self.token_emb_veh(self.token_processor.trajectory_token_veh)
                agent_token_emb_ped = self.token_emb_ped(self.token_processor.trajectory_token_ped)
                agent_token_emb_cyc = self.token_emb_cyc(self.token_processor.trajectory_token_cyc)
                agent_token_emb = torch.zeros(
                    (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
                )
                agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
                agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
                agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]
            else:
                agent_token_emb = self.embedding(agent_token_index)

        else:
            agent_token_emb = None

        # if self.discriminator:
        #     motion_vector_a = torch.cat(
        #         [
        #             1e-4*pos_a[:,:1],
        #             pos_a[:, 1:] - pos_a[:, :-1],
        #         ],
        #         dim=1,
        #     ) [:,-n_step:] # [n_agent, n_step, 2]
        #
        # else:
        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, 2),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        ) [:,-n_step:] # [n_agent, n_step, 2]

        if self.discriminator:
            # feature_a = torch.stack(
            #     [
            #         torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
            #
            #         angle_between_2d_vectors(
            #             ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
            #         ),
            #     ],
            #     dim=-1,
            # )  # [n_agent, n_step, 2]
            u=motion_vector_a[:, :, :2]
            v=head_vector_a

            feature_a = torch.stack(
                [
                    (u*v).sum(dim=-1) ,
                    u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0],
                ],
                dim=-1,
            )  # [n_agent, n_step, 2]
        else:
            feature_a = torch.stack(
                [
                    torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                    ),
                ],
                dim=-1,
            )  # [n_agent, n_step, 2]

        if self.use_mean_speed:
            agent_speed = torch.zeros_like(agent_type)

            if mean_speed is not None:
                mean_speed = torch.clamp_max_(mean_speed, max=19).to(torch.long)//5+1

                if self.training:
                    mask=torch.rand_like(agent_type.float())>0.8
                    agent_speed[mask] = mean_speed[mask]
                else:
                    agent_speed=mean_speed


            categorical_embs = [
                self.type_a_emb(agent_type),
                self.shape_emb(agent_shape),
                self.speed_embed(agent_speed),
            ]  # List of len=2, shape [n_agent, hidden_dim]
        else:
            categorical_embs = [
                self.type_a_emb(agent_type),
                self.shape_emb(agent_shape),
            ]  # List of len=2, shape [n_agent, hidden_dim]


        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        if not self.discriminator:
            feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)
        else:
            feat_a=x_a

        return feat_a, agent_token_emb  # [n_agent, n_step, hidden_dim]

