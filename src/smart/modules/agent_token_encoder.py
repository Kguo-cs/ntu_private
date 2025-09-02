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

        input_dim_x_a = 2
        input_dim_token = 8*5

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        # self.token_emb_veh = MLPEmbedding(
        #     input_dim=input_dim_token, hidden_dim=hidden_dim
        # )
        # self.token_emb_ped = MLPEmbedding(
        #     input_dim=input_dim_token, hidden_dim=hidden_dim
        # )
        # self.token_emb_cyc = MLPEmbedding(
        #     input_dim=input_dim_token, hidden_dim=hidden_dim
        # )

        self.discriminator=discriminator
        self.embedding = nn.Embedding(token_processor.n_token_agent, hidden_dim)

        if not self.discriminator:
            self.fusion_emb = MLPEmbedding(
                input_dim=hidden_dim * 2, hidden_dim=self.hidden_dim
            )

    def forward(
            self,
            agent_token_index,  # [n_agent, n_step]
            trajectory_token_veh,  # [n_token, 8]
            trajectory_token_ped,  # [n_token, 8]
            trajectory_token_cyc,  # [n_token, 8]
            pos_a,  # [n_agent, n_step, 2]
            head_vector_a,  # [n_agent, n_step, 2]
            agent_type,  # [n_agent]
            agent_shape,  # [n_agent, 3]
            inference=False,
    ):
        n_agent, n_step = agent_token_index.shape[0], agent_token_index.shape[1]
        _device = pos_a.device

        agent_token_emb = self.embedding(agent_token_index)

        # veh_mask = agent_type == 0
        # ped_mask = agent_type == 1
        # cyc_mask = agent_type == 2
        # # #  [n_token, hidden_dim]
        # agent_token_emb = torch.zeros(
        #     (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
        # )
        # if len(agent_token_index.shape) == 3:
        #     agent_token_emb[veh_mask] =self.token_emb_veh(agent_token_index[veh_mask])
        #     agent_token_emb[ped_mask] = self.token_emb_ped(agent_token_index[ped_mask])
        #     agent_token_emb[cyc_mask] = self.token_emb_cyc(agent_token_index[cyc_mask])
        # else:
        #     agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh[:,1:].reshape(-1, 40))
        #     agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped[:,1:].reshape(-1, 40))
        #     agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc[:,1:].reshape(-1, 40))
        #     agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        #     agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        #     agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

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
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
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

