import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

import torch.nn.functional as F
from .autoencoder_utils import ResidualMLP, AttentionLayer, AutoEncoderFactorizedAttentionBlock,GeometricLosses,reparameterize
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle

class ScenarioDreamerEncoder(nn.Module):
    """Encoder of the Scenario Dreamer AutoEncoder."""

    def __init__(self, num_encoder_blocks,hidden_dim,latent_dim,num_heads):
        super(ScenarioDreamerEncoder, self).__init__()
        self.num_encoder_blocks=num_encoder_blocks
        self.agent_mlp = ResidualMLP(input_dim=8,   hidden_dim=hidden_dim)
        self.type_a_emb = nn.Embedding(3, hidden_dim)

        # Factorised attention encoder blocks
        self.encoder_transformer_blocks = []
        for l in range(self.num_encoder_blocks):
            encoder_transformer_block = AutoEncoderFactorizedAttentionBlock(
                lane_hidden_dim=hidden_dim,
                lane_feedforward_dim=hidden_dim*4,
                lane_num_heads=num_heads,
                agent_hidden_dim=hidden_dim,
                agent_feedforward_dim=hidden_dim*4,
                agent_num_heads=num_heads,
                lane_conn_hidden_dim=hidden_dim,
                dropout=0.1)

            self.encoder_transformer_blocks.append(encoder_transformer_block)
        self.encoder_transformer_blocks = nn.ModuleList(self.encoder_transformer_blocks)

        # Gaussian latent variable heads
        self.agent_mu = nn.Linear(hidden_dim, latent_dim)
        self.agent_log_var = nn.Linear(hidden_dim, latent_dim)
        self.apply(weight_init)

    def forward(
            self,
            x_agent: torch.Tensor,
            agent_types:torch.Tensor,
            lane_embeddings: torch.Tensor,
            lane_conn_embeddings,
            a2a_edge_index: torch.Tensor,
            l2a_edge_index: torch.Tensor,
            l2l_edge_index: torch.Tensor,
    ):
        agent_embeddings = self.agent_mlp(x_agent)+self.type_a_emb(agent_types)

        for l in range(self.num_encoder_blocks):
            agent_embeddings, lane_embeddings, lane_conn_embeddings = self.encoder_transformer_blocks[l](
                agent_embeddings,
                lane_embeddings,
                lane_conn_embeddings,
                lane_conn_embeddings,
                a2a_edge_index,
                l2l_edge_index,
                l2a_edge_index)


        agent_mu = self.agent_mu(agent_embeddings)
        agent_log_var = self.agent_log_var(agent_embeddings)

        return agent_mu, agent_log_var


class ScenarioDreamerDecoder(nn.Module):
    """Decoder of the Scenario Dreamer AutoEncoder."""

    def __init__(self, num_decoder_blocks,hidden_dim,latent_dim,num_heads):
        super(ScenarioDreamerDecoder, self).__init__()
        self.num_decoder_blocks = num_decoder_blocks

        self.agent_mlp = nn.Linear(latent_dim, hidden_dim)
        self.type_a_emb = nn.Embedding(3, hidden_dim)

        # ------------------- factorized attention decoder blocks ---------------------- #
        self.decoder_transformer_blocks = []
        for l in range(self.num_decoder_blocks):
            decoder_transformer_block = AutoEncoderFactorizedAttentionBlock(
                lane_hidden_dim=hidden_dim,
                lane_feedforward_dim=hidden_dim*4,
                lane_num_heads=num_heads,
                agent_hidden_dim=hidden_dim,
                agent_feedforward_dim=hidden_dim*4,
                agent_num_heads=num_heads,
                lane_conn_hidden_dim=hidden_dim,
                dropout=0.1)
            self.decoder_transformer_blocks.append(decoder_transformer_block)
        self.decoder_transformer_blocks = nn.ModuleList(self.decoder_transformer_blocks)

        # ------------------- output heads -------------------------------- #
        self.pred_agent_states = ResidualMLP(input_dim=hidden_dim,
                                             hidden_dim=hidden_dim,
                                             n_hidden=3,
                                             output_dim=8)
        self.apply(weight_init)


    def forward(
            self,
            x_agent: torch.Tensor,
            agent_types: torch.Tensor,
            lane_embeddings: torch.Tensor,
            a2a_edge_index: torch.Tensor,
            l2l_edge_index: torch.Tensor,
            l2a_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent embeddings into vectorized driving scenes.

        Args:
            x_agent: Tensor *(N_agents, agent_latent_dim)* - latent agent
                embeddings sampled from the encoder.
            x_lane: Tensor *(N_lanes, lane_latent_dim)* - latent lane
                embeddings.
            a2a_edge_index: COO index *(2, E_agent)* for agent-to-agent edges.
            l2l_edge_index: COO index *(2, E_lane)* for lane-to-lane edges.
            l2a_edge_index: COO index *(2, E_cross)* for lane-to-agent edges.

        Returns:
            Tuple containing, in order:

            * **agent_states_pred** - *(N_agents, state_dim)* agent state predictions
            * **agent_types_logits** - *(N_agents, num_agent_types)* logits for
              categorical agent type prediction.
            * **agent_types_pred** - *(N_agents,)* predictions for
              categorical agent type prediction.
            * **lane_states_pred** - *(N_lanes, num_points_per_lane,
              lane_attr)* predicted lane vectors
            * **lane_types_logits** - *(N_lanes, num_lane_types) or None* logits for
              categorical lane type prediction.
            * **lane_types_pred** - *(N_lanes,) or None* predictions for
              categorical lane type prediction.
            * **lane_conn_logits** - *(E_lane, lane_conn_attr)* logits for lane
              connectivity classification.
            * **lane_conn_pred** - *(E_lane, 6)* predictions for lane
              connectivity classification as one-hot vectors.
        """

        # ----------- latent -> hidden-dim projections -------------------- #
        agent_embeddings = self.agent_mlp(x_agent)+self.type_a_emb(agent_types)

        lane_conn_embeddings=None

        # ----------- factorized attention processing ------------------------ #
        for l in range(self.num_decoder_blocks):
            agent_embeddings, lane_embeddings, lane_conn_embeddings = self.decoder_transformer_blocks[l](
                agent_embeddings,
                lane_embeddings,
                lane_conn_embeddings,
                lane_conn_embeddings,
                a2a_edge_index,
                l2l_edge_index,
                l2a_edge_index)

        # ----------- prediction heads ------------------------------------ #
        agent_states_pred = self.pred_agent_states(agent_embeddings)

        return agent_states_pred

class AutoEncoder(nn.Module):
    """Scenario Dreamer AutoEncoder."""

    def __init__(self, num_encoder_blocks,num_decoder_blocks,hidden_dim,latent_dim,num_heads):
        super(AutoEncoder, self).__init__()

        self.encoder = ScenarioDreamerEncoder(num_encoder_blocks,hidden_dim,latent_dim,num_heads)
        self.decoder = ScenarioDreamerDecoder(num_decoder_blocks,hidden_dim,latent_dim,num_heads)

        # loss functions for training variational autoencoder
        self.agent_loss_fn = GeometricLosses['l1']()
        self.lane_loss_fn = GeometricLosses['l1']((1, 2))
        self.agent_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_conn_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.kl_loss_fn = GeometricLosses['kl']()
        self.apply(weight_init)

    def loss(self, x_agent,agent_types,lane_embeddings,batch,batch_pl,lane_conn_embeddings=None,l2l_edge_index=None):

        n_agent=x_agent.shape[0]

        mask = batch[:, None] == batch[None, :]

        src, dst = mask.nonzero(as_tuple=True)

        a2a_edge_index=torch.stack([src, dst], dim=0)

        same_batch = batch_pl[:, None] == batch[None, :]

        pl_src, a_dst = same_batch.nonzero(as_tuple=True)

        a_dst = a_dst + len(lane_embeddings)  # shift polyline indices

        l2a_edge_index=torch.stack([pl_src, a_dst], dim=0)#src, dst

        agent_mu, agent_log_var = self.encoder(
            x_agent,
            agent_types,
            lane_embeddings,
            lane_conn_embeddings,
            a2a_edge_index,
            l2a_edge_index,
            l2l_edge_index
            )

        agent_latents = reparameterize(agent_mu, agent_log_var)

        agent_states_pred = self.decoder(
            agent_latents,
            agent_types,
            lane_embeddings,
            a2a_edge_index,
            l2l_edge_index,
            l2a_edge_index)

        # agent vector regression loss
        agent_loss = self.agent_loss_fn(agent_states_pred, x_agent, batch)
        agent_kl_loss = self.kl_loss_fn(agent_mu, agent_log_var, batch)
        kl_loss = agent_kl_loss

        loss = agent_loss +1e-2 * kl_loss

        return (loss.mean(),agent_loss.mean().detach(),kl_loss.mean().detach())

    def forward_encoder(self, data, return_stats=False, return_lane_embeddings=False):
        """forward pass through the encoder."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
        a2a_edge_index_encoder, l2l_edge_index_encoder, l2a_edge_index_encoder, l2q_edge_index_encoder, x_lane_conn_encoder = get_encoder_edge_indices(
            data)

        encoder_output = self.encoder(
            x_agent,
            x_lane,
            x_lane_conn_encoder,
            a2a_edge_index_encoder,
            l2l_edge_index_encoder,
            l2a_edge_index_encoder,
            l2q_edge_index_encoder,
            agent_batch,
            return_lane_embeddings)

        if return_lane_embeddings:
            return encoder_output
        else:
            agent_mu, lane_mu, agent_log_var, lane_log_var, lane_cond_dis_logits, lane_cond_dis_prob = encoder_output

        if return_stats:
            return agent_mu, lane_mu, agent_log_var, lane_log_var

        agent_latents = reparameterize(agent_mu, agent_log_var)
        lane_latents = reparameterize(lane_mu, lane_log_var)

        return agent_latents, lane_latents, lane_cond_dis_prob

    def forward_decoder(self, agent_latents, lane_latents, data):
        """forward pass through the decoder."""
        a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
        agent_states_pred, agent_types_logits, agent_types_pred, lane_states_pred, lane_types_logits, lane_types_pred, lane_conn_logits, lane_conn_pred = self.decoder(
            agent_latents,
            lane_latents,
            a2a_edge_index,
            l2l_edge_index,
            l2a_edge_index)

        return agent_states_pred, lane_states_pred, agent_types_pred, lane_types_pred, lane_conn_pred

    def forward(self, data, return_latents=False, return_lane_embeddings=False):
        encoder_output = self.forward_encoder(data, return_stats=return_latents,
                                              return_lane_embeddings=return_lane_embeddings)

        if return_latents or return_lane_embeddings:
            return encoder_output
        else:
            agent_latents, lane_latents, lane_cond_dis_prob = encoder_output

        agent_states_pred, lane_states_pred, agent_types_pred, lane_types_pred, lane_conn_pred = self.forward_decoder(
            agent_latents, lane_latents, data)

        return agent_states_pred, lane_states_pred, agent_types_pred, lane_types_pred, lane_conn_pred, lane_cond_dis_prob




