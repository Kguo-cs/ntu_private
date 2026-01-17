import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

import torch.nn.functional as F
from .autoencoder_utils import ResidualMLP, AttentionLayer, AutoEncoderFactorizedAttentionBlock,GeometricLosses,reparameterize
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
import math
from src.smart.layers.relative_transformer import RoFormerBlock, padding

def sinusoidal_embedding(position, D):
    """
    Create sinusoidal positional embeddings for positions 1 to N
    Args:
        N: number of positions (assumes positions 1 to N)
        D: embedding dimension (must be even)
    Returns:
        Tensor of shape [N, D]
    """
   # return 0
    div_term = torch.exp(torch.arange(0, D, 2,device=position.device) * (-math.log(10000.0) / D))  # shape [D/2]

    pe = torch.zeros(len(position), D,device=position.device)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe


def padding_f( feature, batch, batch_num):
    lengths = torch.bincount(batch, minlength=batch_num).tolist()

    padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d
    mask_a_b = torch.any(padding_features_a != 0, dim=-1)

    return  padding_features_a,mask_a_b

def get_edgeindex(batch,batch_pl,num_graphs,use_transformer=False,hidden_dim=128):

    if use_transformer:
        a2a_edge_index = batch
        l2a_edge_index = batch_pl
        l2l_edge_index = num_graphs
    else:

        mask = batch[:, None] == batch[None, :]

        src, dst = mask.nonzero(as_tuple=True)

        a2a_edge_index=torch.stack([src, dst], dim=0)

        same_batch = batch_pl[:, None] == batch[None, :]

        pl_src, a_dst = same_batch.nonzero(as_tuple=True)

        a_dst = a_dst + len(batch_pl)  # shift polyline indices

        l2a_edge_index=torch.stack([pl_src, a_dst], dim=0)#src, dst

        mask = batch_pl[:, None] == batch_pl[None, :]

        src, dst = mask.nonzero(as_tuple=True)

        l2l_edge_index=torch.stack([src, dst], dim=0)


    counts = torch.bincount(batch, minlength=num_graphs)

    pos_idx=torch.arange(batch.size(0), device=batch.device) -  torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)

    pos_idx = sinusoidal_embedding(pos_idx[:,None]+1, hidden_dim)

    return a2a_edge_index, l2a_edge_index,l2l_edge_index,pos_idx



class ScenarioDreamerEncoder(nn.Module):
    """Encoder of the Scenario Dreamer AutoEncoder."""

    def __init__(self, num_encoder_blocks,hidden_dim,latent_dim,num_heads,use_transformer):
        super(ScenarioDreamerEncoder, self).__init__()
        self.num_encoder_blocks=num_encoder_blocks
        self.agent_mlp = ResidualMLP(input_dim=8,   hidden_dim=hidden_dim)
        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.hidden_dim=hidden_dim

        self.use_transformer = use_transformer

        if self.use_transformer:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.hidden_dim*4,
                dropout=0,
                norm_first=True,
                batch_first=True  # nn.Transformer uses (seq_len, batch, dim)
            )

            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=2
            )

        else:

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
                    dropout=0)

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
            agent_pos_idx: torch.Tensor,
            ego_embedding,
            lane_embeddings: torch.Tensor,
            lane_conn_embeddings,
            a2a_edge_index: torch.Tensor,
            l2a_edge_index: torch.Tensor,
            l2l_edge_index: torch.Tensor,
    ):
        agent_embeddings = self.agent_mlp(x_agent)+self.type_a_emb(agent_types)+ego_embedding+agent_pos_idx

        if self.use_transformer:

            feat_a_b,mask_a_b=padding_f(agent_embeddings,a2a_edge_index,l2l_edge_index)
            feat_map,map_mask=padding_f(lane_embeddings,l2a_edge_index,l2l_edge_index)

            agent_embeddings = self.transformer_decoder(
                tgt=feat_a_b,  # self-attention queries
                memory=feat_map,  # cross-attention keys/values
                tgt_key_padding_mask=~mask_a_b,
                memory_key_padding_mask=~map_mask
            )[mask_a_b]

        else:
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

    def __init__(self, num_decoder_blocks,hidden_dim,latent_dim,num_heads,use_transformer):
        super(ScenarioDreamerDecoder, self).__init__()
        self.num_decoder_blocks = num_decoder_blocks
        self.hidden_dim=hidden_dim

        self.agent_mlp = nn.Linear(latent_dim, hidden_dim)
        self.type_a_emb = nn.Embedding(3, hidden_dim)

        self.use_transformer=use_transformer
        if self.use_transformer:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.hidden_dim*4,
                dropout=0,
                norm_first=True,
                batch_first=True  # nn.Transformer uses (seq_len, batch, dim)
            )

            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=2
            )
        else:
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
                    dropout=0)
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
            agent_pos_idx:torch.Tensor,
            ego_embedding,
            lane_embeddings: torch.Tensor,
            a2a_edge_index: torch.Tensor,
            l2l_edge_index: torch.Tensor,
            l2a_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # ----------- latent -> hidden-dim projections -------------------- #
        agent_embeddings = self.agent_mlp(x_agent)+self.type_a_emb(agent_types)+ego_embedding+agent_pos_idx

        if self.use_transformer:

            feat_a_b,mask_a_b=padding_f(agent_embeddings,a2a_edge_index,l2l_edge_index)
            feat_map,map_mask=padding_f(lane_embeddings,l2a_edge_index,l2l_edge_index)

            agent_embeddings = self.transformer_decoder(
                tgt=feat_a_b,  # self-attention queries
                memory=feat_map,  # cross-attention keys/values
                tgt_key_padding_mask=~mask_a_b,
                memory_key_padding_mask=~map_mask
            )[mask_a_b]

        else:

            lane_conn_embeddings = lane_embeddings[l2l_edge_index[0]] + lane_embeddings[l2l_edge_index[1]]

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

        self.hidden_dim=hidden_dim

        latent_dim=8

        self.use_transformer=False

        self.encoder = ScenarioDreamerEncoder(num_encoder_blocks,hidden_dim,latent_dim,num_heads,self.use_transformer)
        self.decoder = ScenarioDreamerDecoder(num_decoder_blocks,hidden_dim,latent_dim,num_heads,self.use_transformer)

        # loss functions for training variational autoencoder
        self.agent_loss_fn = GeometricLosses['l1']()
        self.lane_loss_fn = GeometricLosses['l1']((1, 2))
        self.agent_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_conn_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.kl_loss_fn = GeometricLosses['kl']()
        self.apply(weight_init)



    def loss(self, data):

        x_agent, agent_types,num_graphs, ego_embedding,lane_embeddings, batch, batch_pl=data

        a2a_edge_index, l2a_edge_index,l2l_edge_index,pos_idx=get_edgeindex(batch,batch_pl,num_graphs)
        lane_conn_embeddings = lane_embeddings[l2l_edge_index[0]]+lane_embeddings[l2l_edge_index[1]]

        agent_mu, agent_log_var = self.encoder(
            x_agent,
            agent_types,
            pos_idx,
            ego_embedding,
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
            pos_idx,
            ego_embedding,
            lane_embeddings,
            a2a_edge_index,
            l2l_edge_index,
            l2a_edge_index)

        # agent vector regression loss
        agent_loss = F.l1_loss(agent_states_pred,x_agent)#self.agent_loss_fn(agent_states_pred, x_agent, batch)

        #agent_kl_loss = -0.5 * (1 + agent_log_var - agent_mu ** 2 - agent_log_var.exp())
        agent_kl_loss = self.kl_loss_fn(agent_mu, agent_log_var, batch)
        kl_loss = agent_kl_loss

        loss = agent_loss +1e-2 * kl_loss

        return (loss.mean(),agent_loss.mean().detach(),kl_loss.mean().detach())

    def forward_encoder(self, data, return_latents=False, return_lane_embeddings=False):

        x_agent, agent_types,num_graphs,ego_embedding, lane_embeddings, batch, batch_pl=data

        a2a_edge_index, l2a_edge_index,l2l_edge_index,pos_idx=get_edgeindex(batch,batch_pl,num_graphs)

        lane_conn_embeddings = lane_embeddings[l2l_edge_index[0]]+lane_embeddings[l2l_edge_index[1]]

        encoder_output = self.encoder(
            x_agent,
            agent_types,
            pos_idx,
            ego_embedding,
            lane_embeddings,
            lane_conn_embeddings,
            a2a_edge_index,
            l2a_edge_index,
            l2l_edge_index
            )
        agent_mu, agent_log_var = encoder_output

        if return_latents:
            return agent_mu, agent_log_var

        agent_latents = reparameterize(agent_mu, agent_log_var)

        return agent_latents

    def forward_decoder(self, agent_latents, agent_types,num_graphs, ego_embedding,lane_embeddings,batch, batch_pl):

        a2a_edge_index, l2a_edge_index,l2l_edge_index,pos_idx=get_edgeindex(batch,batch_pl,num_graphs)

        agent_states_pred = self.decoder(
            agent_latents,
            agent_types,
            pos_idx,
            ego_embedding,
            lane_embeddings,
            a2a_edge_index,
            l2l_edge_index,
            l2a_edge_index)

        return agent_states_pred

    def forward(self, data, return_latents=False, return_lane_embeddings=False):
        encoder_output = self.forward_encoder(data, return_stats=return_latents,  return_lane_embeddings=return_lane_embeddings)

        return encoder_output




