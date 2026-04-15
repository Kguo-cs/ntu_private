import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

import torch.nn.functional as F
from src.smart.diffusion.dit.autoencoder_utils import ResidualMLP, AttentionLayer, AutoEncoderFactorizedAttentionBlock,GeometricLosses,reparameterize
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
import math
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_global,
    rotate_to_local,
    weight_init
)
from src.smart.layers import MLPLayer

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
        agent_embeddings = self.agent_mlp(x_agent)+self.type_a_emb(agent_types)+ego_embedding#+agent_pos_idx

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
        agent_log_var = self.agent_log_var(agent_embeddings).clamp(-10, 10)

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
        agent_embeddings = self.agent_mlp(x_agent)+self.type_a_emb(agent_types)+ego_embedding#+agent_pos_idx

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

            lane_conn_embeddings =None #lane_embeddings[l2l_edge_index[0]] + lane_embeddings[l2l_edge_index[1]]

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

        hidden_dim=256
        num_heads=4

        self.use_transformer=False

        self.encoder = ScenarioDreamerEncoder(num_encoder_blocks,hidden_dim,latent_dim,num_heads,self.use_transformer)
        self.decoder = ScenarioDreamerDecoder(num_decoder_blocks,hidden_dim,latent_dim,num_heads,self.use_transformer)

        self.lane_embed= nn.Linear(128+4, hidden_dim)

        # loss functions for training variational auto.yaml
        self.agent_loss_fn = GeometricLosses['l1']()
        self.lane_loss_fn = GeometricLosses['l1']((1, 2))
        self.agent_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_type_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.lane_conn_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)
        self.kl_loss_fn = GeometricLosses['kl']()
        self.apply(weight_init)

        self.use_rel_ego=False

        if self.use_rel_ego:
            self.ego_embed = nn.Linear(9 + 3, hidden_dim)
        else:
            self.ego_embed = MLPLayer(16 + 3, hidden_dim, hidden_dim)


    def process_input(self, tokenized_agent,initial_map_feature):
        batch = tokenized_agent["nonego_batch"]
        num_graphs = tokenized_agent["num_graphs"]

        if self.use_rel_ego:
            ego_pose = tokenized_agent["ego_feat"][:, :-3].reshape(-1, 3, 3)
            type_count = tokenized_agent["ego_feat"][:, -3:][batch]

            all_pos = ego_pose[:, :, :2][batch]
            all_head = ego_pose[:, :, 2][batch]

            theta = torch.atan2(x_agent[:, 3], x_agent[:, 2])

            pos_s = x_agent[:, :2]

            local_ego_pos, local_ego_head = transform_to_local(
                all_pos,
                all_head,
                pos_s,
                theta
            )

            all_features = torch.cat([local_ego_pos.flatten(1, 2), local_ego_head, type_count], dim=-1)

            ego_embedding = self.ego_embed(all_features)
        else:
            ego_embedding = self.ego_embed(tokenized_agent["ego_feat"])
            ego_embedding = ego_embedding[batch]

        batch_pl = initial_map_feature["batch"]
        pos_pl = initial_map_feature["position"]
        orient_pl = initial_map_feature["orientation"]
        feat_map = initial_map_feature["pt_token"]

        lane_embeddings=self.lane_embed(torch.cat([feat_map,pos_pl,orient_pl.cos()[:,None],orient_pl.sin()[:,None]],dim=-1))


        a2a_edge_index, l2a_edge_index,l2l_edge_index,pos_idx=get_edgeindex(batch,batch_pl,num_graphs)


        return lane_embeddings,ego_embedding, a2a_edge_index, l2a_edge_index,l2l_edge_index,pos_idx



    def loss(self, x_agent, tokenized_agent,initial_map_feature):
        agent_types=tokenized_agent["nonego_type"]
        pos_pl = initial_map_feature["position"]
        batch=tokenized_agent["nonego_batch"]

        lane_embeddings, ego_embedding, a2a_edge_index, l2a_edge_index, l2l_edge_index, pos_idx=self.process_input(tokenized_agent,initial_map_feature)

        agent_latents,agent_mu, agent_log_var=self.forward_encoder(x_agent,agent_types,pos_pl, lane_embeddings, ego_embedding, a2a_edge_index, l2a_edge_index, l2l_edge_index, pos_idx)

        agent_states_pred = self.decoder(
            agent_latents,
            agent_types,
            pos_idx,
            ego_embedding,
            lane_embeddings,
            a2a_edge_index,
            l2l_edge_index,
            l2a_edge_index)

        scale=torch.tensor([[32.000, 32.000,  0.500,  0.500, 11.514,  6.312, 57.044,57.044]],device=x_agent.device)

        # x_agent=x_agent/scale

        # agent vector regression loss
        agent_loss = F.l1_loss(agent_states_pred/scale,x_agent/scale)#self.agent_loss_fn(agent_states_pred/scale, x_agent/scale, batch)#=

        #agent_kl_loss = -0.5 * (1 + agent_log_var - agent_mu ** 2 - agent_log_var.exp())
        agent_kl_loss = self.kl_loss_fn(agent_mu, agent_log_var, batch)
        kl_loss = agent_kl_loss

        loss = agent_loss +1e-2 * kl_loss

        # agent_states_pred=agent_states_pred* scale
        # x_agent=x_agent* scale

        pos_error = (agent_states_pred[:, :2] - x_agent[:, :2]).abs().mean()
        head_error = (agent_states_pred[:, 2:4] - x_agent[:, 2:4]).abs().mean()
        shape_error = (agent_states_pred[:, 4:6] - x_agent[:, 4:6]).abs().mean()
        vel_error = (agent_states_pred[:, 6:8] - x_agent[:, 6:8]).abs().mean()

        return loss.mean(),agent_loss.mean().detach(),kl_loss.mean().detach(),pos_error,head_error,shape_error,vel_error,agent_states_pred

    def forward_encoder(self,x_agent,agent_types,pos_pl, lane_embeddings, ego_embedding, a2a_edge_index, l2a_edge_index, l2l_edge_index, pos_idx):

        lane_conn_embeddings =None

        pos_agent=x_agent[:,:2]

        a2a_dist=torch.norm(pos_agent[a2a_edge_index[0]]-pos_agent[a2a_edge_index[1]],dim=-1)

        a2a_edge_index=a2a_edge_index[:,a2a_dist<60]

        a2l_dist=torch.norm(pos_agent[l2a_edge_index[1]-len(pos_pl)]-pos_pl[l2a_edge_index[0]],dim=-1)

        l2a_edge_index=l2a_edge_index[:,a2l_dist<40]

        agent_mu, agent_log_var = self.encoder(
            x_agent,
            agent_types,
            pos_idx,
            ego_embedding,
            lane_embeddings,
            lane_conn_embeddings,
            a2a_edge_index,
            l2a_edge_index,
            l2l_edge_index,
            )

        agent_latents = reparameterize(agent_mu, agent_log_var)

        return agent_latents,agent_mu, agent_log_var

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




