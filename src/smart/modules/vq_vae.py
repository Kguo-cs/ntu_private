from typing import Mapping, Any

import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn

from src.smart.layers.relative_transformer import RoFormerSinusoidalPositionalEmbedding, RoFormerBlock
from .build_edge import radiusGraphNearest, radiusGraphNearest2,nearest_mask,generate_limited_causal_mask
from vector_quantize_pytorch import VectorQuantize
from src.smart.layers import MLPLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))


        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class VQVAE(nn.Module):
    def __init__(self, token_processor,   hidden_dim=128, beta=0.25):
        super(VQVAE, self).__init__()

        num_heads=8

        self.token_processor=token_processor

        self.shift=5

        self.in_proj=nn.Linear(3*self.shift, hidden_dim)

        self.encoder = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.0)

        self.decoder = RoFormerBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.0)

        self.rotary_embedding = RoFormerSinusoidalPositionalEmbedding(hidden_dim=hidden_dim, num_heads=num_heads)

        self.out_proj=nn.Linear(hidden_dim,3*self.shift)

        n_embeddings=2048

        # self.vq = VectorQuantizer(
        #     n_embeddings, hidden_dim, beta)
        self.vq = VectorQuantize(
            dim=hidden_dim,
            codebook_size=n_embeddings,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.  # the weight on the commitment loss
        )

        self.loss_fn = nn.L1Loss()

        self.hidden_dim=hidden_dim

        self.type_a_emb = nn.Embedding(3, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        #self.fuse=nn.Sequential(nn.ReLU(),nn.Linear(hidden_dim*3, hidden_dim))


    def temporal_embed(self, feature, network, n_step, n_current, hist_len, mask):

        causal_mask = generate_limited_causal_mask(n_step, hist_len, device=feature.device)

        time = torch.arange(n_current, n_step + n_current, device=feature.device)[None,:, None]

        sinusoidal_pos=self.rotary_embedding(time=time)

        if mask is not None:
            causal_mask = causal_mask[None,None] | mask[:,None,None,:]

        feature = network(feature, causal_mask, sinusoidal_pos)

        return feature

    def forward(self, data):

        valid = data["agent"]["valid_mask"]  # [n_agent, n_step]
        heading = data["agent"]["heading"]  # [n_agent, n_step]
        pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, n_step, 2]
        vel = data["agent"]["velocity"]  # [n_agent, n_step, 2]
        agent_type=data["agent"]["type"]
        agent_shape=data["agent"]["shape"]


        heading = self.token_processor._clean_heading(valid, heading)

        valid, pos, heading, vel = self.token_processor._extrapolate_agent_to_prev_token_step(
            valid, pos, heading, vel
        )
        # token_agent_shape, token_traj_all, token_traj = self.token_processor._get_agent_shape_and_token_traj(
        #     data["agent"]["type"]
        # )
        #
        # token_dict = self.token_processor._match_agent_token(
        #     valid=valid,
        #     pos=pos,
        #     heading=heading,
        #     agent_shape=token_agent_shape,
        #     token_traj=token_traj,
        # )
        #
        # sampled_pos=token_dict["sampled_pos"]
        # sampled_heading=token_dict["sampled_heading"]
        #
        # sampled_idx=token_dict["sampled_idx"]
        #
        # traj=token_traj_all[np.arange(len(sampled_idx))[:,None],sampled_idx]
        #
        # traj1=traj.reshape(len(traj)*18,-1,2)
        #
        # prev_pos=torch.concat([pos[:,:1],sampled_pos[:,:-1]],dim=1).reshape(-1,2)
        # prev_head=torch.concat([heading[:,:1],sampled_heading[:,:-1]],dim=1).reshape(-1)
        # #valid 0 & 5
        # token_world_gt = transform_to_global(
        #     pos_local=traj1,  # [n_agent, n_token*4, 2]
        #     head_local=None,
        #     pos_now=prev_pos,  # [n_agent, 2]
        #     head_now=prev_head,  # [n_agent]
        # )[0].view(traj.shape)
        # #0,5,10
        #
        # prev_valid=torch.cat([valid[:,:1],token_dict["valid_mask"][:,:-1] ],dim=1)
        #
        # token_valid=prev_valid & token_dict["valid_mask"]
        #
        # sampled_contour=token_world_gt[:,:,1:].reshape(-1,90,4,2)
        # sampled_valid=token_valid[:,:,None].repeat(1,1,5).reshape(-1,90)
        # pred_pos=sampled_contour.mean(-2)
        # diff_xy = sampled_contour[:, :, 0] - sampled_contour[:, :, 3]
        # pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
        # sampled_pose=torch.cat([pred_pos,pred_head[...,None]],dim=-1)

        # smapled_dist=torch.linalg.norm(gt_contour-sampled_contour,dim=-1)[sampled_valid].mean()
        #smapled_dist=0

        # smapled_pos1=sampled_contour.mean(-2)
        #
        # x=smapled_pos1[:,4::5]-sampled_pos#10 is avaliable

        # last_pos=traj[:,:,-1].mean(-2)
        
        # relative_pos=sampled_pos[:,1:]-sampled_pos[:,:-1]
        
        
        # traj1=token_traj[np.arange(len(sampled_idx))[:,None],sampled_idx]
        
        # last_pos1=traj1.mean(-2)

        pose = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

        rel_valid = valid[:, 1:] & valid[:, :-1]

        mask = rel_valid.reshape(-1, 18, self.shift).any(dim=-1)

        type_embedding = 0#self.type_a_emb(agent_type.long())[:,None]
        shape_embedding = 0 #self.shape_emb(agent_shape)[:,None]

        z_q,  commit_loss=self.tokenize(pose,mask,type_embedding,shape_embedding)

        z_q=z_q+type_embedding+shape_embedding

        z_q[~mask]=0

        x_hat = self.temporal_embed(z_q, self.encoder,   n_step=z_q.shape[1], n_current=0, hist_len=6, mask=None)

        out_vel=self.out_proj(x_hat).reshape(-1,90,3)

        out_vel[~valid[:,1:]]=0

        first_valid_step = torch.max(valid, dim=1).indices  # [n_agent]

        out_pose=torch.cumsum(out_vel, dim=1)+pose[torch.arange(len(first_valid_step)),first_valid_step][:,None]

        agent_shape=agent_shape[:,None].repeat(1,pos.shape[1]-1,1)#[:,:,None][:,0]

        gt_contour = cal_polygon_contour(pos[:,1:], heading[:,1:], agent_shape)

        out_contour = cal_polygon_contour(out_pose[:,:,:2], out_pose[:,:,2], agent_shape)

        #sampled_contour=cal_polygon_contour(sampled_pose[:,:,:2], sampled_pose[:,:,2], agent_shape)

        rec_loss=torch.linalg.norm(out_pose-pose[:,1:],ord=1,dim=-1)[valid[:,1:]].mean()

        dist=torch.linalg.norm(gt_contour-out_contour,dim=-1)[valid[:,1:]].mean()

        smapled_dist=0#torch.linalg.norm(gt_contour-sampled_contour,dim=-1)[sampled_valid].mean()

        return commit_loss,rec_loss,dist,smapled_dist

    def tokenize(self,pose,mask,type_embedding,shape_embedding):

        vel = pose[:, 1:] - pose[:, :-1]

        x = self.in_proj(vel.reshape(-1, 18, self.shift * 3))

        x = x + type_embedding+ shape_embedding

        z_e = self.temporal_embed(x, self.encoder, n_step=x.shape[1], n_current=0, hist_len=6, mask=~mask)

        z_q_mask, mask_indices, commit_loss = self.vq(z_e[mask])
        
        #embedding=self.vq.get_codes_from_indices(mask_indices)
        #commit_loss, embedding, perplexity, min_encodings, min_encoding_indices=self.vq(z_e[mask])

        z_q=torch.zeros_like(z_e)
        z_q[mask]=z_q_mask

        return z_q,  commit_loss

    def recontruct(self,indices):
        z_q=self.vq.get_output_from_indices(indices)

        x_hat = self.temporal_embed(z_q, self.encoder,   n_step=z_q.shape[1], n_current=0, hist_len=6, mask=None)

        out_vel=self.out_proj(x_hat).reshape(-1,90,3)

        return 1
