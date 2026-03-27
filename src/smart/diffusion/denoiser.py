import math
import copy
import numpy as np

from typing import Dict, Mapping, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Bernoulli
from src.smart.layers.fourier_embedding import FourierEmbedding

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_global,
    rotate_to_local,
    weight_init
)
import warnings
from torch.nn.modules.container import ModuleList
import copy
from src.smart.layers.relative_transformer import RoFormerBlock,RoFormerDecoder
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.func import functional_call, jvp
from src.smart.utils.cluster import batch_increasing_schedule,allocate_k_per_type
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder,topo_rank_among_edges,project_to_local_frame
from src.smart.layers.relative_transformer import padding
from src.smart.utils.cluster import cluster_point_per_type
from torch_scatter import scatter_sum
from .diffusion_planner.decoder import  DiT
from src.smart.modules.interative_decoder import InterativeDecoder
from src.smart.utils.edge_utils import build_batch
from src.smart.loss.rollout_buffer import RunningMeanStdTorch, get_reward, get_nei_returns, get_return

def batch_histogram_categorical(m_init, bins=8, value_range=None):
    # m_init: (B, 8)
    B, D = m_init.shape
    assert D == 8  # your case

    hist_list = []

    for d in range(D):
        col = m_init[:, d]

        hist = torch.histc(
            col,
            bins=bins,
            min=value_range[0] if value_range else col.min(),
            max=value_range[1] if value_range else col.max()
        )

        hist_list.append(hist)

    hist = torch.stack(hist_list, dim=0)  # (8, bins)

    # normalize → categorical distribution
    probs = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

    return probs  # (8, bins)

class InitDenoiser(nn.Module):

    def __init__(self,
                 token_processor,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 init_timestep: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 diff_type: str,
                 m_dim: int,
                 mean_flow=False
                 ) -> None:
        super(InitDenoiser, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_timestep = init_timestep
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.diff_type = diff_type
        self.m_dim = m_dim

        self.use_roformer=True
        self.use_padding=True
        self.use_all_type=False

        self.num_classes=3

        if self.use_all_type:
            m_delta_dim = 11
        else:
            self.type_a_emb = nn.Embedding(self.num_classes+1, hidden_dim)#
            m_delta_dim = 5+3


        self.use_graph=True
        self.ego_rel = True
        self.use_scale=False
        self.use_dit=False

        noise_dim = 1
        if mean_flow:
            noise_dim = 2
            self.use_padding = True

        self.use_all_pos=token_processor.use_all_pos

        self.use_prev_head=False
        self.use_speed=False

        if self.use_speed:
            m_delta_dim = m_delta_dim-1

        if self.use_prev_head:
            m_delta_dim=m_delta_dim+2

        if self.use_all_pos:
            m_delta_dim=m_delta_dim+94*4-2

        self.output_dim=m_delta_dim
        self.label_drop_prob=0

        self.register_buffer("normal_mean", torch.zeros(1, m_delta_dim))
        self.register_buffer("normal_scale", torch.ones(1, m_delta_dim))

       # self.return_meanstd = ModuleList([copy.deepcopy(RunningMeanStdTorch(shape=(m_delta_dim))) for i in range(3)])

        if self.use_all_pos:
            m_delta_dim=6+4*4

        # self.register_buffer("init_probs", torch.ones(m_delta_dim,100))
        # self.register_buffer("init_min", torch.ones(m_delta_dim))
        # self.register_buffer("init_max", torch.ones(m_delta_dim))

        self.normal_initialized = False


        if self.use_roformer:
            if self.use_dit:
                self.map_embed= MLPLayer(128+2+2, hidden_dim, hidden_dim)

                self.dit = DiT(
                    sde=None,
                    route_encoder=None,
                    # route_encoder=RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate,
                    #                            hidden_dim=config.hidden_dim),
                    depth=3,
                    output_dim=8,  # x, y, cos, sin
                    hidden_dim=hidden_dim,
                    heads=num_heads,
                    dropout=0.1,
                    model_type='x_start',
                    future_length=None
                )
            else:
                self.noise_embedding = MLPLayer(1, hidden_dim, hidden_dim)
                if self.ego_rel:
                    self.proj_in_m_delta = nn.Linear(m_delta_dim-4, self.hidden_dim)#MLPLayer(m_delta_dim-4, hidden_dim, hidden_dim)#
                else:
                    self.proj_in_m_delta = MLPLayer(m_delta_dim, self.hidden_dim,self.hidden_dim)#MLPLayer(m_delta_dim, hidden_dim, hidden_dim)#

                if self.use_graph:
                    self.use_padding = False

                    if self.use_all_pos:
                        self.interative_decoder = InterativeDecoder(hidden_dim,
                                                                    60,
                                                                    40, 60, num_freq_bands,
                                                                    num_layers, num_heads, head_dim,
                                                                    dropout, 0, m_delta_dim,
                                                                    20, 20,
                                                                    token_processor,
                                                                    )

                    else:

                        self.edge_encoder = EdgeEncoder(hidden_dim,
                                                        num_freq_bands,
                                                        use_a2a=True,
                                                        use_pl2a=True
                                                        )
                        self.to_out_m_delta = MLPLayer(hidden_dim, hidden_dim, m_delta_dim)

                        self.pt2a_attn_layers = nn.ModuleList(
                            [
                                AttentionLayer(
                                    hidden_dim=hidden_dim,
                                    num_heads=num_heads,
                                    head_dim=head_dim,
                                    dropout=dropout,
                                    bipartite=True,
                                    has_pos_emb=True,
                                )
                                for _ in range(num_layers)
                            ]
                        )

                        self.a2a_attn_layers = nn.ModuleList(
                            [
                                AttentionLayer(
                                    hidden_dim=hidden_dim,
                                    num_heads=num_heads,
                                    head_dim=head_dim,
                                    dropout=dropout,
                                    bipartite=False,
                                    has_pos_emb=True,
                                )
                                for _ in range(num_layers)
                            ]
                        )
                else:
                    module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                      hist_len=1000000)  # replace with gnn
                    self.entry_formers = ModuleList([copy.deepcopy(module) for i in range(num_layers)])
        else:

            self.proj_in_m_delta = nn.Linear(m_delta_dim, self.hidden_dim)

            self.proj_in_m_delta_2 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.proj_out_m_delta = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

            self.noise_emb = FourierEmbedding(input_dim=noise_dim, hidden_dim=hidden_dim,
                                              num_freq_bands=num_freq_bands)

            # self.interact_pt2m = nn.ModuleList(
            #     [TransformerDecoderLayerDiff(
            #         n_embd=hidden_dim,
            #         n_head=num_heads,
            #         ff_dim=4 * hidden_dim,
            #         dropout=0,
            #         layer_id=i,
            #     ) for i in range(num_layers)])
            module=RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=1000000)  # replace with gnn
            self.interact_pt2m = ModuleList([copy.deepcopy(module) for i in range(num_layers)])
            self.to_out_m_delta = SkipMLP(d_model=hidden_dim)

        self.use_noise=False

        if self.use_noise:
            self.denoising_steps= 20
            self.ft_denoising_steps= 10

            self.learn_explore_noise_from: int = self.denoising_steps-self.ft_denoising_steps
            self.initial_noise_scheduler_type: str = 'learn_decay'
            self.min_logprob_denoising_std: float = 0.08
            self.max_logprob_denoising_std: float = 0.16
            self.learn_explore_time_embedding: bool  = False
            self.device='cuda' if torch.cuda.is_available() else 'cpu'

            self.set_logprob_noise_levels()

            self.use_time_independent_noise = False

            self.init_exploration_noise_net()

        self.apply(weight_init)

    @torch.no_grad()
    def stochastic_interpolate(self, t):
        valid_noise_schedulers = ['vp', 'lin', 'const', 'const_schedule_itr', 'learn_decay']
        if self.initial_noise_scheduler_type == 'vp':
            a = 0.2  # 2.0
            std = torch.sqrt(a * t * (1 - t))
        elif self.initial_noise_scheduler_type == 'lin':
            k = 0.1
            b = 0.0
            std = k * t + b
        elif self.initial_noise_scheduler_type == 'const' or 'const_schedule_itr':
            std = torch.ones_like(t) * self.min_logprob_denoising_std
        else:
            raise ValueError(
                f"Invalid noise scheduler type {self.initial_noise_scheduler_type}, must be in the following: {valid_noise_schedulers}")
        return std

    @torch.no_grad()
    def set_logprob_noise_levels(self, force_level=None, verbose=False):
        '''
        create noise std for logrporbability calcualion.
        generate a tensor `self.logprob_noise_levels` of shape `[1, self.denoising_steps,  self.policy.horizion_steps x self.policy.act_dim]`
        '''
        self.logprob_noise_levels = torch.zeros(self.denoising_steps, device=self.device, requires_grad=False)

        steps = torch.linspace(0, 1 - 1 / self.denoising_steps, self.denoising_steps, device=self.device)
        for i, t in enumerate(steps):
            if force_level:
                self.logprob_noise_levels[i] = torch.tensor(force_level, device=self.device)
            else:
                self.logprob_noise_levels[i] = self.stochastic_interpolate(t)

        self.logprob_noise_levels = self.logprob_noise_levels.clamp(min=self.min_logprob_denoising_std,
                                                                    max=self.max_logprob_denoising_std)

        self.logprob_noise_levels = self.logprob_noise_levels.unsqueeze(0).unsqueeze(-1).repeat(1, 1,8)

    def init_exploration_noise_net(self):
        if self.use_time_independent_noise:
            noise_input_dim = self.policy.cond_enc_dim
            if not self.noise_hidden_dims:
                self.noise_hidden_dims = [16]
        else:
            if self.learn_explore_time_embedding:
                noise_input_dim = self.time_dim_explore + self.policy.cond_enc_dim
                self.time_embedding_explore = nn.Embedding(num_embeddings=self.denoising_steps,
                                                           embedding_dim=self.time_dim_explore,
                                                           device=self.device)
            else:
                noise_input_dim = self.hidden_dim
                # if not self.noise_hidden_dims:
                #     self.noise_hidden_dims = [int(np.sqrt(noise_input_dim ** 2 + self.policy.act_dim_total ** 2))]

        self.explore_noise_net=MLPLayer(self.hidden_dim,self.hidden_dim,self.output_dim)

        # self.explore_noise_net = ExploreNoiseNet(in_dim=noise_input_dim,
        #                                          out_dim=self.output_dim,
        #                                          logprob_denoising_std_range=[self.min_logprob_denoising_std,
        #                                                                       self.max_logprob_denoising_std],
        #                                          device=self.device,
        #                                          hidden_dims=self.hidden_dim,
        #                                          activation_type='Tanh')

    def normalize(self,input):
        return (input - self.normal_mean) / self.normal_scale

    def denormalize(self,input,nonego_type):

        # D, K = self.init_probs.shape
        #
        # idx = torch.multinomial(self.init_probs, len(input),replacement=True).transpose(0,1)#.squeeze(-1)
        # u = torch.rand((len(input),D), device=input.device)
        #
        # width = (self.init_max - self.init_min) / K
        #
        # x = self.init_min[None] + (idx.float() + u) * width[None]

        # for type_idx in range(3):
        #     mask=nonego_type == type_idx
        #     input[mask]=self.return_meanstd[type_idx].denormalize(input[mask])

        # x=self.return_meanstd.denormalize(input)

        input=input* self.normal_scale[None]+self.normal_mean[None]

        return input#[:,None]

    def drop_labels(self, labels,ego_embedding,mode):

        if mode==1:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
            out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        else:
            out=torch.full_like(labels, self.num_classes)

        out1 = ego_embedding#torch.where(drop[:,None].repeat(1,ego_embedding.shape[1]), torch.full_like(ego_embedding, 0), ego_embedding)#ego_embedding#

        return out,out1

    def padding(self, pos, heading, feature, batch, batch_num):
        lengths = torch.bincount(batch, minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        mask = torch.any(padding_features_a != 0, dim=-1)

        return padding_pos_a, padding_heading_a, padding_features_a,mask

    def get_input(self,tokenized_agent,non_ego,nonego_batch,nonego_type):

        batch_ego_pos=tokenized_agent["batch_ego_pos"]
        batch_ego_heading=tokenized_agent["batch_ego_heading"]

        initial_shape = tokenized_agent["initial_shape"][non_ego]
        non_ego_pos=tokenized_agent["initial_pos"][non_ego]
        non_ego_head=tokenized_agent["initial_heading"][non_ego]

        if self.use_all_pos:
            local_allpos,local_allheading = transform_to_local(tokenized_agent["all_pos"],
                                           tokenized_agent["all_heading"],
                                           batch_ego_pos,
                                           batch_ego_heading)

            local_allpos=local_allpos.reshape(-1,19,5,2)

            local_allheading=local_allheading.reshape(-1,19,5)

            inter_pos=local_allpos[:,:,0]

            inter_heading=local_allheading[:,:,0]

            rel_pos,rel_heading=transform_to_local(
                local_allpos[:,:,1:].flatten(0,1),
                local_allheading[:,:,1:].flatten(0,1),
                inter_pos.flatten(0,1),
                inter_heading.flatten(0,1)
            )

            rel_pos=rel_pos.reshape(-1,19,4,2)

            rel_heading=rel_heading.reshape(-1,19,4)

            n_agent=len(rel_pos)

            m_init = torch.cat([inter_pos.reshape(n_agent,-1), inter_heading.reshape(n_agent,-1).cos(),inter_heading.reshape(n_agent,-1).sin(),
                                rel_pos.reshape(n_agent, -1), rel_heading.reshape(n_agent, -1).cos(),rel_heading.reshape(n_agent, -1).sin(),
                                    initial_shape[:, :2]], dim=-1)
        else:
            local_pos, local_heading = transform_to_local(non_ego_pos,
                                                          non_ego_head,
                                                          batch_ego_pos,
                                                          batch_ego_heading,
                                                          )

            head_cosine = torch.cat([local_heading.cos().unsqueeze(-1), local_heading.sin().unsqueeze(-1)],
                                    dim=-1)  # [0,2]

            local_vel = rotate_to_local(tokenized_agent["initial_vel"][non_ego],  non_ego_head)

            #tokenized_agent["nonego_valid"] = None#torch.ones([len(local_vel),8],device=local_vel.device)

            if self.use_prev_head:
                prev_heading = wrap_angle(tokenized_agent["prev_heading"][non_ego],non_ego_head)

                local_vel = torch.cat([local_vel, prev_heading.cos()[:,None],prev_heading.sin()[:,None]], dim=-1)

            m_init = torch.cat([local_pos, head_cosine, initial_shape[:, :2], local_vel], dim=-1)

        tokenized_agent['nonego_type'] = nonego_type

        if self.use_scale:
            diff_input, m_init , nonego_batch= cluster_point_per_type(m_init, nonego_batch, tokenized_agent)
        else:
            diff_input = m_init

        # for type_idx in range(3):
        #
        #     self.return_meanstd[type_idx].update(m_init[nonego_type==type_idx])

        if not self.normal_initialized:

            # valid = ~torch.isnan(m_init)
            # count = valid.sum(0, keepdim=True).clamp_min(1)
            #
            # mean = torch.where(valid, m_init, 0).sum(0, keepdim=True) / count
            # std = torch.sqrt(torch.where(valid, (m_init - mean) ** 2, 0).sum(0, keepdim=True) / count).clamp_min(
            #     1e-8)
            # self.normal_mean.copy_(mean)
            # self.normal_scale.copy_(std)


            self.normal_mean.copy_(torch.mean(m_init, dim=0, keepdim=True))
            self.normal_scale.copy_(torch.std(m_init, dim=0, keepdim=True))
            self.normal_initialized = True

            # probs=batch_histogram_categorical(m_init,bins=100)
            # self.init_probs.copy_(probs)
            # self.init_min.copy_(m_init.amin(0))
            # self.init_max.copy_(m_init.amax(0))

        return diff_input,m_init,nonego_batch

    def forward(self,
                m_delta,
                beta,
                tokenized_agent: HeteroData,
                map_feature: Mapping[str, torch.Tensor],
                eval_mask=None,
                num_samples=1,
                mode=0
                ) -> Dict[str, torch.Tensor]:

        device = m_delta.device
        batch = tokenized_agent["nonego_batch"]
        type = tokenized_agent["nonego_type"]
        ego_embedding = tokenized_agent["ego_embedding"]
        num_graphs = tokenized_agent["num_graphs"]

        if eval_mask is not None:
            batch=batch[eval_mask]
            type=type[eval_mask]
            ego_embedding=ego_embedding[eval_mask]

        type,ego_embedding = self.drop_labels(type,ego_embedding,mode) if self.training else (type,ego_embedding)

        if self.use_roformer:
            m_delta=m_delta.reshape(m_delta.shape[0],-1)
            if self.use_dit:
                pos_pl, orient_pl, map_emb,map_mask=scene_enc

                map_emb=self.map_embed(torch.cat((pos_pl,orient_pl[:,:,None].cos(),orient_pl[:,:,None].sin(),map_emb),dim=-1))

                map_emb[~map_mask]=0

                lengths = torch.bincount(batch, minlength=num_graphs).tolist()

                feat_a = padding(m_delta, lengths, padding_value=0)  # b, n, d

                feat_b=padding(ego_embedding+self.type_a_emb(type), lengths, padding_value=0)

                if len(beta.shape)==0:
                    beta=torch.zeros_like(m_delta[:,0])+beta

                t=padding(beta.reshape(-1), lengths, padding_value=0) [:,0]

                mask = torch.any(feat_b != 0, dim=-1)

                feat_a =self.dit(feat_a, t, map_emb, feat_b)

                res=feat_a[mask][:,None]

            else:

                beta_emb_m = self.noise_embedding(beta.reshape(-1,1)) +self.type_a_emb(type)+ego_embedding

                if self.use_all_pos:
                    shape=m_delta[:,-2:]

                    n_step=19

                    pos_a=m_delta[:,:2*n_step].reshape(-1,n_step,2)

                    inter_heading_cos=m_delta[:,2*n_step:3*n_step]

                    inter_heading_sin=m_delta[:,3*n_step:4*n_step]

                    head_a=torch.atan2(inter_heading_sin,inter_heading_cos)

                    rel_pos=m_delta[:,4*n_step:12*n_step].reshape(-1,n_step,4,2)

                    rel_heading_cos=m_delta[:,12*n_step:16*n_step].reshape(-1,n_step,4)

                    rel_heading_sin=m_delta[:,16*n_step:20*n_step].reshape(-1,n_step,4)

                    n_step=pos_a.shape[1]
                    n_agent=pos_a.shape[0]

                    local_traj=torch.cat([rel_pos,rel_heading_cos[...,None],rel_heading_sin[...,None]],dim=-1).reshape(-1,n_step,16)

                    local_traj[torch.isnan(local_traj)]=-100

                    mask_a=~torch.isnan(head_a)#t,a

                    mask_t=mask_a.transpose(0,1)

                    head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

                    beta_emb_m=beta_emb_m[None].repeat(n_step,1,1)[mask_t]

                    shape=shape[:,None].repeat(1,n_step,1)

                    input_feature = torch.cat([shape,local_traj],dim=-1)

                    feat_a = self.proj_in_m_delta(input_feature).transpose(0,1)[mask_t]

                    feat_a_token = feat_a + beta_emb_m

                    agent_token_emb=None

                    batch_a = tokenized_agent["batch"]
                    batch_s_repeat = batch_a.unsqueeze(1).repeat(1, n_step)

                    batch_s = build_batch(batch_a, tokenized_agent["num_graphs"], n_step).reshape(-1,
                                                                                                  n_agent).transpose(0,
                                                                                                                     1)

                    all_features = [pos_a, head_a, head_vector_a, mask_a, batch_s_repeat, batch_s]

                    res, feat_a, rewards, weight, a2a_feature = self.interative_decoder(all_features,
                                                                                      feat_a_token,
                                                                                      agent_token_emb,
                                                                                      map_feature,
                                                                                      None,
                                                                                      0,
                                                                                      tokenized_agent)
                    pos_s=pos_a.transpose(0,1)[mask_t]

                    theta=head_a.transpose(0,1)[mask_t]

                else:
                    theta = torch.atan2(m_delta[:, 3], m_delta[:, 2])

                    pos_s = m_delta[:, :2]

                    if self.ego_rel:
                        feat_a=self.proj_in_m_delta(m_delta[:,4:])
                    else:
                        feat_a=self.proj_in_m_delta(m_delta)

                    feat_a = feat_a + beta_emb_m

                    if self.use_graph:

                        batch_pl = map_feature["batch"]
                        pos_pl = map_feature["position"]
                        orient_pl = map_feature["orientation"]
                        feat_map = map_feature["pt_token"]

                        head_vector_s = torch.stack([theta.cos(), theta.sin()], dim=-1)

                        edge_index_a2a, r_a2a, dist, relative_pos, r_a2a_nei, center_nei_pos, center_nei_heading = self.edge_encoder.build_interaction_edge(
                            pos_s=pos_s,  # [n_agent, n_step, 2]
                            head_s=theta,  # [n_agent, n_step]
                            head_vector_s=head_vector_s,  # [n_agent, n_step, 2]
                            batch_s=batch,  # [n_agent*n_step]
                            mask=None,  # [n_agent, n_step]
                            max_radius=60,
                            max_num_neighbors=20,
                            agent_train_mask=None,
                            layer_num=self.num_layers,
                            counter_feat_a=None,
                            dis_edge_mask=None
                        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

                        if batch_pl.max().item() != num_graphs - 1:
                            if "non_ego_valid" not in tokenized_agent.keys():
                                batch = tokenized_agent["repeat_batch"]

                                n_step = batch.shape[1]

                                pos_b = pos_s.reshape(n_step, -1, 2)
                                theta_b = theta.reshape(n_step, -1)

                                mask = torch.ones_like(batch).to(torch.bool)
                            else:

                                valid=tokenized_agent["non_ego_valid"]
                                n_step=valid.shape[0]

                                pos_global,theta_global=transform_to_global(
                                    pos_s,
                                    theta,
                                    tokenized_agent["batch_ego_pos"],
                                    tokenized_agent["batch_ego_heading"]
                                )

                                pos_b=torch.zeros([valid.shape[0],valid.shape[1],2],device=device)
                                theta_b=torch.zeros([valid.shape[0],valid.shape[1]],device=device)

                                pos_b[valid]=pos_global
                                theta_b[valid]=theta_global
                                mask=valid.transpose(0,1)
                                batch=tokenized_agent["batch_a"].unsqueeze(1).repeat(1, n_step)

                            pos_s=pos_b.transpose(0,1)
                            theta=theta_b.transpose(0,1)

                            head_vector_s=torch.stack([theta.cos(), theta.sin()], dim=-1)

                        else:
                            mask=None

                        edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                            pos_pl=pos_pl,  # [n_pl, 2]
                            orient_pl=orient_pl,  # [n_pl]
                            pos_a=pos_s,  # [n_agent, n_step, 2]
                            head_a=theta,  # [n_agent, n_step]
                            head_vector_a=head_vector_s,  # [n_agent, n_step, 2]
                            mask=mask,  # [n_agent, n_step]
                            batch_s=batch,  # [n_agent,n_step]
                            batch_pl=batch_pl,  # [n_pl*n_step]
                            pl2a_radius=40,
                            max_num_neighbors=20,
                            agent_train_mask=None,
                            layer_num=self.num_layers
                        )

                        for layer_i in range(self.num_layers):
                            feat_a = self.a2a_attn_layers[layer_i](feat_a, r_a2a, edge_index_a2a)

                            feat_a  = self.pt2a_attn_layers[layer_i]((feat_map, feat_a), r_pl2a, edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

                        res=self.to_out_m_delta(feat_a)

                    else:
                        pos_pl, orient_pl, map_emb,map_mask=map_feature

                        pos_a_b, heading_a_b, feat_a_b, mask_a_b = self.padding(pos_s, theta, feat_a, batch,
                                                                                num_graphs)  # b, n, d

                        for mod in self.entry_formers:
                            feat_a_b = mod(feat_a_b, pos_a_b,
                                           heading_a_b, mask_a_b,
                                           map_emb,
                                           pos_pl,
                                           orient_pl, map_mask
                                           )

                        feat_a = feat_a_b[mask_a_b]

                res_theta=torch.atan2(res[:,3],res[:,2])

                if "non_ego_valid" not in tokenized_agent.keys():
                    pos_s = m_delta[:, :2]
                    theta = torch.atan2(m_delta[:, 3], m_delta[:, 2])

                local_pos,local_theta = transform_to_global(
                    res[:,:2],
                    res_theta,
                    pos_s,
                    theta,
                )

                if self.use_all_pos:
                    new_pos=torch.zeros([n_step,n_agent,2],device=device)
                    new_shape=torch.zeros_like(new_pos)
                    new_theta=torch.zeros([n_step,n_agent],device=device)

                    new_pos[mask_t]=local_pos
                    new_theta[mask_t]=local_theta

                    new_shape[mask_t]=res[:,4:6]

                    mean_shape=new_shape.sum(dim=0)/mask_t.sum(dim=0)[:,None]

                    local_traj=torch.zeros([n_step,n_agent,16],device=device)

                    local_traj[mask_t]=res[:,6:]

                    rel_pos=local_traj[:,:,:8]

                    rel_heading_cos=local_traj[:,:,8:12]

                    rel_heading_sin=local_traj[:,:,12:]

                    res = torch.cat([new_pos.transpose(0,1).reshape(n_agent, -1), new_theta.transpose(0,1).reshape(n_agent, -1).cos(),
                                        new_theta.transpose(0,1).reshape(n_agent, -1).sin(),
                                        rel_pos.transpose(0,1).reshape(n_agent, -1), rel_heading_cos.transpose(0,1).reshape(n_agent, -1),
                                        rel_heading_sin.transpose(0,1).reshape(n_agent, -1).sin(),
                                        mean_shape], dim=-1)[:, None]

                else:
                    local_vel=res[:,6:]

                    res = torch.cat(
                        [local_pos, torch.cos(local_theta)[:, None], torch.sin(local_theta)[:, None], res[:, 4:6],
                         local_vel], dim=-1)[:, None]
        else:
            beta_emb = self.noise_emb(beta)
            # num_agents x 128

            m_delta = self.proj_in_m_delta(m_delta).view(-1, self.hidden_dim)

            if self.use_all_type:
                m_delta = m_delta +ego_embedding

            else:
                m_delta = m_delta + self.type_a_emb(type)+ego_embedding
            m_delta = self.proj_in_m_delta_2(m_delta)

            self.num_samples = num_samples

            if self.use_padding:
                pos_pl, orient_pl,map_mask, map_emb=scene_enc

                lengths=tokenized_agent["lengths"]

                #lengths = torch.bincount(batch, minlength=batch_size).tolist()

                m_delta = padding(m_delta, lengths, padding_value=0)  # b, n, d

                mask_agent = torch.any(m_delta != 0, dim=-1)

                beta_emb_m= padding(beta_emb, lengths, padding_value=0)

            else:
                pos_pl, orient_pl, batch_pl, feat_map=scene_enc

                x_pt = feat_map#.repeat(self.num_samples, 1)
                map_batch_list = batch_pl

                poly_cnt_per_batch = map_batch_list.bincount(minlength=batch_size)
                map_emb_batch = torch.split(x_pt, poly_cnt_per_batch.tolist())

                map_emb = pad_sequence(map_emb_batch, batch_first=True, padding_value=0)

                agent_cnt_per_batch = batch.bincount(minlength=batch_size)
                agent_emb_batch = torch.split(m_delta, agent_cnt_per_batch.tolist())

                m_delta = pad_sequence(agent_emb_batch, batch_first=True, padding_value=0)

                beta_emb_batch = torch.split(beta_emb, agent_cnt_per_batch.tolist())

                beta_emb_m = pad_sequence(beta_emb_batch, batch_first=True, padding_value=0)

            pos_emb = sinusoidal_embedding(m_delta.shape[1], self.hidden_dim).to(device).unsqueeze(0)
            m_delta += pos_emb

            B, N, D = m_delta.shape
            B, N_map, _ = map_emb.shape

            if self.use_padding:
                attn_mask_agent_layers = ~mask_agent
                attn_mask_map_layers = ~map_mask
            else:
                #mask_map_layers = []
                #mask_agent_layers = []

                attn_mask_map_layers = []
                attn_mask_agent_layers = []

                for i in range(batch_size):
                   # mask_attn_map_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                    #mask_attn_map_agent_i = mask_attn_map_agent_i.unsqueeze(-1).expand(-1, N_map)
                    mask_attn_map_pt_i = torch.arange(N_map).to(m_delta.device) < poly_cnt_per_batch[i]
                    attn_mask_map_layers.append(mask_attn_map_pt_i)
                    #mask_attn_map_pt_i = mask_attn_map_pt_i.unsqueeze(0).expand(N, -1)

                    #mask_attn_i = mask_attn_map_agent_i & mask_attn_map_pt_i
                    # mask_map_layers.append(mask_attn_i)

                    mask_attn_agent_i = torch.arange(N).to(m_delta.device) < agent_cnt_per_batch[i]
                    attn_mask_agent_layers.append(mask_attn_agent_i)
                    # mask_attn_agent_i = mask_attn_agent_i.unsqueeze(-1).expand(-1, N)
                    # mask_attn_i = mask_attn_agent_i & mask_attn_agent_i.t()
                    # mask_agent_layers.append(mask_attn_i)

                attn_mask_agent_layers = ~torch.stack(attn_mask_agent_layers)
                attn_mask_map_layers = ~torch.stack(attn_mask_map_layers)
                mask_agent = torch.arange(N).expand(B, N).to(m_delta.device) < agent_cnt_per_batch.unsqueeze(1)  # [B, N]
                #mask_agent = mask.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]


            # attn_mask_agent_layers1 = attn_mask_agent_layers.view(B, 1, N).to(torch.bool)
            # attn_mask_map_layers1 = attn_mask_map_layers.view(B, 1, 1, N_map). \
            #     expand(-1, self.num_heads * 2, N, -1)
            # #
            # # 0: don't attend others
            # if mode == 0:
            #     attn_mask_agent_layers = attn_mask_agent_layers + ~torch.eye(N).to(torch.bool).unsqueeze(0).to(
            #         m_delta.device)

            for i in range(self.num_layers):
                m_delta = m_delta + beta_emb_m

                # m_delta = self.interact_pt2m[i](
                #     tgt=m_delta,
                #     memory=map_emb,
                #     tgt_key_padding_mask=attn_mask_agent_layers,
                #     memory_key_padding_mask=attn_mask_map_layers
                # )

                # m_delta = self.interact_pt2m[i](x=m_delta, map_enc=map_emb,
                #                                 mask=attn_mask_agent_layers1,
                #                                 map_mask=attn_mask_map_layers1)
                m_delta = self.interact_pt2m[i](m_delta, torch.zeros_like(m_delta[:,:,:2]),
                               torch.zeros_like(m_delta[:,:,0]), ~attn_mask_agent_layers,
                               map_emb,
                               torch.zeros_like(map_emb[:,:,:2]),
                               torch.zeros_like(map_emb[:,:,0]), ~attn_mask_map_layers
                               )

            m_out_delta = m_delta[mask_agent]#.view(-1, D)  # [sum(agent_cnt_per_batch), D]

            out_m_delta = self.to_out_m_delta(m_out_delta)
            out_m_delta = out_m_delta.view(-1, self.num_samples, self.hidden_dim)
            res=self.proj_out_m_delta(out_m_delta)

        if self.use_noise :

            noise_std = self.explore_noise_net(feat_a)
            #
            # noise_std=self.denormalize(noise_std[:,None])
            noise_std = torch.exp(noise_std)+1e-3

            if not self.training:
                noise_std=torch.zeros_like(noise_std)

            res=torch.cat([res,noise_std[:,None]],dim=-1)

            if self.training:
                tokenized_agent["noise_feat"]=feat_a

        if torch.all(beta==0):
            tokenized_agent["noise_feat"] = feat_a

        return res

    def get_output(self, pred_init, tokenized_agent, non_ego):

        #pred_init = self.denormalize(pred_init)
        gt_initial_pos = tokenized_agent["initial_pos"].clone()
        gt_initial_heading = tokenized_agent["initial_heading"].clone()

        shape = tokenized_agent["initial_shape"].clone()
        batch_ego_pos = tokenized_agent["batch_ego_pos"]
        batch_ego_heading = tokenized_agent["batch_ego_heading"]

        if self.use_all_pos:
            n_step=19

            shape[non_ego, :2] = pred_init[:, -2:]

            pos_a = pred_init[:, :2 * n_step].reshape(-1, n_step, 2)

            inter_heading_cos = pred_init[:, 2 * n_step:3 * n_step]

            inter_heading_sin = pred_init[:, 3 * n_step:4 * n_step]

            head_a = torch.atan2(inter_heading_sin, inter_heading_cos)

            rel_pos = pred_init[:, 4 * n_step:12 * n_step].reshape(-1, n_step, 4, 2)

            rel_heading_cos = pred_init[:, 12 * n_step:16 * n_step].reshape(-1, n_step, 4)

            rel_heading_sin = pred_init[:, 16 * n_step:20 * n_step].reshape(-1, n_step, 4)

            rel_heading = torch.atan2(rel_heading_sin, rel_heading_cos)


            all_pos,all_heading=transform_to_global(
                pos_a,
                head_a,
                batch_ego_pos,
                batch_ego_heading,
            )

            rel_pos,rel_heading=transform_to_global(
                rel_pos.flatten(0,1),
                rel_heading.flatten(0,1),
                all_pos.flatten(0,1),
                all_heading.flatten(0,1),
            )

            rel_pos=rel_pos.reshape(-1, n_step, 4, 2)
            rel_heading=rel_heading.reshape(-1, n_step, 4)

            gt_initial_pos=torch.cat([all_pos[:,:,None],rel_pos],dim=2)

            gt_initial_heading=torch.cat([all_heading[:,:,None],rel_heading],dim=2)

            gt_initial_pos=gt_initial_pos.reshape(-1, n_step*5,2)[:,:-4]

            gt_initial_heading=gt_initial_heading.reshape(-1,n_step*5)[:,:-4]

            gt_initial_vel= (gt_initial_pos[:,15]-gt_initial_pos[:,10])/0.5
            gt_initial_idx=None
        else:
            pred_trans, pred_head, pred_shape, pred_vel = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4:6], \
                pred_init[..., 6:]

            pred_head = torch.atan2(pred_head[..., 1], pred_head[..., 0])

            shape[non_ego, :2] = pred_shape[:, :2]

            global_pos,global_heading=transform_to_global(
                pred_trans,
                pred_head,
                batch_ego_pos,
                batch_ego_heading,
            )

            if self.use_speed:
                global_pred_vel = torch.stack([global_heading.cos(), global_heading.sin()], dim=-1) * pred_vel
            else:
                global_pred_vel = rotate_to_global(pred_vel[:, :2], global_heading)

            gt_initial_pos[non_ego] = global_pos
            gt_initial_heading[non_ego] = global_heading

            gt_initial_vel = tokenized_agent["initial_vel"].clone()

            gt_initial_vel[non_ego] = global_pred_vel

            rel_vel = rotate_to_local(gt_initial_vel, gt_initial_heading)

            use_corner = False

            if use_corner:
                center_token_traj = tokenized_agent["token_traj"].flatten(1, 2)
            else:
                center_token_traj = tokenized_agent["token_traj"].mean(-2)

            if self.use_prev_head:
                rel_vel_heading=torch.atan2(pred_vel[:, 3], pred_vel[:, 2]) #local heading

                vel_heading = torch.atan2(rel_vel[:, 1], rel_vel[:, 0])

                vel_heading[non_ego]=rel_vel_heading # local heading

                pred_pos=transform_to_global(
                    center_token_traj,
                    None,
                    - rel_vel*0.5,
                    vel_heading,
                )[0]

                if use_corner:
                    token_traj=tokenized_agent["token_traj"]
                    gt_initial_idx = torch.linalg.norm(pred_pos.reshape(token_traj.shape)-token_traj[:,:1], dim=-1).mean(-1).argmin(-1)
                else:
                    gt_initial_idx = torch.linalg.norm(pred_pos, dim=-1).argmin(-1)
            else:
                # vel_heading = torch.atan2(rel_vel[:, 1], rel_vel[:, 0])
                # pred_pos=transform_to_global(
                #     center_token_traj,
                #     None,
                #     - rel_vel*0.5,
                #     vel_heading,
                # )[0]
                # gt_initial_idx = torch.linalg.norm(pred_pos, dim=-1).argmin(-1)

                gt_initial_idx = torch.linalg.norm(center_token_traj - rel_vel[:, None] * 0.5, dim=-1).argmin(-1)

                # import numpy as np
                # import matplotlib.pyplot as plt
                #
                # plt.scatter(x=center_token_traj[0,:,0].cpu().numpy(),y=center_token_traj[0,:,1].cpu().numpy())
                # plt.savefig('/home/ke/code/sim/src/1.png')

            gt_initial_pos,gt_initial_heading,gt_initial_idx=gt_initial_pos[:, None], gt_initial_heading[:, None],gt_initial_idx[:, None]

        return gt_initial_pos,gt_initial_heading,shape,gt_initial_vel,gt_initial_idx


class ExploreNoiseNet(nn.Module):
    '''
    Neural network to generate learnable exploration noise, conditioned on time embeddings and or state embeddings.
    \sigma(s,t) or \sigma(s)
    '''

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 logprob_denoising_std_range: list,  # [min_std, max_std]
                 device,
                 hidden_dims=[16],  # [8]  [32],
                 activation_type='Tanh'
                 ):
        super().__init__()
        self.device = device
        self.mlp_logvar = MLPLayer(in_dim,hidden_dims,out_dim)

        self.set_noise_range(logprob_denoising_std_range)

    def set_noise_range(self, logprob_denoising_std_range: list):
        self.logprob_denoising_std_range = logprob_denoising_std_range
        min_logprob_denoising_std = self.logprob_denoising_std_range[0]
        max_logprob_denoising_std = self.logprob_denoising_std_range[1]
        self.logvar_min = torch.nn.Parameter(
            torch.log(torch.tensor(min_logprob_denoising_std ** 2, dtype=torch.float32, device=self.device)),
            requires_grad=False)
        self.logvar_max = torch.nn.Parameter(
            torch.log(torch.tensor(max_logprob_denoising_std ** 2, dtype=torch.float32, device=self.device)),
            requires_grad=False)

    def forward(self, noise_feature: torch.Tensor):
        '''
        '''
        noise_logvar = self.mlp_logvar(noise_feature)
        noise_std = self.process_noise(noise_logvar)
        return noise_std

    def process_noise(self, noise_logvar):
        '''
        input:
            torch.Tensor([B, Ta , Da])   log \sigma^2
        output:
            torch.Tensor([B, 1, Ta * Da]), sigma, floating point values, bounded in [min_logprob_denoising_std, max_logprob_denoising_std]
        '''
        noise_logvar = noise_logvar
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2.0
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std


class SkipMLP(torch.nn.Module):
    def __init__(self, d_model=128, act_layer=nn.GELU):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)
        self.ac = act_layer()
        self.norm1 = nn.LayerNorm([d_model])
        self.norm2 = nn.LayerNorm([d_model])

    def forward(self, x):
        out = x + self.ac(self.linear(x))
        out = self.norm2(x + self.norm1(out))
        return out
