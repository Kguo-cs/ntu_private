import torch
import torch.nn as nn

import numpy as np
from .diffusion_helpers import FactorizedDiTBlock, FinalLayer, LabelEmbedder, TimestepEmbedder, \
    get_1d_sincos_pos_embed_from_grid, TwoLayerResMLP,get_indices_within_scene,weight_init
from src.smart.diffusion.dit.autoencoder import get_edgeindex
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_global,
    rotate_to_local,
    weight_init
)

class DiT(nn.Module):

    def __init__(self,hidden_dim):
        super(DiT, self).__init__()
        # self.cfg = cfg
        # self.cfg_model = self.cfg.model
        # self.cfg_dataset = self.cfg.dataset

        self.use_rel_ego=False
        self.use_scale=False
        self.use_all_type=False

        self.agent_hidden_dim=hidden_dim
        self.dropout=0
        self.num_heads=8
        self.agent_num_heads=8
        self.num_l2l_blocks=1
        self.num_factorized_dit_blocks=2
        self.agent_latent_dim=8


        self.emb_drop = nn.Dropout(self.dropout)
        # # Condition on scene type
        # self.scene_type_embedder = LabelEmbedder(self.cfg_dataset.num_map_ids * 2, hidden_dim,
        #                                          self.cfg_model.label_dropout) ## 2type: either nocturne_compatible (1) or not (0) used for sampling GPU-Drive compatible scenes
        self.scene_type_embedder = LabelEmbedder(3, hidden_dim,  0.1) ## 2type: either nocturne_compatible (1) or not (0) used for sampling GPU-Drive compatible scenes

        # Condition on number of agents and lanes
        self.num_agents_embedder = LabelEmbedder(350, hidden_dim, 0)
        self.num_lanes_embedder =nn.Linear(1, hidden_dim) #LabelEmbedder(450, hidden_dim, 0)

        # Diffusion timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_dim)
        # Used because agent embedding is smaller than lane embedding
        self.downsample_c = nn.Linear(hidden_dim, self.agent_hidden_dim)

        # Embed agent and lane latents
        # self.lane_embedder = TwoLayerResMLP(self.cfg_model.lane_latent_dim, hidden_dim)
        self.agent_embedder = TwoLayerResMLP(self.agent_latent_dim, self.agent_hidden_dim)

        # # These will be overwritten by sin/cos positional encodings
        # self.pos_emb_lane = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_lanes, hidden_dim),
        #                                  requires_grad=False)
        # self.pos_emb_agent = nn.Parameter(torch.zeros(self.cfg_dataset.max_num_agents, self.agent_hidden_dim),
        #                                   requires_grad=False)

        # factorized dit blocks
        self.blocks = nn.ModuleList([
            FactorizedDiTBlock(
                hidden_dim,
                self.agent_hidden_dim,
                self.num_heads,
                self.agent_num_heads,
                self.dropout,
                mlp_ratio=4,
                num_l2l_blocks=self.num_l2l_blocks
            ) for _ in range(self.num_factorized_dit_blocks)
        ])
        self.register_buffer("normal_mean", torch.zeros(1, 8))
        self.register_buffer("normal_scale", torch.ones(1, 8))

        # noise prediction heads
        self.pred_agent_noise = FinalLayer(self.agent_hidden_dim, self.agent_latent_dim)
        # self.pred_lane_noise = FinalLayer(hidden_dim, self.cfg_model.lane_latent_dim)
        self.initialize_weights()




    def get_input(self,tokenized_agent):

        batch_ego_pos=tokenized_agent["batch_ego_pos"]
        batch_ego_heading=tokenized_agent["batch_ego_heading"]
        non_ego=tokenized_agent["non_ego"]

        initial_shape = tokenized_agent["initial_shape"][non_ego]
        non_ego_pos=tokenized_agent["initial_pos"][non_ego]
        non_ego_head=tokenized_agent["initial_heading"][non_ego]

        local_pos, local_heading = transform_to_local(non_ego_pos,
                                                      non_ego_head,
                                                      batch_ego_pos,
                                                      batch_ego_heading,
                                                      )

        head_cosine = torch.cat([local_heading.cos().unsqueeze(-1), local_heading.sin().unsqueeze(-1)],
                                dim=-1)  # [0,2]

        if "local_vel" in tokenized_agent.keys():
            local_vel=tokenized_agent["local_vel"][non_ego]
        else:
            local_vel = rotate_to_local(tokenized_agent["initial_vel"][non_ego],  non_ego_head)

        m_init = torch.cat([local_pos, head_cosine, initial_shape[:, :2], local_vel], dim=-1)

        diff_input = m_init

        if torch.all(self.normal_mean==0):
            self.normal_mean.copy_(torch.mean(m_init, dim=0, keepdim=True))
            self.normal_scale.copy_(torch.std(m_init, dim=0, keepdim=True))

        return diff_input,m_init

    def normalize(self,input):
        return (input - self.normal_mean[None]) / self.normal_scale[None]

    def denormalize(self,input,nonego_type=None):
        input=input* self.normal_scale[None]+self.normal_mean[None]

        return input#[:,None]

    def initialize_weights(self):
        """ Custom initialization for DiT model"""

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) lane and agent pos_embed by sin-cos embedding:
        # pos_emb_lane = get_1d_sincos_pos_embed_from_grid(self.pos_emb_lane.shape[-1],
        #                                                  np.arange(self.pos_emb_lane.shape[0]))
        # self.pos_emb_lane.data.copy_(torch.from_numpy(pos_emb_lane).float())
        # pos_emb_agent = get_1d_sincos_pos_embed_from_grid(self.pos_emb_agent.shape[-1],
        #                                                   self.cfg_dataset.max_num_lanes + np.arange(
        #                                                       self.pos_emb_agent.shape[0]))
        # self.pos_emb_agent.data.copy_(torch.from_numpy(pos_emb_agent).float())

        # Initialize label embedding table:
        nn.init.normal_(self.scene_type_embedder.embedding_table.weight, std=0.02)

        # Initialize num lane and num agent embedding tables:
        nn.init.normal_(self.num_agents_embedder.embedding_table.weight, std=0.02)
       # nn.init.normal_(self.num_lanes_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            # for l2l_block in block.l2l_blocks:
            #     nn.init.constant_(l2l_block.adaLN_modulation[-1].weight, 0)
            #     nn.init.constant_(l2l_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.a2a_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.l2a_block.adaLN_modulation[-1].bias, 0)
            # nn.init.constant_(block.a2l_block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.a2l_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.pred_agent_noise.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.pred_agent_noise.linear.weight, 0)
        nn.init.constant_(self.pred_agent_noise.linear.bias, 0)

        # nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.pred_lane_noise.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.pred_lane_noise.linear.weight, 0)
        # nn.init.constant_(self.pred_lane_noise.linear.bias, 0)

    def forward(self,#z, t, tokenized_agent, scene_enc
                x_agent,
                agent_timestep,
                data,
                x_lane,
                lane_timestep=None,
                unconditional=False):
        """ Forward pass of the DiT model."""
        agent_batch = data["nonego_batch"]
        nonego_type=data["nonego_type"]
        batch_size=data["num_graphs"]
        ego_embedding=data["ego_embedding"]
        lane_batch=data["lane_batch"]

        agent_timestep=agent_timestep[:,0,0]
        x_agent=x_agent[:,0]

        #agent_batch, lane_batch,batch_size,nonego_type_sorted,ego_embedding=data
        a2a_edge_index, l2a_edge_index,l2l_edge_index,pos_emb_agent=get_edgeindex(agent_batch,lane_batch,batch_size,use_transformer=False,hidden_dim=self.agent_hidden_dim)

        # lane_idx_batch = get_indices_within_scene(lane_batch)
        # agent_idx_batch = get_indices_within_scene(agent_batch)

        # add positional embeddings
        #pos_emb_lane = self.pos_emb_lane[lane_idx_batch]
        #pos_emb_agent = self.pos_emb_agent[agent_idx_batch]
        # x_lane = self.lane_embedder(x_lane[:, 0]) + pos_emb_lane
        #x_agent = self.agent_embedder(x_agent[:, 0]) #+ pos_emb_agent

        x_agent = self.agent_embedder(x_agent)+pos_emb_agent

        # scene_idx = self.cfg_dataset.num_map_ids * data['lg_type'].long() + data['map_id'].long()
        # scene_type = self.scene_type_embedder(scene_idx.long(), train=self.training,
        #                                       force_drop_ids=torch.ones_like(scene_idx) if unconditional else None)
        agent_scene_type = self.scene_type_embedder(nonego_type, train=self.training,
                                              force_drop_ids=torch.ones_like(nonego_type) if unconditional else None)
        # agent_batch = data['agent'].batch
        # lane_batch = data['lane'].batch
        # agent_scene_type = scene_type[agent_batch]
        # lane_scene_type = scene_type[lane_batch]
        num_agents = torch.bincount(agent_batch, minlength=batch_size)
        num_lanes = torch.bincount(lane_batch, minlength=batch_size)

       # print(num_lanes.max(),num_agents.max())

        # num_agents = data['num_agents'].long()
        # num_lanes = data['num_lanes'].long()
        num_agents_emb = self.num_agents_embedder(num_agents, train=self.training)[agent_batch]
        num_lanes_emb =self.num_lanes_embedder(num_lanes[:,None].to(torch.float32))[lane_batch] #self.num_lanes_embedder(num_lanes, train=self.training)[lane_batch]

        if lane_timestep is  None:
            lane_timestep=torch.ones_like(lane_batch)

        # embedding of timestep
        t =self.t_embedder(torch.cat([lane_timestep, agent_timestep], dim=-1))
        # embedding of number of agents and lanes
        n = torch.cat([num_lanes_emb, num_agents_emb], dim=0)
        # embedding of scene type
        y = torch.cat([torch.zeros_like(num_lanes_emb), agent_scene_type+ego_embedding], dim=0)

        # l2l_edge_index = data['lane', 'to', 'lane'].edge_index
        # a2a_edge_index = data['agent', 'to', 'agent'].edge_index
        # l2a_edge_index = data['lane', 'to', 'agent'].edge_index.clone()
        # l2a_edge_index[1] = l2a_edge_index[1] + x_lane.shape[0]

        # conditioning vector for DiT block
        c = t + n+ y
        # # necessary for A2A and L2A attention
        c_small = self.downsample_c(c)

        # apply dropout
        #x_lane = self.emb_drop(x_lane)
        x_agent = self.emb_drop(x_agent)


        # factorized dit block processing
        for block in self.blocks:
            x_lane, x_agent = block(
                x_lane,
                x_agent,
                c,
                c_small,
                l2l_edge_index,
                a2a_edge_index,
                l2a_edge_index)

        # decode the noise as in the original DiT paper
        #c_lane = c[:x_lane.shape[0]]
        c_agent = c_small[x_lane.shape[0]:]
        #x_lane = self.pred_lane_noise(x_lane, c_lane).unsqueeze(1)
        x_agent = self.pred_agent_noise(x_agent, c_agent)#.unsqueeze(1)

        return x_agent[:,None]


    def get_output(self, pred_init, tokenized_agent):
        non_ego=tokenized_agent["non_ego"]
        gt_initial_pos = tokenized_agent["initial_pos"].clone()
        gt_initial_heading = tokenized_agent["initial_heading"].clone()

        shape = tokenized_agent["initial_shape"].clone()
        batch_ego_pos = tokenized_agent["batch_ego_pos"]
        batch_ego_heading = tokenized_agent["batch_ego_heading"]

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

        gt_initial_pos[non_ego] = global_pos
        gt_initial_heading[non_ego] = global_heading

        if "local_vel" in tokenized_agent.keys():
            rel_vel = tokenized_agent["local_vel"].clone()

            rel_vel[non_ego] = pred_vel[:, :2]

            gt_initial_vel=rotate_to_global(rel_vel, gt_initial_heading)

        else:
            global_pred_vel = rotate_to_global(pred_vel[:, :2], global_heading)

            gt_initial_vel = tokenized_agent["initial_vel"].clone()

            gt_initial_vel[non_ego] = global_pred_vel

            rel_vel = rotate_to_local(gt_initial_vel, gt_initial_heading)

        use_corner = False

        if use_corner:
            center_token_traj = tokenized_agent["token_traj"].flatten(1, 2)
        else:
            center_token_traj = tokenized_agent["token_traj"].mean(-2)

        gt_initial_idx = torch.linalg.norm(center_token_traj - rel_vel[:, None] * 0.5, dim=-1).argmin(-1)

        gt_initial_pos,gt_initial_heading,gt_initial_idx=gt_initial_pos[:, None], gt_initial_heading[:, None],gt_initial_idx[:, None]

        return gt_initial_pos,gt_initial_heading,shape,gt_initial_vel,gt_initial_idx
