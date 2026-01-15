
from src.smart.layers import MLPLayer

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
import os
from src.smart.layers.init_diffusion import InitDiffusion
from argparse import ArgumentParser

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from src.smart.layers.autoencoder import AutoEncoder
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from src.smart.layers.autoencoder_utils import reparameterize


class PDInit(nn.Module):

    def __init__(self,token_processor) -> None:
        super(PDInit, self).__init__()

        parser = ArgumentParser()
        parser.add_argument('--train_raw_dir', type=str, default=None)
        parser.add_argument('--val_raw_dir', type=str, default=None)
        parser.add_argument('--test_raw_dir', type=str, default=None)
        parser.add_argument('--train_processed_dir', type=str, default=None)
        parser.add_argument('--val_processed_dir', type=str, default=None)
        parser.add_argument('--test_processed_dir', type=str, default=None)
        parser.add_argument('--accelerator', type=str, default='auto')
        parser.add_argument('--devices', type=str, default="1")
        parser.add_argument('--max_epochs', type=int, default=64)
        parser.add_argument('--check_val_every_n_epoch', type=int, default=1)

        parser.add_argument('--guid_sampling', choices=['no_guid', 'guid'], default='no_guid')
        parser.add_argument('--guid_task',
                            choices=['none', 'goal', 'target_vel', 'target_vego', 'rand_goal_rand', 'rand_goal_rand_o'],
                            default='none')
        parser.add_argument('--guid_method', choices=['none', 'ECM', 'ECMR'], default='none')
        parser.add_argument('--guid_plot', choices=['no_plot', 'plot'], default='no_plot')
        parser.add_argument('--plot', choices=['no_plot', 'plot'], default='no_plot')
        parser.add_argument('--path_pca_V_k', type=str, default='none')

        parser.add_argument('--cond_norm', type=int, default=0)
        parser.add_argument('--cost_param_costl', type=float, default=1.0)
        parser.add_argument('--cost_param_threl', type=float, default=1.0)

        parser.add_argument('--stage', type=str, default='init', choices=['init', 'traj'])

        self.add_model_specific_args(parser)

        args = parser.parse_args()

        self.joint_diffusion = InitDiffusion(args=args)

        hidden_dim=args.hidden_dim

        self.pos_embedding = MLPLayer(2, hidden_dim, hidden_dim)
        self.head_embedding = MLPLayer(2, hidden_dim, hidden_dim)

        self.latent_diffusion=True
        self.learn_autoencoder = token_processor.learn_autoencoder

        if self.latent_diffusion:

            num_encoder_blocks=2
            num_decoder_blocks=2
            latent_dim=8
            num_heads=8

            self.autoencoder=AutoEncoder(num_encoder_blocks,num_decoder_blocks,hidden_dim,latent_dim,num_heads)

        # self.normal_scale = torch.tensor([[80, 80, 1, 1, 22.929/2, 12.527/2, 3, 114.088/2]])
        # self.normal_mean = torch.tensor([[0, 0, 0, 0, 22.929/2, 12.527/2, 3, 114.088/2]])


        # self.normal_scale = torch.tensor([[80, 80, 1, 1, 9, 4, 3, 16]])
        # self.normal_mean = torch.tensor([[0, 0, 0, 0, 9, 4, 3, 16]])
        # min_speed: 0
        # max_speed: 114.088
        # min_length: -0.098
        # max_length: 22.929
        # min_width: 0.096
        # max_width: 12.527

        # normal_scale = torch.tensor([[35.015, 30.428, 35.051, 30.752, 35.069, 30.859,  0.279,  5.282]],device=non_ego.device)
        # normal_mean = torch.tensor([[3.678, 5.166, 3.667, 4.573, 3.401, 4.577, 1.736,  2.799]],device=non_ego.device)
        self.normal_scale = torch.tensor([[35.003, 30.584,  0.769,  0.627,  1.239,  0.380,  0.279,  5.282]])
        self.normal_mean = torch.tensor([[3.539,  4.872,  0.125, -0.002,  4.499,  2.018, 1.736,  2.799]])

        self.apply(weight_init)

    def get_data(self,tokenized_agent,non_ego,batch,initial_type,gt_initial_pos,gt_initial_heading,ego_position,ego_heading):
        shape = tokenized_agent["gt_initial_shape"].clone()

        real_shape = shape[non_ego]

        real_pos, real_heading = transform_to_local(gt_initial_pos[non_ego],
                                                    gt_initial_heading[non_ego],
                                                    ego_position[batch],
                                                    ego_heading[batch],
                                                    )

        init_trans = real_pos[:, :2]

        initial_shape = real_shape[:, :3]

        # initial_contour=cal_polygon_contour(init_trans[:,None,None],real_heading[:,None,None],real_shape[:,None,None,:2])

        delta_rot = real_heading.unsqueeze(-1)

        init_angle = torch.cat([delta_rot.cos(), delta_rot.sin()], dim=-1)  # [0,2]

        init_speed = real_shape[:, -2:].norm(dim=-1)

        m_init = torch.cat([init_trans, init_angle, initial_shape, init_speed[:, None]], dim=-1)

        # m_init = torch.cat([initial_contour[:,0,0,:3].flatten(1,2), initial_shape[:,-1:],init_speed[:,None]], dim=-1)

        m_init = (m_init - self.normal_mean.to(non_ego.device)) / self.normal_scale.to(non_ego.device)  # [-1,1]

        dist = init_trans[:, 0] + init_trans[:, 1]  # +200#torch.norm(init_trans, dim=-1)

        dist_max = dist.max() + 1

        sort_rank = batch.to(torch.float64) * dist_max * 3 + initial_type.to(torch.float64) * dist_max + dist.to(
            torch.float64)  # -ego_mask.float()#+dist#dist sorted

        sort_idx = sort_rank.argsort()

        m_init = m_init[sort_idx]

        tokenized_agent['nonego_type'] = initial_type[sort_idx]

        return m_init

    def forward(self, map_feature, tokenized_agent,map_range=100):

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        gt_initial_pos = tokenized_agent["gt_initial_pos"][:, 0].clone()
        gt_initial_heading = tokenized_agent["gt_initial_heading"][:, 0].clone()

        ego_mask = tokenized_agent["ego_mask"].clone()

        ego_position = gt_initial_pos[ego_mask]
        ego_heading = gt_initial_heading[ego_mask]

        pos_pl, orient_pl = transform_to_local(pos_pl,  # [:,None],
                                               orient_pl,  # [:,None],
                                               ego_position[batch_pl],
                                               ego_heading[batch_pl],
                                               )

        ego_dist = torch.linalg.norm(pos_pl, dim=-1)

        ego_dist_mask = ego_dist < map_range

        pos_pl = pos_pl[ego_dist_mask]
        orient_pl = orient_pl[ego_dist_mask]
        batch_pl = batch_pl[ego_dist_mask]
        feat_map = feat_map[ego_dist_mask]

        init_angle = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)  # [0,2]

        feat_map = feat_map + self.pos_embedding(pos_pl/80) + self.head_embedding(init_angle)

        map_feature = (pos_pl, orient_pl, batch_pl, feat_map)
        non_ego = ~ego_mask

        batch = tokenized_agent["batch"][non_ego].clone()

        initial_type = tokenized_agent["initial_type"][non_ego].clone()

        if self.training:
            m_init=self.get_data(tokenized_agent,non_ego,batch,initial_type,gt_initial_pos,gt_initial_heading,ego_position,ego_heading)
            data = (m_init, tokenized_agent['nonego_type'],0, feat_map, batch, batch_pl)

            if self.learn_autoencoder:
                loss_dict =self.autoencoder.loss(data)

                return loss_dict
            else:
                if self.latent_diffusion:
                    with torch.no_grad():
                        agent_mu, agent_log_var = self.autoencoder.forward(data,return_latents=True)
                        m_init = reparameterize(agent_mu, agent_log_var)

                    # agent_latents_mean=0
                    # agent_latents_std=1
                    # m_init = (agent_latents - agent_latents_mean) / agent_latents_std

                loss_diff_init, pred_init = self.joint_diffusion.get_loss(m_init,tokenized_agent,map_feature,eval_mask=non_ego)

                #pred_init=pred_init*normal_scale+normal_mean
                # num_samples=1
                # pred_trans, pred_head,pred_shape, pred_speed = pred_init[..., :2], pred_init[..., 2:4],pred_init[..., 4:7], pred_init[..., -1]

                # target_origin = init_trans[sort_idx].unsqueeze(1).repeat(1, num_samples, 1)
                # target_theta = init_angle[sort_idx].unsqueeze(1).repeat(1, num_samples,1)
                # target_speed = init_speed[sort_idx].unsqueeze(1).repeat(1, num_samples)
                #
                # loss_trans = torch.nn.HuberLoss()(pred_trans, target_origin)
                # loss_rot2 = torch.nn.HuberLoss()(pred_head, target_theta)
                # loss_speed = torch.nn.HuberLoss()(pred_speed,target_speed)
                #
                # loss_diff_trans = loss_diff_init[..., :2].mean()
                # loss_diff_theta = loss_diff_init[..., 2:4].mean()
                # loss_diff_speed = loss_diff_init[..., -1].mean()

                #weight=torch.tensor([[1,  1,  1, 1,  0.1,  0.1, 0.1,  1]],device=non_ego.device)
                loss_diff_init = loss_diff_init.mean()

                loss_trans=loss_rot2=loss_speed=loss_diff_trans=loss_diff_theta=loss_diff_speed=loss_diff_init

                return loss_diff_init,loss_trans,loss_rot2,loss_speed,loss_diff_trans,loss_diff_theta,loss_diff_speed
        else:
            if self.learn_autoencoder:
                m_init = self.get_data(tokenized_agent, non_ego, batch, initial_type, gt_initial_pos,
                                       gt_initial_heading, ego_position, ego_heading)

                data = (m_init, tokenized_agent['nonego_type'],0, feat_map, batch, batch_pl)

                agent_mu, agent_log_var = self.autoencoder.forward(data, return_latents=True)
                pred_init =reparameterize(agent_mu, agent_log_var)

            else:
                sort_rank = batch.to(torch.float64)  * 3 + initial_type.to(torch.float64)

                sort_idx = sort_rank.argsort()

                tokenized_agent['nonego_type']= initial_type[sort_idx]

                pred_init = self.joint_diffusion.sample(num_samples = 1, data=tokenized_agent, scene_enc=map_feature,
                                                        sampling='ddim',
                                                        stride=10, eval_mask=non_ego,
                                                        if_output_diffusion_process=False,
                                                        reverse_steps=None)[:,0]

            if self.latent_diffusion:
                pred_init = self.autoencoder.forward_decoder(pred_init,   tokenized_agent['nonego_type'],0, feat_map,batch,batch_pl)

            #pred_init=m_init
            pred_init=pred_init*self.normal_scale.to(non_ego.device)+self.normal_mean.to(non_ego.device)

            pred_trans, pred_head,pred_shape, pred_speed = pred_init[..., :2], pred_init[..., 2:4],pred_init[..., 4:7], pred_init[..., -1]
            pred_head = torch.atan2(pred_head[..., 1], pred_head[..., 0])
            # pred_count=pred_init[..., :6].reshape(-1,3,2)
            #
            # # center (diagonal midpoint)
            # pred_trans = 0.5 * (pred_count[:,0] + pred_count[:,2])
            #
            # diff_xy_next = pred_count[:,1] - pred_count[:,2]#left_front, right_front, right_back, left_back
            #
            # # width & length
            # width = torch.norm(pred_count[:,1]-pred_count[:,0],dim=-1)
            # length = torch.norm(diff_xy_next,dim=-1)
            #
            # pred_head = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            #
            # pred_speed=pred_init[..., -1]
            #
            # pred_shape=torch.stack([length,width,pred_init[..., 6]], dim=-1)

            global_pos,global_heading=transform_to_global(
                pred_trans,
                pred_head,
                ego_position[batch],
                ego_heading[batch],
            )

            gt_initial_pos[non_ego]=global_pos
            gt_initial_heading[non_ego]=global_heading

            gt_initial_speed=tokenized_agent["gt_initial_speed"].clone()

            gt_initial_speed[non_ego] =pred_speed

            shape=tokenized_agent["shape"].clone()

            shape[non_ego]=pred_shape[:,:3]

            tokenized_agent["shape"]= shape

            gt_initial_idx=tokenized_agent["gt_initial_idx"][:,0].clone()

            type=tokenized_agent["type"].clone()

            type[non_ego]= tokenized_agent['nonego_type']

            tokenized_agent["type"]= type


            # if self.token_processor.pred_vel:
            #     vel=fake_shape[:,3:]
            #
            #     center_token_traj=tokenized_agent["token_traj"][non_ego].mean(-2)
            #
            #     gt_initial_idx[non_ego]=torch.linalg.norm(center_token_traj-vel[:,None],dim=-1).argmin(-1)
            #     gt_initial_speed[non_ego]=vel.norm(dim=-1)/0.5
            # gt_initial_pos = tokenized_agent["gt_initial_pos"][:, 0]
            # gt_initial_heading = tokenized_agent["gt_initial_heading"][:, 0]
            # gt_initial_speed=tokenized_agent["gt_initial_speed"]

            return gt_initial_pos[:, None], gt_initial_heading[:, None],gt_initial_idx[:, None],gt_initial_speed




    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, default='argoverse_v2')
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--init_timestep', type=int, default=50)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--pl2pl_radius', type=float, default=150)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--submission_dir', type=str, default='./')
        parser.add_argument('--submission_file_name', type=str, default='submission')
        parser.add_argument('--qcnet_map_ckpt_path', type=str, required=False)
        parser.add_argument('--num_denoiser_layers', type=int, default=3)
        parser.add_argument('--num_diffusion_steps', type=int, default=100)
        parser.add_argument('--beta_1', type=float, default=1e-4)
        parser.add_argument('--beta_T', type=float, default=0.05)
        parser.add_argument('--diff_type', default='vd')#['opsd', 'opd', 'vd'])
        parser.add_argument('--sampling', default='ddpm')#['ddpm', 'ddim'])
        parser.add_argument('--sampling_stride', type=int, default=10)
        parser.add_argument('--num_eval_samples', type=int, default=6)
        parser.add_argument('--train_agent', choices=['all', 'eval'], default='all')


        parser.add_argument('--m_dim', type = int,default = 10)

        return parent_parser