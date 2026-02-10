from sympy.testing.pytest import tooslow

from src.smart.layers import MLPLayer

import torch
import torch.nn as nn
from src.smart.layers.diffuser import InitDiffusion
from argparse import ArgumentParser

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from src.smart.layers.autoencoder import AutoEncoder
from src.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from src.smart.my_model.ldm import LDM
from src.smart.utils.earth_match import get_matching_loss
from src.smart.layers.discriminator import InitDiscriminator,InitGeneator
from src.smart.layers.relative_transformer import padding
import torch.nn.functional as F
from torch_geometric.nn.pool import knn_graph,knn

class PDInit(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                num_heads: int,
                num_freq_bands,
                token_processor,
        ) -> None:
        super(PDInit, self).__init__()

        parser = ArgumentParser()
        self.add_model_specific_args(parser)
        args = parser.parse_args()

        self.latent_diffusion=False
        self.use_gan = True


        self.learn_autoencoder = token_processor.learn_autoencoder
        if self.learn_autoencoder:
            self.use_gan = False
            self.latent_diffusion = True

        if self.latent_diffusion:

            num_encoder_blocks=2
            num_decoder_blocks=2
            latent_dim=8
            num_heads=8

            self.autoencoder=AutoEncoder(num_encoder_blocks,num_decoder_blocks,hidden_dim,latent_dim,num_heads)

        self.use_count=False

        if self.use_count:
            self.normal_scale = torch.tensor([[35.105, 29.893, 35.130, 30.152, 35.201, 30.277,  5.155,  0.281]])
            self.normal_mean = torch.tensor([[2.797e+00,  1.178e+00,  2.839e+00,  6.951e-01,  2.621e+00,  6.769e-01,
         2.648e+00, -1.963e-03]])

        else:

            self.normal_scale = torch.tensor([[35.013, 30.234, 0.764, 0.638, 1.326, 0.417, 4.860, 0.230]])
            self.normal_mean = torch.tensor([[2.896e+00, 8.604e-01, 9.726e-02, 9.904e-04, 4.409e+00, 1.989e+00,
                                              2.447e+00, 1.321e-03]])

        if self.latent_diffusion:
            self.agent_latents_scale=torch.tensor([[0.981, 0.982, 0.992, 1.012, 0.979, 0.950, 0.977, 0.975]])
            self.agent_latents_mean=torch.tensor([[0.026,  0.015,  0.001,  0.061,  0.010,  0.030, -0.021,  0.035]])

        # self.agent_latents_scale=torch.tensor([[2.951, 2.383, 3.042, 2.819, 2.614, 2.401, 2.673, 2.773]])
        # self.agent_latents_mean=torch.tensor([[-0.059,  0.043, -0.014,  0.116,  0.314,  0.155,  0.274, -0.091]])


        if self.use_gan:
            self.D=InitDiscriminator(hidden_dim,num_heads,num_freq_bands,token_processor)

        self.use_dit=False
        self.global_step=0
        self.Gamma=1
        self.use_Rp=False

        self.density_conditioned=False

        if self.use_dit:
            self.G = LDM()
        else:
            self.G = InitDiffusion(args=args)

    def padding(self,pos,heading,feature,batch,batch_num):
        lengths = torch.bincount(batch,minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        return padding_pos_a, padding_heading_a, padding_features_a

    def get_data(self,tokenized_agent,non_ego,batch,nonego_type,gt_initial_pos,gt_initial_heading,ego_position,ego_heading):

        real_vel=tokenized_agent["local_vel"][non_ego]

        initial_shape = tokenized_agent["initial_shape"][non_ego]

        non_ego_pos=gt_initial_pos[non_ego]
        non_ego_head=gt_initial_heading[non_ego]

        init_trans, real_heading = transform_to_local(non_ego_pos,
                                                    non_ego_head,
                                                    ego_position[batch],
                                                    ego_heading[batch],
                                                    )

        if self.use_count:
            initial_contour=cal_polygon_contour(init_trans[:,None,None],real_heading[:,None,None],initial_shape[:,None,None,:2])
            m_init = torch.cat([initial_contour[:, 0, 0, :3].flatten(1, 2), real_vel], dim=-1)
        else:
            delta_rot = real_heading.unsqueeze(-1)

            init_angle = torch.cat([delta_rot.cos(), delta_rot.sin()], dim=-1)  # [0,2]

            m_init = torch.cat([init_trans, init_angle, initial_shape[:,:2], real_vel], dim=-1)

        m_init = (m_init - self.normal_mean.to(non_ego.device)) / self.normal_scale.to(non_ego.device)  # [-1,1]

        dist = torch.norm(init_trans, dim=-1)#init_trans[:, 0] + init_trans[:, 1]  # +200#

        dist_max = dist.max().abs() +dist.min().abs()+1

        sort_rank = batch.to(torch.float64) * dist_max * 3 + nonego_type.to(torch.float64) * dist_max + dist.to(
            torch.float64)  # -ego_mask.float()#+dist#dist sorted

        sort_idx = sort_rank.argsort()

        #sort_idx=torch.arange(len(sort_idx))
        #sort_idx = torch.arange(len(sort_idx), device=non_ego.device)

        m_init = m_init[sort_idx]

        tokenized_agent['nonego_type_sorted'] = nonego_type[sort_idx]

        return m_init,sort_idx

    def get_gan_loss(self,m_init,x_pred,map_feature, normal_scale,normal_mean,tokenized_agent,non_ego,rec_loss=None,t=None,t_batch=None):
        if self.D.use_entry_former:
            pos_pl, orient_pl, batch_pl, feat_map=map_feature
            num_graphs=tokenized_agent["num_graphs"]

            pos_pl, orient_pl, feat_map = self.padding(pos_pl, orient_pl, feat_map, batch_pl, num_graphs)

            map_mask = torch.any(feat_map != 0, dim=-1)

        RealSamples = m_init * normal_scale + normal_mean
        FakeSamples = x_pred * normal_scale + normal_mean
        old_nonego_type_sorted = tokenized_agent["nonego_type_sorted"].clone()
        old_batch = tokenized_agent["batch"][non_ego]

        if t is not None:
            gap = -1

            low_noise_mask = t[:, 0] > gap
            low_noise_map_mask = t_batch[:, 0] > gap
            map_feature = (pos_pl[low_noise_map_mask], orient_pl[low_noise_map_mask], feat_map[low_noise_map_mask],
                           map_mask[low_noise_map_mask])
            tokenized_agent["num_graphs"] = low_noise_map_mask.sum()
            _, new_batch = torch.unique(old_batch[low_noise_mask], sorted=True, return_inverse=True)

            tokenized_agent["nonego_batch"] = new_batch
            tokenized_agent["nonego_type_sorted"] = tokenized_agent["nonego_type_sorted"][low_noise_mask]
            RealSamples=RealSamples[low_noise_mask]
            FakeSamples=FakeSamples[low_noise_mask]
        else:
            tokenized_agent["nonego_batch"] = old_batch
            if self.D.use_entry_former:
                map_feature = (pos_pl, orient_pl, feat_map, map_mask)

        agent_n=len(FakeSamples)

        if self.global_step % 10 == 0:

            #RealSamples = RealSamples.detach().requires_grad_(True)
            #FakeSamples = FakeSamples.detach().requires_grad_(True)

            RealLogits,real_weight = self.D(RealSamples, map_feature, tokenized_agent,return_weight=True)
            FakeLogits,fake_weight = self.D(FakeSamples, map_feature, tokenized_agent,return_weight=True)

            # R1Penalty = (self.Gamma / 2) * self.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
            # R2Penalty = (self.Gamma / 2) * self.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)

            if self.use_Rp:
                RelativisticLogits = RealLogits - FakeLogits
                AdversarialLoss = nn.functional.softplus(-RelativisticLogits).mean()
            else:
                FakeLogits, fake_interact_logits = FakeLogits[:agent_n], FakeLogits[agent_n:]
                RealLogits, real_interact_logits = RealLogits[:agent_n], RealLogits[agent_n:]

                fake_bce_loss = F.binary_cross_entropy_with_logits(FakeLogits, torch.zeros_like(FakeLogits),
                                                              reduction='mean')
                real_bce_loss = F.binary_cross_entropy_with_logits(RealLogits, torch.ones_like(RealLogits),
                                                              reduction='mean')
                AdversarialLoss =fake_bce_loss+real_bce_loss
                # AdversarialLoss = FakeLogits.mean() - RealLogits.mean()
                if len(fake_interact_logits) > 0:
                    fake_loss = F.binary_cross_entropy_with_logits(
                        fake_interact_logits,
                        torch.zeros_like(fake_interact_logits),
                        reduction='none'
                    )

                    fake_interact_bce_loss = (fake_loss * fake_weight).sum() / agent_n

                    real_loss = F.binary_cross_entropy_with_logits(
                        real_interact_logits,
                        torch.ones_like(real_interact_logits),
                        reduction='none'
                    )

                    real_interact_bce_loss= (real_loss * real_weight).sum() / agent_n

                    AdversarialLoss =  AdversarialLoss +fake_interact_bce_loss +real_interact_bce_loss#

            w = 1  # 0.1+(1-self.global_step/10000.0)

            R2Penalty = R1Penalty = torch.tensor(0.0, device=RealLogits.device)

            loss = (AdversarialLoss, w * R2Penalty.mean(), w * R1Penalty.mean(),FakeLogits,RealLogits)  # cosine schedule
        else:
            if self.global_step>-1:
                self.D.eval()
                FakeLogits,fake_weight = self.D(FakeSamples, map_feature, tokenized_agent,return_weight=True)

                if self.use_Rp:
                    RealLogits = self.D(RealSamples, map_feature, tokenized_agent)
                    RelativisticLogits = FakeLogits - RealLogits
                    AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
                    loss = AdversarialLoss.mean()
                else:
                    FakeLogits, fake_interact_logits = FakeLogits[:agent_n], FakeLogits[agent_n:]
                    # fake_bce_loss =  F.binary_cross_entropy_with_logits(FakeLogits, torch.zeros_like(FakeLogits),
                    #                                           reduction='mean')
                    fake_bce_loss=FakeLogits
                    loss=-fake_bce_loss.mean()
                    if len(fake_interact_logits) > 0:
                        # fake_loss = F.binary_cross_entropy_with_logits(
                        #     fake_interact_logits,
                        #     torch.zeros_like(fake_interact_logits),
                        #     reduction='none'
                        # )
                        fake_loss =fake_interact_logits

                        fake_interact_bce_loss = (fake_loss * fake_weight).sum() / agent_n

                        loss=loss-fake_interact_bce_loss
                self.D.train()
            else:
                loss=torch.tensor(0.0, device=FakeSamples.device)

            # match_loss,pos_loss,heading_loss,shape_loss,vel_loss=get_matching_loss(tokenized_agent["nonego_type_sorted"], new_batch,
            #                                                                        FakeSamples,RealSamples
            #                                                                        )
            match_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_matching_loss(
                old_nonego_type_sorted,
                old_batch,
                x_pred * normal_scale + normal_mean,
                m_init * normal_scale + normal_mean,
                use_col=False
            )

            # match_loss= pos_loss= heading_loss=shape_loss= vel_loss=torch.tensor(0.0, device=non_ego.device)

            loss = (loss, match_loss, pos_loss, heading_loss, shape_loss, vel_loss)
        self.global_step += 1

        return loss

    def forward(self,  tokenized_agent,map_range=100):

        map_feature=tokenized_agent["initial_map_feature"]

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        gt_initial_pos = tokenized_agent["initial_pos"].clone()
        gt_initial_heading = tokenized_agent["initial_heading"].clone()

        num_graphs=tokenized_agent["num_graphs"]

        ego_mask = tokenized_agent["ego_mask"]
        non_ego = ~ego_mask

        ego_position = gt_initial_pos[ego_mask]
        ego_heading = gt_initial_heading[ego_mask]
        batch = tokenized_agent["batch"][non_ego].clone()

        nonego_type = tokenized_agent["initial_type"][non_ego].clone()

        ego_traj=tokenized_agent["ego_traj"].reshape(len(ego_position),-1,2)

        ego_local_traj=transform_to_local(ego_traj,None,ego_position,ego_heading)[0]

        ego_embedding=self.G.ego_embedding(ego_local_traj.flatten(1,2))
        ego_embedding=ego_embedding[batch]
        # feat_map=feat_map+ego_embedding[batch_pl]
        # ego_embedding=0

        tokenized_agent["ego_embedding"]=ego_embedding

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

        # if self.density_conditioned:
        #     non_ego_pos = gt_initial_pos[non_ego]
        #
        #     init_trans = transform_to_local(non_ego_pos,
        #                                   None,
        #                                   ego_position[batch],
        #                                   ego_heading[batch],
        #                                   )[0]
        #
        #     edge_index = knn(pos_pl, init_trans,  1, batch_x=batch_pl,
        #                      batch_y=batch)  # for each object in y, the nearest point in x
        #
        #     src, dst = edge_index #src is y , dst is x
        #
        #     map_agent_count = torch.bincount(
        #         dst,
        #         minlength=pos_pl.size(0)
        #     )


        #feat_map = feat_map + self.G.pos_embedding(pos_pl) + self.G.head_embedding(init_angle)
        feat_map=self.G.pose_embedding(torch.cat([feat_map, pos_pl,init_angle], dim=-1))#,map_agent_count[:,None]

        map_feature = (pos_pl, orient_pl, batch_pl, feat_map)


        normal_scale=self.normal_scale.to(non_ego.device)
        normal_mean=self.normal_mean.to(non_ego.device)

        if self.training:
            m_init,sort_idx=self.get_data(tokenized_agent,non_ego,batch,nonego_type,gt_initial_pos,gt_initial_heading,ego_position,ego_heading)
            data = (m_init, tokenized_agent['nonego_type_sorted'], num_graphs,ego_embedding,feat_map, batch, batch_pl)
            old_nonego_type_sorted = tokenized_agent["nonego_type_sorted"].clone()
            old_batch = tokenized_agent["batch"][non_ego]

            if self.learn_autoencoder:
                rec_loss,agent_loss,kl_loss,x_pred =self.autoencoder.loss(data)
                if self.use_gan:
                    rec_loss=self.get_gan_loss(m_init,x_pred,map_feature, normal_scale,normal_mean,tokenized_agent,non_ego)
                return rec_loss
            else:
                if self.latent_diffusion:
                    with torch.no_grad():
                        m_init = self.autoencoder.forward_encoder(data)

                    m_init = (m_init - self.agent_latents_mean.to(non_ego.device)) / self.agent_latents_scale.to(non_ego.device)

                loss_diff_init,x_pred ,t_batch,t = self.G.get_loss(m_init, tokenized_agent, map_feature,non_ego)

                if self.use_gan:
                    loss=self.get_gan_loss(m_init,x_pred,map_feature, normal_scale,normal_mean,tokenized_agent,non_ego)
                else:
                    match_loss = pos_loss = heading_loss = shape_loss = vel_loss =collision_loss= torch.tensor(0.0,
                                                                                                device=non_ego.device)

                    match_loss, pos_loss, heading_loss, shape_loss, vel_loss,collision_loss = get_matching_loss(old_nonego_type_sorted,
                                                                                                 old_batch,
                                                                                                 x_pred * normal_scale + normal_mean,
                                                                                                 m_init * normal_scale + normal_mean,
                                                                                                 latent=False
                                                                                                 )

                    # match_loss, pos_loss, heading_loss, shape_loss, vel_loss = get_matching_loss(old_nonego_type_sorted,
                    #                                                                              old_batch,
                    #                                                                              x_pred ,
                    #                                                                              m_init,
                    #                                                                              latent=True
                    #                                                                              )


                   # match_loss=(match_loss/normal_scale).mean()
                   #  #weight=torch.tensor([[[0.1,0.1,0.5,0.5,0.2,0.2,0.2,0.2]]],device=non_ego.device)*normal_mean[None]
                   #  weight=1

                    loss = (match_loss,loss_diff_init.mean(), collision_loss, pos_loss, heading_loss, shape_loss, vel_loss)

                return loss
        else:
            if self.learn_autoencoder:
                m_init,sort_idx = self.get_data(tokenized_agent, non_ego, batch, nonego_type, gt_initial_pos,
                                       gt_initial_heading, ego_position, ego_heading)

                data = (m_init, tokenized_agent['nonego_type_sorted'],num_graphs, ego_embedding,feat_map, batch, batch_pl)

                pred_init = self.autoencoder.forward_encoder(data)
            else:
                sort_rank = batch.to(torch.float64)  * 3 + nonego_type.to(torch.float64)

                sort_idx = sort_rank.argsort()

                tokenized_agent['nonego_type_sorted']= nonego_type[sort_idx]

                pred_init= self.G.sample( tokenized_agent, map_feature,non_ego,num_samples=1,
                                                        sampling='ddim',
                                                        stride=1,
                                                        if_output_diffusion_process=False,
                                                        reverse_steps=None)[:,0]

            if self.latent_diffusion:
                pred_init = pred_init*self.agent_latents_scale.to(non_ego.device)+self.agent_latents_mean.to(non_ego.device)

                pred_init = self.autoencoder.forward_decoder(pred_init,   tokenized_agent['nonego_type_sorted'], num_graphs,ego_embedding,feat_map,batch,batch_pl)

            pred_init=pred_init*normal_scale+normal_mean

            if self.use_count:
                pred_count = pred_init[..., :6].reshape(-1, 3, 2)

                pred_trans = 0.5 * (pred_count[:,0] + pred_count[:,2])

                diff_xy_next = pred_count[:,1] - pred_count[:,2]#left_front, right_front, right_back, left_back

                # width & length
                width = torch.norm(pred_count[:,1]-pred_count[:,0],dim=-1)
                length = torch.norm(diff_xy_next,dim=-1)

                pred_head = torch.atan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

                pred_vel=pred_init[..., -2:]

                pred_shape = torch.stack([length, width], dim=-1)

            else:
                pred_trans, pred_head,pred_shape, pred_vel = pred_init[..., :2], pred_init[..., 2:4],pred_init[..., 4:6], pred_init[..., -2:]
                pred_head = torch.atan2(pred_head[..., 1], pred_head[..., 0])

            global_pos,global_heading=transform_to_global(
                pred_trans,
                pred_head,
                ego_position[batch],
                ego_heading[batch],
            )

            global_pred_vel=transform_to_global(
                pred_vel,
                None,
                global_pos,
                global_heading,
            )[0]-global_pos

            gt_initial_pos[non_ego]=global_pos
            gt_initial_heading[non_ego]=global_heading

            gt_initial_vel=tokenized_agent["initial_vel"].clone()

            gt_initial_vel[non_ego] =global_pred_vel

            shape=tokenized_agent["initial_shape"].clone()

            shape[non_ego,:2]=pred_shape[:,:2]

            tokenized_agent["shape"]= shape

            local_vel=tokenized_agent["local_vel"].clone()

            local_vel[non_ego]=pred_vel

            center_token_traj = tokenized_agent["token_traj"].mean(-2)

            gt_initial_idx = torch.linalg.norm(center_token_traj - local_vel[:, None]*0.5, dim=-1).argmin(-1)

            tokenized_agent["type"][non_ego]= tokenized_agent['nonego_type_sorted']

            tokenized_agent['id'][non_ego]=tokenized_agent['id'][non_ego][sort_idx]

            return gt_initial_pos[:, None], gt_initial_heading[:, None],gt_initial_idx[:, None],gt_initial_vel

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


        parser.add_argument('--m_dim', type = int,default = 10)

        return parent_parser

    def ZeroCenteredGradientPenalty(self,Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([-1])