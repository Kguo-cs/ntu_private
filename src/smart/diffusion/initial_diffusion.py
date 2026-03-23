import torch
import torch.nn as nn
from argparse import ArgumentParser

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_global,
    rotate_to_local,
)
from src.smart.loss.earth_match import get_matching_loss
from src.smart.metrics.gen_metrics import plot_scene
from src.smart.layers import MLPLayer
from src.smart.gan.discriminator import InitDiscriminator, InitGeneator
from src.smart.loss.rollout_buffer import RunningMeanStdTorch, get_reward, get_nei_returns, get_return, \
    get_near_returns, per_scene_zscore_clip,rollout, compute_advantages,get_train_mask,get_reduce_loss
from src.smart.loss.earth_match import gaussian_nll_2d

class InitDiffusion(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                num_heads: int,
                num_freq_bands,
                token_processor,
        ) -> None:
        super(InitDiffusion, self).__init__()

        self.latent_diffusion=False
        self.use_gan = False
        self.use_dit=False
        self.sep_map=False
        self.use_match=True

        self.use_all_pos=token_processor.use_all_pos

        self.learn_autoencoder = token_processor.learn_autoencoder
        if self.learn_autoencoder:
            self.use_gan = False
            self.latent_diffusion = True

        if self.latent_diffusion:
            from .autoencoder import AutoEncoder

            self.autoencoder=AutoEncoder(num_encoder_blocks=2,num_decoder_blocks=2,hidden_dim=hidden_dim,latent_dim=8,num_heads=8)

        if self.use_gan:

            self.D=InitDiscriminator(hidden_dim,num_heads,num_freq_bands,token_processor)

        if self.use_dit:
            from .diffusion import ScaleFlow
        else:
            from .scale_flow import ScaleFlow


        parser = ArgumentParser()
        self.add_model_specific_args(parser)
        args = parser.parse_args()
        self.G = ScaleFlow(args,token_processor)

        self.use_gail=True

        if self.use_gail:
            # self.value_network = MLPLayer(hidden_dim, hidden_dim * 2, 1)
            self.return_meanstd = RunningMeanStdTorch(shape=(1))

            self.D=InitDiscriminator(hidden_dim,num_heads,num_freq_bands,token_processor)

    def forward(self, tokenized_agent):
        num_graphs = tokenized_agent["num_graphs"]

        ego_mask = tokenized_agent["ego_mask"]
        non_ego = ~ego_mask

        if self.use_all_pos:
            non_ego=torch.ones_like(non_ego)

        ego_position = tokenized_agent["initial_pos"][ego_mask]
        ego_heading = tokenized_agent["initial_heading"][ego_mask]
        nonego_batch = tokenized_agent["batch"][non_ego]
        tokenized_agent["batch_ego_pos"] = ego_position[nonego_batch]
        tokenized_agent["batch_ego_heading"] = ego_heading[nonego_batch]

        nonego_type = tokenized_agent["initial_type"][non_ego].clone()

        ego_traj=tokenized_agent["ego_traj"].reshape(len(ego_position),-1,2)

        ego_local_traj=transform_to_local(ego_traj,None,ego_position,ego_heading)[0].flatten(1,2)

        num_types = 3  # since types are 0,1,2

        idx = nonego_batch * num_types + nonego_type

        type_counts = torch.bincount(
            idx,
            minlength=num_graphs * num_types
        ).view(-1, num_types)

        tokenized_agent["type_counts"]=type_counts

        ego_local_traj=torch.cat([ego_local_traj,type_counts],dim=-1)

        ego_embedding=self.G.ego_embedding(ego_local_traj)

        if "initial_map_feature" not in tokenized_agent.keys():
            map_feature = tokenized_agent["map_feature"]
            if self.use_all_pos:
                initial_map_feature =map_feature
            else:
                batch_pl = map_feature["batch"]

                pos_pt = map_feature["position"]

                ego_pos = ego_position.reshape(-1,batch_pl.max().item()+1,2)

                dist=torch.norm(ego_pos[:,batch_pl]-pos_pt[None],dim=-1).amin(0)

                initial_map_feature = {}

                for key in map_feature.keys():
                    initial_map_feature[key] = map_feature[key][dist < 100]

            tokenized_agent["initial_map_feature"] = initial_map_feature
        else:
            initial_map_feature=tokenized_agent["initial_map_feature"]

        batch_pl = initial_map_feature["batch"]
        pos_pl = initial_map_feature["position"]
        orient_pl = initial_map_feature["orientation"]
        feat_map = initial_map_feature["pt_token"]

        if batch_pl.max().item()==num_graphs-1:
            pos_pl, orient_pl = transform_to_local(pos_pl,  # [:,None],
                                                   orient_pl,  # [:,None],
                                                   ego_position[batch_pl],
                                                   ego_heading[batch_pl],
                                                   )

        if self.G.net.use_padding  or (self.use_gan and self.D.use_entry_former):
            map_feature = self.G.net.padding(pos_pl, orient_pl, feat_map, batch_pl, tokenized_agent["num_graphs"])
        else:
            map_feature={
                "pt_token": feat_map,
                "position": pos_pl,
                "orientation": orient_pl,
                "batch": batch_pl,
            }

        if self.training:

            diff_input,m_init,nonego_batch=self.G.net.get_input(tokenized_agent,non_ego,nonego_batch,nonego_type)

            ego_embedding = ego_embedding[nonego_batch]

            tokenized_agent["nonego_batch"]=nonego_batch
            tokenized_agent["ego_embedding"] = ego_embedding

            # data=(m_init, tokenized_agent['nonego_type_sorted'], num_graphs,ego_embedding,feat_map, nonego_batch, batch_pl)

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

                loss_diff_init,x_pred ,expert_state,denom,t = self.G.get_loss(diff_input, tokenized_agent, map_feature,None)

                if self.use_gan:
                    loss=self.D.get_gan_loss(m_init,self.G.net.denormalize(x_pred),map_feature, tokenized_agent,denom)
                else:
                    if self.use_match:
                        match_loss, pos_loss, heading_loss, shape_loss, vel_loss, collision_loss = get_matching_loss(
                            tokenized_agent,
                            x_pred,
                            m_init,
                            x_pred,
                            self.G.net.normalize(m_init),
                            denom,
                            all_state=False,
                            use_col=False,
                            use_all_type=False
                            )
                    else:
                        pos_loss = heading_loss = shape_loss = vel_loss =collision_loss= torch.tensor(0.0, device=non_ego.device)

                        match_loss= loss_diff_init.mean()

                    if self.use_gail:
                        expert_dis_loss,_ = self.D.get_reward(m_init,t,tokenized_agent, map_feature,"expert")

                        with torch.no_grad():
                            pred_init, x_list,z_list, step_list,t_list = self.G.sample(tokenized_agent, map_feature, None)

                        print(z_list[-1].shape,m_init.shape)

                        agent_dis_loss, agent_rewards = self.D.get_reward(z_list[-1], t, tokenized_agent,
                                                                          map_feature, "agent")

                        agent_action=torch.cat(x_list,dim=1).transpose(0, 1).flatten(0,1)   #action_list

                        agent_state=torch.cat(z_list,dim=1)

                        t=torch.cat(t_list,dim=1).transpose(0, 1).flatten(0,1)

                        n_step=agent_state.shape[1]-1

                        batch=tokenized_agent["nonego_batch"]

                        tokenized_agent["repeat_batch"] = batch.unsqueeze(1).repeat(1, n_step) #n_agent ,n_step

                        batch = torch.stack(
                            [
                                batch + num_graphs * t
                                for t in range(n_step)
                            ],
                            dim=1,
                        ).transpose(0, 1).flatten(0,1)  # [n_agent*n_step]

                        tokenized_agent["nonego_batch"]=batch

                        tokenized_agent["nonego_type_sorted"]=tokenized_agent["nonego_type_sorted"][None].repeat(n_step,1).flatten(0,1)

                        agent_input_state=agent_state[:,:-1].transpose(0, 1).flatten(0,1) #t,a

                        agent_next_state=agent_state[:,1:].transpose(0, 1).flatten(0,1) #t,a

                        tokenized_agent["num_graphs"]=num_graphs*n_step

                        tokenized_agent["ego_embedding"]=ego_embedding[None].repeat(n_step,1,1).flatten(0,1)

                        #agent_dis_loss,agent_rewards = self.D.get_reward(agent_next_state, t, tokenized_agent, map_feature,"agent")

                        x_pred = self.G.net(agent_input_state, t, tokenized_agent, map_feature, mode=1)[:,0]

                        agent_log_prob=-gaussian_nll_2d(x_pred[:,:8], x_pred[:,8:], agent_action)

                        feat_a = tokenized_agent["noise_feat"]  # [-2]

                        value = self.G.value_network(feat_a)[..., 0].view(n_step,-1)

                        rewards=torch.zeros_like(value)

                        rewards[-1]=agent_rewards

                        advantages, value_loss = compute_advantages(rewards, value)

                        advantages=advantages.view(-1)

                        self.return_meanstd.update(advantages.detach())

                        advantages = self.return_meanstd.normalize(advantages)

                        ppo_loss = -(agent_log_prob * advantages).mean()

                        policy_loss = match_loss + ppo_loss + 1e-3 * value_loss  # - 0.01 * agent_entropy.mean()

                        match_loss = expert_dis_loss + agent_dis_loss + policy_loss

                    loss = (match_loss,loss_diff_init.mean(), collision_loss, pos_loss, heading_loss, shape_loss, vel_loss)

                return loss
        else:
            ego_embedding = ego_embedding[nonego_batch]
            tokenized_agent["ego_embedding"] = ego_embedding
            tokenized_agent["nonego_batch"]=nonego_batch

            if self.learn_autoencoder:
                m_init,sort_idx = self.get_data(tokenized_agent, non_ego, nonego_batch, nonego_type, gt_initial_pos,
                                       gt_initial_heading, ego_position, ego_heading)

                data = (m_init, tokenized_agent['nonego_type_sorted'],num_graphs, ego_embedding,feat_map, nonego_batch, batch_pl)

                pred_init = self.autoencoder.forward_encoder(data)
            else:
                tokenized_agent['nonego_type_sorted']= nonego_type

                pred_init, x_list,z_list, step_list,t_list = self.G.sample( tokenized_agent, map_feature,None)


            if self.latent_diffusion:
                pred_init = pred_init*self.agent_latents_scale.to(non_ego.device)+self.agent_latents_mean.to(non_ego.device)

                pred_init = self.autoencoder.forward_decoder(pred_init,tokenized_agent['nonego_type_sorted'], num_graphs,ego_embedding,feat_map,nonego_batch,batch_pl)

            gt_initial_pos,gt_initial_heading,shape,gt_initial_vel,gt_initial_idx=self.G.net.get_output(
                pred_init, tokenized_agent, non_ego
            )

            return gt_initial_pos, gt_initial_heading,gt_initial_idx,shape,gt_initial_vel
        
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
