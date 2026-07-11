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
from src.smart.gan.discriminator import InitDiscriminator
from src.smart.loss.rollout_buffer import RunningMeanStdTorch, get_reward, get_nei_returns, get_return, \
    get_near_returns, per_scene_zscore_clip,rollout, compute_advantages,get_train_mask,get_reduce_loss
from .scale_flow import ScaleFlow
from src.smart.diffusion.dit.autoencoder import AutoEncoder
from src.smart.diffusion.dit.ldm import LDM
from torch_scatter import scatter_sum,scatter_mean
from .diffusion_utils import multi_circle_collision_loss_mem_efficient

class InitDiffusion(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                num_heads: int,
                num_freq_bands,
                token_processor,
                gail
        ) -> None:
        super(InitDiffusion, self).__init__()

        self.use_all_pos=token_processor.use_all_pos
        self.pred_all_pos=token_processor.pred_all_pos
        self.token_processor=token_processor

        parser = ArgumentParser()
        self.add_model_specific_args(parser)
        args, _ = parser.parse_known_args()

        self.learn_autoencoder = False
        self.latent_diffusion = False
        self.sep_map=False

        if self.learn_autoencoder or self.latent_diffusion:

            self.autoencoder=AutoEncoder(num_encoder_blocks=2,num_decoder_blocks=2,hidden_dim=256,latent_dim=8,num_heads=4)

        if self.latent_diffusion:
            self.ldm=True
        else:
            self.ldm=False

        self.ldm=False

        if self.ldm:
            self.G1=LDM()
        else:
            self.G1 = ScaleFlow(args,token_processor,gail)

        self.use_gan = False

        self.use_rl=False

        if  self.use_gan:
            self.return_meanstd = RunningMeanStdTorch(shape=(1))

            self.D=InitDiscriminator(hidden_dim,num_heads,num_freq_bands,token_processor)


    def forward(self, tokenized_agent,resampling=False):

        num_graphs = tokenized_agent["num_graphs"]
        ego_mask = tokenized_agent["ego_mask"]
        ego_position = tokenized_agent["initial_pos"][ego_mask]
        ego_heading = tokenized_agent["initial_heading"][ego_mask]

        if "ego_feat" not in tokenized_agent.keys():
            init_agent_batch = tokenized_agent["batch"]
            tokenized_agent["init_agent_batch"] = init_agent_batch

            tokenized_agent["batch_ego_pos"] = ego_position[init_agent_batch]
            tokenized_agent["batch_ego_heading"] = ego_heading[init_agent_batch]

            init_agent_type = tokenized_agent["type"].long()
            tokenized_agent["init_agent_type"] = init_agent_type

            if "ego_traj" not in tokenized_agent.keys():
                if self.G1.model.use_rel_ego:
                    ego_pos2 = tokenized_agent["ego_pos2"]
                    ego_heading2 = tokenized_agent["ego_heading2"]

                    ego_local_pos2, ego_local_heading2 = transform_to_local(
                        ego_pos2,
                        ego_heading2,
                        ego_position,
                        ego_heading,
                    )

                    local_ego_traj = torch.cat(
                        [ego_local_pos2, ego_local_heading2[:, :, None]],
                        dim=-1,
                    ).flatten(1, 2)
                else:
                    local_ego_traj = tokenized_agent["local_ego_traj"]
            else:
                ego_traj = tokenized_agent["ego_traj"].reshape(len(ego_position), -1, 2)

                local_ego_traj = transform_to_local(
                    ego_traj,
                    None,
                    ego_position,
                    ego_heading,
                )[0].flatten(1, 2)

            num_types = 3
            idx = init_agent_batch * num_types + init_agent_type

            type_counts = torch.bincount(
                idx,
                minlength=num_graphs * num_types,
            ).view(-1, num_types)

            tokenized_agent["type_counts"] = type_counts

            ego_feat = torch.cat([local_ego_traj, type_counts], dim=-1)
            tokenized_agent["ego_feat"] = ego_feat
        else:
            ego_feat = tokenized_agent["ego_feat"]
            init_agent_batch = tokenized_agent["init_agent_batch"]

        if not self.G1.model.use_rel_ego and not self.learn_autoencoder:
            ego_embedding=self.G1.ego_embedding1(ego_feat)
            ego_embedding = ego_embedding[init_agent_batch]

            tokenized_agent["ego_embedding"] = ego_embedding

        if "initial_map_feature" not in tokenized_agent.keys():
            map_feature = tokenized_agent["map_feature"]
            if self.use_all_pos:
                initial_map_feature =map_feature
            else:
                batch_pl = map_feature["batch"]
                pos_pt = map_feature["position"]

                if batch_pl.numel() == 0:
                    initial_map_feature = {key: value for key, value in map_feature.items()}
                else:
                    ego_pos = ego_position.reshape(-1, num_graphs, 2)
                    dist = torch.norm(ego_pos[:, batch_pl] - pos_pt[None], dim=-1).amin(0)
                    initial_map_feature = {
                        key: value[dist < self.token_processor.init_map_range] for key, value in map_feature.items()
                    }

            batch_pl = initial_map_feature["batch"]
            pos_pl = initial_map_feature["position"]
            orient_pl = initial_map_feature["orientation"]
            feat_map = initial_map_feature["pt_token"]

            if batch_pl.numel() > 0 and batch_pl.max().item() == num_graphs - 1:
                pos_pl, orient_pl = transform_to_local(pos_pl,  # [:,None],
                                                       orient_pl,  # [:,None],
                                                       ego_position[batch_pl],
                                                       ego_heading[batch_pl],
                                                       )
            feat_map = self.G1.model.lane_embed(feat_map)

            initial_map_feature = {
                "pt_token": feat_map,
                "position": pos_pl,
                "orientation": orient_pl,
                "batch": batch_pl,
            }

            tokenized_agent["initial_map_feature"] = initial_map_feature
        else:
            initial_map_feature=tokenized_agent["initial_map_feature"]

        if self.training:
            diff_input,diff_output=self.G1.model.get_input(tokenized_agent)

            if self.learn_autoencoder:
                return self.autoencoder.loss(diff_input, tokenized_agent, initial_map_feature)
            else:
                if self.latent_diffusion:
                    with torch.no_grad():
                        diff_input = self.autoencoder.forward_encoder(diff_input,tokenized_agent,initial_map_feature)[0]

                if self.use_rl:
                    with torch.no_grad():
                        pred_init, x_list = self.G1.sample(tokenized_agent, initial_map_feature, None)

                    col_reward,end_idx,start_idx=multi_circle_collision_loss_mem_efficient(pred_init,None, init_agent_batch,None)

                    N = len(pred_init)

                    col_reward_end = scatter_sum(
                        col_reward,
                        end_idx,
                        dim=0,
                        dim_size=N
                    )

                    col_reward_start = scatter_sum(
                        col_reward,
                        start_idx,
                        dim=0,
                        dim_size=N
                    )

                    col_reward_agent=-col_reward_end-col_reward_start
                    #
                    # batch=tokenized_agent["init_agent_batch"]
                    #
                    # same_batch = batch[:, None] == batch[None, :]
                    # not_self = ~torch.eye(len(batch), dtype=torch.bool, device=batch.device)
                    # edge_mask = same_batch & not_self
                    #
                    # pos=pred_init[:,:2]
                    #
                    # shape=pred_init[:, 4:6]
                    #
                    # r=torch.linalg.norm(shape,dim=-1)
                    #
                    # d=torch.linalg.norm(pos[:,None]-pos[None],dim=-1)
                    #
                    # r2=(r[:,None]+r[None])/2
                    #
                    # col_reward=d>r2
                    # col_reward[~edge_mask]=True
                    #
                    # col_reward_agent=torch.all(col_reward,dim=-1)

                    advantage=(col_reward_agent==0).float() #-0.5#col_reward <0 collision 0

                    tokenized_agent["noncol_rate"]=advantage

                    #advantages=(advantage-advantage.mean())/(advantage.std()+1e-5)#(advantage-0.771)/0.42

                    tokenized_agent["advantages"]=advantage#advantage conditioned

                loss,x_pred ,z,t = self.G1.get_loss(diff_input, diff_output,tokenized_agent, initial_map_feature)

            match_loss, collision_loss, pos_loss, heading_loss, shape_loss, vel_loss=loss

            if self.use_gan:
                return  (diff_input,x_pred,match_loss.mean()+collision_loss.mean(),z,t, tokenized_agent)

            loss = (match_loss.mean(), collision_loss.mean(), pos_loss.mean(), heading_loss.mean(), shape_loss.mean(), vel_loss.mean())

            return loss
        else:
            if self.learn_autoencoder:
                diff_input, diff_output = self.G1.model.get_input(tokenized_agent)

                pred_init = self.autoencoder.loss(diff_input, tokenized_agent, initial_map_feature)[-1]
            else:
                if resampling:
                    pred_init, x_list = self.G1.self_resample_initial_state(tokenized_agent, initial_map_feature, 0)
                else:
                    pred_init, x_list = self.G1.sample(tokenized_agent, initial_map_feature, None,
                                                       # infer_steps=10,
                                                       # sampling_mode="heun_low_noise",
                                                       # schedule_power=2.0,
                                                       )

                if self.latent_diffusion:
                    pred_init = self.autoencoder.forward_decoder(pred_init, tokenized_agent, initial_map_feature)

            gt_initial_pos, gt_initial_heading, shape, gt_initial_vel, gt_initial_idx = self.G1.model.get_output(
                pred_init, tokenized_agent
            )
            return gt_initial_pos, gt_initial_heading, gt_initial_idx, shape, gt_initial_vel

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('QCNet')
        parser.add_argument('--dataset', type=str, default='argoverse_v2')
        parser.add_argument('--input_dim', type=int, default=2)
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--output_dim', type=int, default=2)
        parser.add_argument('--output_head', action='store_true')
        parser.add_argument('--init_timestep', type=int, default=50)
        parser.add_argument('--num_freq_bands', type=int, default=64)
        parser.add_argument('--num_map_layers', type=int, default=1)
        parser.add_argument('--num_agent_layers', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--head_dim', type=int, default=16)
        parser.add_argument('--dropout', type=float, default=0)
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
