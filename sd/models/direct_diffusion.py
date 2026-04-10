import os
import pickle
from sd.utils.train_helpers import create_lambda_lr_cosine, create_lambda_lr_linear

import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from src.smart.utils import transform_to_local,transform_to_global,wrap_angle

torch.set_printoptions(sci_mode=False)

from sd.nn_modules.agent_diffuser import Agent_Diffuser

from sd.utils.data_container import get_batches, get_features, get_edge_indices, get_encoder_edge_indices
from src.smart.loss.earth_match import get_matching_loss
from sd.utils.data_helpers import unnormalize_scene, normalize_latents, unnormalize_latents, convert_batch_to_scenarios, reorder_indices
from sd.utils.metrics_helpers import convert_data_to_unified_format, compute_lane_metrics, compute_agent_metrics
from tqdm import tqdm
import gzip
from sd.models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder
from sd.nn_modules.ldm import LDM

from torch.optim.lr_scheduler import LambdaLR
import math
from sd.utils.losses import GeometricLosses
from sd.utils.data_helpers import sample_latents, reorder_indices
from sd.utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from sd.models.scenario_dreamer_ldm import ScenarioDreamerLDM
from sd.utils.diffusion_helpers import (
    cosine_beta_schedule,
    extract
)
from sd.utils.viz import visualize_batch

# this ensures CPUs are not suboptimally utilized
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))



def resample_polyline_torch_batch(points, num_points=20):
    """
    points: (B, N, 2)
    returns: (B, num_points, 2)
    """
    B, N, _ = points.shape

    # Segment differences
    diffs = points[:, 1:] - points[:, :-1]                  # (B, N-1, 2)
    distances = torch.norm(diffs, dim=-1)                   # (B, N-1)

    # Cumulative arc-length
    cumulative = torch.cat([
        torch.zeros(B, 1, device=points.device),
        torch.cumsum(distances, dim=1)
    ], dim=1)                                               # (B, N)

    total_length = cumulative[:, -1]                        # (B,)

    # Handle degenerate polylines (all points same)
    mask = total_length < 1e-6

    # Target distances
    target = torch.linspace(
        0, 1, num_points, device=points.device
    )[None, :] * total_length[:, None]                      # (B, num_points)

    # Searchsorted per batch
    idx = torch.searchsorted(cumulative, target, right=True) - 1
    idx = idx.clamp(0, N - 2)                               # (B, num_points)

    # Gather cumulative distances
    left = torch.gather(cumulative, 1, idx)                 # (B, num_points)
    right = torch.gather(cumulative, 1, idx + 1)

    denom = (right - left).clamp_min(1e-6)
    t = (target - left) / denom                             # (B, num_points)

    # Gather points
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 2)      # (B, num_points, 2)

    p0 = torch.gather(points, 1, idx_expanded)              # (B, num_points, 2)
    p1 = torch.gather(points, 1, idx_expanded + 1)

    new_points = p0 + t.unsqueeze(-1) * (p1 - p0)

    # Handle degenerate case: repeat first point
    if mask.any():
        new_points[mask] = points[mask, 0:1].repeat(1, num_points, 1)

    return new_points


class Direct_diffusion(pl.LightningModule):
    """PyTorch Lightning module for ScenarioDreamer AutoEncoder model."""

    def __init__(self, cfg,cfg_ldm):
        super(Direct_diffusion, self).__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.cfg_dataset = self.cfg.dataset

        self.use_latent=False
        self.eval_set =os.environ["PROJECT_ROOT"]+"/metadata/waymo_eval_set.pkl"

        self.lane_conn_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)

        self.scenarios={}

        self.use_diffusion=False

        if self.use_latent:
            self.cfg_model = cfg_ldm.model
            self.cfg_ldm = cfg_ldm
            # self.autoencoder = ScenarioDreamerAutoEncoder.load_from_checkpoint(self.cfg_model.autoencoder_path,
            #                                                                    cfg=cfg, map_location='cpu',
            #                                                                    weights_only=False)  # 🔥 key fix)
            # self.diff_model = LDM(cfg_ldm).load_from_checkpoint(os.environ["PROJECT_ROOT"]+"/checkpoints/scenario_dreamer_ldm_base_waymo/last-001.ckpt",
            #                                                                    cfg=cfg, map_location='cpu',
            #                                                                    weights_only=False)  # 🔥 key fix)

            self.model=ScenarioDreamerLDM.load_from_checkpoint(os.environ["PROJECT_ROOT"]+"/checkpoints/scenario_dreamer_ldm_base_waymo/last-001.ckpt", cfg=cfg_ldm, cfg_ae=cfg,weights_only=False)

            self.autoencoder=self.model.autoencoder
            self.diff_model=self.model.diff_model
        else:
            self.diff_model = Agent_Diffuser(cfg.model)
            self.lane_loss_fn = GeometricLosses['l2']((1))
            self.agent_loss_fn = GeometricLosses['l2']((1))

            if self.use_diffusion:
                self.ldm = LDM(cfg_ldm)

        # nocturne-compatible metadata (stored in latent cache)
        if self.cfg.eval.cache_latents.enable_caching and self.cfg.dataset_name == 'waymo':
            with open(self.cfg.eval.cache_latents.nocturne_train_filenames_path, 'rb') as f:
                nocturne_train_filenames = pickle.load(f)
            with open(self.cfg.eval.cache_latents.nocturne_val_filenames_path, 'rb') as f:
                nocturne_val_filenames = pickle.load(f)
            self.nocturne_compatible_filenames = nocturne_train_filenames + nocturne_val_filenames

        agent_dim=10

        self.register_buffer("agent_mean", torch.zeros(1, agent_dim))
        self.register_buffer("agent_scale", torch.ones(1, agent_dim))

        lane_dim=42

        self.register_buffer("lane_mean", torch.zeros(1, lane_dim))
        self.register_buffer("lane_scale", torch.ones(1, lane_dim))

        self.t_eps=0.01

        self.lane_steps=100

        self.agent_steps=100

        self.lane_sampling_temperature=0.75

    def process_features(self, x, t_batch, batch, mean, scale):
        if torch.all(mean==0):
            mean.copy_(x.mean(0, keepdim=True))
            scale.copy_(x.std(0, keepdim=True))

        raw_noise=torch.randn_like(x)
        t = t_batch[batch]

        if self.use_diffusion:
            x= (x-mean)/scale
            z =  extract(self.ldm.sqrt_alphas_cumprod, t, x.shape) * x + extract(self.ldm.sqrt_one_minus_alphas_cumprod, t, x.shape) * raw_noise
            z= z * scale + mean
            denom=1
        else:
            noise = raw_noise * scale + mean

            z = (1 - t) * noise + t * x
            denom = (1 - t).clamp_min(self.t_eps)

        return z,  denom,raw_noise

    def process_lane(self,x_lane_states1):

        x_lane = resample_polyline_torch_batch(x_lane_states1)

        lane_pos = x_lane[:, 0]
        dx = x_lane[:, 1, 0] - x_lane[:, 0, 0]
        dy = x_lane[:, 1, 1] - x_lane[:, 0, 1]

        lane_heading = torch.atan2(dy, dx)

        x_lane = transform_to_local(
            x_lane,
            None,
            lane_pos,
            lane_heading
        )[0]  # {"none": 0, "pred": 1, "succ": 2, "left": 3, "right": 4, "self": 5}

        # rel_lane = x_lane[:, 1:] - x_lane[:, :-1]
        #
        # dist = torch.norm(rel_lane, dim=-1)  # (N, 19)
        #
        # # shared distance (your assumption)
        # d = dist.mean(dim=1, keepdim=True)  # (N, 1)
        #
        # # segment headings
        # theta = torch.atan2(rel_lane[..., 1], rel_lane[..., 0])  # (N, 19)
        #
        # # heading differences
        # dtheta = wrap_angle(theta[:, 1:] - theta[:, :-1])  # (N, 18)
        #
        # lane_cosine = torch.stack([dtheta.cos(), dtheta.sin()], dim=-1).reshape(-1, 36)
        # rel_lane=torch.cat([d, d, lane_cosine],dim=1)

        x_lane = torch.cat([lane_pos, lane_heading.cos()[:, None], lane_heading.sin()[:, None],
                            x_lane[:,-1:].flatten(1,2),x_lane[:,1:-1].flatten(1,2)],
                           dim=-1)

        # original=self.get_original_lane(x_lane)

        return x_lane

    def training_step(self, data, batch_idx):
        """ Training step for the model. Computes the loss and logs it to WandB."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)

        if self.use_latent:

            with torch.no_grad():
                agent_mu, lane_mu, agent_log_var, lane_log_var = self.autoencoder.model.forward(data, return_latents=True)

                data['agent'].x=agent_mu
                data['agent'].log_var=agent_log_var
                data['lane'].x=lane_mu
                data['lane'].log_var=lane_log_var

                data['agent'].latents, data['lane'].latents = sample_latents(
                    data,
                    self.cfg_ldm.dataset.agent_latents_mean,
                    self.cfg_ldm.dataset.agent_latents_std,
                    self.cfg_ldm.dataset.lane_latents_mean,
                    self.cfg_ldm.dataset.lane_latents_std,
                    normalize=True)  # sample normalized latents for training

            loss_dict = self.diff_model.loss(data)
            self._log_losses(loss_dict, split='train')
            loss=loss_dict['loss']
        else:

            x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
            a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)
            scene_idx = 2 * data['lg_type'].long() + data['map_id'].long()

            non_self_mask=l2l_edge_index[0]!=l2l_edge_index[1]

            l2l_edge_index=l2l_edge_index[:,non_self_mask]
            lane_conn_batch=lane_conn_batch[non_self_mask]
            x_lane_conn=x_lane_conn[non_self_mask,:5]

            x_agent=x_agent[:,[0,1,3,4,5,6,2,7,8,9]] #x,y, speed,cosθ,sinθ ,length, width,type

            x_lane=self.process_lane(x_lane_states)

            if self.use_diffusion:
                t = torch.randint(0, self.lane_steps, (data.num_graphs,), device=x_agent.device)
            else:
                t = torch.rand((data.num_graphs, 1), device=agent_batch.device)

            z_agent, agent_denom,agent_noise = self.process_features(
                x_agent, t, agent_batch,
                self.agent_mean, self.agent_scale,
            )

            z_lane, lane_denom,lane_noise = self.process_features(
                x_lane, t, lane_batch,
                self.lane_mean, self.lane_scale,
            )

            agent_pred,lane_pred,lane_conn_logits=self.diff_model(z_agent,z_lane,x_lane,l2l_edge_index,t,agent_batch,lane_batch,scene_idx)

            lane_conn_loss=self.lane_conn_loss_fn(lane_conn_logits, x_lane_conn, lane_conn_batch).mean()

            self.log('train/lane_conn_loss', lane_conn_loss, on_step=True, batch_size=1)

            # if self.use_diffusion:
            #     match_loss = self.agent_loss_fn(agent_pred, agent_noise, agent_batch).mean()
            #     match_loss1 = self.lane_loss_fn(lane_pred, lane_noise, lane_batch).mean()
            # else:
            ego_mask = agent_batch[1:] != agent_batch[:-1]
            ego_mask = torch.cat([torch.ones_like(ego_mask[:1]), ego_mask])

            agent_pred[ego_mask, :6] = self.diff_model.agent_encoder.ego_shape[:, :6]
            agent_pred[ego_mask, -3:] = self.diff_model.agent_encoder.ego_shape[:, -3:]

            match_loss, pos_loss, heading_loss, shape_loss, vel_loss, _ = get_matching_loss(
                agent_batch,
                agent_pred,
                x_agent,
                agent_denom,
                scale=self.agent_scale,
                all_state=True,
                use_all_type=True,
                use_match=True
               # w_shape=1,
            )

            match_loss1, pos_loss1, heading_loss1, shape_loss1, vel_loss1, _ = get_matching_loss(
                lane_batch,
                lane_pred,
                x_lane,
                lane_denom,
                all_state=False,
                use_all_type=True,
               # use_match=True
            )
                #lane_pred=self.get_original_lane(lane_pred)

                #match_loss1=F.l1_loss(lane_pred,x_lane_states)


            self.log('train/pos_loss', pos_loss, on_step=True, batch_size=1)
            self.log('train/heading_loss', heading_loss, on_step=True, batch_size=1)
            self.log('train/shape_loss', shape_loss, on_step=True, batch_size=1)
            self.log('train/vel_loss', vel_loss, on_step=True, batch_size=1)

            self.log('train/match_loss', match_loss, on_step=True, batch_size=1)
            self.log('train/match_loss1', match_loss1, on_step=True, batch_size=1)
            self.log('train/pos_loss1', pos_loss1, on_step=True, batch_size=1)
            self.log('train/heading_loss1', heading_loss1, on_step=True, batch_size=1)
            self.log('train/vel_loss1', vel_loss1, on_step=True, batch_size=1)

            loss = match_loss + 10 * match_loss1 + 10 * lane_conn_loss

            self.log('train/loss', loss, on_step=True, batch_size=1)

        return loss

    def get_original_lane(self,z_lane):

        # initial heading
        heading0 = torch.atan2(z_lane[:, 3], z_lane[:, 2])  # (N,)

        # shared step length
        # d = z_lane[:, 4:6].mean(1)  # (N,)
        #
        # # delta angles (cos/sin form)
        # dtheta_vec = z_lane[:, 6:].reshape(-1, 18, 2)  # (N,18,2)
        #
        # # normalize for safety
        # norm = torch.norm(dtheta_vec, dim=-1, keepdim=True).clamp_min(1e-6)
        # dtheta_vec = dtheta_vec / norm
        #
        # # recover angle increments
        # dtheta = torch.atan2(dtheta_vec[..., 1], dtheta_vec[..., 0])  # (N,18)
        #
        # # reconstruct full theta sequence
        # # first delta = 0
        # zero = torch.zeros_like(d[:, None])  # (N,1)
        #
        # theta = torch.cumsum(
        #     torch.cat([zero, dtheta], dim=1),  # (N,19)
        #     dim=1
        # ) #+ heading0[:, None]  # absolute heading
        #
        # # segment vectors
        # dx = torch.cos(theta) * d[:, None]  # (N,19)
        # dy = torch.sin(theta) * d[:, None]
        #
        # lane_rel = torch.stack([dx, dy], dim=-1)  # (N,19,2)
        #
        # # lane_cosine = z_lane[:, 6:].reshape(-1, 18, 2)
        # #
        # # lane_cosine=torch.cat([torch.zeros_like(lane_cosine[:,:1]), lane_cosine], dim=1)
        # #
        # # lane_cosine[:, 0, 0] = 1
        # #
        # # lane_rel = lane_cosine / torch.norm(lane_cosine, dim=-1, keepdim=True).clamp_min(1e-6) * dist[:, None, None]
        #
        # lane_local = torch.cumsum(lane_rel, dim=1)  # z_lane[:, 4:].reshape(-1,19,2),

        lane_local=z_lane[:, 4:].reshape(-1,19,2)

        lane_local = transform_to_global(
            lane_local,
            None,
            z_lane[:, :2],
            heading0
        )[0]

        lane_samples = torch.cat([z_lane[:, None, :2], lane_local[:,1:],lane_local[:,:1]], dim=1)

        return lane_samples

    def sample_step_diffusion(self,  z, pred, t_local, mean, scale, ldm, temperature=1.0):
        # normalize
        z_norm = (z - mean) / scale

        # predict x0
        x0 = (pred-mean)/scale#ldm.predict_start_from_noise(z_norm, t=t_local, noise=pred)

        # posterior
        model_mean, logvar = ldm.q_posterior(
            x_start=x0,
            x_t=z_norm,
            t=t_local
        )

        # noise
        noise = torch.randn_like(z_norm)

        mask = (t_local != 0).float().view(-1, *([1] * (z.dim() - 1)))

        z_norm = model_mean + mask * logvar.exp().sqrt() * noise * temperature

        # clip
        z_norm = torch.clamp(z_norm, -ldm.cfg_model.diffusion_clip, ldm.cfg_model.diffusion_clip)

        return z_norm * scale + mean

    def sample_step_flow(self,z, pred, t, t_next, scale, eps):
        denom = (1.0 - t).clamp_min(eps)
        v = (pred - z) / denom
        return z + (t_next - t) * v

    def sample_step_sde(self,z, pred, t, t_next, mean, scale, noise_level, sde_fn):
        z_norm = (z - mean) / scale
        v = (pred - z) / (1.0 - t).clamp_min(1e-6)
        v = v / scale

        z_norm, *_ = sde_fn(
            1 - t,
            1 - t_next,
            -v,
            z_norm,
            noise_level=noise_level
        )

        return z_norm * scale + mean

    def sample_block(
            self,
            z,
            steps,
            timesteps,
            batch,
            mean,
            scale,
            c,
            temperature=1.0,
            eps=1e-3,
            noise_level=None,
            use_sde=False,
            pred_v=None
    ):
        for i in range(steps):
            t = timesteps[i]
            t_batch = t.expand(batch.max() + 1)
            t_local = t_batch[batch]

            if pred_v=="lane":
                pred = self.diff_model.pred_lane(z,t_batch,batch,c)
            else:
                pred = self.diff_model.pred_agent(z,t_batch,batch,c)

            if self.use_diffusion:
                z = self.sample_step_diffusion(
                    z, pred, t_local, mean, scale, self.ldm, temperature
                )

            else:
                t_next = timesteps[i + 1]
                t_next_batch = t_next.expand(batch.max() + 1)
                t_next_local = t_next_batch[batch]

                if use_sde:
                    z = self.sample_step_sde(
                        z, pred, t_local, t_next_local, mean, scale,
                        noise_level[:, i], self.sde_step_with_logprob
                    )
                else:
                    z = self.sample_step_flow(z, pred, t, t_next, scale, eps)

        return z

    @torch.no_grad()
    def generate(self,data,batch_idx):

        if self.use_latent:
            with self.model.ema.average_parameters():

                data['lane'].x=torch.empty((data['lane'].x.shape[0], self.cfg_model.lane_latent_dim))
                data['agent'].x=torch.empty((data['agent'].x.shape[0], self.cfg_model.agent_latent_dim))

                data = data.to(self.device)
                agent_latents, lane_latents = self.diff_model.forward(data, mode='initial_scene')
                agent_latents, lane_latents = unnormalize_latents(
                    agent_latents,
                    lane_latents,
                    self.cfg_ldm.dataset.agent_latents_mean,
                    self.cfg_ldm.dataset.agent_latents_std,
                    self.cfg_ldm.dataset.lane_latents_mean,
                    self.cfg_ldm.dataset.lane_latents_std
                )

                agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples = self.autoencoder.model.forward_decoder(
                    agent_latents,
                    lane_latents,
                    data)

                agent_samples, lane_samples = unnormalize_scene(
                    agent_samples,
                    lane_samples,
                    fov=self.cfg_dataset.fov,
                    min_speed=self.cfg_dataset.min_speed,
                    max_speed=self.cfg_dataset.max_speed,
                    min_length=self.cfg_dataset.min_length,
                    max_length=self.cfg_dataset.max_length,
                    min_width=self.cfg_dataset.min_width,
                    max_width=self.cfg_dataset.max_width,
                    min_lane_x=self.cfg_dataset.min_lane_x,
                    min_lane_y=self.cfg_dataset.min_lane_y,
                    max_lane_x=self.cfg_dataset.max_lane_x,
                    max_lane_y=self.cfg_dataset.max_lane_y)

        else:
            agent_batch, lane_batch, lane_conn_batch = get_batches(data)
            a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)

            x_agent= data['agent'].x
            x_lane= data['lane'].x

            non_self_mask=l2l_edge_index[0]!=l2l_edge_index[1]

            l2l_edge_index=l2l_edge_index[:,non_self_mask]

            if self.use_diffusion:
                timesteps = torch.arange(self.lane_steps-1,-1,-1, device=agent_batch.device)
            else:
                timesteps = torch.linspace(0, 1, self.lane_steps + 1, device=agent_batch.device)

            scene_idx = 2 * data['lg_type'].long() + data['map_id'].long()

            num_agents = torch.bincount(agent_batch)
            num_lanes = torch.bincount(lane_batch)

            num_agents_emb = self.diff_model.num_agents_embedder(num_agents, train=self.training)
            num_lanes_emb = self.diff_model.num_lanes_embedder(num_lanes, train=self.training)

            scene_type = self.diff_model.scene_type_embedder(scene_idx.long(),
                                                  train=self.training)  # , force_drop_ids=torch.ones_like(scene_idx))

            c=(l2l_edge_index, scene_type+ num_lanes_emb)

            z_lane = torch.randn_like(x_lane)*self.lane_scale*self.lane_sampling_temperature+self.lane_mean

            z_lane = self.sample_block(
                z=z_lane,
                steps=self.lane_steps,
                timesteps=timesteps,
                batch=lane_batch,
                mean=self.lane_mean,
                scale=self.lane_scale,
                temperature=self.lane_sampling_temperature,
                c=c,
                pred_v="lane"
            )

            #z_lane=self.process_lane(x_lane)

            map_feature,lane_conn_logits=self.diff_model.predict_con(z_lane,l2l_edge_index,lane_batch,scene_type+ num_lanes_emb)

            lane_conn_pred = torch.argmax(lane_conn_logits, dim=1)

            lane_conn_pred_all=torch.full((non_self_mask.shape[0],), 5, device=lane_conn_pred.device)

            lane_conn_pred_all[non_self_mask]=lane_conn_pred

            lane_conn_samples =  F.one_hot(lane_conn_pred_all, num_classes=6)

            lane_samples=self.get_original_lane(z_lane)

            z_agent =  torch.randn_like(x_agent)*self.agent_scale+self.agent_mean

            if self.use_diffusion:
                timesteps = torch.arange(self.agent_steps-1,-1,-1, device=agent_batch.device)
                noise_level=0
            else:
                timesteps = torch.linspace(0, 1, self.agent_steps + 1, device=agent_batch.device)

                noise_level = torch.zeros(len(z_agent), self.agent_steps, 1, device=agent_batch.device)

                t_rand = torch.randint(0, self.agent_steps, (data.num_graphs,), device=agent_batch.device)

                t_rand=t_rand[agent_batch]

                noise_level[torch.arange(len(z_agent)), t_rand, 0] = 0.7

            ego_mask = agent_batch[1:] != agent_batch[:-1]
            ego_mask = torch.cat([torch.ones_like(ego_mask[:1]), ego_mask])

            c=(map_feature, scene_type+ num_agents_emb)

            z_agent = self.sample_block(
                z=z_agent,
                steps=self.agent_steps,
                timesteps=timesteps,
                batch=agent_batch,
                mean=self.agent_mean,
                scale=self.agent_scale,
                noise_level=noise_level,
                c=c,
                pred_v='agent'
            )

            z_agent[ego_mask, :6] = self.diff_model.agent_encoder.ego_shape[:, :6]
            z_agent[ego_mask, -3:] = self.diff_model.agent_encoder.ego_shape[:, -3:]

            agent_samples= z_agent[:,[0,1,6,2,3,4,5]]# [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
            agent_types = torch.argmax(z_agent[:,-3:], dim=1)

        data['agent'].x = agent_samples
        data['lane'].x = lane_samples
        data['agent'].type = torch.nn.functional.one_hot(agent_types, num_classes=self.cfg_dataset.num_agent_types)
        data['lane', 'to', 'lane'].type =   lane_conn_samples
        lane_types=None

        visualize=True
        if visualize:
            print(f"Visualizing batch {batch_idx}...")

            num_samples_to_visualize = 4

            images_to_log = visualize_batch(
                num_samples_to_visualize,
                agent_samples,
                lane_samples,
                agent_types,
                lane_types,
                lane_conn_samples,
                data,
                save_dir=None,
                epoch=0,
                batch_idx=batch_idx,
                save_wandb=True)
            self.logger.experiment.log(images_to_log)


        batch_of_scenarios = convert_batch_to_scenarios(
            data,
            batch_size=data.num_graphs,
            batch_idx=batch_idx,
            cache_dir=None,
            conditioning_filenames=None,
            cache_samples=False,
            cache_lane_types=self.cfg.dataset_name == 'nuplan',
            mode='initial_scene',
        )
        self.scenarios.update(batch_of_scenarios)


    def validation_step(self, data, batch_idx):
        """ Validation step for the model. Computes the loss and logs it to WandB."""
        self.generate(data,batch_idx)

    def on_validation_epoch_end(self):
        self.compute_metrics()

        self.scenarios={}

    def compute_metrics(self):
        """Compute metrics given the generated samples and the ground truth samples."""
        #sample_paths = [os.path.join(self.samples_path, file) for file in os.listdir(self.samples_path)]

        with open(self.eval_set, 'rb') as f:
            gt_sample_filenames = pickle.load(f)['files']

        if self.cfg.dataset_name == 'nuplan':
            gt_sample_ids = [os.path.splitext(file)[0] for file in gt_sample_filenames]

        num_samples = len(self.scenarios)
        gt_sample_filenames = gt_sample_filenames[:num_samples]
        num_gt_samples = len(gt_sample_filenames)
        assert num_samples == num_gt_samples, "Number of samples and ground truth samples do not match."

        print("Number of evaluated samples (real/generated): ", num_samples)
        samples = []
        gt_samples = []
        print("Converting samples to unified format for metrics computation...")
        for i,data in enumerate(self.scenarios.values()):
            #data=self.scenarios[i]
            # with open(sample_paths[i], 'rb') as f:
            #     data = pickle.load(f)
            sample = convert_data_to_unified_format(data, dataset_name=f"{self.cfg.dataset_name}")

            if self.cfg.dataset_name == 'waymo':
                # agent and lane gt data are loaded from the preprocessed scenario dreamer waymo data
                with open(os.path.join(os.environ["PROJECT_ROOT"]+'/checkpoints/scenario_dreamer_ae_preprocess_waymo/test', gt_sample_filenames[i]), 'rb') as f:
                    gt_data = pickle.load(f)
            else:
                # the gt agent data comes from the preprocessed scenario dreamer nuplan data
                sample_id = gt_sample_ids[i]
                with open(os.path.join(self.cfg.eval.metrics.gt_agent_test_dir, f'{sample_id}_0.pkl'), 'rb') as f:
                    gt_agent_data = pickle.load(f)

                # As the lane graph is preprocessed slightly differently between SLEDGE and scenario dreamer,
                # for fairest comparison with SLEDGE we process the gt lane graphs following the SLEDGE preprocessing scheme (this requires
                # loading from the SLEDGE preprocessed nuplan data)
                # We could preprocess the gt lane graphs using the scenario dreamer preprocessing scheme,
                # but then we wouldn't know if performance improvement compared to SLEDGE is attributed to the GT lane graph preprocessing
                # being more aligned with scenario dreamer.
                # In practice, we find both preprocessing schemes yield very similar performance.
                with gzip.open(os.path.join(self.cfg.eval.metrics.gt_lane_test_dir, gt_sample_filenames[i]), 'rb') as f:
                    gt_lane_data = pickle.load(f)

                gt_data = gt_lane_data
                # add agent data to the gt lane data
                gt_data['agent_states'] = gt_agent_data['agent_states']
                gt_data['agent_types'] = gt_agent_data['agent_types']
                gt_data['lg_type'] = gt_agent_data['lg_type']

            gt_sample = convert_data_to_unified_format(gt_data, dataset_name=f'{self.cfg.dataset_name}_gt')

            if len(sample['G']) > 0:
                samples.append(sample)
            gt_samples.append(gt_sample)

        lane_metrics = compute_lane_metrics(samples=samples, gt_samples=gt_samples)
        agent_metrics = compute_agent_metrics(samples=samples, gt_samples=gt_samples)

        print("--------------------------------------------------------------------------")
        print("Lane metrics: ", ["{}: {:.2f}".format(k, v) for (k, v) in lane_metrics.items()])
        print("Agent metrics: ", ["{}: {:.2f}".format(k, v) for (k, v) in agent_metrics.items()])
        print("--------------------------------------------------------------------------")

        # metrics = {
        #     'lane_metrics': lane_metrics,
        #     'agent_metrics': agent_metrics
        # }
        for key,value in lane_metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        for key,value in agent_metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

        # # save metrics to file
        # metrics_path = os.path.join(self.cfg.eval.metrics.metrics_save_path, 'metrics.pkl')
        # with open(metrics_path, 'wb') as f:
        #     pickle.dump(metrics, f)
        # print(f"Metrics saved to {metrics_path}")


    # def on_before_optimizer_step(self, optimizer):
    #     """ Called before the optimizer step. Logs the gradient norms for each layer."""
    #     # Compute the 2-norm for each layer
    #     norms_encoder = grad_norm(self.diff_model, norm_type=2)
    #     self.log_dict(norms_encoder)
    def sde_step_with_logprob(
            self,
            sigma,
            sigma_prev,
            model_output: torch.FloatTensor,
            sample: torch.FloatTensor,
            noise_level = 0.7,
            prev_sample=None,
            sde_type = 'sde',
            return_sqrt_dt= False,

    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
        process from the learned model outputs (most often the predicted velocity).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned flow model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """

        dt = sigma_prev - sigma

        if sde_type == 'sde':
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_prev, sigma))) * noise_level

            # our sde
            prev_sample_mean = sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt) + model_output * (
                    1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt

            if prev_sample is None:
                variance_noise = torch.randn_like(model_output)

                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

            sqrt_term = std_dev_t * torch.sqrt(torch.clamp(-1 * dt, min=0))
            log_prob = (
                    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (sqrt_term ** 2))
                    - torch.log(sqrt_term.clamp_min(1e-8))
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif sde_type == 'cps':
            std_dev_t = sigma_prev * torch.sin(noise_level * math.pi / 2)  # sigma_t in paper
            pred_original_sample = sample - sigma * model_output  # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
                sigma_prev ** 2 - std_dev_t ** 2)

            if prev_sample is None:
                variance_noise = torch.randn_like(model_output)

                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            # remove all constants
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)/( 2*std_dev_t.clamp_min(0.05) ** 2)

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        if return_sqrt_dt:
            return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(torch.clamp(-1 * dt, min=0))
        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    ### Taken largely from QCNet repository: https://github.com/ZikangZhou/QCNet
    def configure_optimizers(self):
        """ Configure the optimizer and learning rate scheduler for the model."""
        self.lr_warmup_steps=0
        self.lr_min_ratio=1e-2
        self.lr_total_steps=128


        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                        self.lr_min_ratio
                        + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                    1.0
                    + math.cos(
                math.pi
                * min(
                    1.0,
                    (current_step - self.lr_warmup_steps)
                    / (self.lr_total_steps - self.lr_warmup_steps),
                )
            )
            )
        optimizer = torch.optim.AdamW(self.diff_model.parameters(), lr=5e-4)

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [lr_scheduler]
        #
        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
        #                             nn.LSTMCell, nn.GRU, nn.GRUCell)
        # blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        # for module_name, module in self.named_modules():
        #     for param_name, param in module.named_parameters():
        #         full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
        #         if 'bias' in param_name:
        #             no_decay.add(full_param_name)
        #         elif 'weight' in param_name:
        #             if isinstance(module, whitelist_weight_modules):
        #                 decay.add(full_param_name)
        #             elif isinstance(module, blacklist_weight_modules):
        #                 no_decay.add(full_param_name)
        #         elif not ('weight' in param_name or 'bias' in param_name):
        #             no_decay.add(full_param_name)
        # param_dict = {param_name: param for param_name, param in self.named_parameters()}
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0
        #
        # optim_groups = [
        #     {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
        #      "weight_decay": self.cfg.train.weight_decay},
        #     {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
        #      "weight_decay": 0.0},
        # ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay,
        #                               betas=(self.cfg.train.beta_1, self.cfg.train.beta_2), eps=self.cfg.train.epsilon)
        #
        # if self.cfg.train.lr_schedule == 'cosine':
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                                                   lr_lambda=create_lambda_lr_cosine(self.cfg))
        # elif self.cfg.train.lr_schedule == 'linear':
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                                                   lr_lambda=create_lambda_lr_linear(self.cfg))
        #
        # return [optimizer], {"scheduler": scheduler,
        #                      "interval": "step",
        #                      "frequency": 1}

    def _log_losses(self, loss_dict, split='train', batch_size=None):
        """ Log the losses to WandB."""
        if split == 'train':
            on_step = True
            on_epoch = False
            key_lambda = lambda s: s  # no change
        elif split == 'val':
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f'val_{s}'  # add val_ prefix
        elif split == 'test':
            on_step = False
            on_epoch = True
            key_lambda = lambda s: f'test_{s}'  # add test_ prefix

        for k, v in loss_dict.items():
            if k == 'loss':
                v = v.item()

            self.log(key_lambda(k), v, prog_bar=True, on_step=on_step, on_epoch=on_epoch, sync_dist=True,
                     batch_size=batch_size)

        if split == 'train':
            cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', cur_lr, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

