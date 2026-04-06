import os
import pickle
from sd.utils.train_helpers import create_lambda_lr_cosine, create_lambda_lr_linear

import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

from src.smart.utils import transform_to_local,transform_to_global

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

# this ensures CPUs are not suboptimally utilized
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))


class Direct_diffusion(pl.LightningModule):
    """PyTorch Lightning module for ScenarioDreamer AutoEncoder model."""

    def __init__(self, cfg,cfg_ldm):
        super(Direct_diffusion, self).__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.cfg_dataset = self.cfg.dataset

        self.use_latent=True

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

        # self.register_buffer("lane_con_mean", torch.zeros(1, 6))
        # self.register_buffer("lane_con_scale", torch.ones(1, 6))

        lane_dim=44

        self.register_buffer("lane_mean", torch.zeros(1, lane_dim))
        self.register_buffer("lane_scale", torch.ones(1, lane_dim))
        self.t_eps=0.01

        self.eval_set =os.environ["PROJECT_ROOT"]+"/metadata/waymo_eval_set.pkl"

        self.lane_conn_loss_fn = GeometricLosses['cross_entropy'](apply_mean=False)

        self.scenarios={}

    def training_step(self, data, batch_idx):
        """ Training step for the model. Computes the loss and logs it to WandB."""
        agent_batch, lane_batch, lane_conn_batch = get_batches(data)

        if self.use_latent:

            with torch.no_grad():
                agent_mu, lane_mu, agent_log_var, lane_log_var = self.autoencoder.model.forward(data, return_latents=True)

                scene_type = data['lg_type']
                road_points = data['lane'].x
                agent_states = data['agent'].x

                agent_mu2=[]
                agent_log_var2=[]
                lane_mu2=[]
                lane_log_var2=[]
                agent_partition_mask2=[]
                lane_partition_mask2=[]


                for i in range(data.num_graphs):
                    n_lanes=len(road_points[lane_batch==i])

                    edge_index_lane_to_lane = get_edge_index_complete_graph(n_lanes)

                    agent_mu1, agent_log_var1, lane_mu1, lane_log_var1, edge_index_lane_to_lane1, agent_partition_mask1, lane_partition_mask1 = reorder_indices(
                        agent_mu[agent_batch==i].cpu().numpy(),
                        agent_log_var[agent_batch==i].cpu().numpy(),
                        lane_mu[lane_batch==i].cpu().numpy(),
                        lane_log_var[lane_batch==i].cpu().numpy(),
                        edge_index_lane_to_lane,
                        agent_states[agent_batch==i].cpu().numpy(),
                        road_points[lane_batch==i].cpu().numpy(),
                        scene_type[i],
                        dataset='waymo')

                    agent_mu2.append(torch.tensor(agent_mu1))
                    agent_log_var2.append(torch.tensor(agent_log_var1))
                    lane_mu2.append(torch.tensor(lane_mu1))
                    lane_log_var2.append(torch.tensor(lane_log_var1))
                    agent_partition_mask2.append(torch.tensor(agent_partition_mask1))
                    lane_partition_mask2.append(torch.tensor(lane_partition_mask1))

                device=agent_mu.device

                data['agent'].x=torch.cat(agent_mu2).to(device)
                data['agent'].log_var=torch.cat(agent_log_var2).to(device)
                data['lane'].x=torch.cat(lane_mu2).to(device)
                data['lane'].log_var=torch.cat(lane_log_var2).to(device)
                data['agent'].partition_mask=torch.cat(agent_partition_mask2).to(device)
                data['lane'].partition_mask=torch.cat(lane_partition_mask2).to(device)

                data['agent'].latents, data['lane'].latents = sample_latents(
                    data,
                    self.cfg_ldm.dataset.agent_latents_mean,
                    self.cfg_ldm.dataset.agent_latents_std,
                    self.cfg_ldm.dataset.lane_latents_mean,
                    self.cfg_ldm.dataset.lane_latents_std,
                    normalize=True)  # sample normalized latents for training

            data["map_id"] = data['nocturne_compatible']

            loss_dict = self.diff_model.loss(data)
            self._log_losses(loss_dict, split='train')
            loss=loss_dict['loss']
        else:

            x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)
            a2a_edge_index, l2l_edge_index, l2a_edge_index = get_edge_indices(data)

            x_agent=x_agent[:,[0,1,3,4,5,6,2,7,8,9]] #x,y, speed,cosθ,sinθ ,length, width,type

            lane_pos=x_lane_states.mean(1)
            dx = x_lane_states[:, 1:, 0] - x_lane_states[:, :-1, 0]
            dy = x_lane_states[:, 1:, 1] - x_lane_states[:, :-1, 1]

            sin = torch.sin(torch.atan2(dy, dx))
            cos = torch.cos(torch.atan2(dy, dx))

            lane_heading = torch.atan2(sin.mean(1), cos.mean(1))

            x_lane=transform_to_local(
                x_lane_states,
                None,
                lane_pos,
                lane_heading
            )[0].reshape(-1,40)

            x_lane=torch.cat([lane_pos,lane_heading.cos()[:,None],lane_heading.sin()[:,None],x_lane],dim=-1)

            if torch.all(self.agent_mean==0):
                self.agent_mean.copy_(torch.mean(x_agent, dim=0, keepdim=True))
                self.agent_scale.copy_(torch.std(x_agent, dim=0, keepdim=True))

                self.lane_mean.copy_(torch.mean(x_lane, dim=0, keepdim=True))
                self.lane_scale.copy_(torch.std(x_lane, dim=0, keepdim=True))

                #self.lane_con_mean.copy_(torch.mean(x_lane_conn, dim=0, keepdim=True))
               # self.lane_con_scale.copy_(torch.std(x_lane_conn, dim=0, keepdim=True))


            agent_noise = torch.randn_like(x_agent)*self.agent_scale+self.agent_mean

            lane_noise = torch.randn_like(x_lane)*self.lane_scale+self.lane_mean

            # lane_con_noise=torch.randn_like(x_lane_conn)*self.lane_con_scale+self.lane_con_mean

            t_batch = torch.rand((data.num_graphs,1), device=agent_noise.device)  # t ~ U[0,1]

            agent_t=t_batch[agent_batch]

            lane_t=t_batch[lane_batch]

            z_agent = (1 - agent_t) * agent_noise + agent_t * x_agent  # large t, low noise        target velocity e-x = (z-x)/(1-t)

            z_lane = (1 - lane_t) * lane_noise + lane_t * x_lane  # large t, low noise        target velocity e-x = (z-x)/(1-t)

            lane_pred, agent_pred=self.diff_model(z_agent,z_lane,t_batch,agent_batch,lane_batch)

            lane_conn_logits=self.diff_model.predict_con(x_lane,l2l_edge_index,lane_batch)

            lane_conn_loss=self.lane_conn_loss_fn(lane_conn_logits, x_lane_conn, lane_conn_batch).mean()

            self.log('train/lane_conn_loss', lane_conn_loss, on_step=True, batch_size=1)

            denom = (1 - agent_t).clamp_min(self.t_eps)

            match_loss, pos_loss, heading_loss, shape_loss, vel_loss, _ = get_matching_loss(
                agent_batch,
                agent_pred,
                x_agent,
                denom,
                scale=self.agent_scale,
                all_state=True,
                use_all_type=True,
               # w_shape=1,
            )

            denom = (1 - lane_t).clamp_min(self.t_eps)

            match_loss1, pos_loss1, heading_loss1, shape_loss1, vel_loss1, _ = get_matching_loss(
                lane_batch,
                lane_pred,
                x_lane,
                denom,
                all_state=True,
                use_all_type=True
            )
            self.log('train/match_loss', match_loss, on_step=True, batch_size=1)
            self.log('train/pos_loss', pos_loss, on_step=True, batch_size=1)
            self.log('train/heading_loss', heading_loss, on_step=True, batch_size=1)
            self.log('train/shape_loss', shape_loss, on_step=True, batch_size=1)
            self.log('train/vel_loss', vel_loss, on_step=True, batch_size=1)

            self.log('train/match_loss1', match_loss1, on_step=True, batch_size=1)
            self.log('train/pos_loss1', pos_loss1, on_step=True, batch_size=1)
            self.log('train/heading_loss1', heading_loss1, on_step=True, batch_size=1)
            self.log('train/vel_loss1', vel_loss1, on_step=True, batch_size=1)

            loss=match_loss1+match_loss+lane_conn_loss

            self.log('train/loss', loss, on_step=True, batch_size=1)

        return loss

    @torch.no_grad()
    def generate(self,data,batch_idx):

        if self.use_latent:

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

            z_agent =  torch.randn_like(x_agent)*self.agent_scale+self.agent_mean

            z_lane = torch.randn_like(x_lane)*self.lane_scale+self.lane_mean

            steps=20

            timesteps = torch.linspace(0, 1, steps + 1, device=agent_batch.device)

            for i in range(steps):
                t = timesteps[i]
                t_next = timesteps[i + 1]

                lane_pred, agent_pred=self.diff_model(z_agent,z_lane,t[None,None].repeat(data.num_graphs, 1),agent_batch,lane_batch)

                denom = (1.0 - t).clamp_min(self.t_eps)
                v_agent = (agent_pred - z_agent) / denom
                v_lane = (lane_pred - z_lane) / denom

                z_agent=z_agent+(t_next-t)*v_agent
                z_lane=z_lane+(t_next-t)*v_lane

            lane_conn_logits=self.diff_model.predict_con(z_lane,l2l_edge_index,lane_batch)
            lane_conn_pred = torch.argmax(lane_conn_logits, dim=1)
            lane_conn_samples =  F.one_hot(lane_conn_pred, num_classes=6)

            pred_head=torch.atan2(z_lane[:, 3], z_lane[:, 2])

            lane_samples = transform_to_global(
                z_lane[:, 4:].reshape(-1,20,2),
                None,
                z_lane[:, :2],
                pred_head
            )[0]

            # #x_agent : agent state + type   x,y, cosθ,sinθ ,length, width,speed,type-> x,y, speed,cosθ,sinθ ,length, width,type

            agent_samples= z_agent[:,[0,1,6,2,3,4,5]]# [pos_x, pos_y, speed, cos(heading), sin(heading), length, width]
            agent_types = torch.argmax(z_agent[:,-3:], dim=1)

        data['agent'].x = agent_samples
        data['lane'].x = lane_samples
        data['agent'].type = torch.nn.functional.one_hot(agent_types, num_classes=self.cfg_dataset.num_agent_types)
        data['lane', 'to', 'lane'].type =   lane_conn_samples

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

    ### Taken largely from QCNet repository: https://github.com/ZikangZhou/QCNet
    def configure_optimizers(self):
        """ Configure the optimizer and learning rate scheduler for the model."""
        #self.lr_warmup_steps=0

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

       # lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer]#, [lr_scheduler]

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

