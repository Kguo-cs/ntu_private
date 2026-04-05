import os
import pickle
from sd.utils.train_helpers import create_lambda_lr_cosine, create_lambda_lr_linear
from sd.datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
from sd.datasets.nuplan.dataset_autoencoder_nuplan import NuplanDatasetAutoEncoder
from torch_geometric.loader import DataLoader

import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from torch_geometric.data import Batch

from smart.utils import transform_to_local

torch.set_printoptions(sci_mode=False)

from sd.utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph
from sd.utils.data_helpers import unnormalize_scene
from sd.utils.viz import visualize_batch

from sd.nn_modules.agent_diffuser import Agent_Diffuser

from sd.utils.data_container import get_batches, get_features, get_edge_indices, get_encoder_edge_indices

# this ensures CPUs are not suboptimally utilized
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))


class Direct_diffusion(pl.LightningModule):
    """PyTorch Lightning module for ScenarioDreamer AutoEncoder model."""

    def __init__(self, cfg):
        super(Direct_diffusion, self).__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.cfg_dataset = self.cfg.dataset
        self.model = Agent_Diffuser(cfg.model)

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

        lane_dim=44

        self.register_buffer("lane_mean", torch.zeros(1, lane_dim))
        self.register_buffer("lane_scale", torch.ones(1, lane_dim))
        self.t_eps=0.01


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

    def training_step(self, data, batch_idx):
        """ Training step for the model. Computes the loss and logs it to WandB."""
        num_graphs=data["num_graphs"]

        agent_batch, lane_batch, lane_conn_batch = get_batches(data)
        x_agent, x_agent_states, x_agent_types, x_lane, x_lane_states, x_lane_types, x_lane_conn = get_features(data)


        #x_agent : agent state + type  x,y, speed,cosθ,sinθ ,length, width,type
        #x_lane:  2* 20*2

        # agent_pos=x_agent_states[:2]
        # heading=torch.atan2(x_agent_states[3], x_agent_states[2])

        lane_pos=x_lane_states.mean(1)
        lane_heading = torch.atan2(
            x_lane_states[:,1:, 1] - x_lane_states[:,:-1, 1],
            x_lane_states[:,1:, 0] - x_lane_states[:,:-1, 0],
        ).mean(1)

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


        agent_noise = torch.randn_like(x_agent)*self.agent_scale+self.agent_mean

        lane_noise = torch.randn_like(x_lane)*self.lane_scale+self.lane_mean

        t_batch = torch.rand(num_graphs, device=agent_noise.device)[:,  None]  # t ~ U[0,1]

        agent_t=t_batch[agent_batch]

        lane_t=t_batch[lane_batch]

        z_agent = (1 - agent_t) * agent_noise + agent_t * x_agent  # large t, low noise        target velocity e-x = (z-x)/(1-t)

        z_lane = (1 - lane_t) * lane_noise + agent_t * x_lane  # large t, low noise        target velocity e-x = (z-x)/(1-t)

        clean_agent,clean_lane=self.model(z_agent,z_lane,t_batch,agent_batch,lane_batch)




        #loss_dict = self.model.loss(data)
        self._log_losses(loss_dict, split='train')

        return loss_dict['loss']

    def validation_step(self, data, batch_idx):
        """ Validation step for the model. Computes the loss and logs it to WandB."""
        loss_dict = self.model.loss(data)
        self._log_losses(loss_dict, split='val', batch_size=data.batch_size)

        if self.cfg.train.num_samples_to_visualize > 0 and batch_idx == 0 and self.trainer.is_global_zero:
            num_samples = self.cfg.train.num_samples_to_visualize
            assert num_samples <= data.batch_size, f"num_samples ({num_samples}) must be less than or equal to batch size ({data.batch_size})"

            # retrieve first num_samples samples from the batch
            indices = torch.arange(num_samples)
            subset_data_list = data.index_select(indices)
            subset_data = Batch.from_data_list(subset_data_list)

            # forward the model to get reconstructed samples
            agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples, _ = self.forward(subset_data)
            save_dir = self.cfg.train.viz_dir

            print(f"Visualizing batch {batch_idx}...")
            images_to_log = visualize_batch(num_samples,
                                            agent_samples,
                                            lane_samples,
                                            agent_types,
                                            lane_types,
                                            lane_conn_samples,
                                            subset_data,
                                            save_dir,
                                            self.current_epoch,
                                            batch_idx,
                                            self.cfg.train.track)
            if self.cfg.train.track:
                self.logger.experiment.log(images_to_log)

            print('finished visualizing')

    def test_step(self, data, batch_idx):
        """ Test step for the model. Either computes test loss or caches latents based on configuration."""
        if self.cfg.eval.cache_latents.enable_caching:
            self._cache_latents(data)
        else:
            loss_dict = self.model.loss(data)
            self._log_losses(loss_dict, split='test', batch_size=data.batch_size)
            if self.cfg.eval.num_samples_to_visualize > 0 and batch_idx == 0:
                num_samples = self.cfg.eval.num_samples_to_visualize
                assert num_samples <= data.batch_size, f"num_samples ({num_samples}) must be less than or equal to batch size ({data.batch_size})"

                # retrieve first num_samples samples from the batch
                indices = torch.arange(num_samples)
                subset_data_list = data.index_select(indices)
                subset_data = Batch.from_data_list(subset_data_list)

                # forward the model to get reconstructed samples
                agent_samples, lane_samples, agent_types, lane_types, lane_conn_samples, _ = self.forward(subset_data)
                save_dir = self.cfg.eval.viz_dir

                print(f"Visualizing batch {batch_idx}...")
                visualize_batch(
                    num_samples,
                    agent_samples,
                    lane_samples,
                    agent_types,
                    lane_types,
                    lane_conn_samples,
                    subset_data,
                    save_dir,
                    self.current_epoch,
                    batch_idx,
                    False,
                    visualize_lane_graph=self.cfg.eval.visualize_lane_graph)

    def on_before_optimizer_step(self, optimizer):
        """ Called before the optimizer step. Logs the gradient norms for each layer."""
        # Compute the 2-norm for each layer
        norms_encoder = grad_norm(self.model, norm_type=2)
        self.log_dict(norms_encoder)

    ### Taken largely from QCNet repository: https://github.com/ZikangZhou/QCNet
    def configure_optimizers(self):
        """ Configure the optimizer and learning rate scheduler for the model."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.cfg.train.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay,
                                      betas=(self.cfg.train.beta_1, self.cfg.train.beta_2), eps=self.cfg.train.epsilon)

        if self.cfg.train.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                          lr_lambda=create_lambda_lr_cosine(self.cfg))
        elif self.cfg.train.lr_schedule == 'linear':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                          lr_lambda=create_lambda_lr_linear(self.cfg))

        return [optimizer], {"scheduler": scheduler,
                             "interval": "step",
                             "frequency": 1}