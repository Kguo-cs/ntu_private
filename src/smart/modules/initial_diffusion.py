# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import pytorch_lightning as pl
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
import os
from .init_diffusion import InitDiffusion
from argparse import ArgumentParser

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    transform_to_local,
    wrap_angle,
)

class PDInit(nn.Module):

    def __init__(self) -> None:
        super(PDInit, self).__init__()

        parser = ArgumentParser()
        # parser.add_argument('--root', type=str, required=True)
        # parser.add_argument('--train_batch_size', type=int, required=True)
        # parser.add_argument('--val_batch_size', type=int, required=True)
        # parser.add_argument('--test_batch_size', type=int, required=True)
        # parser.add_argument('--shuffle', type=bool, default=True)
        # parser.add_argument('--num_workers', type=int, default=8)
        # parser.add_argument('--pin_memory', type=bool, default=True)
        # parser.add_argument('--persistent_workers', type=bool, default=True)
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

    def forward(self, map_feature, tokenized_agent,map_range=100):

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        batch_num = tokenized_agent["num_graphs"]

        gt_initial_pos = tokenized_agent["gt_initial_pos"][:, 0]
        gt_initial_heading = tokenized_agent["gt_initial_heading"][:, 0]

        ego_mask = tokenized_agent["ego_mask"]

        ego_position = gt_initial_pos[ego_mask]
        ego_heading = gt_initial_heading[ego_mask]

        pos_pl, orient_pl = transform_to_local(pos_pl,  # [:,None],
                                               orient_pl,  # [:,None],
                                               ego_position[batch_pl],
                                               ego_heading[batch_pl],
                                               )

        ego_dist = torch.linalg.norm(pos_pl, dim=-1)

        ego_dist_mask = ego_dist < map_range

        pos_pl = pos_pl[ego_dist_mask]/map_range
        orient_pl = orient_pl[ego_dist_mask]
        batch_pl = batch_pl[ego_dist_mask]
        feat_map = feat_map[ego_dist_mask]

        map_feature = (pos_pl, orient_pl, batch_pl, feat_map)

        non_ego = ~ego_mask

        batch = tokenized_agent["batch"][non_ego]

        shape = tokenized_agent["gt_initial_shape"]

        real_shape = shape[non_ego]

        real_pos, real_heading = transform_to_local(gt_initial_pos[non_ego],
                                                    gt_initial_heading[non_ego],
                                                    ego_position[batch],
                                                    ego_heading[batch],
                                                    )


        init_trans = real_pos[:, :2]/map_range

        delta_rot = real_heading.unsqueeze(-1)

        init_angle = torch.cat([delta_rot.cos(), delta_rot.sin()], dim=-1)

        init_speed = real_shape[:,-2:].norm(dim=-1)/25-1

        m_init = torch.cat([init_trans, init_angle, init_speed[:,None]], dim=-1)

        num_samples=1

        loss_diff_init, pred_init = self.joint_diffusion.get_loss(m_init,tokenized_agent,map_feature)

        pred_trans, pred_head, pred_speed = pred_init[..., :2]*map_range, pred_init[..., 2:4], pred_init[..., 4]*25+25

        target_origin = init_trans.unsqueeze(1).repeat(1, num_samples, 1)
        target_theta = init_angle.unsqueeze(1).repeat(1, num_samples,1)
        target_speed = init_speed.unsqueeze(1).repeat(1, num_samples)

        loss_trans = torch.nn.HuberLoss()(pred_trans, target_origin)
        loss_rot2 = torch.nn.HuberLoss()(pred_head, target_theta)
        loss_speed = torch.nn.HuberLoss()(pred_speed,target_speed)

        loss_diff_trans = loss_diff_init[..., :2].mean()
        loss_diff_theta = loss_diff_init[..., 2:4].mean()
        loss_diff_speed = loss_diff_init[..., 4].mean()
        loss_diff_init = loss_diff_init.mean()

        return loss_diff_init,loss_trans,loss_rot2,loss_speed,loss_diff_trans,loss_diff_theta,loss_diff_speed

    def validation_step(self,
                        data,
                        batch_idx):
        if self.guid_sampling == 'no_guid':
            self.validation_step_norm(data, batch_idx)
        elif self.guid_sampling == 'guid':
            self.validation_step_guid(data, batch_idx)

    def load_vars(self, device):
        s_mean = np.load(self.path_pca_s_mean)
        self.s_mean = torch.tensor(s_mean).to(device)
        VT_k = np.load(self.path_pca_VT_k)

        self.VT_k = torch.tensor(VT_k).to(device)
        if self.path_pca_V_k != 'none':
            V_k = np.load(self.path_pca_V_k)
            self.V_k = torch.tensor(V_k).to(device)
        else:
            self.V_k = self.VT_k.transpose(0, 1)
        latent_mean = np.load(self.path_pca_latent_mean)
        self.latent_mean = torch.tensor(latent_mean).to(device)
        latent_std = np.load(self.path_pca_latent_std) * 2
        self.latent_std = torch.tensor(latent_std).to(device)

    def validation_step_norm(self,
                             data,
                             batch_idx):
        print_flag = False
        if batch_idx % 100 == 0:
            print_flag = True

        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        scene_enc = self.qcnet_mapencoder(data)

        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        if self.s_mean == None:
            self.load_vars(self.device)
        mask = data['agent']['mask']

        gt_n = gt[mask][..., :self.output_dim]
        device = gt_n.device

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        gt_eval = gt[eval_mask]

        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples

        if_output_diffusion_process = False

        if if_output_diffusion_process:
            reverse_steps = self.num_diffusion_steps
            pred_modes, pred_delta = self.joint_diffusion.sample(num_samples, data=data, scene_enc=scene_enc,
                                                                 sampling=self.sampling,
                                                                 stride=self.sampling_stride, eval_mask=eval_mask,
                                                                 if_output_diffusion_process=if_output_diffusion_process,
                                                                 reverse_steps=reverse_steps)

            inter_latents = pred_modes[::1]
            inter_trajs = []
            for latent in inter_latents:
                inter_trajs.append(latent)

            pred_modes = pred_modes[-1]

        else:

            reverse_steps = None
            pred_init = self.joint_diffusion.sample(num_samples, data=data, scene_enc=scene_enc,
                                                    sampling=self.sampling,
                                                    stride=self.sampling_stride, eval_mask=eval_mask,
                                                    if_output_diffusion_process=if_output_diffusion_process,
                                                    reverse_steps=reverse_steps)
        agent_batch = data['agent']['batch'][eval_mask]

        pred_trans, pred_head, pred_speed = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4]

        map_min = data['map_min'].view(-1, 3)[..., :2].unsqueeze(1)[agent_batch]
        map_max = data['map_max'].view(-1, 3)[..., :2].unsqueeze(1)[agent_batch]

        pred_trans = (pred_trans + 1) * (map_max - map_min) / 2 + map_min

        if True in torch.isnan(pred_trans):
            print('nan')
            print(data_batch)
            exit()

        # joint mode clustering
        batch_idx = data['agent']['batch'][eval_mask]
        num_scenes = batch_idx[-1].item() + 1
        num_agents_per_scene = pred_trans.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)]).type(
            torch.int64)

        all_rel_origin_eval = data['agent']['position'][eval_mask, self.init_timestep, :2]
        all_rel_theta_eval = data['agent']['heading'][eval_mask, self.init_timestep]
        all_rel_rot_mat = self.create_rot_mat(all_rel_theta_eval, all_rel_theta_eval.shape[0])
        all_rel_rot_mat_inv = all_rel_rot_mat.permute(0, 2, 1)

        pred_rot_mat = self.create_rot_mat_cossin(pred_head, pred_head.shape[0])
        pred_rot_mat_inv = pred_rot_mat.permute(0, 2, 1)

        if self.eval_line.device != device:
            self.eval_line = self.eval_line.to(device)
        rec_traj_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                      pred_rot_mat_inv).unsqueeze(1) + pred_trans.reshape(-1, 1, 1, 2)
        gt_eval_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                     all_rel_rot_mat_inv) + all_rel_origin_eval[:, :2].reshape(-1, 1, 2)

        trans_loss = torch.nn.MSELoss()(pred_trans.squeeze(1), all_rel_origin_eval)
        dist_from_gt = torch.norm(pred_trans.squeeze(1) - all_rel_origin_eval, dim=-1)
        self.dist_from_gt_all += dist_from_gt.sum().item()
        self.cnt += dist_from_gt.size(0)

        self.log('val_trans_loss', trans_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)
        agent_batch = data['agent']['batch'][eval_mask]
        pred_angle = torch.atan2(pred_head[..., 1], pred_head[..., 0])
        gt_init = torch.cat([all_rel_origin_eval, all_rel_theta_eval.unsqueeze(-1)], dim=-1)
        pred_init = torch.cat([pred_trans, pred_angle.unsqueeze(-1)], dim=-1)

        self.JSD_LOCAL_DENSITY.update(pred=pred_init, gt=gt_init, agent_batch=agent_batch, data=data)
        self.log('val/jsd_local_density', self.JSD_LOCAL_DENSITY, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)
        self.JSD_MAP_DIST.update(pred=pred_init, gt=gt_init, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/jsd_map_dist', self.JSD_MAP_DIST, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)
        hist_angle_gt, hist_angle_pred = self.JSD_MAP_DIST.get_angle_hist()

        self.JSD_MAP_ANGLE.update(hist_angle_gt, hist_angle_pred)
        self.log('val/jsd_map_angle', self.JSD_MAP_ANGLE, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.JSD_INTERACTIVE.update(pred_init, gt_init, agent_batch=agent_batch)
        self.log('val/jsd_interactive', self.JSD_INTERACTIVE, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        gt_init_speed = data['agent']['init_speed'][eval_mask]
        self.JSD_SPEED.update(pred_speed, gt_init_speed)
        self.log('val/jsd_speed', self.JSD_SPEED, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.OffRoad.update(pred=pred_trans, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/offroad_rate', self.OffRoad, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.NearestEdge.update(pred=pred_trans, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/nearest_edge_dist', self.NearestEdge, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.Collision.update(pred=pred_trans, agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate', self.Collision, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.OffRoad_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/offroad_rate_gt', self.OffRoad_gt, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.NearestEdge_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch, map_pts=data['map_point'])
        self.log('val/nearest_edge_dist_gt', self.NearestEdge_gt, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.Collision_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch,
                                 agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate_gt', self.Collision_gt, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)
        if print_flag:
            print(
                f'GT: collision: {self.Collision_gt.compute().item()}, nearest_edge:{self.NearestEdge_gt.compute().item()}, offroad:{self.OffRoad_gt.compute().item()}')
            mean_dist = self.dist_from_gt_all / self.cnt
            print(
                f'Gen: collision: {self.Collision.compute().item()}, nearest_edge:{self.NearestEdge.compute().item()}, offroad:{self.OffRoad.compute().item()}, dist: {mean_dist}')

    def validation_step_guid(self,
                             data,
                             batch_idx):
        print_flag = False
        if batch_idx % 1 == 0:
            print_flag = True

        data_batch = batch_idx
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        scene_enc = self.qcnet_mapencoder(data)
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        if self.s_mean == None:
            self.load_vars(gt.device)

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['mask']
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        gt_eval = gt[eval_mask]

        self.joint_diffusion.eval()
        num_samples = self.num_eval_samples

        task = self.guid_task
        if task == 'none':
            cond_gen = None
            grad_guid = None

            vel = (gt_eval[:, 1:, :2] - gt_eval[:, :-1, :2]).detach().clone()
            vel = torch.abs(vel)
            max_vel = vel.max(-2)[0]

            vel = (gt_eval[:, 1:, :2] - gt_eval[:, :-1, :2]).detach().clone()
            mean_vel = vel.mean(-2)

        elif task == 'map':
            goal_point = gt_eval[:, -1, :2].detach().clone()

            grad_guid = [data, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None

        elif task == 'map_collision':

            grad_guid = [data, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
        elif task == 'original':

            grad_guid = [data, self.s_mean, self.V_k, self.VT_k, self.latent_mean, self.latent_std]
            cond_gen = None
        else:
            raise print('unseen tasks.')

        guid_method = self.guid_method  # none ECM ECMR
        guid_param = {}
        guid_param['task'] = task
        guid_param['guid_method'] = guid_method
        cost_param = {'cost_param_costl': self.cost_param_costl, 'cost_param_threl': self.cost_param_threl}
        guid_param['cost_param'] = cost_param

        sub_folder = self.ckpt_path.split('/')[-3]
        os.makedirs('visual/' + sub_folder, exist_ok=True)

        pred_init = self.joint_diffusion.sample(num_samples, data=data, scene_enc=scene_enc,
                                                sampling=self.sampling,
                                                stride=self.sampling_stride, eval_mask=eval_mask,
                                                grad_guid=grad_guid, cond_gen=cond_gen,
                                                guid_param=guid_param)

        if True in torch.isnan(pred_init):
            print('nan')
            print(data_batch)
            exit()

        pred_trans, pred_head, pred_speed = pred_init[..., :2], pred_init[..., 2:4], pred_init[..., 4]

        batch_idx = data['agent']['batch'][eval_mask]
        map_min = data['map_min'].view(-1, 3)[..., :2].unsqueeze(1)[batch_idx]
        map_max = data['map_max'].view(-1, 3)[..., :2].unsqueeze(1)[batch_idx]
        pred_trans = (pred_trans + 1) * (map_max - map_min) / 2 + map_min
        device = pred_trans.device
        num_scenes = batch_idx[-1].item() + 1
        num_agents_per_scene = batch_idx.new_tensor([(batch_idx == i).sum() for i in range(num_scenes)])

        all_rel_origin_eval = data['agent']['position'][eval_mask, self.init_timestep, :2]
        all_rel_theta_eval = data['agent']['heading'][eval_mask, self.init_timestep]
        all_rel_rot_mat = self.create_rot_mat(all_rel_theta_eval, all_rel_theta_eval.shape[0])
        all_rel_rot_mat_inv = all_rel_rot_mat.permute(0, 2, 1)

        pred_rot_mat = self.create_rot_mat_cossin(pred_head, pred_head.shape[0])
        pred_rot_mat_inv = pred_rot_mat.permute(0, 2, 1)

        if self.eval_line.device != device:
            self.eval_line = self.eval_line.to(device)
        rec_traj_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                      pred_rot_mat_inv).unsqueeze(1) + pred_trans.reshape(-1, 1, 1, 2)
        gt_eval_world = torch.matmul(self.eval_line.repeat(gt_eval.shape[0], 1, 1),
                                     all_rel_rot_mat_inv) + all_rel_origin_eval[:, :2].reshape(-1, 1, 2)
        trans_loss = torch.nn.MSELoss()(pred_trans.squeeze(1), all_rel_origin_eval)
        dist_from_gt = torch.norm(pred_trans.squeeze(1) - all_rel_origin_eval, dim=-1)
        self.dist_from_gt_all += dist_from_gt.sum().item()
        self.cnt += dist_from_gt.size(0)

        self.log('val_trans_loss', trans_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)
        agent_batch = data['agent']['batch'][eval_mask]

        self.OffRoad.update(pred=pred_trans, agent_batch=agent_batch, scenario_id_list=data['scenario_id'])
        self.log('val/offroad_rate', self.OffRoad, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.NearestEdge.update(pred=pred_trans, agent_batch=agent_batch, scenario_id_list=data['scenario_id'])
        self.log('val/nearest_edge_dist', self.NearestEdge, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.Collision.update(pred=pred_trans, agent_batch=agent_batch, agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate', self.Collision, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.OffRoad_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch, scenario_id_list=data['scenario_id'])
        self.log('val/offroad_rate_gt', self.OffRoad_gt, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.NearestEdge_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch,
                                   scenario_id_list=data['scenario_id'])
        self.log('val/nearest_edge_dist_gt', self.NearestEdge_gt, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)

        self.Collision_gt.update(pred=all_rel_origin_eval, agent_batch=agent_batch,
                                 agent_type=data['agent']['type'][eval_mask])
        self.log('val/collision_rate_gt', self.Collision_gt, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=len(data['scenario_id']), sync_dist=True)
        if print_flag:
            print(
                f'GT: collision: {self.Collision_gt.compute().item()}, nearest_edge:{self.NearestEdge_gt.compute().item()}, offroad:{self.OffRoad_gt.compute().item()}')
            mean_dist = self.dist_from_gt_all / self.cnt
            print(
                f'Gen: collision: {self.Collision.compute().item()}, nearest_edge:{self.NearestEdge.compute().item()}, offroad:{self.OffRoad.compute().item()}, dist: {mean_dist}')


    def configure_optimizers(self):
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

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, weight_decay=self.weight_decay, lr=self.lr)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.lr,
                                                        steps_per_epoch=self.trainer.estimated_stepping_batches // self.trainer.max_epochs,
                                                        # Or len(train_dataloader) if you know it
                                                        epochs=self.trainer.max_epochs)

        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',  # or 'epoch', depending on when you want to step the scheduler
            'frequency': 1
        }]


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
