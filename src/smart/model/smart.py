# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import random
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from waymo_open_dataset.utils.sim_agents.submission_specs import ChallengeType

from src.smart.metrics import (
    CrossEntropy,
    TokenCls,
    WOSACMetrics,
    WOSACSubmission,
    minADE,
)
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo,get_map_features
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts
from src.smart.plot.plot_bird.plot_bird import plot_bird_from_tensors
from src.smart.metrics.bird_metrics import compute_bird_metrics,MetricDict
from src.smart.plot.plot_rollout import plot_rollout_frames
from src.smart.metrics.wosac_metrics import WOSACMetrics
import time
from src.smart.metrics.gen_metrics import compute_gen_samples,compute_agent_metrics
import numpy as np
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors,
    weight_init,
    rotate_to_global
)
import os
from src.smart.tokens.token_processor import TokenProcessor


class SMART(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop

        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder,token_processor=self.token_processor, n_token_agent=self.token_processor.n_token_agent,
            finetune=model_config.finetune
        )

        self.training_rollout_len=18

        if  model_config.finetune:
            for p in self.encoder.map_encoder.parameters():
                p.requires_grad = False

            if self.token_processor.pred_init:
                if self.token_processor.learn_init:
                    if not self.encoder.gail or self.training_rollout_len==1:
                        for p in self.encoder.agent_encoder.parameters():
                            p.requires_grad = False

                    if self.encoder.init_decoder.G1.use_ref:
                        for p in self.encoder.init_decoder.G1.ref_model.parameters():
                            p.requires_grad = False
                else:
                    for p in self.encoder.init_decoder.parameters():
                        p.requires_grad = False

        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        if self.token_processor.pred_init:
            self.challenge_type=ChallengeType.SCENARIO_GEN
            working_dir = os.getcwd()
            if (('keguo' in working_dir) or ("guoke" in working_dir)):
                self.para_num=32
            else:
                self.para_num=1
            self.n_rollout_closed_val=2
        else:
            self.challenge_type=ChallengeType.SIM_AGENTS
            self.para_num=32
            self.n_rollout_closed_val=8

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.wosac_metrics = WOSACMetrics("val_closed",challenge_type=self.challenge_type)
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)


        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"
        self.video_dir.mkdir(exist_ok=True, parents=True)

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

        if self.wosac_submission.is_active:
            self.n_rollout_closed_val=32

        self.minADE0=0
        self.minADE0_num=0

        self.all_data=[]

        self.metric_logger=MetricDict()
       # self.wosac_submission.save_sub_file()

        self.samples = []
        self.gt_samples = []
        self.gt_dist=None

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            train_mask=data["agent"]["train_mask"],  # [n_agent]
            current_epoch=self.current_epoch,
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)

            attention_weight=self.encoder.agent_encoder.interative_decoder.a2a_attn_layers[0].attention_weight

            edge_weight=self.encoder.agent_encoder.interative_decoder.a2a_attn_layers[0].egde_weight

            edge_index_a2a,relative_pos=pred["edge_index_a2a"]

            sampled_pos=tokenized_agent["sampled_pos"]

            valid_mask=tokenized_agent["valid_mask"]
            self.all_data.append((attention_weight,edge_weight,edge_index_a2a,relative_pos,valid_mask,sampled_pos,pred["agent_q"]))

        # ! closed-loop vlidation
        if self.global_rank == 0 and self.val_closed_loop:
            pred_traj, pred_z, pred_head, pred_sizes, pred_vels = [], [], [], [], []
            pred_z_list_for_vis = []
            if self.encoder.sep_map:
                map_feature = self.encoder.map_encoder1(tokenized_map,tokenized_agent=tokenized_agent)
                tokenized_agent["initial_map_feature"] = map_feature

            map_feature = self.encoder.map_encoder(tokenized_map)
            tokenized_agent["map_feature"]=map_feature

            for _ in range(self.n_rollout_closed_val):

                pred = self.encoder.agent_encoder.inference(self.encoder.init_decoder,
                    tokenized_agent, map_feature,self.validation_rollout_sampling
                )

                if self.token_processor.traj_diffusion:
                    self.encoder.traj_diffuser.sample(pred, map_feature)

                pred_traj.append(pred["pred_traj_10hz"])

                if self.n_vis_batch > 0 and "pred_z_list" in tokenized_agent:
                    pred_z_list_global = self._pred_z_list_ego_local_to_global(
                        pred_z_list=tokenized_agent["pred_z_list"],
                        tokenized_agent=tokenized_agent,
                    )

                    pred_z_list_for_vis.append(
                        pred_z_list_global.detach().cpu()
                    )

                if not self.token_processor.use_bird:
                    pred_z.append(pred["pred_z_10hz"])
                    pred_head.append(pred["pred_head_10hz"])

                if self.challenge_type == ChallengeType.SCENARIO_GEN:
                    pred_sizes.append(pred["shape"])
                    pred_vels.append(pred["initial_local_vel"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]
            if len(pred_z_list_for_vis) > 0:
                # [N_agent, N_rollout, N_gen_step, D]
                pred_z_list_for_vis = torch.stack(pred_z_list_for_vis, dim=1)
            else:
                pred_z_list_for_vis = None

            if self.challenge_type == ChallengeType.SCENARIO_GEN:
                pred_sizes=torch.stack(pred_sizes, dim=1)[:,:,None].repeat(1,1,pred_traj.shape[2],1)
                pred_head=wrap_angle(pred_head)
                pred_sizes=torch.clamp_min(pred_sizes,min=0.1)

                # pred_traj=data["agent"]["position"][:, None,: ,:2].repeat(1,self.n_rollout_closed_val,1,1)
                # pred_head=data["agent"]["heading"][:, None,:].repeat(1,self.n_rollout_closed_val,1)
                # pred_sizes=data["agent"]["shape"][:, None,None].repeat(1,self.n_rollout_closed_val,pred_traj.shape[2],1)

                if not self.wosac_submission.is_active:
                    compute_gen_samples(data, tokenized_agent, pred_traj, pred_vels, pred_head, pred_sizes, self.samples,
                                        self.gt_samples,self.gt_dist)
            else:
                pred_traj=pred_traj[:,:,-80:]
                pred_z=pred_z[:,:,-80:]
                pred_head=pred_head[:,:,-80:]

            # ! WOSAC
            scenario_rollouts = None
            if self.wosac_submission.is_active:  # ! save WOSAC submission
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    pred_sizes=pred_sizes,
                    global_rank=self.global_rank,
                )

                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():  # single gpu fix
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()

            else:  # ! compute metrics, disable if save WOSAC submission
                if self.challenge_type != ChallengeType.SCENARIO_GEN:
                    self.minADE.update(
                        pred=pred_traj,
                        target=data["agent"]["position"][
                            :, self.num_historical_steps :, : pred_traj.shape[-1]
                        ],
                        target_valid=data["agent"]["valid_mask"][
                            :, self.num_historical_steps :
                        ],
                    ) #minimum sum distance

                # WOSAC metrics
                if batch_idx < self.n_batch_wosac_metric or batch_idx <self.n_vis_batch:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], device
                        ),
                        agent_id=tokenized_agent['id'],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                        pred_sizes=pred_sizes,
                    )
                    # mask=torch.bincount(tokenized_agent['batch'])<20
                    # valid_eval=torch.nonzero(mask)[:,0]
                    # scenario_rollouts = [scenario_rollouts[i] for i in valid_eval.tolist()]
                    # tfrecord_path=[ data["tfrecord_path"][i] for i in valid_eval.tolist()]
                    tfrecord_path=data["tfrecord_path"]
                    if self.n_vis_batch==0:
                        if self.challenge_type == ChallengeType.SCENARIO_GEN:
                            scenario_rollouts=scenario_rollouts[:64]
                        if len(scenario_rollouts) > self.para_num:
                            for i in range(np.ceil(len(scenario_rollouts) / self.para_num).astype(int)):  # 64
                                print(i)# [05:45<00:00] para  [05:27<00:00] resample 64: [06:11<00:00,
                                self.wosac_metrics.update(tfrecord_path[self.para_num * i:self.para_num * (i + 1)],
                                                          scenario_rollouts[self.para_num * i:self.para_num * (i + 1)])
                        else:
                            self.wosac_metrics.update(tfrecord_path,   scenario_rollouts)

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is not None:
                    for _i_sc in range(self.n_vis_scenario):
                        print('visualize', _i_sc)
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"step_{self.global_step}_batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                        )
                        # _vis.save_video_scenario_rollout(
                        #     scenario_rollouts[_i_sc], self.n_vis_rollout,
                        # )

                        scenario_pred_z_list = None

                        if pred_z_list_for_vis is not None:
                            agent_batch = data["agent"]["batch"].detach().cpu()
                            scenario_agent_mask = agent_batch == _i_sc

                            scenario_pred_z_list = pred_z_list_for_vis[scenario_agent_mask]
                            # shape: [N_agent_in_scenario, N_rollout, N_gen_step, D]

                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc],
                            self.n_vis_rollout,
                            pred_z_list=scenario_pred_z_list,
                            crop_size_m=200.0,
                            add_subtitle=True,
                        )

    def on_validation_epoch_end(self):

        if self.val_open_loop:

            torch.save(self.all_data,"all_data.pt")


        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()

                if self.challenge_type==ChallengeType.SCENARIO_GEN:
                    t1 = time.time()
                    print('metric compute start.')

                    self.result,self.gt_dist = compute_agent_metrics(self.samples, self.gt_samples, self.gt_dist,self.n_vis_batch>0)

                    self.samples=[]

                    for key, value in self.result.items():
                        self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                                 rank_zero_only=True)

                    print('metric compute time:', time.time() - t1)
                else:
                    epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()#ADE is all the sum distance for all agent

                if self.global_rank == 0:
                    if self.token_processor.use_bird:

                        epoch_wosac_metrics['val_closed/minADE'] = self.minADE0/self.minADE0_num

                        final_metrics = self.metric_logger.compute()

                        for key,value in final_metrics.items():
                            epoch_wosac_metrics['val_closed/'+key] = value

                        epoch_wosac_metrics['val_closed/scene_likelihood'] = (
                                                                                     epoch_wosac_metrics[
                                                                                         'val_closed/linear_speed_likelihood1'] +
                                                                                     epoch_wosac_metrics[
                                                                                         'val_closed/angular_speed_likelihood1'] +
                                                                                     epoch_wosac_metrics[
                                                                                         'val_closed/linear_acceleration_likelihood1'] +
                                                                                     epoch_wosac_metrics[
                                                                                         'val_closed/angular_acceleration_likelihood1'] +
                                                                                     epoch_wosac_metrics['val_closed/distance_likelihood1']+
                                                                                     epoch_wosac_metrics['val_closed/polar_likelihood1']+
                                                                                     epoch_wosac_metrics['val_closed/heading_likelihood1']
                                                                             ) / 7

                        epoch_wosac_metrics['val_closed/scene_emd'] = (
                                                                              epoch_wosac_metrics[
                                                                                  'val_closed/angular_acceleration_emd'] +
                                                                              epoch_wosac_metrics[
                                                                                  'val_closed/linear_speed_emd'] +
                                                                              epoch_wosac_metrics[
                                                                                  'val_closed/angular_speed_emd'] +
                                                                              epoch_wosac_metrics[
                                                                                  'val_closed/linear_acceleration_emd'] +
                                                                              epoch_wosac_metrics[
                                                                                  'val_closed/distance_emd'] +
                                                                              epoch_wosac_metrics['val_closed/polar_emd'] +
                                                                              epoch_wosac_metrics['val_closed/heading_emd']
                                                                      ) / 7.0

                        self.metric_logger.reset()

                    scalar_metrics = {
                        str(key): _to_python_scalar(value, str(key))
                        for key, value in epoch_wosac_metrics.items()
                    }

                    for key, value in scalar_metrics.items():#minADE is the time average distance for evaluated agent
                        self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        map_feature = self.encoder.map_encoder(tokenized_map)
        tokenized_agent["map_feature"] = map_feature

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head ,pred_sizes= [], [], [],[]
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.agent_encoder.inference(
                tokenized_agent, map_feature,  # post_sampling=True
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])
            pred_sizes.append(pred["shape"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]
        pred_sizes=torch.stack(pred_sizes, dim=1)[:,:,None].repeat(1,1,pred_traj.shape[2],1)

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            pred_sizes=pred_sizes,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def _pred_z_list_ego_local_to_global(self, pred_z_list, tokenized_agent):
        """
        Convert generated initial states from ego-local coordinates to Waymo global coordinates.

        Args:
            pred_z_list:
                [N_agent, N_gen_step, D] or [N_agent, N_rollout, N_gen_step, D]

                Expected layout for D >= 4:
                    0: x_local
                    1: y_local
                    2: cos(local_heading)
                    3: sin(local_heading)
                    4: length
                    5: width
                    6: vx
                    7: vy

            tokenized_agent:
                Must contain per-agent scene index and per-scene ego pose.

        Returns:
            pred_z_list_global:
                Same shape as pred_z_list, but position and heading are global.
        """
        z = pred_z_list.clone()

        if z.numel() == 0:
            return z

        # ------------------------------------------------------------
        # 1. Find each generated agent's scene index.
        # ------------------------------------------------------------
        if "nonego_batch" in tokenized_agent:
            agent_batch = tokenized_agent["nonego_batch"]
        elif "batch" in tokenized_agent:
            agent_batch = tokenized_agent["batch"]
        else:
            raise KeyError(
                "Cannot transform pred_z_list to global coordinates: "
                "tokenized_agent must contain 'nonego_batch' or 'batch'."
            )

        # ------------------------------------------------------------
        # 2. Find ego pose per scene.
        #
        # Prefer batch_ego_pos / batch_ego_heading if your tokenizer stores them.
        # These should be:
        #     batch_ego_pos:     [num_graphs, 2]
        #     batch_ego_heading: [num_graphs]
        # ------------------------------------------------------------
        if "batch_ego_pos" in tokenized_agent:
            ego_pos = tokenized_agent["batch_ego_pos"][..., :2]
        elif "ego_pos" in tokenized_agent:
            ego_pos = tokenized_agent["ego_pos"][..., :2][agent_batch]
        else:
            raise KeyError(
                "Cannot transform pred_z_list to global coordinates: "
                "tokenized_agent must contain 'batch_ego_pos' or 'ego_pos'."
            )

        if "batch_ego_heading" in tokenized_agent:
            ego_heading = tokenized_agent["batch_ego_heading"]
        elif "ego_heading" in tokenized_agent:
            ego_heading = tokenized_agent["ego_heading"][agent_batch]
        else:
            raise KeyError(
                "Cannot transform pred_z_list to global coordinates: "
                "tokenized_agent must contain 'batch_ego_heading' or 'ego_heading'."
            )

        # ------------------------------------------------------------
        # 3. Convert position.
        # ------------------------------------------------------------
        if z.ndim == 3:
            # [N_agent, N_gen_step, D]
            local_pos = z[..., :2]  # [N, G, 2]

            global_pos = transform_to_global(
                pos_local=local_pos,
                head_local=None,
                pos_now=ego_pos,
                head_now=ego_heading,
            )[0]

            z[..., :2] = global_pos

            # --------------------------------------------------------
            # 4. Convert heading cos/sin.
            # --------------------------------------------------------
            local_heading = torch.atan2(z[..., 3], z[..., 2])
            global_heading = wrap_angle(local_heading + ego_heading[:, None])

            z[..., 2] = torch.cos(global_heading)
            z[..., 3] = torch.sin(global_heading)

            # --------------------------------------------------------
            # 5. Optional: convert velocity if vx/vy are ego-local.
            # --------------------------------------------------------
            local_vel = z[..., 6:8]  # [N, G, 2]
            global_vel = rotate_to_global(
                local_vel,
                global_heading,
            )
            z[..., 6:8] = global_vel

        elif z.ndim == 4:
            # [N_agent, N_rollout, N_gen_step, D]
            n_agent, n_rollout, n_gen_step, dim = z.shape

            z_flat = z.reshape(n_agent, n_rollout * n_gen_step, dim)

            local_pos = z_flat[..., :2]

            global_pos = transform_to_global(
                pos_local=local_pos,
                head_local=None,
                pos_now=ego_pos,
                head_now=ego_heading,
            )[0]

            z_flat[..., :2] = global_pos

            local_heading = torch.atan2(z_flat[..., 3], z_flat[..., 2])
            global_heading = wrap_angle(local_heading + ego_heading[:, None])

            z_flat[..., 2] = torch.cos(global_heading)
            z_flat[..., 3] = torch.sin(global_heading)

            local_vel = z_flat[..., 6:8]
            global_vel = rotate_to_global(
                local_vel,
                global_heading[:, None],
            )
            z_flat[..., 6:8] = global_vel

            z = z_flat.reshape(n_agent, n_rollout, n_gen_step, dim)

        else:
            raise ValueError(
                f"pred_z_list must have shape [N,G,D] or [N,R,G,D], got {z.shape}"
            )

        return z

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()

import numbers

def _to_python_scalar(value, name: str) -> float:
    """Convert a metric value to a plain finite Python float."""

    if isinstance(value, torch.Tensor):
        value = value.detach()

        if value.numel() != 1:
            raise ValueError(
                f"Metric {name!r} must be scalar, but got "
                f"shape={tuple(value.shape)}, numel={value.numel()}."
            )

        value = value.float().cpu().item()

    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"Metric {name!r} must be scalar, but got "
                f"NumPy shape={value.shape}, size={value.size}."
            )
        value = value.item()

    elif isinstance(value, np.generic):
        value = value.item()

    elif isinstance(value, numbers.Number):
        value = float(value)

    else:
        raise TypeError(
            f"Unsupported metric type for {name!r}: {type(value).__name__}"
        )

    value = float(value)


    return value
