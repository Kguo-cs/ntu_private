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
from src.smart.modules.build_edge import insert_ego
from src.smart.metrics.gen_metrics import compute_gen_samples,compute_agent_metrics
import numpy as np

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

        self.use_smart=model_config.smart
        self.use_bird=model_config.bird

        if self.use_smart:
            from src.smart.tokens.smart_token_processsor import TokenProcessor
        elif self.use_bird:
            from src.smart.tokens.token_bird_processor import TokenProcessor
        else:
            from src.smart.tokens.token_processor import TokenProcessor

        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder,token_processor=self.token_processor, n_token_agent=self.token_processor.n_token_agent
        )

        if self.use_smart:
            set_model_for_finetuning(self.encoder, model_config.finetune)

        if  self.encoder.agent_encoder.learn_init_only and self.encoder.agent_encoder.learn_init:
            for p in self.encoder.parameters():
                p.requires_grad = False

            for p in self.encoder.agent_encoder.init_decoder.parameters():
                p.requires_grad = True

            if self.encoder.sep_map:

                for p in self.encoder.map_encoder1.parameters():
                    p.requires_grad = True

            if not self.encoder.agent_encoder.use_gan and self.encoder.agent_encoder.init_decoder.latent_diffusion and not self.encoder.agent_encoder.init_decoder.learn_autoencoder:
                for p in self.encoder.agent_encoder.init_decoder.autoencoder.parameters():
                    p.requires_grad = False
                for p in self.encoder.agent_encoder.init_decoder.G.pose_embedding.parameters():
                    p.requires_grad = False
                for p in self.encoder.agent_encoder.init_decoder.G.ego_embedding.parameters():
                    p.requires_grad = False
                # for p in self.encoder.map_encoder.parameters():
                #     p.requires_grad = False

        # else:
        #     for p in self.encoder.map_encoder.parameters():
        #             p.requires_grad = False
        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        if self.token_processor.pred_init and self.encoder.agent_encoder.learn_init:
            self.challenge_type=ChallengeType.SCENARIO_GEN
            self.para_num=2
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
        # self.log("val_closed/wosac_likelihood/metametric", float("-inf"), prog_bar=False, on_epoch=True,
        #          rank_zero_only=True)

        if self.wosac_submission.is_active:
            self.n_rollout_closed_val=32

        self.all_time=0
        self.all_count=0
        self.minADE0=0
        self.minADE0_num=0

        self.all_data=[]

        self.metric_logger=MetricDict()
        self.samples = []
        self.gt_samples = []

        #self.wosac_submission.save_sub_file()

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
        # torch.random.manual_seed(1)
        # if batch_idx not in [1819]:#[28,109,164,242,402,729,842,1819]: #3500
        #     return
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
            pred_traj, pred_z, pred_head,new_agent,pred_sizes,pred_speeds = [], [], [],[],[],[]
            # tokenized_map, tokenized_agent = self.token_processor(data)
            map_feature = self.encoder.map_encoder(tokenized_map)

            if self.encoder.agent_encoder.learn_init :
                if self.encoder.sep_map:
                    map_feature1 = self.encoder.map_encoder1(tokenized_map,tokenized_agent)
                else:
                    map_feature1 = self.encoder.map_encoder(tokenized_map,tokenized_agent)

                tokenized_agent["initial_map_feature"]=map_feature1


            for _ in range(self.n_rollout_closed_val):

                pred = self.encoder.agent_encoder.inference(
                    tokenized_agent, map_feature,self.validation_rollout_sampling
                )
                pred_traj.append(pred["pred_traj_10hz"])

                if not self.token_processor.use_bird:
                    pred_z.append(pred["pred_z_10hz"])
                    pred_head.append(pred["pred_head_10hz"])

                    if "new_agent" in pred.keys():
                        new_agent.append(pred["new_agent"])

                if self.challenge_type == ChallengeType.SCENARIO_GEN:
                    pred_sizes.append(pred["shape"])
                    pred_speeds.append(pred["initial_speed"])


            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            if not self.token_processor.use_bird:
                pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
                pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

                if len(new_agent):
                    new_agent=torch.stack(new_agent, dim=1).cpu().numpy()

            if self.challenge_type == ChallengeType.SCENARIO_GEN:
                pred_sizes=torch.stack(pred_sizes, dim=1)[:,:,None].repeat(1,1,pred_traj.shape[2],1)

                compute_gen_samples(data, tokenized_agent, pred_traj, pred_speeds, pred_head, pred_sizes, self.samples,
                                    self.gt_samples)

            else:
                pred_traj=pred_traj[:,:,-80:]
                pred_z=pred_z[:,:,-80:]
                pred_head=pred_head[:,:,-80:]

            if self.token_processor.use_bird :
                save_path = self.video_dir / f"step_{self.global_step}_batch_{batch_idx:02d}"

                if batch_idx < self.n_vis_batch:
                    batch=pred["batch"]
                    plot_bird_from_tensors(pred_traj[batch==0],tokenized_agent['sampled_pos'][batch==0],
                              data["agent"]["position"][:,self.num_historical_steps :][batch==0], data["agent"]["valid_mask"][:,self.num_historical_steps :][batch==0],
                                           show=False,      save_path=save_path
                              )

                (heading_likelihoods,polar_likelihoods,distance_likelihoods,linear_speed_likelihoods, linear_acc_likelihoods, angular_speed_likelihoods,
                 angular_acceleration_likelihoods,num_diff,num_entry_diff, num_exit_diff)=compute_bird_metrics(pred_traj, data["agent"]["position"][:,self.num_historical_steps :],
                                     data["agent"]["valid_mask"][:,self.num_historical_steps :],
                                        tokenized_agent["batch"],batch_idx < self.n_vis_batch,save_path=save_path)

                target = data["agent"]["position"][
                         :, self.num_historical_steps:, : pred_traj.shape[-1]
                         ]
                target_valid = data["agent"]["valid_mask"][
                               :, self.num_historical_steps:
                               ]

                current_valid=target_valid[:,0]

                pred=pred_traj[current_valid]
                target=target[current_valid]
                target_valid=target_valid[current_valid]

                pred_valid_mask = (pred!=10000).any(dim=-1).any(dim=1) #any false

                target_valid = target_valid & pred_valid_mask

                dist = torch.norm(pred - target.unsqueeze(1), p=2, dim=-1)
                dist2 = (dist * target_valid.unsqueeze(1)).sum(-1).amin(-1)  # [n_agent]

                self.minADE0 += (dist2 / (target_valid.sum(-1) + 1e-6)).sum() # [n_agent]
                self.minADE0_num+=len(dist2)

                metric_dict = {
                    "linear_speed_likelihood1": linear_speed_likelihoods[1].mean().item(),
                    "linear_acceleration_likelihood1": linear_acc_likelihoods[1].mean().item(),
                    "angular_speed_likelihood1": angular_speed_likelihoods[1].mean().item(),
                    "angular_acceleration_likelihood1": angular_acceleration_likelihoods[1].mean().item(),
                    "heading_likelihood1": heading_likelihoods[1].mean().item(),
                    "polar_likelihood1": polar_likelihoods[1].mean().item(),
                    "distance_likelihood1": distance_likelihoods[1].mean().item(),

                    "linear_speed_emd": linear_speed_likelihoods[2].mean().item(),
                    "linear_acceleration_emd": linear_acc_likelihoods[2].mean().item(),
                    "angular_speed_emd": angular_speed_likelihoods[2].mean().item(),
                    "angular_acceleration_emd": angular_acceleration_likelihoods[2].mean().item(),
                    "heading_emd": heading_likelihoods[2].mean().item(),
                    "polar_emd": polar_likelihoods[2].mean().item(),
                    "distance_emd": distance_likelihoods[2].mean().item(),

                    "num_diff_mean": num_diff.mean().item(),
                    "num_entry_diff_mean": num_entry_diff.mean().item(),
                    "num_exit_diff_mean": num_exit_diff.mean().item(),
                }

                self.metric_logger.update(metric_dict)

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
                            scenario_rollouts=scenario_rollouts[:32]
                        if len(scenario_rollouts) > self.para_num:
                            for i in range(np.ceil(len(scenario_rollouts) / self.para_num).astype(int)):  # 64
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
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout,new_agent[_i_sc*100:(_i_sc+1)*100],
                        )

            # if self.n_rollout_closed_val ==1 and not self.use_bird:
            #     scenario_metrics=self.wosac_metrics.pool_scenario_metrics[0]
            #
            #     simulated_collision_rate=scenario_metrics.simulated_collision_rate
            #
            #     collision_indication_likelihood=scenario_metrics.collision_indication_likelihood
            #     #simulated_offroad_rate = scenario_metrics.simulated_offroad_rate
            #     #print(collision_indication_likelihood,simulated_collision_rate)
            #
            #     if collision_indication_likelihood<0.99 and simulated_collision_rate>0 : #simulated_collision_rate<0.99 :#or simulated_offroad_rate>0       #242
            #         disc_out = self.encoder.discriminator.predict_agent(None,
            #                                                             pred["token_mask"],
            #                                                             pred["valid_mask"],
            #                                                             pred["sampled_pos"],
            #                                                             pred["sampled_heading"],
            #                                                             tokenized_agent,
            #                                                             map_feature,
            #                                                             abs_time=tokenized_agent["abs_time"])
            #         ego_logits, interact_logits = disc_out[0]
            #
            #         edge_index_a2a=disc_out[1] [0]       #t,a
            #
            #         n_step=18
            #
            #         ego_logits=ego_logits.reshape(18,-1)
            #         n_agent=ego_logits.shape[1]
            #
            #         ego_index = torch.where(tokenized_agent["ego_mask"])[0][0]
            #
            #         scene_realism=ego_logits[:,ego_index]
            #
            #         mask=pred["valid_mask"]
            #
            #         src,dst=edge_index_a2a[0],edge_index_a2a[1]
            #
            #         flat_mask = mask.transpose(0, 1).flatten(0, 1)
            #
            #         kept_nodes = torch.nonzero(flat_mask, as_tuple=True)[0]  # shape [M]
            #
            #         dst_all = kept_nodes[dst]
            #
            #         dst_agent=dst_all % n_agent
            #
            #         ego_mask=dst_agent ==ego_index
            #
            #         src_ego=src[ego_mask]
            #
            #         src_all=kept_nodes[src_ego]
            #
            #         interact_realism= torch.zeros([n_step*n_agent],device=src_all.device)
            #
            #         interact_realism[src_all] = interact_logits[ego_mask]
            #
            #         interact_realism=interact_realism.reshape(n_step,n_agent)
            #
            #         #print(torch.all(interact_realism[:,ego_index]==0))
            #
            #         interact_realism[:,ego_index]=scene_realism
            #
            #         interact_realism=torch.sigmoid(interact_realism)
            #         save_path=self.video_dir  / f"step_{self.global_step}_batch_{batch_idx:02d}.pdf"
            #
            #        # if interact_realism.min()>0.45:
            #         plot_rollout_frames( tokenized_agent,
            #                                 data["tfrecord_path"][0],
            #                                 interact_realism.transpose(0,1).cpu(),
            #                                 pred,
            #                                  save_path=save_path
            #                             )


    def on_validation_epoch_end(self):

        if self.val_open_loop:

            torch.save(self.all_data,"all_data.pt")


        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()

                if self.challenge_type!=ChallengeType.SCENARIO_GEN:
                   epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()#ADE is all the sum distance for all agent

                else:
                    self.result=compute_agent_metrics(samples=self.samples, gt_samples=self.gt_samples,vis=False)

                    for key, value in self.result.items():
                        self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True,
                                 rank_zero_only=True)

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

                    for key, value in epoch_wosac_metrics.items():#minADE is the time average distance for evaluated agent
                        self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

           # print("Callback metrics:", self.trainer.callback_metrics.keys())

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

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.agent_encoder.inference(
                tokenized_agent, map_feature,  # post_sampling=True
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            pred_sizes=[],
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

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()