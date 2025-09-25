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
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

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
from src.smart.plot.plot_rollout import plot_rollout_frames,plot_rollout_frames1,plot_rollout_frames_pair

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

        if self.use_smart:
            from src.smart.tokens.smart_token_processsor import TokenProcessor
        else:
            from src.smart.tokens.token_processor import TokenProcessor

        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder,token_processor=self.token_processor, n_token_agent=self.token_processor.n_token_agent
        )

        if self.use_smart:
            set_model_for_finetuning(self.encoder, model_config.finetune)
        # else:
        #     for p in self.encoder.map_encoder.parameters():
        #             p.requires_grad = False

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling
        # self.log("val_closed/wosac_likelihood/metametric", float("-inf"), prog_bar=False, on_epoch=True,
        #          rank_zero_only=True)

        self.all_time=0
        self.all_count=0

        # self.wosac_submission.save_sub_file()
        #
        # print(1/0)

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

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            )

            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.global_rank == 0 and self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            #tokenized_map,tokenized_agent = self.encoder.preprocess(tokenized_map, tokenized_agent)
            map_feature = self.encoder.map_encoder(tokenized_map)

            if self.encoder.use_vae:
                # logits = self.encoder.prior_net.predict_agent(tokenized_agent["sampled_idx"][:, :2],
                #                                       tokenized_agent["goal_idx"],
                #                                       tokenized_agent["valid_mask"][:, :2],
                #                                       tokenized_agent["sampled_pos"][:, :2],
                #                                       tokenized_agent["sampled_heading"][:, :2],
                #                                       tokenized_agent,
                #                                       map_feature,
                #                                       tokenized_agent["light_idx"],
                #                                       None)[0]  # [all_valid]
                # logits_p = logits[:, -1:]
                #
                # mu=logits_p[:,:,:self.encoder.agent_encoder.k_dim]
                # logvar=logits_p[:,:,self.encoder.agent_encoder.k_dim:]
                # std = torch.exp(0.5 * logvar)
                prior_dist = self.encoder.post_net.predict_agent(tokenized_agent["sampled_idx"][:, :2],
                                                         tokenized_agent["goal_idx"],
                                                         tokenized_agent["valid_mask"][:, :2],
                                                         tokenized_agent["sampled_pos"][:, :2],
                                                         tokenized_agent["sampled_heading"][:, :2],
                                                         tokenized_agent,
                                                         map_feature,
                                                         tokenized_agent["light_idx"],
                                                         None)[0]  # [all_valid]

                mu = prior_dist[:, :, :self.encoder.agent_encoder.k_dim]
                logvar = prior_dist[:, :, self.encoder.agent_encoder.k_dim:]  # self.log_std#torch.zeros_like(mu)#logits_p[:,:,self.k_dim:]
                std = torch.exp(0.5 * logvar)

                # mu=torch.zeros([len(tokenized_agent["sampled_idx"]),1,self.encoder.agent_encoder.k_dim],device=self.device)
                # std=torch.ones_like(mu)

            # probs = logits_p.softmax(-1)

            #t1=time.time()

            for _ in range(self.n_rollout_closed_val):

                if self.encoder.use_vae:
                    latent_z = mu + torch.randn_like(std) * std

                    tokenized_agent["latent_z"] = latent_z

                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
                #latent_z = torch.multinomial(probs, 1) # [B]
                #tokenized_agent["latent_z"] = latent_z

                # pred = self.encoder.agent_encoder.inference(
                #     tokenized_agent, map_feature,self.validation_rollout_sampling
                # )
                # first set (top row)
                # pred = torch.load("./waymo_data/pred.pt")
                # scenario_path_A = data["tfrecord_path"][0]
                # sampled_pos = pred["sampled_pos"]  # torch.round(tokenized_agent["sampled_pos"]*10)/10##
                # sampled_heading = pred[
                #     "sampled_heading"]  # torch.round(wrap_angle(tokenized_agent["sampled_heading"])/np.pi*30)*np.pi/30#
                #
                # disc_out = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                #                                                     None,
                #                                                     tokenized_agent["valid_mask"],  # expert_
                #                                                     sampled_pos,
                #                                                     sampled_heading,
                #                                                     tokenized_agent,
                #                                                     map_feature,
                #                                                     [],
                #                                                     None,
                #                                                     # latent_z=tokenized_agent["latent_z"]
                #                                                     )  # [0]#Metrics-Guided Adversarial Training
                # ego_rewards_A, nei_sum_rewards_A = disc_out[2]  # .detach()
                #
                # disc_val_A = (ego_rewards_A + nei_sum_rewards_A).detach().cpu().numpy()  # shape [N, K]
                #
                # # second set (bottom row)
                # tokenized_agent_B, pred_B, data_B, map_feature_B = torch.load("./waymo_data/pred2.pt")
                # scenario_path_B = data_B["tfrecord_path"][0]
                #
                # sampled_pos = pred_B["sampled_pos"]  # torch.round(tokenized_agent["sampled_pos"]*10)/10##
                # sampled_heading = pred_B[
                #     "sampled_heading"]  # torch.round(wrap_angle(tokenized_agent["sampled_heading"])/np.pi*30)*np.pi/30#
                #
                # disc_out = self.encoder.discriminator.predict_agent(tokenized_agent_B["sampled_idx"],
                #                                                     None,
                #                                                     tokenized_agent_B["valid_mask"],  # expert_
                #                                                     sampled_pos,
                #                                                     sampled_heading,
                #                                                     tokenized_agent_B,
                #                                                     map_feature_B,
                #                                                     [],
                #                                                     None,
                #                                                     # latent_z=tokenized_agent["latent_z"]
                #                                                     )  # [0]#Metrics-Guided Adversarial Training
                # ego_rewards_B, nei_sum_rewards_B = disc_out[2]  # .detach()
                #
                # disc_val_B = (ego_rewards_B + nei_sum_rewards_B).detach().cpu().numpy()
                #
                # plot_rollout_frames_pair(
                #     tokenized_agent, scenario_path_A, disc_val_A, pred,
                #     tokenized_agent_B, scenario_path_B, disc_val_B, pred_B,
                #     frames=(30, 50, 70, 90),
                #     radius_m=45.0,
                #     vmin=0.0, vmax=2.0,  # shared color scale
                #     cmap_name="RdYlGn"
                # )

                # pred=torch.load("./smart/plot/pred.pt")
                #
                # scenario_path=data["tfrecord_path"][0]
                #
                # sampled_pos = pred["sampled_pos"]  # torch.round(tokenized_agent["sampled_pos"]*10)/10##
                # sampled_heading = pred[
                #     "sampled_heading"]  # torch.round(wrap_angle(tokenized_agent["sampled_heading"])/np.pi*30)*np.pi/30#
                #
                # disc_out = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                #                                                     None,
                #                                                     tokenized_agent["valid_mask"],  # expert_
                #                                                     sampled_pos,
                #                                                     sampled_heading,
                #                                                     tokenized_agent,
                #                                                     map_feature,
                #                                                     [],
                #                                                     None,
                #                                                     # latent_z=tokenized_agent["latent_z"]
                #                                                     )  # [0]#Metrics-Guided Adversarial Training
                #
                #
                # ego_rewards, nei_sum_rewards = disc_out[2]  # .detach()
                #
                # rewards = ego_rewards + nei_sum_rewards
                #
                # print(rewards.max())
                # print(rewards.min())
                #
                # disc_val=rewards.cpu().numpy() #[:,:,0]
                #
                # plot_rollout_frames(tokenized_agent,scenario_path,disc_val,pred)
                #
                # tokenized_agent,pred,data,map_feature=torch.load("./smart/plot/pred2.pt")
                #
                # #torch.save((tokenized_agent,pred,data,map_feature),'pred1.pt')#26
                #
                # scenario_path=data["tfrecord_path"][0]
                #
                # sampled_pos = pred["sampled_pos"]  # torch.round(tokenized_agent["sampled_pos"]*10)/10##
                # sampled_heading = pred[
                #     "sampled_heading"]  # torch.round(wrap_angle(tokenized_agent["sampled_heading"])/np.pi*30)*np.pi/30#
                #
                # disc_out = self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
                #                                                     None,
                #                                                     tokenized_agent["valid_mask"],  # expert_
                #                                                     sampled_pos,
                #                                                     sampled_heading,
                #                                                     tokenized_agent,
                #                                                     map_feature,
                #                                                     [],
                #                                                     None,
                #                                                     # latent_z=tokenized_agent["latent_z"]
                #                                                     )  # [0]#Metrics-Guided Adversarial Training
                #
                #
                # agent_num = len(sampled_pos) * 16
                # ego_rewards, nei_sum_rewards = disc_out[2]  # .detach()
                #
                # rewards = ego_rewards + nei_sum_rewards
                #
                # print(rewards.max())
                # print(rewards.min())
                #
                # disc_val=rewards.cpu().numpy() #[:,:,0]
                #
                # plot_rollout_frames1(tokenized_agent,scenario_path,disc_val,pred)

                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]
            #print(data.scenario_id)
            # pred_traj=torch.load("/home/ke/code/catk/src/waymo_data/pred_traj.pt").cuda()
            # pred_z=torch.load("/home/ke/code/catk/src/waymo_data/pred_z.pt").cuda()
            # pred_head=torch.load("/home/ke/code/catk/src/waymo_data/pred_head.pt").cuda()
            #self.all_time+=time.time()-t1
          #  self.all_count+=    self.n_rollout_closed_val*16

            #print(time.time()-t1)
            #self.wosac_metrics = WOSACMetrics("val_closed")

            # torch.save(pred_traj.cpu(),"pred_traj.pt")
            # torch.save(pred_z.cpu(),"pred_z.pt")
            # torch.save(pred_head.cpu(),"pred_head.pt")

            #print(self.all_time/self.all_count)

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
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][
                        :, self.num_historical_steps :
                    ],
                )

                # WOSAC metrics
                if batch_idx < self.n_batch_wosac_metric:
                    device = pred_traj.device
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], device
                        ),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )
                    if len(scenario_rollouts) > 32:
                        self.wosac_metrics.update(data["tfrecord_path"][:20], scenario_rollouts[:20])
                        self.wosac_metrics.update(data["tfrecord_path"][20:], scenario_rollouts[20:])
                    else:
                        self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

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
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        # for _path in _vis.video_paths:
                        #     self.logger.log_video(
                        #         "/".join(_path.split("/")[-3:]), [_path]
                        #     )
                    #print(time.time()-t1)

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    # epoch_wosac_metrics["epoch"] = (
                    #     self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    # )
                    # self.logger.log_metrics(epoch_wosac_metrics)
                    #print("Logged keys:", epoch_wosac_metrics.keys())

                    for key, value in epoch_wosac_metrics.items():
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
            # pred = self.encoder.inference(
            #     tokenized_map, tokenized_agent, self.validation_rollout_sampling
            # )
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