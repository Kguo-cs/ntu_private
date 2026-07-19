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

from __future__ import annotations

import math
import numbers
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
import torch
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from waymo_open_dataset.utils.sim_agents.submission_specs import ChallengeType

from src.smart.metrics import CrossEntropy, TokenCls, WOSACSubmission, minADE
from src.smart.metrics.bird_metrics import MetricDict
from src.smart.metrics.gen_metrics import compute_agent_metrics, compute_gen_samples
from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_global, wrap_angle
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


def _cfg(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _freeze(module: torch.nn.Module | None) -> None:
    if module is not None:
        module.requires_grad_(False)
        module.eval()


def _cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(x) for x in value)
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    return value


def _shape_over_time(shape: Tensor, steps: int) -> Tensor:
    if shape.ndim == 2:
        return shape[:, None].expand(-1, steps, -1)
    if shape.ndim == 3 and shape.shape[1] in (1, steps):
        return shape.expand(-1, steps, -1)
    raise ValueError(f"Cannot align shape {tuple(shape.shape)} with {steps} steps.")


def _rotate(v: Tensor, angle: Tensor) -> Tensor:
    view = [len(angle)] + [1] * (v.ndim - 2)
    c, s = angle.cos().reshape(view), angle.sin().reshape(view)
    x, y = v[..., 0], v[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], -1)


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"Metric {name!r} is not scalar: {tuple(value.shape)}.")
        value = value.detach().float().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Metric {name!r} is not scalar: {value.shape}.")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    elif not isinstance(value, numbers.Number):
        raise TypeError(f"Unsupported metric {name!r}: {type(value).__name__}.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric {name!r} is not finite: {result}.")
    return result


class SMART(LightningModule):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(model_config.lr)
        self.lr_warmup_steps = int(model_config.lr_warmup_steps)
        self.lr_total_steps = int(model_config.lr_total_steps)
        self.lr_min_ratio = float(model_config.lr_min_ratio)
        self._check_lr_config()

        self.num_historical_steps = int(
            model_config.decoder.num_historical_steps
        )
        self.val_open_loop = bool(model_config.val_open_loop)
        self.val_closed_loop = bool(model_config.val_closed_loop)

        self.token_processor = TokenProcessor(**model_config.token_processor)
        self.encoder = SMARTDecoder(
            **model_config.decoder,
            token_processor=self.token_processor,
            n_token_agent=self.token_processor.n_token_agent,
            finetune=model_config.finetune,
        )

        self.training_rollout_len = int(_cfg(model_config, "training_rollout_len", 14))
        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling
        self._configure_finetuning(bool(model_config.finetune))

        self.n_vis_batch = int(model_config.n_vis_batch)
        self.n_vis_scenario = int(model_config.n_vis_scenario)
        self.n_vis_rollout = int(model_config.n_vis_rollout)
        self.n_batch_wosac_metric = int(model_config.n_batch_wosac_metric)

        scenario_gen = bool(self.token_processor.pred_init)
        self.challenge_type = (
            ChallengeType.SCENARIO_GEN if scenario_gen else ChallengeType.SIM_AGENTS
        )
        self.n_rollout_closed_val = int(
            _cfg(model_config, "n_rollout_closed_val", 2 if scenario_gen else 8)
        )
        self.metric_chunk_size = int(
            _cfg(model_config, "metric_chunk_size", 64 if scenario_gen else 32)
        )
        self.max_metric_scenarios = int(
            _cfg(model_config, "max_metric_scenarios", 64 if scenario_gen else 0)
        )
        self.compute_mmd = bool(_cfg(model_config, "compute_mmd", True))
        if self.metric_chunk_size <= 0 or self.n_rollout_closed_val <= 0:
            raise ValueError("Rollout count and metric chunk size must be positive.")

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)  # compatibility
        self.wosac_metrics = WOSACMetrics(
            "val_closed", challenge_type=self.challenge_type
        )
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)
        if self.wosac_submission.is_active:
            self.n_rollout_closed_val = int(_cfg(model_config, "submission_rollouts", 32))

        self.video_dir = self._video_dir()
        self.all_data: list[Any] = []
        self.metric_logger = MetricDict()
        self.samples: list[Any] = []
        self.gt_samples: list[Any] = []
        self.gt_dist = None
        self.result: dict[str, Any] = {}

        # Kept for older bird-metric code and IQ/GAIL subclasses.
        self.minADE0, self.minADE0_num, self.log_epoch = 0.0, 0, -1

    def _check_lr_config(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0 <= self.lr_min_ratio <= 1:
            raise ValueError("lr_min_ratio must be in [0, 1].")
        if self.lr_warmup_steps < 0 or self.lr_total_steps <= self.lr_warmup_steps:
            raise ValueError("Require 0 <= warmup_steps < total_steps.")

    def _configure_finetuning(self, enabled: bool) -> None:
        if not enabled:
            return
        _freeze(self.encoder.map_encoder)
        if not self.token_processor.pred_init:
            return
        if not self.token_processor.learn_init:
            _freeze(self.encoder.init_decoder)
            return
        if not self.encoder.gail or self.training_rollout_len == 1:
            _freeze(self.encoder.agent_encoder)
        generator = getattr(self.encoder.init_decoder, "G1", None)
        if generator is not None and getattr(generator, "use_ref", False):
            _freeze(getattr(generator, "ref_model", None))

    @staticmethod
    def _video_dir() -> Path:
        try:
            root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        except Exception:
            root = Path.cwd()
        path = Path(root) / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _global_zero(self) -> bool:
        trainer = getattr(self, "_trainer", None)
        return trainer is None or bool(trainer.is_global_zero)

    def validation_step(self, data, batch_idx):
        # if batch_idx<161:
        #     return None
        tokenized_map, tokenized_agent = self.token_processor(data)

        if self.val_open_loop and self._global_zero:
            self._save_open_loop_batch(tokenized_map, tokenized_agent)

        # All ranks must enter submission synchronization. Normal validation
        # keeps the original rank-zero-only behavior.
        run_closed = self.val_closed_loop and (
            self.wosac_submission.is_active or self._global_zero
        )
        if run_closed:
            self._validate_closed_loop(data, tokenized_map, tokenized_agent, batch_idx)

    def _save_open_loop_batch(self, tokenized_map, agent) -> None:
        pred = self.encoder(tokenized_map, agent)
        decoder = self.encoder.agent_encoder.interative_decoder
        layers = getattr(decoder, "a2a_attn_layers", [])
        layer = layers[0] if len(layers) else None
        self.all_data.append(_cpu({
            "attention_weight": getattr(layer, "attention_weight", None),
            "edge_weight": getattr(
                layer, "edge_weight", getattr(layer, "egde_weight", None)
            ),
            "edge_index_a2a": pred.get("edge_index_a2a"),
            "sampled_pos": agent.get("sampled_pos"),
            "valid_mask": agent.get("valid_mask"),
            "agent_q": pred.get("agent_q"),
        }))

    def _validate_closed_loop(self, data, tokenized_map, agent, batch_idx: int) -> None:
        out = self._rollouts(tokenized_map, agent)

        if self.challenge_type == ChallengeType.SCENARIO_GEN:
            if not self.wosac_submission.is_active:
                compute_gen_samples(
                    data, agent,
                    out["traj"], out["vel"], out["head"], out["size"],
                    self.samples, self.gt_samples, self.gt_dist,
                    compute_mmd=self.compute_mmd,
                )
        else:
            self.minADE.update(
                pred=out["traj"],
                target=data["agent"]["position"][
                    :, self.num_historical_steps :, : out["traj"].shape[-1]
                ],
                target_valid=data["agent"]["valid_mask"][
                    :, self.num_historical_steps :
                ],
            )

        if self.wosac_submission.is_active:
            scenarios = self._submission_update(data, out, batch_idx)
        else:
            scenarios = self._metric_update(data, agent, out, batch_idx)

        if self._global_zero and batch_idx < self.n_vis_batch:
            self._visualize(data, agent, scenarios, out["z_list"], batch_idx)

    @torch.no_grad()
    def _rollouts(self, tokenized_map, agent) -> dict[str, Any]:
        if getattr(self.encoder, "sep_map", False):
            agent["initial_map_feature"] = self.encoder.init_map_encoder(
                tokenized_map, tokenized_agent=agent
            )
        map_feature = self.encoder.map_encoder(tokenized_map)
        agent["map_feature"] = map_feature

        traj, z, head, size, vel, z_list = [], [], [], [], [], []
        for _ in range(self.n_rollout_closed_val):
            rollout_agent = dict(agent)  # prevent cross-rollout dictionary mutation
            pred = self.encoder.inference(rollout_agent)
            trajectory = pred["pred_traj_10hz"]
            traj.append(trajectory)
            head.append(self._heading(pred, trajectory, rollout_agent))
            z.append(self._height(pred, trajectory, rollout_agent))
            size.append(_shape_over_time(pred["shape"], trajectory.shape[1]))
            vel.append(pred.get(
                "initial_local_vel", self._initial_velocity(trajectory)
            ))

            generated = rollout_agent.get("pred_z_list")
            if self.n_vis_batch > 0 and generated is not None:
                z_list.append(
                    self._pred_z_list_ego_local_to_global(
                        generated, rollout_agent
                    ).detach().cpu()
                )

        out = {
            "traj": torch.stack(traj, 1),
            "z": torch.stack(z, 1),
            "head": wrap_angle(torch.stack(head, 1)),
            "size": torch.stack(size, 1).clamp_min(0.1),
            "vel": torch.stack(vel, 1),
            "z_list": torch.stack(z_list, 1) if z_list else None,
        }

        if self.challenge_type == ChallengeType.SIM_AGENTS:
            steps = min(80, out["traj"].shape[2])
            for key in ("traj", "z", "head", "size"):
                out[key] = out[key][:, :, -steps:]
        return out

    @staticmethod
    def _heading(pred, trajectory: Tensor, agent) -> Tensor:
        if "pred_head_10hz" in pred:
            return pred["pred_head_10hz"]
        if trajectory.shape[1] == 0:
            return trajectory.new_empty(trajectory.shape[:2])

        delta = trajectory[:, 1:] - trajectory[:, :-1]
        inferred = torch.atan2(delta[..., 1], delta[..., 0])
        initial = agent.get("initial_heading", agent.get("sampled_heading"))
        if initial is None:
            raise KeyError("Cannot infer heading without an initial heading.")
        if initial.ndim > 1:
            initial = initial[:, 0]
        return torch.cat([initial[:, None], inferred], 1)

    @staticmethod
    def _height(pred, trajectory: Tensor, agent) -> Tensor:
        if "pred_z_10hz" in pred:
            return pred["pred_z_10hz"]
        if "gt_z_raw" not in agent:
            raise KeyError("Missing both pred_z_10hz and gt_z_raw.")
        return agent["gt_z_raw"][:, None].expand(-1, trajectory.shape[1])

    @staticmethod
    def _initial_velocity(trajectory: Tensor) -> Tensor:
        if trajectory.shape[1] < 2:
            return trajectory.new_zeros((len(trajectory), 2))
        return (trajectory[:, 1] - trajectory[:, 0]) / 0.1

    def _submission_update(self, data, out, batch_idx: int):
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=out["traj"], pred_z=out["z"],
            pred_head=out["head"], pred_sizes=out["size"],
            global_rank=self.global_rank,
        )
        synced = self.wosac_submission.compute()
        scenarios = None
        if self._global_zero:
            synced = {
                k: v[0] if isinstance(v, list) else v
                for k, v in synced.items()
            }
            scenarios = get_scenario_rollouts(**synced)
            self.wosac_submission.i_file = batch_idx
            self.wosac_submission.aggregate_rollouts(scenarios)
        self.wosac_submission.reset()
        return scenarios

    def _metric_update(self, data, agent, out, batch_idx: int):
        if batch_idx >= max(self.n_batch_wosac_metric, self.n_vis_batch):
            return None

        scenarios = get_scenario_rollouts(
            scenario_id=get_scenario_id_int_tensor(
                data["scenario_id"], out["traj"].device
            ),
            agent_id=agent["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=out["traj"], pred_z=out["z"],
            pred_head=out["head"], pred_sizes=out["size"],
        )

        # Metrics and visualization are independent; enabling visualization no
        # longer disables WOSAC metric updates.
        if batch_idx < self.n_batch_wosac_metric:
            paths = list(data["tfrecord_path"])
            count = min(len(paths), len(scenarios))
            if self.max_metric_scenarios > 0:
                count = min(count, self.max_metric_scenarios)
            paths, metric_scenarios = paths[:count], scenarios[:count]
            for start in range(0, count, self.metric_chunk_size):
                end = min(start + self.metric_chunk_size, count)
                self.wosac_metrics.update(
                    paths[start:end], metric_scenarios[start:end]
                )
        return scenarios

    def _visualize(self, data, agent, scenarios, z_list, batch_idx: int) -> None:
        if scenarios is None:
            return
        paths = data["tfrecord_path"]
        count = min(self.n_vis_scenario, len(paths), len(scenarios))

        generated_batch = None
        if z_list is not None:
            for key in ("nonego_batch", "batch"):
                batch = agent.get(key)
                if batch is not None and len(batch) == z_list.shape[0]:
                    generated_batch = batch.detach().cpu()
                    break
            if generated_batch is None:
                raise ValueError("Cannot align generated states with scenarios.")

        for index in range(count):
            states = None if z_list is None else z_list[generated_batch == index]
            VisWaymo(
                scenario_path=paths[index],
                save_dir=self.video_dir / (
                    f"step_{self.global_step}_batch_{batch_idx:02d}"
                    f"-scenario_{index:02d}"
                ),
            ).save_video_scenario_rollout(
                scenarios[index], self.n_vis_rollout,
                pred_z_list=states, crop_size_m=200.0, add_subtitle=True,
            )

    def on_validation_epoch_end(self) -> None:
        if self.val_open_loop:
            if self._global_zero and self.all_data:
                torch.save(self.all_data, self.video_dir.parent / "all_data.pt")
            self.all_data.clear()

        if not self.val_closed_loop:
            return
        if self.wosac_submission.is_active:
            if self._global_zero:
                self.wosac_submission.save_sub_file()
            return
        if not self._global_zero:
            return

        metrics = (
            self.wosac_metrics.compute()
            if self.n_batch_wosac_metric > 0 else {}
        )
        if self.challenge_type == ChallengeType.SCENARIO_GEN:
            if self.samples:
                start = time.time()
                self.result, self.gt_dist = compute_agent_metrics(
                    self.samples, self.gt_samples, self.gt_dist,
                    self.n_vis_batch > 0,
                )
                metrics.update(self.result)
                print(f"metric compute time: {time.time() - start:.2f}s")
            self.samples.clear()
            self.gt_samples.clear()
        else:
            metrics["val_closed/ADE"] = self.minADE.compute()

        if self.token_processor.use_bird:
            self._append_bird_metrics(metrics)

        for key, value in metrics.items():
            self.log(
                str(key), _scalar(value, str(key)),
                on_step=False, on_epoch=True, prog_bar=True,
                sync_dist=False, rank_zero_only=True,
            )
        self.wosac_metrics.reset()
        self.minADE.reset()

    def _append_bird_metrics(self, metrics: dict[str, Any]) -> None:
        if self.minADE0_num > 0:
            metrics["val_closed/minADE"] = self.minADE0 / self.minADE0_num
        metrics.update({
            f"val_closed/{key}": value
            for key, value in self.metric_logger.compute().items()
        })
        self._average(metrics, "val_closed/scene_likelihood", [
            "val_closed/linear_speed_likelihood1",
            "val_closed/angular_speed_likelihood1",
            "val_closed/linear_acceleration_likelihood1",
            "val_closed/angular_acceleration_likelihood1",
            "val_closed/distance_likelihood1",
            "val_closed/polar_likelihood1",
            "val_closed/heading_likelihood1",
        ])
        self._average(metrics, "val_closed/scene_emd", [
            "val_closed/angular_acceleration_emd",
            "val_closed/linear_speed_emd",
            "val_closed/angular_speed_emd",
            "val_closed/linear_acceleration_emd",
            "val_closed/distance_emd",
            "val_closed/polar_emd",
            "val_closed/heading_emd",
        ])
        self.metric_logger.reset()

    @staticmethod
    def _average(metrics: dict, output: str, keys: Sequence[str]) -> None:
        if all(key in metrics for key in keys):
            metrics[output] = sum(metrics[key] for key in keys) / len(keys)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("The model has no trainable parameters.")
        optimizer = torch.optim.Adam(params, lr=self.lr)

        def schedule(step: int) -> float:
            if self.lr_warmup_steps and step < self.lr_warmup_steps:
                progress = step / self.lr_warmup_steps
                return self.lr_min_ratio + (1 - self.lr_min_ratio) * progress
            progress = (step - self.lr_warmup_steps) / (
                self.lr_total_steps - self.lr_warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1 + math.cos(math.pi * progress)
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }

    def test_step(self, data, batch_idx):
        if not self.wosac_submission.is_active:
            raise RuntimeError("test_step requires an active WOSAC submission.")
        tokenized_map, agent = self.token_processor(data)
        out = self._rollouts(tokenized_map, agent)
        self._submission_update(data, out, batch_idx)

    def _pred_z_list_ego_local_to_global(
        self, pred_z_list: Tensor, agent: Mapping[str, Tensor]
    ) -> Tensor:
        """Convert [N,G,D] or [N,R,G,D] ego-frame states to global states."""
        if pred_z_list.ndim not in (3, 4) or pred_z_list.shape[-1] < 4:
            raise ValueError(
                "pred_z_list must be [N,G,D] or [N,R,G,D] with D >= 4."
            )
        if pred_z_list.numel() == 0:
            return pred_z_list.clone()

        z = pred_z_list.clone()
        num_agents = len(z)
        batch = next((
            agent[key].to(z.device, torch.long)
            for key in ("nonego_batch", "batch")
            if key in agent and len(agent[key]) == num_agents
        ), None)
        if batch is None:
            raise KeyError("No batch tensor matches pred_z_list agents.")

        ego_pos = self._ego_value(
            agent, batch, num_agents, "batch_ego_pos", "ego_pos"
        )[..., :2].to(z)
        ego_head = self._ego_value(
            agent, batch, num_agents, "batch_ego_heading", "ego_heading"
        ).to(z).reshape(num_agents)

        shape = z.shape
        flat = z.reshape(num_agents, -1, shape[-1])
        flat[..., :2] = transform_to_global(
            pos_local=flat[..., :2], head_local=None,
            pos_now=ego_pos, head_now=ego_head,
        )[0]

        local_head = torch.atan2(flat[..., 3], flat[..., 2])
        global_head = wrap_angle(local_head + ego_head[:, None])
        flat[..., 2], flat[..., 3] = global_head.cos(), global_head.sin()

        # vx/vy are documented as ego-frame, not agent-frame velocities.
        if flat.shape[-1] >= 8:
            flat[..., 6:8] = _rotate(flat[..., 6:8], ego_head)
        return flat.reshape(shape)

    @staticmethod
    def _ego_value(agent, batch, num_agents, scene_key, fallback_key):
        value = agent.get(scene_key)
        if value is not None:
            if batch.numel() and int(batch.max()) >= len(value):
                raise ValueError(f"{scene_key} has too few scene entries.")
            return value[batch.to(value.device)]

        value = agent.get(fallback_key)
        if value is None:
            raise KeyError(f"Missing {scene_key!r} and {fallback_key!r}.")
        if len(value) == num_agents:
            return value
        if not batch.numel() or int(batch.max()) < len(value):
            return value[batch.to(value.device)]
        raise ValueError(f"Cannot align {fallback_key!r} with generated agents.")

    def on_test_epoch_end(self) -> None:
        if self._global_zero:
            self.wosac_submission.save_sub_file()