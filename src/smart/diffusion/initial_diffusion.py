"""Initial-state flow wrapper for SMART."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_sum
from torch_ema import ExponentialMovingAverage

from src.smart.utils import transform_to_local

from .diffusion_utils import multi_circle_collision_loss_mem_efficient
from .scale_flow import Flow


class InitDiffusion(nn.Module):
    """Prepare ego/map context and delegate learning to ``ScaleFlow``."""

    NUM_TYPES = 3

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_freq_bands: int,
        token_processor,
        gail: bool,
        model_args: Optional[Any] = None,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        edge_embedding_type: str = "fourier",
        sigma_h: Optional[float] = None,
        heading_noise: str = "gaussian",
        heading_objective: str = "x0",
        heading_flow_loss_weight: float = 1.0,
        velocity_representation: str = "vector",
        time_embedding_type: str = "legacy",
        time_embedding_scale: float = 99.0,
        count_embedding_type: str = "none",
        count_lane_source: str = "map_tokens",
        count_max_num_agents: int = 128,
        count_max_num_lanes: int = 1024,
        map_embedding_type: str = "none",
        map_id_source: str = "fixed",
        map_id: int = 0,
        map_lg_type: Optional[int] = 0,
        map_label_dropout: float = 0.1,
        size_representation: str = "linear",
        speed_loss_weight: float = 0.0,
        speed_loss_scale: Optional[float] = None,
        invalid_size_policy: str = "mask",
        pos_source: str = "gaussian",
        shape_source: str = "gaussian",
        velocity_source: str = "gaussian",
        fix_ego: bool = True,
        generate_type: bool = False,
        type_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if token_processor is None:
            raise ValueError("token_processor is required.")
        if edge_embedding_type not in ("fourier", "mlp"):
            raise ValueError("edge_embedding_type must be 'fourier' or 'mlp'.")

        self.token_processor = token_processor
        self.edge_embedding_type = edge_embedding_type
        if velocity_representation not in ("vector", "speed"):
            raise ValueError("velocity_representation must be vector or speed")
        self.velocity_representation = velocity_representation
        if size_representation not in ("linear", "log"):
            raise ValueError("size_representation must be linear or log")
        self.size_representation = size_representation
        self.invalid_size_policy = invalid_size_policy

        # Compatibility flags used by SMART/SMART_GAIL.
        self.learn_autoencoder = False
        self.latent_diffusion = False
        self.ldm = False
        self.use_gan = False

        args = self._make_args( )
        args.velocity_representation = velocity_representation
        args.size_representation = size_representation
        if not isinstance(fix_ego, bool):
            raise ValueError("fix_ego must be boolean")
        self.fix_ego = args.fix_ego = fix_ego
        if not isinstance(generate_type, bool):
            raise ValueError("generate_type must be boolean")
        self.generate_type = args.generate_type = generate_type
        self.type_loss_weight = args.type_loss_weight = float(type_loss_weight)
        self._type_head_missing_on_load = False
        args.invalid_size_policy = invalid_size_policy
        self.pos_source = args.pos_source = pos_source
        self.shape_source = args.shape_source = shape_source
        self.velocity_source = args.velocity_source = velocity_source
        self.speed_loss_weight = float(speed_loss_weight)
        self.speed_loss_scale = None if speed_loss_scale is None else float(speed_loss_scale)
        args.speed_loss_weight = self.speed_loss_weight
        args.speed_loss_scale = self.speed_loss_scale
        if velocity_representation == "speed":
            args.input_dim = 7
        args.edge_embedding_type = edge_embedding_type
        self.time_embedding_type = time_embedding_type
        self.time_embedding_scale = float(time_embedding_scale)
        args.time_embedding_type = self.time_embedding_type
        args.time_embedding_scale = self.time_embedding_scale
        self.count_embedding_type = count_embedding_type
        self.count_lane_source = count_lane_source
        self.count_max_num_agents = count_max_num_agents
        self.count_max_num_lanes = count_max_num_lanes
        args.count_embedding_type = count_embedding_type
        args.count_lane_source = count_lane_source
        args.count_max_num_agents = count_max_num_agents
        args.count_max_num_lanes = count_max_num_lanes
        self.map_embedding_type = map_embedding_type
        self.map_id_source = map_id_source
        self.map_id = map_id
        self.map_lg_type = map_lg_type
        self.map_label_dropout = float(map_label_dropout)
        args.map_embedding_type = self.map_embedding_type
        args.map_id_source = self.map_id_source
        args.map_id = self.map_id
        args.map_lg_type = self.map_lg_type
        args.map_label_dropout = self.map_label_dropout
        self.sigma_h = None if sigma_h is None else float(sigma_h)
        args.sigma_h = self.sigma_h
        self.heading_noise = heading_noise
        args.heading_noise = heading_noise
        self.heading_objective = heading_objective
        self.heading_flow_loss_weight = float(heading_flow_loss_weight)
        args.heading_objective = heading_objective
        args.heading_flow_loss_weight = self.heading_flow_loss_weight
        self.G1 = Flow(args, token_processor, gail)
        if map_embedding_type == "scenario_dreamer" and (map_id_source == "metadata" or map_lg_type is None):
            # Forward labels even in the generic Init path where the separate
            # scenario_dreamer_init tokenization flag is disabled.
            token_processor.init_map_id_conditioning = True

        self.use_rl = bool(args.use_rl)
        self.sampling_steps = int(args.sampling_steps)
        self.branch_steps = getattr(args, "branch_steps", None)

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
        self.ema = None
        self._pending_ema_state = None
        if self.use_ema:
            self.reset_ema()
        # Children load after this module's extra state. Bind EMA to the loaded
        # generator only after all its parameters have been restored.
        self.register_load_state_dict_post_hook(self._restore_ema_after_load)

    def reset_ema(self):
        """Start averaging from the current generator, without training history."""
        if self.use_ema or self.ema is not None:
            self.ema = ExponentialMovingAverage(self.G1.parameters(), decay=self.ema_decay)

    def _move_ema(self):
        parameter = next(self.G1.parameters())
        self.ema.to(device=parameter.device, dtype=parameter.dtype)

    @torch.no_grad()
    def update_ema(self):
        """Called after an optimizer step, including gradient accumulation."""
        if self.ema is None and self.use_ema:
            self.reset_ema()
        if self.ema is not None:
            self._move_ema()
            self.ema.update(self.G1.parameters())

    def get_extra_state(self):
        return {
            "size_representation": self.size_representation,
            "generate_type": self.generate_type,
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "ema_parameter_names": list(dict(self.G1.named_parameters())) if self.ema is not None else None,
        }

    def set_extra_state(self, state):
        if not isinstance(state, Mapping):
            raise ValueError("InitDiffusion extra state must be a mapping")
        saved_size_mode = state.get("size_representation", "linear")
        if saved_size_mode != self.size_representation:
            raise ValueError(f"InitDiffusion checkpoint size_representation={saved_size_mode!r} "
                             f"does not match configured {self.size_representation!r}; "
                             "log size models require training with log targets")
        # A mode change adds/removes the head. Warm-start physical weights,
        # but start EMA afresh for the new parameter layout.
        self._pending_ema_state = state if state.get("generate_type", False) == self.generate_type else None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        extra_key = prefix + "_extra_state"
        self._pending_ema_state = None
        if self.generate_type:
            required = [prefix + "G1.model.to_out_type." + name
                        for name in self.G1.model.to_out_type.state_dict()]
            self._type_head_missing_on_load = any(key not in state_dict for key in required)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        # Legacy checkpoints have no EMA extra state. Retain strict checking of
        # every generator weight; exempt only the newly introduced extra key.
        if extra_key not in state_dict and extra_key in missing_keys:
            missing_keys.remove(extra_key)
            if self.size_representation != "linear" and any(
                    key.startswith(prefix + "G1.") for key in state_dict):
                error_msgs.append("Legacy InitDiffusion checkpoints use linear sizes; "
                                  "cannot load them as a log size model")
            # A backbone-only checkpoint can initialize a new log model;
            # its missing generator weights remain subject to normal checks.

    def _restore_ema_after_load(self, module, incompatible_keys):
        state = self._pending_ema_state
        self._pending_ema_state = None
        saved = state.get("ema") if state is not None else None
        if saved is None:
            self.reset_ema()
            return
        parameters = dict(self.G1.named_parameters())
        names = state.get("ema_parameter_names")
        if names is not None and list(parameters) != names:
            raise ValueError("InitDiffusion EMA parameter names do not match the generator")
        shadows = saved.get("shadow_params", [])
        if len(shadows) != len(parameters) or any(
                not torch.is_tensor(shadow) or shadow.shape != parameter.shape
                for shadow, parameter in zip(shadows, parameters.values())):
            raise ValueError("InitDiffusion EMA parameter shapes do not match the generator")
        self.ema_decay = float(saved["decay"])
        self.ema = ExponentialMovingAverage(parameters.values(), decay=self.ema_decay)
        self.ema.load_state_dict(saved)

    @staticmethod
    def _make_args(
    ) -> SimpleNamespace:
        """Create deterministic ScaleFlow settings without parsing process CLI."""
        values = {
            "input_dim": 8,
            "hidden_dim": 256,
            "num_heads": 8,
            "dropout": 0,
            "num_denoiser_layers": 3,
            "num_branch_steps": 1,
            "branch_steps": [0,1,2,3,4,5,6,7,8],#5,6,71,3,5,7,4,4,1,2,3,4,5,10,11,12,13,
            "sampling_steps": 20,
            "use_rl": False,
        }
        return SimpleNamespace(**values)

    # ------------------------------------------------------------------
    # Ego context
    # ------------------------------------------------------------------
    @staticmethod
    def _require(data, key: str):
        if key not in data:
            raise KeyError(f"tokenized_agent is missing {key!r}.")
        return data[key]

    def _scene_ego_pose(
        self,
        agent,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        batch = self._require(agent, "batch")
        ego_mask = self._require(agent, "ego_mask")
        pos = self._require(agent, "initial_pos")
        heading = self._require(agent, "initial_heading")
        num_graphs = int(self._require(agent, "num_graphs"))

        scene_pos = pos[ego_mask]
        scene_heading = heading[ego_mask]

        # Store scene-level values under the batch_ego_* names.
        agent["batch_ego_pos"] = scene_pos[batch]
        agent["batch_ego_heading"] = scene_heading[batch]
        return scene_pos, scene_heading, batch, num_graphs

    def _prepare_ego_context(
        self,
        agent,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        scene_pos, scene_heading, batch, num_graphs = (
            self._scene_ego_pose(agent)
        )
        if "ego_feat" not in agent and "ego_pos2" in agent:
            local_pos, local_heading = transform_to_local(
                self._require(agent, "ego_pos2"),
                self._require(agent, "ego_heading2"),
                scene_pos,
                scene_heading,
            )
            local_trajectory = torch.cat(
                [local_pos, local_heading[..., None]],
                dim=-1,
            ).flatten(1)

            if self.generate_type:
                counts = torch.bincount(batch, minlength=num_graphs)
                type_count = torch.stack((counts, torch.zeros_like(counts), torch.zeros_like(counts)), -1)
            else:
                agent_type = self._require(agent, "type").long()
                type_id = batch * self.NUM_TYPES + agent_type
                type_count = torch.bincount(type_id, minlength=num_graphs * self.NUM_TYPES).reshape(num_graphs, self.NUM_TYPES)
            type_count = type_count.to(local_trajectory)

            feature = torch.cat([local_trajectory, type_count], dim=-1)
            agent["ego_feat"] = feature

        if self.generate_type and "ego_feat" in agent:
            # Also sanitize cached contexts prepared before this option was enabled.
            counts = torch.bincount(batch, minlength=num_graphs).to(agent["ego_feat"])
            counts = torch.stack((counts, torch.zeros_like(counts), torch.zeros_like(counts)), -1)
            agent["ego_feat"] = torch.cat((agent["ego_feat"][:, :-3], counts), -1)
        return scene_pos, scene_heading, batch, num_graphs

    # ------------------------------------------------------------------
    # Initial map context
    # ------------------------------------------------------------------
    def _initial_map_feature(
        self,
        agent,
        scene_pos: Tensor,
        scene_heading: Tensor,
        num_graphs: int,
    ):
        # Retain unprojected features when EMA is tracked so repeated sampling
        # (or switching EMA off) cannot reuse another weight version's embedding.
        raw_feature = agent.get("_initial_map_raw_feature")
        if raw_feature is not None:
            result = dict(raw_feature, pt_token=self.G1.model.lane_embed(raw_feature["pt_token"]))
            agent["initial_map_feature"] = result
            return result
        cached = agent.get("initial_map_feature")
        if cached is not None:
            feature = cached["pt_token"]
            is_raw = agent.pop("_initial_map_feature_is_raw", False)
            if is_raw or feature.shape[-1] != self.G1.model.hidden_dim:
                if self.ema is not None or self.use_ema:
                    agent["_initial_map_raw_feature"] = dict(cached)
                result = dict(cached, pt_token=self.G1.model.lane_embed(feature))
                agent["initial_map_feature"] = result
                return result
            return cached

        map_feature = self._require(agent, "map_feature")
        batch = map_feature["batch"]
        position = map_feature["position"]
        orientation = map_feature["orientation"]
        feature = map_feature["pt_token"]

        if batch.numel():
            distance = torch.linalg.vector_norm(
                position[..., :2] - scene_pos[batch],
                dim=-1,
            )
            keep = distance < float(self.token_processor.init_map_range)
            batch = batch[keep]
            position = position[keep]
            orientation = orientation[keep]
            feature = feature[keep]

            # Always transform non-empty map data. The old code skipped every
            # scene when the final scene happened to contain no map points.
            position, orientation = transform_to_local(
                position,
                orientation,
                scene_pos[batch],
                scene_heading[batch],
            )

        raw_feature = {
            "pt_token": feature,
            "position": position,
            "orientation": orientation,
            "batch": batch,
        }
        if self.ema is not None or self.use_ema:
            agent["_initial_map_raw_feature"] = raw_feature
        result = dict(raw_feature, pt_token=self.G1.model.lane_embed(feature))
        agent["initial_map_feature"] = result
        return result


    def _collision_advantage(
        self,
        agent,
        map_feature,
        batch: Tensor,
    ) -> None:
        with torch.no_grad():
            sample = self.G1.sample(
                agent,
                map_feature,
                self.sampling_steps,
                self.branch_steps
            )
            collision, dst, src = (
                multi_circle_collision_loss_mem_efficient(
                    sample,
                    None,
                    batch,
                    None,
                )
            )
            penalty = scatter_sum(
                collision,
                dst,
                dim=0,
                dim_size=len(sample),
            )
            penalty += scatter_sum(
                collision,
                src,
                dim=0,
                dim_size=len(sample),
            )
            advantage = (penalty <= 0).to(sample.dtype)

        agent["noncol_rate"] = advantage
        agent["advantages"] = advantage

    @staticmethod
    def _mean(value: Tensor, name: str) -> Tensor:
        if not torch.is_tensor(value) or value.numel() == 0:
            raise ValueError(f"{name} must be a non-empty tensor.")
        value = value.mean()
        if not torch.isfinite(value):
            raise FloatingPointError(f"{name} is not finite.")
        return value

    def _train(
        self,
        agent,
        map_feature,
        batch: Tensor,
    ):
        diff_input, _ = self.G1.model.get_input(agent)

        if self.use_rl:
            self._collision_advantage(agent, map_feature, batch)

        loss = self.G1.get_loss(
            diff_input,
            agent,
            map_feature,
        )

        names = (
            "match_loss",
            "collision_loss",
            "pos_loss",
            "heading_loss",
            "shape_loss",
            "velocity_loss",
        )
        result = tuple(self._mean(value, name) for value, name in zip(loss, names))
        self._type_head_missing_on_load = False
        return result

    def _infer(
        self,
        agent,
        map_feature,
    ):
        if self.generate_type and self._type_head_missing_on_load:
            raise ValueError("Loaded checkpoint has no trained InitDiffusion type head; "
                             "train/finetune with generate_type=true before evaluation, "
                             "or set generate_type=false for a legacy checkpoint")
        sample = self.G1.sample(
            agent,
            map_feature,
            self.sampling_steps,
            self.branch_steps
        )
        pos, heading, shape, velocity, token_index = (
            self.G1.model.get_output(sample, agent)
        )
        return pos, heading, token_index, shape, velocity

    def forward(self, tokenized_agent):
        use_average = not self.training and self.use_ema
        if use_average:
            if self.ema is None:
                self.reset_ema()
            self._move_ema()
        context = self.ema.average_parameters(self.G1.parameters()) if use_average else nullcontext()
        with torch.set_grad_enabled(torch.is_grad_enabled() and not use_average), context:
            scene_pos, scene_heading, batch, num_graphs = self._prepare_ego_context(tokenized_agent)
            map_feature = self._initial_map_feature(tokenized_agent, scene_pos, scene_heading, num_graphs)
            if self.training:
                return self._train(tokenized_agent, map_feature, batch)
            return self._infer(tokenized_agent, map_feature)
