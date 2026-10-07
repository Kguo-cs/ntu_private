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
    ) -> None:
        super().__init__()
        if token_processor is None:
            raise ValueError("token_processor is required.")
        if edge_embedding_type not in ("fourier", "mlp"):
            raise ValueError("edge_embedding_type must be 'fourier' or 'mlp'.")

        self.token_processor = token_processor
        self.edge_embedding_type = edge_embedding_type

        # Compatibility flags used by SMART/SMART_GAIL.
        self.learn_autoencoder = False
        self.latent_diffusion = False
        self.ldm = False
        self.use_gan = False

        args = self._make_args( )
        args.edge_embedding_type = edge_embedding_type
        self.sigma_h = None if sigma_h is None else float(sigma_h)
        args.sigma_h = self.sigma_h
        self.heading_noise = heading_noise
        args.heading_noise = heading_noise
        self.heading_objective = heading_objective
        self.heading_flow_loss_weight = float(heading_flow_loss_weight)
        args.heading_objective = heading_objective
        args.heading_flow_loss_weight = self.heading_flow_loss_weight
        self.G1 = Flow(args, token_processor, gail)

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
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "ema_parameter_names": list(dict(self.G1.named_parameters())) if self.ema is not None else None,
        }

    def set_extra_state(self, state):
        if not isinstance(state, Mapping):
            raise ValueError("InitDiffusion extra state must be a mapping")
        self._pending_ema_state = state

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        extra_key = prefix + "_extra_state"
        self._pending_ema_state = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        # Legacy checkpoints have no EMA extra state. Retain strict checking of
        # every generator weight; exempt only the newly introduced extra key.
        if extra_key not in state_dict and extra_key in missing_keys:
            missing_keys.remove(extra_key)

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

            agent_type = self._require(agent, "type").long()

            type_id = batch * self.NUM_TYPES + agent_type
            type_count = torch.bincount(
                type_id,
                minlength=num_graphs * self.NUM_TYPES,
            ).reshape(num_graphs, self.NUM_TYPES)
            type_count = type_count.to(local_trajectory)

            feature = torch.cat([local_trajectory, type_count], dim=-1)
            agent["ego_feat"] = feature

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
        return tuple(
            self._mean(value, name)
            for value, name in zip(loss, names)
        )

    def _infer(
        self,
        agent,
        map_feature,
    ):
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
