"""Rectified Flow with timestep-adaptive SDE noise.

State convention:
    vector: [x, y, heading_cos, heading_sin, length, width, agent_vx, agent_vy]
    speed:  [x, y, heading_cos, heading_sin, length, width, speed]
    size_representation=log replaces length/width with their natural logs;
    internal states stay in log units until physical output/collision decoding.

Flow convention:
    x0 ~ data, x1 ~ noise
    Euclidean fields: x_t = (1 - t) * x0 + t * x1
    Circular heading: theta_t = wrap(theta0 + t * wrap(theta1 - theta0))
    Euclidean fields predict x0. Circular heading optionally predicts angular
    velocity with a separate scalar flow matching objective.

Training uses t in [0, 1]. Generation starts from noise at t=1 and
integrates backward to data at t=0. The SDE/PPO transition uses the same
Rectified-Flow time directly; no complementary ``1 - t`` variable is used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from src.smart.diffusion.diffusion_utils import (
    get_closest_sum_idx_fast,
    get_diff_loss,multi_circle_collision_loss_mem_efficient
)
from src.smart.utils import weight_init, wrap_angle
import copy
from .denoiser import InitDenoiser
from .ego_conditioning import EGO_FIELDS, resolve_ego_conditioning


def _noise_endpoint(model, standard_noise: Tensor, sigma_h: Optional[float],
                    heading_noise: str = "gaussian", *,
                    pos_source: str = "gaussian", shape_source: str = "gaussian",
                    velocity_source: str = "gaussian") -> Tensor:
    """Draw selected fields uniformly in standardized model coordinates.

    U(-sqrt(3), sqrt(3)) has the same mean and variance as N(0, 1).
    Size fields follow the model's linear/log representation; velocity fields
    are vx/vy or scalar speed. Heading has its own independent source option.
    """
    source_fields = (("pos_source", pos_source, slice(0, 2)),
                     ("shape_source", shape_source, slice(4, 6)),
                     ("velocity_source", velocity_source, slice(6, None)))
    for name, source, _ in source_fields:
        if source not in ("gaussian", "uniform"):
            raise ValueError(f"{name} must be gaussian or uniform")
    if any(source == "uniform" for _, source, _ in source_fields):
        standard_noise = standard_noise.clone()
        for _, source, fields in source_fields:
            if source == "uniform":
                standard_noise[:, fields] = (2 * torch.rand_like(standard_noise[:, fields]) - 1) * math.sqrt(3)
    noise = model.denormalize(standard_noise)
    if heading_noise == "circular":
        theta = 2 * torch.pi * torch.rand_like(standard_noise[:, 0]) - torch.pi
        heading = torch.stack((theta.cos(), theta.sin()), dim=-1)
        noise = torch.cat((noise[:, :2], heading, noise[:, 4:]), dim=-1)
    elif sigma_h is not None:
        noise = torch.cat((noise[:, :2], sigma_h * standard_noise[:, 2:4], noise[:, 4:]), dim=-1)
    return noise


def _circular_interpolate(clean: Tensor, noise: Tensor, time: Tensor) -> Tensor:
    """Interpolate other fields linearly and headings along the shortest arc."""
    theta0 = torch.atan2(clean[:, 3], clean[:, 2])
    theta1 = torch.atan2(noise[:, 3], noise[:, 2])
    delta = wrap_angle(theta1 - theta0)
    theta = wrap_angle(theta0 + time[:, 0] * delta)
    state = (1. - time) * clean + time * noise
    return torch.cat((state[:, :2], torch.stack((theta.cos(), theta.sin()), dim=-1),
                      state[:, 4:]), dim=-1)


class Flow(nn.Module):
    """Initial-state flow with simple timestep-adaptive branch noise."""

    def __init__(
        self,
        args,
        token_processor,
        gail: bool = False,
    ) -> None:
        super().__init__()
        self.fix_ego = getattr(args, "fix_ego", True)
        if not isinstance(self.fix_ego, bool):
            raise ValueError("fix_ego must be boolean")
        conditions = resolve_ego_conditioning(self.fix_ego, **{
            f"fix_ego_{field}": getattr(args, f"fix_ego_{field}", None)
            for field in EGO_FIELDS
        })
        for name, value in conditions.items():
            setattr(self, name, value)
        self.use_ego_embedding = getattr(args, "use_ego_embedding", False)
        if not isinstance(self.use_ego_embedding, bool):
            raise ValueError("use_ego_embedding must be boolean")
        self.ego_context_heading_encoding = getattr(args, "ego_context_heading_encoding", "angle")
        self.generate_type = getattr(args, "generate_type", False)
        if not isinstance(self.generate_type, bool):
            raise ValueError("generate_type must be boolean")
        self.type_loss_weight = float(getattr(args, "type_loss_weight", 1.0))
        if not math.isfinite(self.type_loss_weight) or self.type_loss_weight <= 0:
            raise ValueError("type_loss_weight must be finite and positive")
        self.velocity_representation = getattr(args, "velocity_representation", "vector")
        self.size_representation = getattr(args, "size_representation", "linear")
        if self.velocity_representation not in ("vector", "speed"):
            raise ValueError("velocity_representation must be vector or speed")
        self.speed_loss_weight = float(getattr(args, "speed_loss_weight", 0.0))
        if not math.isfinite(self.speed_loss_weight) or self.speed_loss_weight < 0:
            raise ValueError("speed_loss_weight must be finite and nonnegative")
        speed_loss_scale = getattr(args, "speed_loss_scale", None)
        self.speed_loss_scale = None if speed_loss_scale is None else float(speed_loss_scale)
        if self.speed_loss_scale is not None and (not math.isfinite(self.speed_loss_scale)
                                                  or self.speed_loss_scale <= 0):
            raise ValueError("speed_loss_scale must be finite and positive, or null for RMS speed")
        if self.speed_loss_weight > 0 and self.velocity_representation != "vector":
            raise ValueError("speed magnitude loss requires velocity_representation=vector; scalar speed is already directly supervised")
        state_dim = 7 if self.velocity_representation == "speed" else args.input_dim
        for name in ("pos_source", "shape_source", "velocity_source"):
            source = getattr(args, name, "gaussian")
            if source not in ("gaussian", "uniform"):
                raise ValueError(f"{name} must be gaussian or uniform")
            setattr(self, name, source)
        self.heading_noise = getattr(args, "heading_noise", "gaussian")
        if self.heading_noise not in ("gaussian", "circular"):
            raise ValueError("heading_noise must be gaussian or circular")
        self.heading_objective = getattr(args, "heading_objective", "x0")
        if self.heading_objective not in ("x0", "angular_velocity"):
            raise ValueError("heading_objective must be x0 or angular_velocity")
        if self.heading_objective == "angular_velocity" and self.heading_noise != "circular":
            raise ValueError("angular_velocity requires heading_noise=circular")
        self.heading_x0_loss = getattr(args, "heading_x0_loss", "vector_mse")
        if self.heading_x0_loss not in ("vector_mse", "angle_mse"):
            raise ValueError("heading_x0_loss must be vector_mse or angle_mse")
        if self.heading_x0_loss == "angle_mse" and self.heading_objective != "x0":
            raise ValueError("heading_x0_loss=angle_mse requires heading_objective=x0")
        self.heading_flow_loss_weight = float(getattr(args, "heading_flow_loss_weight", 1.0))
        if not math.isfinite(self.heading_flow_loss_weight) or self.heading_flow_loss_weight <= 0:
            raise ValueError("heading_flow_loss_weight must be finite and positive")
        sigma_h = getattr(args, "sigma_h", None)
        self.sigma_h = None if sigma_h is None else float(sigma_h)
        if self.sigma_h is not None and (not math.isfinite(self.sigma_h) or self.sigma_h <= 0):
            raise ValueError("sigma_h must be finite and positive, or null for empirical heading noise")
        # Euclidean x0 prediction, with an optional scalar circular velocity head.
        self.model = InitDenoiser(
            token_processor,
            input_dim=state_dim,
            hidden_dim=args.hidden_dim,
            output_dim=state_dim,
            num_layers=args.num_denoiser_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            edge_embedding_type=getattr(args, "edge_embedding_type", "fourier"),
            heading_velocity=self.heading_objective == "angular_velocity",
            velocity_representation=self.velocity_representation,
            size_representation=self.size_representation,
            fix_ego=self.fix_ego,
            **conditions,
            generate_type=self.generate_type,
            use_ego_embedding=self.use_ego_embedding,
            ego_context_heading_encoding=self.ego_context_heading_encoding,
            invalid_size_policy=getattr(args, "invalid_size_policy", "mask"),
            time_embedding_type=getattr(args, "time_embedding_type", "legacy"),
            time_embedding_scale=getattr(args, "time_embedding_scale", 99.0),
            count_embedding_type=getattr(args, "count_embedding_type", "none"),
            count_lane_source=getattr(args, "count_lane_source", "map_tokens"),
            count_max_num_agents=getattr(args, "count_max_num_agents", 128),
            count_max_num_lanes=getattr(args, "count_max_num_lanes", 1024),
            map_embedding_type=getattr(args, "map_embedding_type", "none"),
            map_id_source=getattr(args, "map_id_source", "fixed"),
            map_id=getattr(args, "map_id", 0),
            map_lg_type=getattr(args, "map_lg_type", 0),
            map_label_dropout=getattr(args, "map_label_dropout", 0.1),
        )

        self.t_eps = 0.05
        self.token_processor=token_processor

        # --------------------------------------------------------------
        # Initial-state policy optimization
        # --------------------------------------------------------------
        self.use_sde = bool( gail and getattr(token_processor, "learn_init", False) )

        # --------------------------------------------------------------
        # Multi-branch sampling
        # --------------------------------------------------------------
        self.num_branch_steps = int(
            getattr(args, "num_branch_steps", 1)
        )
        self.fixed_branch_steps = self._parse_fixed_branch_steps(
            getattr(args, "branch_steps", None)
        )
        self.use_refiner = token_processor.use_refiner
        if self.generate_type and (self.use_sde or self.use_refiner or getattr(args, "use_rl", False)):
            raise ValueError("generate_type supports supervised deterministic Flow; SDE/RL/refiner is unsupported")
        if not all(conditions[f"fix_ego_{field}"] for field in EGO_FIELDS[:4]) and (
            self.use_sde or self.use_refiner or getattr(args, "use_rl", False)
        ):
            raise ValueError("fix_ego=false or partial ego supports supervised Flow training and evaluation; SDE/RL/refiner requires fixed ego")
        if self.speed_loss_weight > 0 and self.use_refiner:
            raise ValueError("speed magnitude loss applies to supervised Flow reconstruction; refiner uses its own policy objective")
        if self.size_representation == "log" and (self.use_sde or self.use_refiner or getattr(args, "use_rl", False)):
            raise ValueError("log size representation supports supervised Flow training and evaluation; SDE/RL/refiner requires linear sizes")
        if self.velocity_representation == "speed" and (self.use_sde or self.use_refiner):
            raise ValueError("speed representation supports deterministic supervised Flow; SDE/refiner requires vector")

        if self.use_refiner:
            self.use_sde=False
            self.use_dit=False

            if self.use_dit:
                from .dit.dit import DiT
                self.refine_model = DiT(args.hidden_dim )
            else:
                self.refine_model = InitDenoiser(
                    token_processor,
                    input_dim=args.input_dim,
                    hidden_dim=args.hidden_dim,
                    output_dim=args.input_dim*2,#,
                    num_layers=1,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    x_pred=False,
                    ego_context_heading_encoding=self.ego_context_heading_encoding,
                    edge_embedding_type=getattr(args, "edge_embedding_type", "fourier"),
                    time_embedding_type=getattr(args, "time_embedding_type", "legacy"),
                    time_embedding_scale=getattr(args, "time_embedding_scale", 99.0),
                    count_embedding_type=getattr(args, "count_embedding_type", "none"),
                    count_lane_source=getattr(args, "count_lane_source", "map_tokens"),
                    count_max_num_agents=getattr(args, "count_max_num_agents", 128),
                    count_max_num_lanes=getattr(args, "count_max_num_lanes", 1024),
                    map_embedding_type=getattr(args, "map_embedding_type", "none"),
                    map_id_source=getattr(args, "map_id_source", "fixed"),
                    map_id=getattr(args, "map_id", 0),
                    map_lg_type=getattr(args, "map_lg_type", 0),
                    map_label_dropout=getattr(args, "map_label_dropout", 0.1),
                )

            # normalized-space exploration std
            # self.refiner_log_std = nn.Parameter(
            #     torch.full(
            #         (args.input_dim,),
            #         math.log( 0.1  ),
            #     ),
            #     # requires_grad=False
            # )
            # refiner mean 最大修正量，normalized space
            self.refiner_delta_scale = 0.2

        if self.use_sde and any(getattr(self, name) == "uniform"
                                for name in ("pos_source", "shape_source", "velocity_source")):
            raise ValueError("Uniform sources support deterministic flow sampling; the SDE/PPO drift requires gaussian sources")
        if self.heading_noise == "circular" and self.use_sde:
            raise ValueError("Circular heading supports deterministic flow sampling; the SDE/PPO transition requires heading_noise=gaussian")
        self.apply(weight_init)
        # The outer generic initialization also visits the time MLP. Restore
        # SD's Normal(0, .02) weights for both denoisers afterwards.
        self.model.reset_time_embedding_parameters()
        if self.use_refiner:
            self.refine_model.reset_time_embedding_parameters()

    @staticmethod
    def _parse_fixed_branch_steps(
        value,
    ) -> Optional[tuple[int, ...]]:
        if value is None:

            return None

        if torch.is_tensor(value):
            value = (
                value.detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
        ):
            raise TypeError(
                "branch_steps must be a sequence "
                "of timestep indices."
            )

        steps = tuple(
            sorted(
                {
                    int(step)
                    for step in value
                }
            )
        )

        if not steps:
            raise ValueError(
                "branch_steps cannot be empty."
            )

        return steps

    # ==================================================================
    # Standard flow helpers
    # ==================================================================
    def _sample_noise(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        noise = _noise_endpoint(self.model, torch.randn_like(x), getattr(self, "sigma_h", None),
                                getattr(self, "heading_noise", "gaussian"),
                                pos_source=getattr(self, "pos_source", "gaussian"),
                                shape_source=getattr(self, "shape_source", "gaussian"),
                                velocity_source=getattr(self, "velocity_source", "gaussian"))

        fixed = self._conditioned_state_mask(tokenized_agent, x)
        noise = torch.where(fixed, x, noise)

        # Ego always retains its own source row, independently of whether its
        # state is fixed or its role is embedded. Hungarian only permutes
        # non-ego sources using the existing scene/type rule.
        matching_excluded = (self.model._ego_role_mask(tokenized_agent)
                             if getattr(self, "use_ego_embedding", False)
                             else tokenized_agent["ego_mask"].bool())
        movable = ~matching_excluded
        movable_noise = noise[movable]
        matched_index = get_closest_sum_idx_fast(
            movable_noise/self.model.normal_scale,
            x[movable]/self.model.normal_scale,
            {
                "batch": tokenized_agent["batch"][movable],
                "type": tokenized_agent["type"][movable],
            },
            all_state=True,
            use_all_type=False,#getattr(self, "generate_type", False),
        )
        noise[movable] = movable_noise[matched_index]

        return noise

        # return self.model.denormalize(
        #     noise[matched_index]
        # )
        #

    def _sample_time(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> Tensor:
        batch = tokenized_agent["batch"].long()
        num_graphs = int(
            tokenized_agent["num_graphs"]
        )

        scene_time = torch.rand(
            (num_graphs, 1),
            device=x.device,
            dtype=x.dtype,
        )

        return scene_time[batch]

    def _conditioned_agent_mask(self, tokenized_agent) -> Tensor:
        """Rows with no generated fields; only these use zero timestep."""
        mask = tokenized_agent["ego_mask"].bool()
        fully_fixed = all(self._ego_field_fixed(field) for field in EGO_FIELDS[:4])
        fully_fixed &= not getattr(self, "generate_type", False) or self._ego_field_fixed("type")
        return mask if fully_fixed else torch.zeros_like(mask)

    def _ego_field_fixed(self, field) -> bool:
        # Older lightweight Flow fixtures only provided the master flag.
        if not hasattr(self, "fix_ego"):
            return True
        return getattr(self, f"fix_ego_{field}", self.fix_ego)

    def _conditioned_state_mask(self, tokenized_agent, reference: Tensor) -> Tensor:
        fields = reference.new_zeros(reference.shape[-1], dtype=torch.bool)
        for field, indices in (("position", slice(0, 2)), ("heading", slice(2, 4)),
                               ("shape", slice(4, 6)), ("velocity", slice(6, None))):
            fields[indices] = self._ego_field_fixed(field)
        return tokenized_agent["ego_mask"].bool()[:, None] & fields[None]

    def _conditioned_type_mask(self, tokenized_agent) -> Tensor:
        mask = tokenized_agent["ego_mask"].bool()
        return mask if self._ego_field_fixed("type") else torch.zeros_like(mask)

    def _fix_conditioned_agents(
        self,
        clean: Tensor,
        latent: Tensor,
        time: Tensor,
        tokenized_agent: HeteroData,
    ) -> None:
        ego_mask = self._conditioned_agent_mask(tokenized_agent)
        fixed = self._conditioned_state_mask(tokenized_agent, clean)
        latent[fixed] = clean[fixed]
        time[ego_mask] = 0.0

    def _prepare_supervised_batch(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
    ) -> tuple[Tensor, Tensor, Tensor]:
        noise = self._sample_noise(
            x,
            tokenized_agent,
        )

        time = self._sample_time(
            x,
            tokenized_agent,
        )

        self._fix_conditioned_agents(
            x,
            noise,
            time,
            tokenized_agent,
        )

        # Rectified Flow: x0=data, x1=noise.
        if self.heading_noise == "circular":
            latent = _circular_interpolate(x, noise, time)
        else:
            latent = (1.0 - time) * x + time * noise
        latent = torch.where(self._conditioned_state_mask(tokenized_agent, x), x, latent)

        if getattr(self, "generate_type", False):
            labels = F.one_hot(tokenized_agent["type"].long(), 3).to(x)
            type_noise = torch.randn_like(labels)
            conditioned = self._conditioned_type_mask(tokenized_agent)
            type_noise[conditioned] = labels[conditioned]
            tokenized_agent["_init_diffusion_type_source"] = type_noise
            type_state = (1. - time) * labels + time * type_noise
            tokenized_agent["_init_diffusion_type_state"] = torch.where(conditioned[:, None], labels, type_state)
        return noise, time, latent

    def _model_velocity(
        self,
        latent: Tensor,
        time: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        eval_mask: Optional[Tensor] = None,
        mode: int = 1,
        use_map_condition: bool = True,
    ) -> tuple[Tensor, Tensor]:
        prediction = self.model(
            latent,
            time,
            tokenized_agent,
            map_feature,
            eval_mask,
            mode=mode,
            use_map_condition=use_map_condition,
        )

        return self._prediction_velocity(latent, time, prediction)

    def _prediction_velocity(
        self, latent: Tensor, time: Tensor, prediction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode mixed Euclidean x0 / circular angular-velocity outputs."""
        x0 = prediction[:, : latent.shape[-1]]
        if self.heading_objective == "angular_velocity":
            theta = torch.atan2(latent[:, 3], latent[:, 2])
            omega = prediction[:, self.model.m_delta_dim]
            theta0 = wrap_angle(theta - time[:, 0] * omega)
            x0 = torch.cat((x0[:, :2], torch.stack((theta0.cos(), theta0.sin()), dim=-1),
                            x0[:, 4:]), dim=-1)

        velocity = (
            latent - x0
        ) / time.clamp_min(
            self.t_eps
        )

        if self.heading_noise == "circular":
            theta = torch.atan2(latent[:, 3], latent[:, 2])
            if self.heading_objective == "x0":
                theta0 = torch.atan2(x0[:, 3], x0[:, 2])
                denominator = time[:, 0].clamp_min(torch.finfo(time.dtype).eps)
                omega = wrap_angle(theta - theta0) / denominator
            omega = torch.where(time[:, 0] > 0, omega, torch.zeros_like(omega))
            tangent = torch.stack((-theta.sin(), theta.cos()), dim=-1) * omega[:, None]
            velocity = torch.cat((velocity[:, :2], tangent, velocity[:, 4:]), dim=-1)
        return velocity, x0

    def get_ref_mean_std(self,base,tokenized_agent: HeteroData) -> tuple[Tensor, Tensor,Tensor]:

        if "prediction"  in tokenized_agent:
            prediction=tokenized_agent["prediction"]
        else:
            prediction = self.refine_model(
                base,
                torch.zeros_like(base[:, :1]),
                tokenized_agent,
                tokenized_agent["initial_map_feature"],
            )

        delta_mu = (
                self.refiner_delta_scale
                * torch.tanh(prediction[:, :base.shape[-1]])
        )

        # delta_mu[:, 4:6] = 0.0  # don't touch length / width

        # delta_mu=prediction[:,:base.shape[-1]]

        #log_std=self.refiner_log_std.clamp(  math.log(0.03),  math.log(0.3)  ).expand_as(delta_mu)
        log_std = prediction[:, base.shape[-1]:]#
        # min_log_std = math.log(0.03)
        # max_log_std = math.log(0.3)
        #
        # log_std = min_log_std + (
        #         0.5 * (torch.tanh(prediction[:,base.shape[-1]:]) + 1.0)
        #         * (max_log_std - min_log_std)
        # )
        std = log_std.exp()

        return std, delta_mu, log_std

    def get_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ):

        if self.token_processor.use_refiner:
            non_ego = ~tokenized_agent["ego_mask"].bool()

            base = tokenized_agent["refiner_base"]
            old_action = tokenized_agent["refiner_action"]

            std, delta_mu, log_std=self.get_ref_mean_std(
                base,
                tokenized_agent )

            active_dims = [0, 1, 2, 3, 4,5,6, 7]

            dist = torch.distributions.Normal(
                delta_mu[non_ego][:, active_dims],
                std[non_ego][:, active_dims],
            )

            log_prob = dist.log_prob(
                old_action[non_ego][:, active_dims]
            ).sum(dim=-1)

            advantage = tokenized_agent["advantages"][0][non_ego].detach()

            pg_loss = -(log_prob * advantage).mean()

            # Keep correction small.

            residual_loss =delta_mu[non_ego].square().mean()

            #eps = torch.randn_like(delta_mu)

            # eps[:,4:6]=0
            # else:
            #     eps=torch.zeros_like(delta_mu)
            tokenized_agent["delta_mu"]=delta_mu
            tokenized_agent["std"]=std

            #delta = delta_mu + std * eps

            #res = delta * self.model.normal_scale
            #
            # refine_mean = res + base
            #
            # refine_mean = torch.where(
            #     tokenized_agent["ego_mask"][:,None],
            #     base,
            #     refine_mean,
            # )
            #
            # # res[:,4:]=base[:,4:] +res[:,4:]
            # #
            # # refine_mean =self.encoder.init_decoder.G1.model.output_transform(res, base[:, :2], torch.atan2(base[:, 3], base[:, 2]))
            #
            # edge_loss, end_idx, start_idx = multi_circle_collision_loss_mem_efficient(
            #     refine_mean, tokenized_agent["batch"]
            # )
            # collision_loss = edge_loss.mean()

            # Don't let exploration std explode.
            std_loss = (
                    log_std[:,active_dims] - math.log(0.1)
            ).square().mean()

            # std_sq = torch.exp(2.0 * log_std)
            #
            # ref_std=0.1
            #
            # ref_log_std = math.log(ref_std)
            # ref_var = ref_std ** 2
            #
            # kl_per_dim = 0.5 * (
            #         (std_sq + delta_mu.square()) / ref_var
            #         - 1.0
            #         + 2.0 * (ref_log_std - log_std)
            # )
            #
            # # sum over action dimensions, mean over agents
            # kl = kl_per_dim.sum(dim=-1)[non_ego].mean()

            shape_delta = delta_mu[non_ego, 4:6]

            shape_loss = (
                    shape_delta
                    / torch.tensor(
                [0.20, 0.10],
                device=shape_delta.device,
            )
            ).square().mean()

            rl_loss = (
                    pg_loss
                   # +kl*0.1
                    + 0.02 * residual_loss
                    + 0.1 * std_loss
                    +shape_loss
                    #+  100* collision_loss
            )

            tokenized_agent["rl_loss"] = rl_loss

            match_loss= col_loss= pos_loss= heading_loss=shape_loss=vel_loss=torch.zeros_like(rl_loss)

            loss=(match_loss, col_loss, pos_loss, heading_loss, shape_loss, vel_loss)

            # loss = self._supervised_loss(
            #     x,
            #     tokenized_agent,
            #     map_feature,
            # )

            return loss

        loss=self._supervised_loss(
            x,
            tokenized_agent,
            map_feature,
        )

        if "advantages" in tokenized_agent:
            if self.use_sde:
                rl_loss = self._sde_advantage_loss(
                    tokenized_agent,
                    map_feature,
                )
            else:
                rl_loss = self._direct_advantage_loss(
                    tokenized_agent,
                    map_feature,
                )

            tokenized_agent["rl_loss"] = rl_loss

        return loss

    def _speed_magnitude_loss(self, prediction: Tensor, target: Tensor,
                              time: Tensor, ego_mask: Tensor):
        """Dimensionless speed MSE, averaged over movable interior-time agents.

        The existing vector reconstruction supplies direction gradients, including
        when the predicted vector is exactly zero. This auxiliary term supervises
        the physical norm before velocity normalization or output transforms.
        """
        predicted_velocity = prediction[:, 6:8]
        target_velocity = target[:, 6:8].detach()
        if predicted_velocity.dtype in (torch.float16, torch.bfloat16):
            predicted_velocity = predicted_velocity.float()
        if target_velocity.dtype in (torch.float16, torch.bfloat16):
            target_velocity = target_velocity.float()
        speed_error = (torch.linalg.vector_norm(predicted_velocity, dim=-1)
                       - torch.linalg.vector_norm(target_velocity, dim=-1))
        if self.speed_loss_scale is None:
            # E[||v||^2] = sum(E[v]^2 + Var[v]). Reuse the checkpointed
            # population moments; no batch-dependent or learnable denominator.
            mean = self.model.normal_mean[0, 6:8].detach().to(speed_error)
            std = self.model.normal_scale[0, 6:8].detach().to(speed_error)
            scale = (mean.square() + std.square()).sum().sqrt().clamp_min(1.)
        else:
            scale = speed_error.new_tensor(self.speed_loss_scale)
        scene_time = time.reshape(-1)
        active = (~ego_mask.reshape(-1).bool()) & (scene_time > 0) & (scene_time < 1)
        error = (speed_error / scale).square()
        loss = torch.where(active, error, torch.zeros_like(error)).sum() / active.sum().clamp_min(1)
        return loss, scale

    def _supervised_loss(
        self,
        x: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ):
        tokenized_agent.pop("_init_diffusion_speed_metrics", None)
        tokenized_agent.pop("_init_diffusion_type_metrics", None)
        tokenized_agent.pop("_init_diffusion_type_logits", None)
        noise, time, latent = (
            self._prepare_supervised_batch(
                x,
                tokenized_agent,
            )
        )

        if self.heading_objective == "angular_velocity":
            prediction = self.model(latent, time, tokenized_agent, map_feature)
            _, x0 = self._prediction_velocity(latent, time, prediction)
        else:
            _, x0 = self._model_velocity(latent, time, tokenized_agent, map_feature)

        fixed = self._conditioned_state_mask(tokenized_agent, x)

        # Avoid in-place modification of model output.
        x0 = torch.where(
            fixed,
            x.detach(),
            x0,
        )

        loss_options = {}
        if self.heading_x0_loss == "angle_mse":
            loss_options["heading_x0_loss"] = self.heading_x0_loss
        if self.size_representation == "log":
            # Reconstruction uses log sizes; collision circles use meters.
            loss_options["state_to_physical"] = self.model.state_to_physical
            size_valid = tokenized_agent.get("_init_diffusion_size_valid_mask")
            if size_valid is not None:
                reconstruction_mask = torch.ones((len(x), 8), dtype=torch.bool, device=x.device)
                reconstruction_mask[:, 4:6] = size_valid
                loss_options["reconstruction_mask"] = reconstruction_mask
                loss_options["collision_valid_mask"] = size_valid.all(-1)
        if self.heading_objective == "angular_velocity":
            loss_options["reconstruction_dims"] = ((0, 1, 4, 5, 6)
                if self.velocity_representation == "speed" else (0, 1, 4, 5, 6, 7))
        loss_prediction, loss_target, loss_scale = x0, x, self.model.normal_scale
        if self.velocity_representation == "speed":
            # The shared loss/collision API is 8D. Append a dummy zero, so the
            # scalar speed is supervised directly, without a direction loss.
            loss_prediction = torch.cat((x0, torch.zeros_like(x0[:, :1])), dim=-1)
            loss_target = torch.cat((x, torch.zeros_like(x[:, :1])), dim=-1)
            loss_scale = torch.cat((loss_scale, torch.ones_like(loss_scale[:, :1])), dim=-1)
        loss = get_diff_loss(
            tokenized_agent,
            loss_prediction,
            loss_target,
            time,
            self.t_eps,
            scale=loss_scale,
            use_col=True,
            x_pred=True,
            **loss_options,
        )

        if self.velocity_representation == "speed":
            total, collision, position, heading, shape, _ = loss
            speed_loss = (x0[:, 6] - x[:, 6]).square()
            loss = (total, collision, position, heading, shape, speed_loss)

        if self.heading_objective == "angular_velocity":
            theta0 = torch.atan2(x[:, 3], x[:, 2])
            theta1 = torch.atan2(noise[:, 3], noise[:, 2])
            target = wrap_angle(theta1 - theta0)
            active = (~fixed[:, 2]) & (time[:, 0] > 0) & (time[:, 0] < 1)
            # Uniform flow-time weighting; no 1/t^3 weighting or wrapping
            # the prediction error. The target is d(theta_t)/dt = delta.
            heading_loss = torch.where(active, (prediction[:, self.model.m_delta_dim] - target).square(),
                                       torch.zeros_like(target))
            total, collision, position, _, shape, velocity = loss
            loss = (total + self.heading_flow_loss_weight * heading_loss, collision,
                    position, heading_loss, shape, velocity)

        if self.speed_loss_weight > 0:
            magnitude_loss, magnitude_scale = self._speed_magnitude_loss(x0, x, time, fixed[:, 6])
            weighted_loss = self.speed_loss_weight * magnitude_loss
            # A scalar active-agent mean adds once after the wrapper reduces
            # reconstruction, preserving the existing six-item return contract.
            loss = (loss[0] + weighted_loss, *loss[1:])
            tokenized_agent["_init_diffusion_speed_metrics"] = {
                "loss": magnitude_loss.detach(),
                "weighted_loss": weighted_loss.detach(),
                "scale": magnitude_scale.detach(),
            }

        if getattr(self, "generate_type", False):
            logits = tokenized_agent["_init_diffusion_type_logits"]
            active = (~self._conditioned_type_mask(tokenized_agent)) & (time[:, 0] > 0) & (time[:, 0] < 1)
            labels = tokenized_agent["type"].long()
            per_agent = F.cross_entropy(logits.float(), labels, reduction="none")
            type_loss = torch.where(active, per_agent, torch.zeros_like(per_agent)).sum() / active.sum().clamp_min(1)
            weighted_loss = self.type_loss_weight * type_loss
            loss = (loss[0] + weighted_loss, *loss[1:])
            accuracy = ((logits.argmax(-1) == labels) & active).sum().float() / active.sum().clamp_min(1)
            tokenized_agent["_init_diffusion_type_metrics"] = {
                "loss": type_loss.detach(), "weighted_loss": weighted_loss.detach(),
                "accuracy": accuracy.detach(),
            }
        return loss

    def _sde_advantage_loss(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        (
            current,
            next_sample,
            old_log_prob,
        ) = tokenized_agent["sde_z"]

        (
            time,
            next_time,
        ) = tokenized_agent["sde_t"]

        saved_noise_level = tokenized_agent.get(
            "sde_noise_level"
        )

        if current.ndim == 2:
            current = current[:, None]
            next_sample = next_sample[:, None]
            time = time[:, None]
            next_time = next_time[:, None]

            if old_log_prob is not None:
                old_log_prob = old_log_prob[:, None]

            if saved_noise_level is not None:
                saved_noise_level = saved_noise_level[:, None]

        num_agents, num_branches, _ = current.shape

        non_ego = ~tokenized_agent[
            "ego_mask"
        ].bool()

        if not torch.any(non_ego):
            return current.new_zeros(())

        tokenized_agent=self.repeat_input_copy(tokenized_agent,num_branches)

        velocities, pred_x0 = self._model_velocity(
            current.transpose(0, 1).flatten(0, 1),
            time.transpose(0, 1).flatten(0, 1),
            tokenized_agent,
            map_feature,
        )

        velocities = velocities.reshape(
            num_branches,
            num_agents,
            -1,
        ).transpose(0, 1)  # [A, B, D]

        branch_noise = saved_noise_level.detach()

        # delta_t = time - next_time

        active = (
                non_ego[:, None]
                & (branch_noise.amax(dim=-1) > 0)
        )

        (
            _,
            selected_log_prob,
            _,
            _,
        ) = self.sde_step_with_logprob(
            time=time[active],
            next_time=next_time[active],
            model_output=velocities[active],
            sample=current[active],
            noise_level=branch_noise[active],
            prev_sample=next_sample[active],
        )

        advantages = tokenized_agent["advantages"].transpose(0, 1) #a,t

        selected_advantage = advantages[active]

        # time_for_ratio = torch.where(
        #     time >= 1.0,
        #     torch.full_like(time, 0.9),
        #     time,
        # )
        #
        # diffusion = (
        #         torch.sqrt(
        #             time.clamp_min(0.0)
        #             / (1.0 - time_for_ratio)
        #         )
        #         * branch_noise#noise_level
        # )#smaller t , smaller std
        #
        # step_std_dim = (
        #         diffusion * torch.sqrt(torch.abs(delta_t))
        # )
        #
        # step_std = step_std_dim.mean(dim=-1)
        #
        # selected_step_std = 1/step_std[active]  # [N]
        #
        # weight = selected_step_std / (
        #         selected_step_std.mean().detach() + 1e-8
        # )
        #v_norm=velocities[active].norm(dim=-1).mean()

        weight=1

        loss=-(weight*
            selected_log_prob
            * selected_advantage
        ).mean()

        return loss#+v_norm*0.01

    def _direct_advantage_loss(
            self,
            tokenized_agent: HeteroData,
            map_feature: Mapping[str, Tensor],
    ) -> Tensor:
        sampled_x0 = tokenized_agent["gen_z"].detach()

        _, time, latent = self._prepare_supervised_batch(
            sampled_x0,
            tokenized_agent,
        )

        _, pred_x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        non_ego = ~tokenized_agent["ego_mask"].bool()

        if not torch.any(non_ego):
            return sampled_x0.new_zeros(())

        advantage = (
            tokenized_agent["advantages"].transpose(0, 1)[:,0][non_ego]
            .detach()
        )

        # Assuming advantage has already been normalized.
        advantage = advantage.clamp(-2.0, 2.0)

        pred = pred_x0[non_ego]
        target_endpoint = sampled_x0[non_ego]

        alpha_pos = 0.10
        alpha_neg = 0.05

        coeff = torch.where(
            advantage >= 0,
            alpha_pos * advantage,
            alpha_neg * advantage,
        )

        coeff = coeff.clamp(
            min=-0.10,
            max=0.20,
        )

        target_x0 = (
                pred.detach()
                + coeff[..., None]
                * (
                        target_endpoint
                        - pred.detach()
                )
        )

        loss = 0.5 * (
                pred - target_x0.detach()
        ).square().mean()

        return loss
    # ==================================================================
    # Multi-branch sampling
    # ==================================================================
    def _choose_branch_steps(
        self,
        num_graphs: int,
        total_steps: int,
        device: torch.device,
        branch_steps: Optional[
            int | Sequence[int] | Tensor
        ],
    ) -> Tensor:
        if (
            branch_steps is None
            and self.fixed_branch_steps is not None
        ):
            branch_steps = self.fixed_branch_steps

        if torch.is_tensor(branch_steps):
            if branch_steps.ndim == 0:
                branch_steps = int(
                    branch_steps.item()
                )
            else:
                branch_steps = (
                    branch_steps.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                )

        if (
            isinstance(branch_steps, Sequence)
            and not isinstance(
                branch_steps,
                (str, bytes),
            )
        ):
            selected = sorted(
                {
                    int(step)
                    for step in branch_steps
                }
            )

            return torch.tensor(
                selected,
                device=device,
                dtype=torch.long,
            )[None].expand(
                num_graphs,
                -1,
            )

        count = (
            self.num_branch_steps
            if branch_steps is None
            else int(branch_steps)
        )

        count = min(
            count,
            total_steps,
        )

        # Sampling without replacement for every scene.
        random_score = torch.rand(
            num_graphs,
            total_steps-1,
            device=device,
        )

        selected = random_score.argsort(
            dim=1
        )[:, :count]

        return selected

    @staticmethod
    def _gather_steps(
        values: Tensor,
        step_index: Tensor,
    ) -> Tensor:
        """Gather [N, T, ...] at per-agent indices [N, B]."""
        if values.ndim < 2:
            raise ValueError(
                "values must have shape [N, T, ...]."
            )

        if (
            step_index.ndim != 2
            or step_index.shape[0] != values.shape[0]
        ):
            raise ValueError(
                "step_index must have shape [N, B]."
            )

        index = step_index

        for _ in range(values.ndim - 2):
            index = index.unsqueeze(-1)

        index = index.expand(
            *step_index.shape,
            *values.shape[2:],
        )

        return values.gather(
            dim=1,
            index=index,
        )

    @torch.no_grad()
    def _sample_step(
        self,
        latent: Tensor,
        time_scalar: Tensor,
        next_time_scalar: Tensor,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        branch_mask: Optional[Tensor] = None,
    ):
        num_agents = latent.shape[0]

        time = torch.full(
            (num_agents, 1),
            time_scalar,
            device=latent.device,
            dtype=latent.dtype,
        )
        next_time = torch.full_like(time, next_time_scalar)

        self._fix_conditioned_agents(
            tokenized_agent["expert_input"],
            latent,
            time,
            tokenized_agent,
        )

        ego_mask = self._conditioned_agent_mask(tokenized_agent)
        next_time[ego_mask] = 0.0

        velocity, x0 = self._model_velocity(
            latent,
            time,
            tokenized_agent,
            map_feature,
        )

        if getattr(self, "generate_type", False):
            type_state = tokenized_agent["_init_diffusion_type_state"]
            type_clean = tokenized_agent["_init_diffusion_type_logits"].float().softmax(-1).to(type_state)
            type_velocity = (type_state - type_clean) / time.clamp_min(self.t_eps)
            next_type = type_state + (next_time - time) * type_velocity
            fixed_type = self._conditioned_type_mask(tokenized_agent)
            next_type[fixed_type] = tokenized_agent["_init_diffusion_type_source"][fixed_type]
            tokenized_agent["_init_diffusion_type_state"] = next_type
        next_latent = latent + (next_time - time) * velocity
        if self.heading_noise == "circular":
            theta = torch.atan2(latent[:, 3], latent[:, 2])
            tangent = torch.stack((-theta.sin(), theta.cos()), dim=-1)
            omega = (velocity[:, 2:4] * tangent).sum(dim=-1)
            theta_next = wrap_angle(theta + (next_time - time)[:, 0] * omega)
            heading = torch.stack((theta_next.cos(), theta_next.sin()), dim=-1)
            fixed_heading = self._conditioned_state_mask(tokenized_agent, latent)[:, 2:4]
            heading = torch.where(fixed_heading, tokenized_agent["expert_input"][:, 2:4], heading)
            next_latent = torch.cat((next_latent[:, :2], heading, next_latent[:, 4:]), dim=-1)
        log_prob = latent.new_zeros(num_agents)
        used_noise_level = latent.new_zeros(latent.shape)

        if (
            self.use_sde
            and branch_mask is not None
            and branch_mask.any()
            and self.token_processor.learn_init
            and "gt_z_raw" not in tokenized_agent
        ):
            noise_level =0.5 #self.get_adaptive_noise_level(time, next_time)
            noise_level = noise_level * branch_mask[:, None].to(latent.dtype)

            stochastic = branch_mask.bool() & (~ego_mask)
            stochastic &= noise_level.amax(dim=-1) > 0

            if torch.any(stochastic):
                (
                    stochastic_next,
                    stochastic_log_prob,
                    _,
                    _,
                ) = self.sde_step_with_logprob(
                    time=time[stochastic],
                    next_time=next_time[stochastic],
                    model_output=velocity[stochastic],
                    sample=latent[stochastic],
                    noise_level=noise_level[stochastic],
                )
                next_latent[stochastic] = stochastic_next
                log_prob[stochastic] = stochastic_log_prob

                # Expand [N,1] scheduled noise to [N,D] for replay storage.
                used_noise_level[stochastic] = noise_level[stochastic]

        fixed = self._conditioned_state_mask(tokenized_agent, latent)
        next_latent = torch.where(fixed, tokenized_agent["expert_input"], next_latent)
        x0 = torch.where(fixed, tokenized_agent["expert_input"], x0)
        return (
            next_latent,
            x0,
            time,
            next_time,
            log_prob,
            used_noise_level,
        )

    @torch.no_grad()
    def sample(
        self,
        tokenized_agent: HeteroData,
        map_feature: Mapping[str, Tensor],
        steps: int = 20,
        branch_steps: Optional[
            int | Sequence[int] | Tensor
        ] = None,
    ) -> Tensor:

        agent_batch = tokenized_agent[
            "batch"
        ].long()

        num_graphs = int(
            tokenized_agent["num_graphs"]
        )

        num_agents = agent_batch.numel()
        if getattr(self, "use_ego_embedding", False):
            self.model._ego_role_mask(tokenized_agent)

        ego_mask = self._conditioned_agent_mask(tokenized_agent)

        if self.size_representation == "log":
            # Fit source statistics and encode any raw cached target before
            # drawing the log-space endpoint. This also preserves ego sizes.
            expert_input, _ = self.model.get_input(tokenized_agent)
            tokenized_agent["expert_input"] = expert_input
            tokenized_agent["_init_diffusion_size_representation"] = "log"

        latent = torch.randn(
            num_agents,
            self.model.m_delta_dim,
            device=agent_batch.device,
            dtype=self.model.normal_scale.dtype,
        )

        latent = _noise_endpoint(self.model, latent, self.sigma_h, self.heading_noise,
                                 pos_source=self.pos_source, shape_source=self.shape_source,
                                 velocity_source=self.velocity_source)
        tokenized_agent["gen_noise"]=latent.clone()

        if self.velocity_representation == "speed" or "expert_input" not in tokenized_agent:
            expert_input, _ = self.model.get_input(
                tokenized_agent
            )
            tokenized_agent[
                "expert_input"
            ] = expert_input

        if getattr(self, "generate_type", False):
            for key in ("_init_diffusion_type_logits", "_init_diffusion_type_metrics",
                        "_init_diffusion_generated_type"):
                tokenized_agent.pop(key, None)
            type_source = torch.randn(num_agents, 3, device=latent.device, dtype=latent.dtype)
            fixed_type = self._conditioned_type_mask(tokenized_agent)
            if fixed_type.any():
                type_source[fixed_type] = F.one_hot(tokenized_agent["type"][fixed_type].long(), 3).to(type_source)
            tokenized_agent["_init_diffusion_type_source"] = type_source
            tokenized_agent["_init_diffusion_type_state"] = type_source.clone()

        # Generation: start from x1~noise at t=1 and integrate to x0 at t=0.
        timesteps = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            device=agent_batch.device,
            dtype=latent.dtype,
        )

        if self.use_sde:
            graph_branch_steps = (
                self._choose_branch_steps(
                    num_graphs,
                    steps,
                    agent_batch.device,
                    branch_steps,
                )
            )

            agent_branch_steps = (
                graph_branch_steps[
                    agent_batch
                ]
            )

            step_branch_mask = torch.zeros(
                num_agents,
                steps,
                device=latent.device,
                dtype=torch.bool,
            )
            step_branch_mask.scatter_(
                dim=1,
                index=agent_branch_steps,
                value=True,
            )
            step_branch_mask[ego_mask] = False

            latent_history = [
                latent.clone()
            ]
            time_history = []
            next_time_history = []
            log_prob_history = []
            noise_level_history = []
            feature_history = []

        for step in range(steps):
            (
                latent,
                _,
                time,
                next_time,
                log_prob,
                used_noise_level,
            ) = self._sample_step(
                latent,
                timesteps[step],
                timesteps[step + 1],
                tokenized_agent,
                map_feature,
                branch_mask=(
                    step_branch_mask[:, step]
                    if self.use_sde
                    else None
                ),
            )

            if self.use_sde:
                latent_history.append(
                    latent.clone()
                )
                time_history.append(time)
                next_time_history.append(
                    next_time
                )
                log_prob_history.append(
                    log_prob
                )
                noise_level_history.append(
                    used_noise_level
                )

                feature_history.append(
                    tokenized_agent[
                        "noise_feat_cur"
                    ].clone()
                )

            elif (
                step == 0
                and "noise_feat_cur"
                in tokenized_agent
            ):
                tokenized_agent[
                    "noise_feat"
                ] = tokenized_agent[
                    "noise_feat_cur"
                ][:,None]

       # del tokenized_agent["agent_type_embed"]

        fixed = self._conditioned_state_mask(tokenized_agent, latent)
        latent = torch.where(fixed, tokenized_agent["expert_input"], latent)

        if getattr(self, "generate_type", False):
            # Decode the final clean prediction, rather than a residual noisy state.
            types = tokenized_agent["_init_diffusion_type_logits"].argmax(-1)
            fixed_type = self._conditioned_type_mask(tokenized_agent)
            types[fixed_type] = tokenized_agent["_init_diffusion_type_source"][fixed_type].argmax(-1)
            tokenized_agent["_init_diffusion_generated_type"] = types
            tokenized_agent["type"] = types
        if not self.use_sde:
            tokenized_agent["gen_z"] = latent

            if self.use_refiner:
                base=latent

                std, delta_mu, log_std = self.get_ref_mean_std(
                    base,
                    tokenized_agent)

                if "gt_z_raw" not in tokenized_agent and self.token_processor.use_noise:
                    eps = torch.randn_like(delta_mu)

                    #eps[:,4:6]=0
                else:
                     eps=torch.zeros_like(delta_mu)

                delta = delta_mu + std * eps

                # refiner_scale = torch.tensor([
                #     0.50,  # dx_long, m
                #     0.30,  # dy_lat,  m
                #     0.05,  # cos-heading residual
                #     0.08,  # sin-heading residual
                #     0.30,  # length residual, m
                #     0.15,  # width residual,  m
                #     0.60,  # dv_long, m/s
                #     0.30,  # dv_lat,  m/s
                # ])*5
                # refiner_scale = torch.tensor([
                #     0.30,  # dx_long, m
                #     0.20,  # dy_lat,  m
                #     0.03,  # cos-heading residual
                #     0.05,  # sin-heading residual
                #     0.20,  # length residual, m
                #     0.10,  # width residual,  m
                #     0.40,  # dv_long, m/s
                #     0.20,  # dv_lat,  m/s
                # ])*5
                # refiner_scale = torch.tensor([
                #     0.80,  # dx_long, m
                #     0.50,  # dy_lat,  m
                #     0.08,  # cos-heading residual
                #     0.12,  # sin-heading residual
                #     0.50,  # length residual, m
                #     0.25,  # width residual,  m
                #     1.00,  # dv_long, m/s
                #     0.50,  # dv_lat,  m/s
                # ])*5
                # --------------------------------------
                # normalized residual -> raw residual
                # --------------------------------------
                res =  delta *self.model.normal_scale#refiner_scale[None].to(delta.device)#

                latent=base +res

                # res[:, 4:] = base[:, 4:] + res[:, 4:]
                # res[:, 2] = res[:, 2] + 1
                #
                # latent = self.model.output_transform(res, base[:, :2],torch.atan2(base[:, 3], base[:, 2]))

                # tokenized_agent["log_prob"] = dist.log_prob(latent)[~ego_mask].sum(dim=-1)
                latent[ego_mask] = tokenized_agent["expert_input"  ][ego_mask]

                tokenized_agent["refiner_base"] = base
                tokenized_agent["refiner_action"] = delta

                tokenized_agent["noise_feat"]=tokenized_agent[ "noise_feat_cur" ][:,None]
                tokenized_agent["refined_z"] = latent

            return latent

        latent_stack = torch.stack(
            latent_history,
            dim=1,
        )

        time_stack = torch.stack(
            time_history,
            dim=1,
        )

        next_time_stack = torch.stack(
            next_time_history,
            dim=1,
        )

        log_prob_stack = torch.stack(
            log_prob_history,
            dim=1,
        )

        noise_level_stack = torch.stack(
            noise_level_history,
            dim=1,
        )

        feature_stack = torch.stack(
            feature_history,
            dim=1,
        )

        selected_current = self._gather_steps(
            latent_stack[:, :-1],
            agent_branch_steps,
        )

        selected_next = self._gather_steps(
            latent_stack[:, 1:],
            agent_branch_steps,
        )

        selected_time = self._gather_steps(
            time_stack,
            agent_branch_steps,
        )

        selected_next_time = self._gather_steps(
            next_time_stack,
            agent_branch_steps,
        )

        selected_log_prob = self._gather_steps(
            log_prob_stack,
            agent_branch_steps,
        )

        selected_noise_level = self._gather_steps(
            noise_level_stack,
            agent_branch_steps,
        )

        selected_features = self._gather_steps(
            feature_stack,
            agent_branch_steps,
        )

        tokenized_agent["sde_z"] = (
            selected_current,
            selected_next,
            selected_log_prob.detach(),
        )

        tokenized_agent["sde_t"] = (
            selected_time,
            selected_next_time,
        )

        tokenized_agent["gen_z"]=latent

        tokenized_agent[ "sde_noise_level"] = selected_noise_level.detach()

        tokenized_agent["noise_feat"] = selected_features

        return latent

    def sde_step_with_logprob(
            self,
            time: Tensor,
            next_time: Tensor,
            model_output: Tensor,
            sample: Tensor,
            noise_level=0.7,
            prev_sample: Optional[Tensor] = None,
    ):
        """SDE transition directly in raw data space."""

        eps = 1e-5

        scale = self.model.normal_scale.to(
            device=sample.device,
            dtype=sample.dtype,
        ).clamp_min(eps)

        mean = self.model.normal_mean.to(
            device=sample.device,
            dtype=sample.dtype,
        )

        dt = next_time - time

        time_for_ratio = torch.where(
            time >= 1.0,
            torch.full_like(time, 0.9),
            time,
        )

        diffusion = (
                torch.sqrt(
                    time.clamp_min(0.0)
                    / (1.0 - time_for_ratio).clamp_min(eps)
                )
                * noise_level#noise_level
        )
        #diffusion=0.05/torch.sqrt(-dt)

        safe_time = time.clamp_min(eps)

        # Normalized-space transition coefficients.
        A = (
                1.0
                + diffusion.square()
                / (2.0 * safe_time)
                * dt
        )

        B = (
                    1.0
                    + diffusion.square()
                    * (1.0 - time)
                    / (2.0 * safe_time)
            ) * dt

        # Equivalent mean directly in raw space.
        next_mean = (
                mean
                + A * (sample - mean)
                + B * model_output
        )

        # Raw-space transition std.
        transition_std = (
                diffusion
                * torch.sqrt((-dt).clamp_min(eps))
                * scale
        ).clamp_min(eps)

        # Sample transition.
        if prev_sample is None:
            next_sample = (
                    next_mean
                    + transition_std
                    * torch.randn_like(sample)
            )
        else:
            next_sample = prev_sample

        # Raw-space Gaussian log probability.
        residual = (
                next_sample.detach()
                - next_mean
        )

        log_prob_element = (
                -0.5
                * (
                        residual.square()
                        / transition_std.square()
                        + 2.0 * torch.log(transition_std)
                        + math.log(2.0 * math.pi)
                )
        )

        log_prob = log_prob_element.sum(
            dim=tuple(range(1, log_prob_element.ndim))
        )

        return (
            next_sample,
            log_prob,
            next_mean,
            transition_std,
        )

    def repeat_input_copy(self, tokenized_agent, n_step):
        out = copy.copy(tokenized_agent)

        num_graphs = tokenized_agent["num_graphs"]
        batch = tokenized_agent["batch"]

        out["repeat_batch"] = batch.unsqueeze(1).repeat(1, n_step)

        repeated_batch = torch.stack(
            [batch + num_graphs * k for k in range(n_step)],
            dim=1,
        ).transpose(0, 1).flatten(0, 1)

        out["batch"] = repeated_batch
        # out["agent_type_embed"] = tokenized_agent["agent_type_embed"][None].repeat(
        #     n_step, 1,1
        # ).flatten(0, 1)
        out["num_graphs"] = num_graphs * n_step

        out["ego_feat"] = tokenized_agent["ego_feat"][None].repeat(
            n_step, 1, 1
        ).flatten(0, 1)

        return out
