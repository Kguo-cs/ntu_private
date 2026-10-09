import math
import logging
import copy
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn

from src.smart.layers import MLPLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.modules.edge_encoder import EdgeEncoder
from src.smart.scenario_dreamer.core.dit_layers import LabelEmbedder, TimestepEmbedder
from src.smart.scenario_dreamer.preprocessed import read_vectorworld_map_metadata
from src.smart.utils import (
    transform_to_global,
    transform_to_local,
    wrap_angle,
    rotate_to_global,
    rotate_to_local,
    weight_init,
)
from .noise_schedule import LearnableGroupedPowerSchedule
import torch.nn.functional as F

class InitDenoiser(nn.Module):
    """Cleaned initial-state denoiser.

    Kept public API:
        - normalize
        - denormalize
        - get_input
        - forward
        - get_output

    Removed unsupported/dead branches from the previous implementation:
        - DiT path
        - use_all_pos path
        - bin-normalization path
        - non-RoFormer path
        - previous-heading/speed/condition branches
        - return/cfg conditioning branches
        - unused padding/SkipMLP/ExploreNoiseNet code

    ``edge_embedding_type`` selects "fourier" or "mlp" for agent-agent
    and map-agent geometry. ``time_embedding_type`` independently selects
    the legacy encoding or Scenario Dreamer's scalar TimestepEmbedder.
    Optional SD count embeddings condition agents and map tokens by scene size.
    Optional SD scene-type embeddings share a graph/map label across both types.

    MeanFlow/iMF support:
        When ``mean_flow=True``, the model output is interpreted as the
        interval-average velocity u(z_t,t,r). The current time remains ``beta``;
        the interval length h=r-t is read from ``tokenized_agent["meanflow_h"]``
        and embedded additively.
    """

    def __init__(
        self,
        token_processor,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        x_pred: bool = True,
        edge_embedding_type: str = "fourier",
        heading_velocity: bool = False,
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
        invalid_size_policy: str = "mask",
        fix_ego: bool = True,
        generate_type: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        num_freq_bands = hidden_dim//2
        self.num_layers = num_layers
        self.num_heads = num_heads
        head_dim = hidden_dim//num_heads
        self.dropout = dropout
        self.x_pred = x_pred
        self.heading_velocity = heading_velocity
        if velocity_representation not in ("vector", "speed"):
            raise ValueError("velocity_representation must be vector or speed")
        if not isinstance(fix_ego, bool):
            raise ValueError("fix_ego must be boolean")
        self.fix_ego = fix_ego
        if not isinstance(generate_type, bool):
            raise ValueError("generate_type must be boolean")
        self.generate_type = generate_type
        self.velocity_representation = velocity_representation
        if size_representation not in ("linear", "log"):
            raise ValueError("size_representation must be linear or log")
        self.size_representation = size_representation
        if invalid_size_policy not in ("mask", "error"):
            raise ValueError("invalid_size_policy must be mask or error")
        self.invalid_size_policy = invalid_size_policy
        self._warned_invalid_sizes = False
        state_dim = 7 if velocity_representation == "speed" else 8
        if velocity_representation == "speed" and (not x_pred or input_dim != 7 or output_dim != 7):
            raise ValueError("speed representation requires a 7D x0 state predictor")
        if heading_velocity and (not x_pred or output_dim != state_dim):
            raise ValueError("heading_velocity requires an x0 state predictor matching the representation")
        self.token_processor = token_processor
        self.edge_embedding_type = edge_embedding_type
        if time_embedding_type not in ("legacy", "scenario_dreamer"):
            raise ValueError("time_embedding_type must be legacy or scenario_dreamer")
        self.time_embedding_type = time_embedding_type
        self.time_embedding_scale = float(time_embedding_scale)
        if not math.isfinite(self.time_embedding_scale) or self.time_embedding_scale <= 0:
            raise ValueError("time_embedding_scale must be finite and positive")

        if count_embedding_type not in ("none", "scenario_dreamer"):
            raise ValueError("count_embedding_type must be none or scenario_dreamer")
        if count_lane_source not in ("map_tokens", "scenario_dreamer"):
            raise ValueError("count_lane_source must be map_tokens or scenario_dreamer")
        for name, value in (("count_max_num_agents", count_max_num_agents),
                            ("count_max_num_lanes", count_max_num_lanes)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.count_embedding_type = count_embedding_type
        self.count_lane_source = count_lane_source
        self.count_max_num_agents = count_max_num_agents
        self.count_max_num_lanes = count_max_num_lanes

        if map_embedding_type not in ("none", "scenario_dreamer"):
            raise ValueError("map_embedding_type must be none or scenario_dreamer")
        if map_id_source not in ("fixed", "metadata"):
            raise ValueError("map_id_source must be fixed or metadata")
        if not isinstance(map_id, int) or isinstance(map_id, bool) or map_id not in (0, 1):
            raise ValueError("map_id must be integer 0 or 1")
        if map_lg_type is not None and (not isinstance(map_lg_type, int)
                or isinstance(map_lg_type, bool) or map_lg_type not in (0, 1)):
            raise ValueError("map_lg_type must be integer 0 or 1, or null for per-scene metadata")
        self.map_embedding_type = map_embedding_type
        self.map_id_source = map_id_source
        self.map_id = map_id
        self.map_lg_type = map_lg_type
        self.map_label_dropout = float(map_label_dropout)
        if not math.isfinite(self.map_label_dropout) or not 0. <= self.map_label_dropout <= 1.:
            raise ValueError("map_label_dropout must be finite and between 0 and 1")

        self.label_drop_prob = 0.0
        self.map_drop_prob=0.0

        self.num_classes = 3
        self.shape_dim = 2
        self.m_delta_dim = input_dim
        self.output_dim =output_dim

        self.register_buffer("normal_mean", torch.zeros(1, self.m_delta_dim))
        self.register_buffer("normal_scale", torch.ones(1, self.m_delta_dim))

        # Different groups can still use different schedules.
        # self.schedule = LearnableGroupedPowerSchedule(
        #     group_dims=(2, 2, 2, self.m_delta_dim - 6)
        # )
        #
        self.type_a_emb = nn.Embedding(self.num_classes , hidden_dim)
        self.noise_embedding = (
            TimestepEmbedder(hidden_dim)
            if time_embedding_type == "scenario_dreamer"
            else MLPLayer(self.m_delta_dim, hidden_dim, hidden_dim)
        )
        # Legacy features and their nonpersistent buffers retain the old
        # checkpoint layout. SD mode uses the bundled 256D features instead.
        # These sin/cos pairs at pi*2**k survive legacy MLP's LayerNorm.
        time_dims = torch.arange(self.m_delta_dim)
        self.register_buffer(
            "_time_frequencies",
            math.pi * 2.0 ** (time_dims // 2),
            persistent=False,
        )
        self.register_buffer(
            "_time_cos_mask", time_dims.remainder(2).bool(), persistent=False
        )

        if count_embedding_type == "scenario_dreamer":
            # SD indexes integer scene counts directly, including zero. Its
            # tables have no label dropout; the generic weight_init below also
            # matches SD's Normal(0, .02) initialization for embeddings.
            self.num_agents_embedder = LabelEmbedder(count_max_num_agents + 1, hidden_dim, 0)
            self.num_lanes_embedder = LabelEmbedder(count_max_num_lanes + 1, hidden_dim, 0)

        if map_embedding_type == "scenario_dreamer":
            # Waymo SD conditions on 2 * lg_type + map_id (four classes),
            # plus the null label when classifier-free label dropout is used.
            self.scene_type_embedder = LabelEmbedder(4, hidden_dim, self.map_label_dropout)

        if self.x_pred:
            self.proj_in_m_delta = nn.Linear(self.m_delta_dim - 4, hidden_dim)
        else:
            self.proj_in_m_delta = nn.Linear(self.m_delta_dim, hidden_dim)

        # Ego-context embedding. The input is:
        #   local ego poses relative to the generated agent + per-scene type count.
        # For the current tokenization, ego pose part is 9 and type-count part is 3.
        self.ego_dim = 9
        self.ego_embed = MLPLayer(self.ego_dim + 3, hidden_dim, hidden_dim)

        self.edge_encoder = EdgeEncoder(
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            use_a2a=True,
            use_pl2a=True,
            embedding_type=edge_embedding_type,
        )

        self.lane_embed = MLPLayer(128, hidden_dim, hidden_dim)

        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=self.dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=self.dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.to_out_m_delta = MLPLayer(hidden_dim, hidden_dim, self.output_dim)
        if self.generate_type:
            self.to_out_type = MLPLayer(hidden_dim, hidden_dim, self.num_classes)
        if heading_velocity:
            # rad / unit flow time; shared graph features, separate scalar head.
            self.to_out_heading_velocity = MLPLayer(hidden_dim, hidden_dim, 1)

        self.apply(weight_init)
        self.reset_time_embedding_parameters()

    def reset_time_embedding_parameters(self) -> None:
        """Restore the released SD time MLP initialization after generic init."""
        if self.time_embedding_type == "scenario_dreamer":
            for index in (0, 2):
                layer = self.noise_embedding.mlp[index]
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    # ---------------------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------------------
    def state_to_model(self, state: torch.Tensor) -> torch.Tensor:
        """Encode physical length/width without modifying the input tensor."""
        if self.size_representation == "linear":
            return state
        size = state[..., 4:6]
        if not torch.isfinite(size).all() or not (size > 0).all():
            raise ValueError("log size representation requires finite, positive length and width")
        # Keep log/exp reliable under mixed precision; all other fields retain
        # their meaning, and concatenation promotes the state if necessary.
        if size.dtype in (torch.float16, torch.bfloat16):
            size = size.float()
        return torch.cat((state[..., :4], size.log(), state[..., 6:]), dim=-1)

    def _encode_input_state(self, state: torch.Tensor, agent) -> torch.Tensor:
        """Treat corrupt training size coordinates as missing annotations."""
        if self.size_representation == "linear":
            return state
        size = state[:, 4:6]
        valid = torch.isfinite(size) & (size > 0)
        if valid.all():
            agent.pop("_init_diffusion_size_valid_mask", None)
            agent.pop("_init_diffusion_size_metrics", None)
            return self.state_to_model(state)
        rows = (~valid).any(-1).nonzero(as_tuple=True)[0]
        examples = []
        for row in rows[:5].tolist():
            example = {"agent": row, "length_width": size[row].detach().cpu().tolist()}
            for name in ("batch", "type"):
                if name in agent:
                    example[name] = int(agent[name][row])
            examples.append(example)
        if not self.training or self.invalid_size_policy == "error":
            raise ValueError("log size representation requires finite, positive length and width; "
                             f"invalid annotations: {examples}")

        # The fill is only an input condition, never a reconstruction target or
        # normalizer observation. Prefer a same-type geometric mean, then a
        # batch geometric mean, then SMART's nominal token dimensions (L,W).
        size = size.float() if size.dtype in (torch.float16, torch.bfloat16) else size
        log_size = torch.where(valid, size, torch.ones_like(size)).log()
        # Generated types must not leak into noisy inputs through annotation fills.
        kinds = (torch.zeros(len(size), device=size.device, dtype=torch.long)
                 if self.generate_type else agent.get("type", torch.zeros(len(size), device=size.device, dtype=torch.long)).long())
        nominal = size.new_tensor(((4.8, 2.), (1., 1.), (2., 1.)))[kinds.clamp(0, 2)].log()
        filled = log_size.clone()
        for dim in range(2):
            observed = valid[:, dim]
            fallback = nominal[:, dim].clone()
            if observed.any():
                fallback[:] = log_size[observed, dim].mean()
            for kind in range(3):
                members = kinds == kind
                known = members & observed
                if known.any():
                    fallback[members] = log_size[known, dim].mean()
            filled[:, dim] = torch.where(observed, log_size[:, dim], fallback)
        agent["_init_diffusion_size_valid_mask"] = valid
        agent["_init_diffusion_size_metrics"] = {
            "invalid_fields": (~valid).sum().to(dtype=filled.dtype).detach(),
            "invalid_agents": rows.numel(),
        }
        if not self._warned_invalid_sizes:
            logging.getLogger(__name__).warning(
                "Missing size annotations in log-size training: %s. Masking these "
                "coordinates in size loss/statistics and their GT collision pairs; "
                "all agents and other targets are retained.", examples)
            self._warned_invalid_sizes = True
        return torch.cat((state[:, :4], filled, state[:, 6:]), dim=-1)

    def state_to_physical(self, state: torch.Tensor) -> torch.Tensor:
        """Decode sizes for physical geometry/output, retaining gradients."""
        if self.size_representation == "linear":
            return state
        size = state[..., 4:6]
        if size.dtype in (torch.float16, torch.bfloat16):
            size = size.float()
        size = size.exp()
        if not torch.isfinite(size).all() or not (size > 0).all():
            raise FloatingPointError("exp(log size) produced a non-finite or zero physical size")
        return torch.cat((state[..., :4], size, state[..., 6:]), dim=-1)

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.normal_scale.clamp_min(1e-6)
        return (input - self.normal_mean) / scale

    def denormalize(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.normal_scale.clamp_min(1e-6)
        return input * scale + self.normal_mean

    def _maybe_init_normalizer(self, diff_output: torch.Tensor, size_valid_mask=None) -> None:
        if not torch.all(self.normal_mean == 0):
            # if self.normal_scale[0][0]>15:#20
            #     self.normal_scale[:, :2] = self.normal_scale[:, :2] * 0.8
            # if self.normal_scale[0][2]>1.5:
            #     self.normal_scale[:, 2:6] = self.normal_scale[:, 2:6] * 0.5
            #     # self.normal_scale[:, :2] = self.normal_scale[:, :2] * 2
            return

        with torch.no_grad():
            mean = torch.mean(diff_output, dim=0, keepdim=True)

            scale = torch.std(
                diff_output,
                dim=0,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-6)

            if size_valid_mask is not None:
                # Missing annotations and their input fills must not influence
                # the log-size source distribution. No observations => N(0,1).
                for dim in range(2):
                    observed = diff_output[size_valid_mask[:, dim], 4 + dim]
                    if observed.numel():
                        mean[0, 4 + dim] = observed.mean()
                        scale[0, 4 + dim] = observed.std(unbiased=False).clamp_min(1e-6)
                    else:
                        mean[0, 4 + dim] = 0.
                        scale[0, 4 + dim] = 1.
            self.normal_mean.copy_(mean)

            # Keep the old scaling heuristic.
            #scale[:, 2:4] = scale[:, 2:4] * 4
            #scale[:, :2] = scale[:, :2] * 2

            self.normal_scale.copy_(scale)

    # ---------------------------------------------------------------------
    # Tokenized-agent input construction
    # ---------------------------------------------------------------------
    def get_input(self, tokenized_agent,expert_data=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build local all-agent initial state.

        Returns:
            diff_input:
                Initial state used as source distribution input.
            diff_output:
                Target state to reconstruct / generate.

        Vector mode has shape [N_agent, 8]:
            [local_x, local_y, cos(local_heading), sin(local_heading),
             length, width, agent_frame_vx, agent_frame_vy]
        Speed mode has shape [N_agent, 7], replacing vx/vy with their norm.
        In log size mode, fields 4:6 contain log(length), log(width).

        Notes:
            The previous implementation used a mask that was immediately
            set to all True in ``InitDiffusion.forward``. This cleaned version
            therefore treats every agent as part of the
            initial-state generation set and uses explicit all-agent metadata:
                ``batch`` and ``type``.
        """
        if "expert_input" in tokenized_agent.keys():
            state = tokenized_agent["expert_input"]
            cached_size_mode = tokenized_agent.get("_init_diffusion_size_representation")
            if cached_size_mode is not None and cached_size_mode != self.size_representation:
                raise ValueError("cached expert_input size representation does not match the denoiser")
            if self.velocity_representation == "speed":
                if state.shape[-1] == 8:
                    tokenized_agent["_init_diffusion_ego_local_velocity"] = state[:, 6:8].clone()
                    state = torch.cat((state[:, :6], torch.linalg.vector_norm(state[:, 6:8], dim=-1, keepdim=True)), dim=-1)
                    tokenized_agent["expert_input"] = state
                elif state.shape[-1] != 7:
                    raise ValueError("speed expert_input must have 7 state fields or 8 vector fields")
                elif "local_vel" in tokenized_agent:
                    tokenized_agent.setdefault("_init_diffusion_ego_local_velocity", tokenized_agent["local_vel"][:, :2].clone())
            if self.size_representation == "log":
                cached_valid = tokenized_agent.get("_init_diffusion_size_valid_mask")
                if cached_size_mode == "log" and cached_valid is not None and not cached_valid.all() and (
                        not self.training or self.invalid_size_policy == "error"):
                    rows = (~cached_valid).any(-1).nonzero(as_tuple=True)[0].tolist()
                    raise ValueError("cached log expert_input contains missing size annotations; "
                                     f"evaluation/invalid_size_policy=error require valid sizes, agent rows={rows}")
                if cached_size_mode is None:
                    state = self._encode_input_state(state, tokenized_agent)
                    tokenized_agent["expert_input"] = state
                    tokenized_agent["_init_diffusion_size_representation"] = "log"
                self._maybe_init_normalizer(state, tokenized_agent.get("_init_diffusion_size_valid_mask"))
            elif self.velocity_representation == "speed":
                self._maybe_init_normalizer(state)
            return state, state

        batch_ego_pos = tokenized_agent["batch_ego_pos"]
        batch_ego_heading = tokenized_agent["batch_ego_heading"]
        shape = tokenized_agent["shape"]

        agent_pos = tokenized_agent["initial_pos"]
        agent_head = tokenized_agent["initial_heading"]
        local_vel = tokenized_agent["local_vel"]
        motion = local_vel[:, :2]
        if self.velocity_representation == "speed":
            tokenized_agent["_init_diffusion_ego_local_velocity"] = motion.clone()
            motion = torch.linalg.vector_norm(motion, dim=-1, keepdim=True)

        local_pos, local_heading = transform_to_local(
            agent_pos,
            agent_head,
            batch_ego_pos,
            batch_ego_heading,
        )

        heading_vec = torch.stack(
            [local_heading.cos(), local_heading.sin()],
            dim=-1,
        )

        m_init = torch.cat(
            [
                local_pos,
                heading_vec,
                shape[:, :2],
                motion,
            ],
            dim=-1,
        )

        m_init = self._encode_input_state(m_init, tokenized_agent)
        diff_input = m_init
        diff_output = m_init

        self._maybe_init_normalizer(diff_output, tokenized_agent.get("_init_diffusion_size_valid_mask"))

        return diff_input, diff_output


    def _format_beta(self, beta: torch.Tensor, n_agent: int) -> torch.Tensor:
        if beta.ndim == 3:
            beta = beta[:, 0]
        elif beta.ndim == 1:
            beta = beta[:, None]

        if beta.shape[0] != n_agent:
            raise ValueError(
                f"beta first dim must match agents: beta={tuple(beta.shape)}, N={n_agent}."
            )

        if beta.shape[-1] == 1:
            beta = beta.expand(-1, self.m_delta_dim)

        return beta

    def _embed_time(self, beta: torch.Tensor, n_agent: int) -> torch.Tensor:
        """Embed flow time (possibly expanded over state dimensions)."""
        if self.time_embedding_type == "scenario_dreamer":
            if (beta.ndim not in (1, 2, 3)
                    or (beta.ndim == 3 and beta.shape[1] != 1)
                    or (beta.ndim > 1 and beta.shape[-1] not in (1, self.m_delta_dim))):
                raise ValueError("Scenario Dreamer time requires one scalar per agent, optionally expanded over state dimensions")
            time = self._format_beta(beta, n_agent)
            # Scalar Flow times expand with stride 0, so the usual train/sample
            # path needs no GPU synchronization to check repeated columns.
            scalar = time[:, :1]
            if time.stride(-1) != 0 and not torch.equal(time, scalar.expand_as(time)):
                raise ValueError("Scenario Dreamer time requires one scalar per agent; per-state timesteps must agree")
            # Flow t=0 is clean and t=1 is noise. The released SD model uses
            # indices 0..99; retain fractional indices for continuous Flow.
            timestep = scalar[:, 0] * self.time_embedding_scale
            features = self.noise_embedding.timestep_embedding(
                timestep, self.noise_embedding.frequency_embedding_size
            )
            # SD computes frequency features in float32. Match the MLP dtype
            # for explicitly converted models while retaining autocast support.
            features = features.to(dtype=self.noise_embedding.mlp[0].weight.dtype)
            return self.noise_embedding.mlp(features)
        time = self._format_beta(1.0 - beta, n_agent)
        phase = time * self._time_frequencies.to(time)
        features = torch.where(self._time_cos_mask, phase.cos(), phase.sin())
        return self.noise_embedding(features)

    @staticmethod
    def _scene_node_counts(node_batch: torch.Tensor, num_graphs: int,
                           max_count: int, name: str) -> torch.Tensor:
        if (not torch.is_tensor(node_batch) or node_batch.ndim != 1
                or node_batch.dtype not in (torch.int8, torch.int16, torch.int32,
                                           torch.int64, torch.uint8)):
            raise ValueError(f"{name} batch must be a 1D integer tensor")
        if num_graphs < 0:
            raise ValueError("num_graphs must be nonnegative")
        if node_batch.numel() and (node_batch.min() < 0 or node_batch.max() >= num_graphs):
            raise ValueError(f"{name} batch IDs must be in [0, num_graphs)")
        counts = torch.bincount(node_batch.long(), minlength=num_graphs)
        if (counts > max_count).any():
            raise ValueError(
                f"{name} count exceeds the embedding limit {max_count}; "
                f"increase count_max_num_{name} for both training and evaluation"
            )
        return counts

    def _embed_scene_counts(self, tokenized_agent, map_feature):
        """Return scene count embeddings using one fixed lane-count definition."""
        num_graphs = int(tokenized_agent["num_graphs"])
        # Use the complete conditioned scene, including ego and agents excluded
        # by eval_mask, rather than the current forward's subset of agent rows.
        agent_counts = self._scene_node_counts(
            tokenized_agent["batch"], num_graphs, self.count_max_num_agents, "agents"
        )
        if self.count_lane_source == "scenario_dreamer":
            sd_map = tokenized_agent.get("sd_map")
            if sd_map is None or "batch" not in sd_map:
                raise ValueError(
                    "count_lane_source=scenario_dreamer requires sd_map lane batch metadata; "
                    "use map_tokens consistently for token-only training caches"
                )
            lane_batch = sd_map["batch"]
        else:
            lane_batch = map_feature["batch"]
        lane_counts = self._scene_node_counts(
            lane_batch, num_graphs, self.count_max_num_lanes, "lanes"
        )
        return (self.num_agents_embedder(agent_counts, train=self.training),
                self.num_lanes_embedder(lane_counts, train=self.training))

    def _embed_scene_type(self, tokenized_agent):
        """Embed explicit SD graph/map categories once for all scene nodes."""
        num_graphs = int(tokenized_agent["num_graphs"])
        device = tokenized_agent["batch"].device
        if self.map_id_source == "metadata":
            ids, valid, _ = read_vectorworld_map_metadata(tokenized_agent, num_graphs)
            if not bool(valid.all()):
                missing = (~valid).nonzero(as_tuple=True)[0].tolist()
                raise ValueError(
                    f"map_id_source=metadata requires valid map_id labels for every scene; "
                    f"missing scenes {missing[:5]}. Use map_id_source=fixed explicitly for unlabeled caches"
                )
            ids = ids.to(device=device)
        else:
            # This is a configured category, not an inferred missing-label value.
            ids = torch.full((num_graphs,), self.map_id, device=device, dtype=torch.long)
        if self.map_lg_type is None:
            sd_map = tokenized_agent.get("sd_map")
            kinds = sd_map.get("lg_type") if sd_map is not None else None
            if kinds is None:
                kinds = tokenized_agent.get("lg_type")
            if kinds is None:
                raise ValueError("map_lg_type=null requires per-scene lg_type metadata")
            kinds = torch.as_tensor(kinds, device=device).reshape(-1)
            if kinds.numel() != num_graphs or not bool(((kinds == 0) | (kinds == 1)).all()):
                raise ValueError("lg_type metadata must provide one 0/1 label per scene")
            kinds = kinds.long()
        else:
            kinds = torch.full((num_graphs,), self.map_lg_type, device=device, dtype=torch.long)
        labels = 2 * kinds + ids
        # One call gives one dropout decision per scene, shared by agent/map
        # nodes; calling the embedder separately would produce different masks.
        return self.scene_type_embedder(labels, train=self.training)

    def _ego_context_embedding(
        self,
        pos_s: torch.Tensor,
        theta: torch.Tensor,
        batch: torch.Tensor,
        tokenized_agent,
    ) -> torch.Tensor:

        ego_feat = tokenized_agent["ego_feat"]

        ego_pose = ego_feat[:, :-3]
        if self.generate_type:
            counts = torch.bincount(tokenized_agent["batch"].long(), minlength=len(ego_feat)).to(ego_feat)
            type_count = torch.stack((counts, torch.zeros_like(counts), torch.zeros_like(counts)), -1)[batch]
        else:
            type_count = ego_feat[:, -3:][batch]

        # [num_graphs, 3, 3] -> [N_agent, 3, 3]
        # Last dim is expected to be [x, y, heading].
        ego_pose = ego_pose.reshape(-1, 3, 3)
        all_pos = ego_pose[:, :, :2][batch]
        all_head = ego_pose[:, :, 2][batch]

        local_ego_pos, local_ego_head = transform_to_local(
            all_pos,
            all_head,
            pos_s,
            theta,
        )

        local_ego_head=wrap_angle(local_ego_head)

        ego_features = torch.cat(
            [
                local_ego_pos.flatten(1, 2),
                local_ego_head,#.cos(),
             #   local_ego_head.sin(),
                type_count,
            ],
            dim=-1,
        )

        return self.ego_embed(ego_features)

    def _original_state_embedding(
        self,
        m_delta: torch.Tensor,
        beta: torch.Tensor,
        agent_type_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Embed state, time and type, retaining the noisy heading magnitude.

        The x0 predictor encodes position and heading angle through geometric
        relations. Its continuous state projection also needs heading magnitude:
        unlike a clean heading, the linearly noised pair need not have unit norm.
        """
        if self.x_pred:
            feat_a = self.proj_in_m_delta(m_delta[:, 4:])
        else:
            feat_a = self.proj_in_m_delta(m_delta)
        feat_a = feat_a + self._embed_time(beta, m_delta.shape[0])
        feat_a = feat_a + agent_type_embed
        return feat_a

    def _embed_agents(
        self,
        m_delta: torch.Tensor,
        beta: torch.Tensor,
        agent_type: torch.Tensor,
        batch: torch.Tensor,
        tokenized_agent,
        mode: int,
        type_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        theta = torch.atan2(m_delta[:, 3], m_delta[:, 2])
        pos_s = m_delta[:, :2]

        if not self.generate_type and self.label_drop_prob > 0:
            if self.training and mode == 1:
                drop = torch.rand(agent_type.shape[0], device=agent_type.device) < self.label_drop_prob
                agent_type =torch.where(drop, torch.full_like(agent_type, self.num_classes), agent_type)
                #ego_embedding=torch.where(drop[:, None], torch.full_like(ego_embedding, 0), ego_embedding)
            elif mode == 0:
                agent_type = torch.full_like(agent_type, self.num_classes)

        # if "agent_type_embed"  in tokenized_agent and not self.training:
        #     agent_type_embed=tokenized_agent["agent_type_embed"]
        # else:
        if self.generate_type:
            if type_state is None or type_state.shape != (len(m_delta), self.num_classes):
                raise ValueError("generate_type requires an [N, 3] noisy type state")
            agent_type_embed = type_state.to(self.type_a_emb.weight) @ self.type_a_emb.weight
        else:
            agent_type_embed=self.type_a_emb(agent_type)
        #tokenized_agent["agent_type_embed"]=agent_type_embed

        if self.time_embedding_type == "legacy":
            beta = self._format_beta(beta, m_delta.shape[0])
        # SD validates the original scalar layout before any dimensions are
        # removed; malformed grouped times must not be silently truncated.

        feat_a = self._original_state_embedding(
            m_delta=m_delta,
            beta=beta,
            agent_type_embed=agent_type_embed,
        )

        ego_embedding = self._ego_context_embedding(
            pos_s=pos_s,
            theta=theta,
            batch=batch,
            tokenized_agent=tokenized_agent,
        )

        feat_a = feat_a + ego_embedding

        return feat_a, pos_s, theta

    # ---------------------------------------------------------------------
    # Graph denoising
    # ---------------------------------------------------------------------
    def _apply_graph_attention(
            self,
            feat_a: torch.Tensor,
            pos_s: torch.Tensor,
            theta: torch.Tensor,
            batch: torch.Tensor,
            tokenized_agent,
            map_feature: Mapping[str, torch.Tensor],
            num_graphs: int,
            use_map_condition: bool = True,
    ) -> torch.Tensor:
        head_vector_s = torch.stack(
            [theta.cos(), theta.sin()],
            dim=-1,
        )

        edge_index_a2a, r_a2a, *_ = self.edge_encoder.build_interaction_edge(
            pos_s=pos_s,
            head_s=theta,
            head_vector_s=head_vector_s,
            batch_s=batch,
            mask=None,
            max_radius=300,
            max_num_neighbors=30,
            agent_train_mask=None,
            layer_num=self.num_layers,
        )

        if use_map_condition:
            batch_pl = map_feature["batch"]
            pos_pl = map_feature["position"]
            orient_pl = map_feature["orientation"]
            feat_map = map_feature["pt_token"]

            # A missing final map scene does not imply a temporal batch.
            # Only use the legacy temporal alignment with explicit metadata.
            temporal_map = "repeat_batch" in tokenized_agent or "agent_valid" in tokenized_agent
            if temporal_map and batch_pl.numel() > 0 and int(batch_pl.max().item()) != num_graphs - 1:
                if "agent_valid" not in tokenized_agent:
                    batch_for_map = tokenized_agent["repeat_batch"]
                    n_step = batch_for_map.shape[1]

                    pos_for_map = pos_s.reshape(n_step, -1, 2).transpose(0, 1)
                    theta_for_map = theta.reshape(n_step, -1).transpose(0, 1)
                    mask_for_map = torch.ones_like(batch_for_map, dtype=torch.bool)
                else:
                    valid = tokenized_agent["agent_valid"]
                    n_step = valid.shape[0]

                    pos_global, theta_global = transform_to_global(
                        pos_s,
                        theta,
                        tokenized_agent["batch_ego_pos"],
                        tokenized_agent["batch_ego_heading"],
                    )

                    pos_b = torch.zeros(
                        [valid.shape[0], valid.shape[1], 2],
                        device=pos_s.device,
                        dtype=pos_s.dtype,
                    )
                    theta_b = torch.zeros(
                        [valid.shape[0], valid.shape[1]],
                        device=theta.device,
                        dtype=theta.dtype,
                    )

                    pos_b[valid] = pos_global
                    theta_b[valid] = theta_global

                    pos_for_map = pos_b.transpose(0, 1)
                    theta_for_map = theta_b.transpose(0, 1)
                    mask_for_map = valid.transpose(0, 1)
                    batch_for_map = tokenized_agent["batch_a"].unsqueeze(1).repeat(
                        1,
                        n_step,
                    )
            else:
                pos_for_map = pos_s
                theta_for_map = theta
                mask_for_map = None
                batch_for_map = batch

            head_vector_for_map = torch.stack(
                [theta_for_map.cos(), theta_for_map.sin()],
                dim=-1,
            )

            edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                pos_pl=pos_pl,
                orient_pl=orient_pl,
                pos_a=pos_for_map,
                head_a=theta_for_map,
                head_vector_a=head_vector_for_map,
                mask=mask_for_map,
                batch_s=batch_for_map,
                batch_pl=batch_pl,
                pl2a_radius=300,
                max_num_neighbors=30,
                agent_train_mask=None,
                layer_num=self.num_layers,
            )

        for layer_i in range(self.num_layers):
            feat_a = self.a2a_attn_layers[layer_i](
                feat_a,
                r_a2a,
                edge_index_a2a,
            )

            if use_map_condition:
                feat_a = self.pt2a_attn_layers[layer_i](
                    (feat_map, feat_a),
                    r_pl2a,
                    edge_index_pl2a,
                )

        if self.generate_type:
            tokenized_agent["_init_diffusion_type_logits"] = self.to_out_type(feat_a)
        state = self.to_out_m_delta(feat_a)
        if self.heading_velocity:
            return torch.cat((state, self.to_out_heading_velocity(feat_a)), dim=-1)
        return state

    def output_transform(self,res,cur_pos,cur_theta):
        # Flow reconstructs heading from theta_t - t * omega. Do not evaluate
        # atan2 on the unused clean-heading output in angular-velocity mode.
        res_theta = (torch.zeros_like(cur_theta) if self.heading_velocity
                     else torch.atan2(res[:, 3], res[:, 2]))

        local_pos, local_theta = transform_to_global(
            res[:, :2],
            res_theta,
            cur_pos,
            cur_theta,
        )

        res = torch.cat(
            [
                local_pos,
                torch.cos(local_theta)[:, None],
                torch.sin(local_theta)[:, None],
                res[:, 4:],
            ],
            dim=-1,
        )
        return res

    def forward(
        self,
        m_delta: torch.Tensor,
        beta: torch.Tensor,
        tokenized_agent,
        map_feature: Mapping[str, torch.Tensor],
        eval_mask: torch.Tensor=None,
        mode: int = 1,
        use_map_condition: Optional[bool] = True,
    ) -> torch.Tensor:
        m_delta = m_delta.reshape(m_delta.shape[0], -1)

        batch = tokenized_agent["batch"]
        agent_type = tokenized_agent["type"]
        type_state = tokenized_agent.get("_init_diffusion_type_state") if self.generate_type else None
        num_graphs = tokenized_agent["num_graphs"]

        if eval_mask is not None:
            m_delta = m_delta[eval_mask]
            beta = beta[eval_mask]
            batch = batch[eval_mask]
            agent_type = agent_type[eval_mask]
            if type_state is not None:
                type_state = type_state[eval_mask]

        feat_a, pos_s, theta = self._embed_agents(
            m_delta=m_delta,
            beta=beta,
            agent_type=agent_type,
            batch=batch,
            tokenized_agent=tokenized_agent,
            mode=mode,
            type_state=type_state,
        )

        if self.count_embedding_type == "scenario_dreamer":
            agent_count_embedding, lane_count_embedding = self._embed_scene_counts(
                tokenized_agent, map_feature
            )
            # Match SD's per-node-type routing: agent count conditions agents,
            # lane count conditions map nodes, then reaches agents via L2A.
            feat_a = feat_a + agent_count_embedding[batch]
            map_feature = dict(
                map_feature,
                pt_token=map_feature["pt_token"] + lane_count_embedding[map_feature["batch"]],
            )
            # Do not modify cached map tokens: every denoising step starts from
            # the same context and adds the count embedding exactly once.

        if self.map_embedding_type == "scenario_dreamer":
            scene_embedding = self._embed_scene_type(tokenized_agent)
            feat_a = feat_a + scene_embedding[batch]
            map_feature = dict(
                map_feature,
                pt_token=map_feature["pt_token"] + scene_embedding[map_feature["batch"]],
            )

        if use_map_condition:
            if self.training and self.map_drop_prob > 0:
                use_map_condition = (
                        torch.rand((), device=m_delta.device) >= self.map_drop_prob
                )
            else:
                use_map_condition = True

        res = self._apply_graph_attention(
            feat_a=feat_a,
            pos_s=pos_s,
            theta=theta,
            batch=batch,
            tokenized_agent=tokenized_agent,
            map_feature=map_feature,
            num_graphs=num_graphs,
            use_map_condition=use_map_condition
        )

        if self.x_pred :
            res =self.output_transform(res, pos_s, theta)

        #ego_mask = tokenized_agent.get("ego_mask", None)
        #if not self.training and len(beta)==len(ego_mask): #and torch.all(beta[~ego_mask] == 0):
        tokenized_agent["noise_feat_cur"] = feat_a

        return res

    def get_output(self, pred_init: torch.Tensor, tokenized_agent):
        """Convert generated local all-agent initial state back to global fields."""
        if self.generate_type:
            types = tokenized_agent.get("_init_diffusion_generated_type")
            if types is None or types.shape != (len(pred_init),):
                raise ValueError("generate_type output requires types from Flow.sample")
            tokenized_agent["type"] = types
            shapes, all_tokens, final_tokens = self.token_processor._get_agent_tokens(types)
            tokenized_agent.update(token_agent_shape=shapes, token_traj_all=all_tokens,
                                   token_traj=final_tokens)
        batch_ego_pos = tokenized_agent["batch_ego_pos"]
        batch_ego_heading = tokenized_agent["batch_ego_heading"]

        pred_init = self.state_to_physical(pred_init)
        pred_trans = pred_init[..., :2]
        pred_head = pred_init[..., 2:4]
        pred_shape = pred_init[..., 4:6]
        if self.velocity_representation == "speed":
            if pred_init.shape[-1] != 7:
                raise ValueError("speed output requires a 7D generated state")
            # Noisy/clean regression values remain unconstrained in training.
            # Only the physical output is nonnegative and aligned with heading.
            speed = pred_init[..., 6:7].clamp_min(0.)
            pred_vel = torch.cat((speed, torch.zeros_like(speed)), dim=-1)
            ego_velocity = tokenized_agent.get("_init_diffusion_ego_local_velocity",
                                                tokenized_agent.get("local_vel"))
            if self.fix_ego and ego_velocity is not None and "ego_mask" in tokenized_agent:
                pred_vel = torch.where(tokenized_agent["ego_mask"].bool()[:, None],
                                       ego_velocity[:, :2].to(pred_vel), pred_vel)
        else:
            pred_vel = pred_init[..., 6:8]

        pred_heading = torch.atan2(pred_head[..., 1], pred_head[..., 0])

        global_pos, global_heading = transform_to_global(
            pred_trans,
            pred_heading,
            batch_ego_pos,
            batch_ego_heading,
        )

        gt_initial_pos = global_pos[:, None]
        gt_initial_heading = global_heading[:, None]

        #center_token_traj = tokenized_agent["token_traj"].mean(-2)
        # gt_initial_idx = torch.linalg.norm(
        #     center_token_traj - pred_vel[:, None] * 0.5,
        #     dim=-1,
        # ).argmin(-1)

        # token_traj: [N, K, 4, 2], endpoint contour.
        token_end_contour = tokenized_agent["token_traj"]

        token_dt = self.token_processor.shift * 0.1

        token_vel_current = self.token_processor.token_velocity_in_current_frame(
            token_end_contour,
            token_dt,
        )

        gt_initial_idx = torch.linalg.vector_norm(
            token_vel_current - pred_vel[:, None],
            dim=-1,
        ).argmin(dim=-1)

        #local_vel = center_token_traj[torch.arange(len(gt_initial_idx), device=gt_initial_idx.device), gt_initial_idx]

        global_vel= rotate_to_global(pred_vel,global_heading)

        return (
            gt_initial_pos,
            gt_initial_heading,
            pred_shape,
            global_vel,
            gt_initial_idx[:, None],
        )
