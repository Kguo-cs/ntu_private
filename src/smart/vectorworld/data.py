"""Use SMART's exact Scenario Dreamer graphs with VectorWorld's motion AE.

Static states, lane topology, frame transforms and row ordering are shared with
Scenario Dreamer. Motion is an additional body-frame history feature; an initial
state or a tokenized map alone cannot provide its training labels.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from src.smart.scenario_dreamer.data import build_graph as _build_sd_graph
from src.smart.scenario_dreamer.generation import build_generation_graph as _build_sd_generation_graph
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene, read_vectorworld_map_metadata


def normalize_motion(motion_raw, *, max_displacement=12.0):
    """Normalize native [x1,y1,...] body-frame history using the released AE.

    Physical zero history maps to [1,0,...,1,0], rather than normalized zero.
    """
    motion = torch.as_tensor(motion_raw, dtype=torch.float32)
    if motion.ndim != 2 or motion.shape[1] % 2 or not torch.isfinite(motion).all():
        raise ValueError("VectorWorld motion must be a finite [agents, 2 * points] tensor")
    if not np.isfinite(max_displacement) or max_displacement <= 0:
        raise ValueError("max_displacement must be finite and positive")
    result = motion.clone()
    result[:, 0::2] = (2 * motion[:, 0::2] / max_displacement + 1).clamp(-1, 1)
    result[:, 1::2] = (2 * motion[:, 1::2] / max_displacement).clamp(-1, 1)
    return result


def compute_motion_code(position, heading, velocity, valid_mask, scene_timestep, *,
                        num_points=6, history_max_m=12.0, d_static=0.5,
                        v_static=0.2, t_hist_max=8):
    """Compute the native arc-length sampled history from real SMART trajectories.

    Coordinates can be world or any rigid scene frame: output is in each current
    agent's body frame. Gaps use the native convention of connecting valid past
    observations. Returns physical motion, static labels and label availability.
    A real stationary/short-history observation is a valid zero-motion label;
    absence of the current observation is explicitly unavailable.
    """
    device = position.device if isinstance(position, torch.Tensor) else torch.device("cpu")
    if int(num_points) != num_points or num_points < 2:
        raise ValueError("num_points must be an integer >= 2")
    if history_max_m <= 0 or t_hist_max < 1:
        raise ValueError("history_max_m and t_hist_max must be positive")
    def array(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    pos, yaw, vel, exists = map(array, (position, heading, velocity, valid_mask))
    if pos.ndim != 3 or pos.shape[-1] < 2:
        raise ValueError("position must have shape [agents, timesteps, >=2]")
    n, t, _ = pos.shape
    if yaw.shape != (n, t) or vel.shape != (n, t, 2) or exists.shape != (n, t):
        raise ValueError("Inconsistent VectorWorld trajectory feature shapes")
    times = np.asarray(array(scene_timestep), dtype=np.int64).reshape(-1)
    if times.size == 1:
        times = np.repeat(times, n)
    if times.shape != (n,) or np.any(times < 0) or np.any(times >= t):
        raise ValueError("scene_timestep must select one available trajectory timestep per agent")
    exists = exists.astype(bool)
    # Invalid/padded positions are allowed; real history must always be finite.
    if not all(np.isfinite(x[exists]).all() for x in (pos[..., :2], yaw, vel)):
        raise ValueError("Non-finite valid VectorWorld trajectory observations")
    raw = np.zeros((n, 2 * int(num_points)), dtype=np.float32)
    static = np.ones(n, dtype=bool)
    available = exists[np.arange(n), times].copy()
    for i, current in enumerate(times):
        if not available[i]:
            continue
        history = np.flatnonzero(exists[i, :current + 1])
        recent = history[-t_hist_max:]
        displacement = np.linalg.norm(pos[i, recent, :2] - pos[i, current, :2], axis=-1)
        speed = np.linalg.norm(vel[i, recent], axis=-1)
        if len(history) < 2 or displacement.max() < d_static or speed.mean() < v_static:
            continue
        xy = pos[i, history, :2] - pos[i, current, :2]
        c, s = np.cos(yaw[i, current]), np.sin(yaw[i, current])
        xy = np.stack((c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]), -1)
        xy[-1] = 0
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))))
        if arc[-1] < 1e-3:
            continue
        start = int(np.searchsorted(arc, arc[-1] - history_max_m, side="left")) if arc[-1] > history_max_m else 0
        xy, arc = xy[start:], arc[start:] - arc[start]
        if arc[-1] < 1e-3:
            continue
        samples = np.linspace(0.0, float(arc[-1]), num=int(num_points), dtype=np.float32)
        points = np.stack((np.interp(samples, arc, xy[:, 0]), np.interp(samples, arc, xy[:, 1])), -1).astype(np.float32)
        points[-1] = 0
        raw[i] = points.reshape(-1)
        static[i] = False
    return tuple(torch.from_numpy(x).to(device) for x in (raw, static, available))


def _motion_features(agent, motion_dim, *, motion_missing, is_training):
    n = len(agent["initial_pos"])
    device = agent["initial_pos"].device
    if not isinstance(motion_dim, int) or motion_dim < 0 or motion_dim % 2:
        raise ValueError("motion_dim must be a nonnegative even integer")
    if motion_missing not in ("error", "static_masked"):
        raise ValueError("motion_missing must be error or static_masked")
    if motion_missing == "static_masked" and is_training:
        raise ValueError("static_masked is evaluation-only: the released AE cannot mask missing motion supervision")
    if motion_dim == 0:
        return (torch.empty((n, 0), device=device), torch.ones(n, device=device, dtype=torch.bool),
                torch.ones(n, device=device, dtype=torch.bool), "disabled")
    if "vectorworld_motion" in agent:
        motion = torch.as_tensor(agent["vectorworld_motion"], device=device, dtype=torch.float32).clone()
        source = "native_normalized"
    elif "vectorworld_motion_raw" in agent:
        motion = normalize_motion(torch.as_tensor(agent["vectorworld_motion_raw"], device=device))
        source = "native_raw"
    elif "vectorworld_history" in agent:
        raw, static, available = compute_motion_code(**agent["vectorworld_history"], num_points=motion_dim // 2)
        motion = normalize_motion(raw).to(device)
        source = "trajectory_history"
    else:
        motion = normalize_motion(torch.zeros((n, motion_dim), device=device))
        available = torch.zeros(n, device=device, dtype=torch.bool)
        source = "missing"
    if motion.shape != (n, motion_dim) or not torch.isfinite(motion).all():
        raise ValueError(f"VectorWorld motion must have finite shape {(n, motion_dim)}, got {tuple(motion.shape)}")
    if source not in ("trajectory_history", "missing"):
        available = torch.as_tensor(agent.get("vectorworld_motion_valid_mask", torch.ones(n)), device=device, dtype=torch.bool)
        static = torch.as_tensor(agent.get("vectorworld_motion_is_static", torch.zeros(n)), device=device, dtype=torch.bool).clone()
    elif source == "missing":
        static = torch.ones(n, device=device, dtype=torch.bool)
    available, static = available.to(device), static.to(device)
    if available.shape != (n,) or static.shape != (n,):
        raise ValueError("VectorWorld motion validity/static masks must have one element per agent")
    if not available.all():
        count = int((~available).sum())
        if motion_missing == "error":
            raise ValueError(f"VectorWorld motion_dim={motion_dim} requires real trajectory history or native agent_motion_raw; {count}/{n} agents have no motion labels. Initial-state-only SD/SMART caches cannot train the released motion AE.")
        # An explicit inference placeholder, never a motion reconstruction target.
        motion[~available] = normalize_motion(torch.zeros((count, motion_dim), device=device))
        static[~available] = True
        warnings.warn(f"VectorWorld evaluation uses stationary encoder placeholders for {count}/{n} agents without real motion; motion_valid_mask=False. This is not the native motion conditioning protocol.", RuntimeWarning, stacklevel=3)
        source += "+static_masked"
    return motion, static, available, source


def build_graph(agent, tokens, cfg, *, map_source="exact", motion_dim=12,
                motion_missing="error", mode="joint", is_training=True):
    """Return native AE graph, native-row -> SMART-row IDs, centers and angles.

    All lg_type values are supported for training. Lane-conditioned evaluation
    requires full, non-partitioned lanes. Partition masks retain AE semantics;
    the flow wrapper separately fixes all lane latents for lane conditioning.
    """
    if mode not in ("joint", "lane_conditioned"):
        raise ValueError("VectorWorld graph mode must be joint or lane_conditioned")
    motion, static, available, source = _motion_features(agent, motion_dim, motion_missing=motion_missing, is_training=is_training)
    # SD latent-cache posteriors have different dimensions and a different AE.
    # Build geometry from the cache, but always encode with VectorWorld's AE.
    inputs = {key: value for key, value in agent.items() if key != "sd_cached_posterior"}
    graph, rows, centers, angles = _build_sd_graph(inputs, tokens, cfg, map_source=map_source)
    map_ids, map_valid, map_sources = read_vectorworld_map_metadata(agent, int(agent["num_graphs"]))
    map_ids, map_valid = map_ids.to(graph.map_id.device), map_valid.to(graph.map_id.device)
    graph.map_id = torch.where(map_valid, map_ids, graph.map_id)
    fallback = f"fallback_config{int(agent.get('sd_map_id', 0))}"
    graph.vectorworld_map_sources = [source if bool(valid) else fallback
                                   for source, valid in zip(map_sources, map_valid.tolist())]
    unique_sources = set(graph.vectorworld_map_sources)
    graph.vectorworld_map_source = (next(iter(unique_sources)) if len(unique_sources) == 1
                                   else "mixed_metadata_and_config" if not bool(map_valid.all()) else "per_scene_metadata")
    if mode == "lane_conditioned" and not is_training and bool((graph.lg_type != 0).any()):
        raise ValueError("VectorWorld lane-conditioned evaluation requires lg_type=0 full (non-partitioned) lanes")
    # SD's graph builder already emits bool True for before-partition nodes.
    # VectorWorld BEFORE_PARTITION=1; do not invert these boolean masks.
    graph["agent"].partition_mask = graph["agent"].partition_mask.bool()
    graph["lane"].partition_mask = graph["lane"].partition_mask.bool()
    graph["agent"].motion = motion[rows]
    graph["agent"].motion_valid_mask = available[rows]
    graph["agent"].motion_is_static = static[rows]
    graph.vectorworld_motion_source = source
    counts = torch.zeros(int(agent["num_graphs"]), device=rows.device, dtype=torch.long)
    counts.scatter_add_(0, graph["agent"].batch, (~graph["agent"].partition_mask).long())
    graph.num_agents_after_origin = torch.where(graph.lg_type == 1, counts, torch.zeros_like(counts))
    return graph, rows, centers, angles


def build_generation_graph(counts, *, agent_latent_dim, lane_latent_dim, device, dtype):
    """Reuse the official count-prior graph without reference motion or geometry."""
    graph, rows, centers, angles = _build_sd_generation_graph(counts, agent_latent_dim=agent_latent_dim,
        lane_latent_dim=lane_latent_dim, device=device, dtype=dtype)
    graph["agent"].partition_mask = torch.zeros(len(graph["agent"].x), device=device, dtype=torch.bool)
    graph["lane"].partition_mask = torch.zeros(len(graph["lane"].x), device=device, dtype=torch.bool)
    graph.num_agents_after_origin = torch.zeros(len(counts), device=device, dtype=torch.long)
    graph.vectorworld_motion_source = "unconditioned_generation"
    graph.vectorworld_map_source = "official_count_prior"
    graph.vectorworld_map_sources = ["official_count_prior"] * len(counts)
    return graph, rows, centers, angles
