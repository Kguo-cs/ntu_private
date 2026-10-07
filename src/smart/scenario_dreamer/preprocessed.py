"""Read official Waymo AE pickles through SMART's existing data pipeline."""
from pathlib import Path

import torch


def read_vectorworld_map_metadata(scene, num_scenes=1):
    """Read Waymo category labels without replacing missing metadata with GT.

    Nocturne compatibility is the official Waymo LDM map category. Generic
    map_id is accepted when that field is unavailable. Internal validity masks
    let native scenes with and without labels share a PyG batch.
    """
    num_scenes = int(num_scenes)
    # HeteroData.__contains__ also sees node-store attributes, while [] would
    # create an empty node store for such a global lookup. Inspect global/node
    # stores explicitly to avoid losing nested labels or mutating the batch.
    top = scene._global_store if hasattr(scene, "_global_store") else scene
    containers = [(top, "metadata")]
    for key in ("scenario_dreamer", "scene_metadata", "metadata"):
        if key in top:
            nested = top[key]
        elif key in getattr(scene, "node_types", ()):
            nested = scene[key]
        else:
            continue
        if hasattr(nested, "keys"):
            containers.append((nested, f"metadata_{key}"))
    for key in ("vectorworld_map_id", "nocturne_compatible", "map_id"):
        for container, prefix in containers:
            if key not in container:
                continue
            labels = torch.as_tensor(container[key]).reshape(-1)
            if labels.numel() != num_scenes or not bool(((labels == 0) | (labels == 1)).all()):
                raise ValueError(f"VectorWorld {key} must provide one 0/1 label per scene")
            valid = torch.ones(num_scenes, dtype=torch.bool, device=labels.device)
            if key == "vectorworld_map_id" and "vectorworld_map_valid_mask" in container:
                valid = torch.as_tensor(container["vectorworld_map_valid_mask"], device=labels.device).reshape(-1)
                if valid.shape != labels.shape or not bool(((valid == 0) | (valid == 1)).all()):
                    raise ValueError("VectorWorld map validity must provide one boolean per scene")
                valid = valid.bool()
            sources = [f"{prefix}_{key}"] * num_scenes
            if key == "vectorworld_map_id" and "vectorworld_map_source" in container:
                source = container["vectorworld_map_source"]
                sources = [source] * num_scenes if isinstance(source, str) else list(source)
                if len(sources) != num_scenes or any(not isinstance(x, str) for x in sources):
                    raise ValueError("VectorWorld map sources must align with scenes")
            return labels.long(), valid, sources
    return torch.zeros(num_scenes, dtype=torch.long), torch.zeros(num_scenes, dtype=torch.bool), ["missing"] * num_scenes


def adapt_preprocessed_scene(scene, filename):
    """Keep physical SD-local states and explicit map edges; ego becomes last.

    The official pickle has one snapshot, not a world-frame trajectory. Identity
    transforms keep the decoder in the saved ego-+Y coordinate system.
    """
    states = torch.as_tensor(scene["agent_states"], dtype=torch.float64)
    types = torch.as_tensor(scene["agent_types"])
    lanes = torch.as_tensor(scene["road_points"], dtype=torch.float64)
    edges = torch.as_tensor(scene["edge_index_lane_to_lane"], dtype=torch.long)
    relations = torch.as_tensor(scene["road_connection_types"])
    n, l = int(scene["num_agents"]), int(scene["num_lanes"])
    kind = int(scene["lg_type"])
    if n < 1 or l < 1 or states.shape != (n, 7) or types.shape != (n, 3):
        raise ValueError(f"Malformed official agent states/counts in {filename}")
    if lanes.shape != (l, 20, 2) or edges.shape != (2, l * l) or relations.shape != (l * l, 6):
        raise ValueError(f"Malformed official lane graph in {filename}")
    if kind not in (0, 1) or edges.min() < 0 or edges.max() >= l:
        raise ValueError(f"Invalid official graph type/indices in {filename}")
    if not all(torch.isfinite(x).all() for x in (states, types, lanes, relations)):
        raise ValueError(f"Non-finite official features in {filename}")
    order = torch.cat((torch.arange(1, n), torch.zeros(1, dtype=torch.long)))
    timestep = int(scene["scene_timestep"])
    result = {
        "scenario_id": Path(filename).stem,
        "scenario_dreamer_cache_file": Path(filename).name,
        "scene_timestep": timestep,
        "generation_scene_timestep": timestep,
        "sd_center_world": torch.zeros(1, 2, dtype=torch.float64),
        "sd_rotation_angle": torch.zeros(1, dtype=torch.float64),
        "sd_lg_type": kind,
        "sd_agent": {"x": states[order], "type": types.argmax(-1)[order], "num_nodes": n},
        "sd_lane": {"x": lanes, "num_nodes": l},
        ("sd_lane", "to", "sd_lane"): {"edge_index": edges, "type": relations.argmax(-1)},
    }
    map_ids, map_valid, map_sources = read_vectorworld_map_metadata(scene)
    result["vectorworld_map_id"] = map_ids
    result["vectorworld_map_valid_mask"] = map_valid
    result["vectorworld_map_source"] = map_sources[0]
    # VectorWorld's native snapshot schema extends SD with real motion labels.
    # Preserve them in the same ego-last order as states for downstream models.
    if "agent_motion_raw" in scene:
        motion = torch.as_tensor(scene["agent_motion_raw"], dtype=torch.float32)
        available = torch.as_tensor(scene.get("agent_motion_valid_mask", torch.ones(n)), dtype=torch.bool)
        static = torch.as_tensor(scene.get("agent_motion_is_static", torch.zeros(n)), dtype=torch.bool)
        if (motion.ndim != 2 or motion.shape[0] != n or motion.shape[1] % 2
                or available.shape != (n,) or static.shape != (n,)
                or not torch.isfinite(motion).all()):
            raise ValueError(f"Malformed native motion features in {filename}")
        result["sd_agent"].update(motion_raw=motion[order], motion_valid_mask=available[order],
                                  motion_is_static=static[order])
    return result


def tokenize_preprocessed_agents(data, processor):
    """Supply initial states without map tokenization or invented GT motion."""
    if not (processor.scenario_dreamer_init and processor.pred_init):
        raise ValueError("Official AE samples require pred_init=true and scenario_dreamer_init=true")
    raw = data["sd_agent"]
    state = raw.x.float()
    heading = torch.atan2(state[:, 4], state[:, 3])
    ego = processor._make_ego_mask(raw.batch)
    agent_type = raw.type.long()
    shapes, all_tokens, final_tokens = processor._get_agent_tokens(agent_type)
    agent = {
        "batch": raw.batch, "type": agent_type, "ego_mask": ego,
        "initial_pos": state[:, :2], "initial_heading": heading,
        "local_vel": torch.stack((state[:, 2], torch.zeros_like(state[:, 2])), -1),
        "shape": state[:, 5:7], "sd_states": raw.x,
        "ego_pos2": state[ego, None, :2].expand(-1, 3, -1),
        "ego_heading2": heading[ego, None].expand(-1, 3),
        "token_agent_shape": shapes, "token_traj": final_tokens, "token_traj_all": all_tokens,
        "initial_scene_only": True,
    }
    if "posterior_mu" in raw:
        agent["sd_cached_posterior"] = {
            "encoder_fingerprint": data["sd_latent_cache_fingerprint"],
            "agent_mu": raw.posterior_mu, "agent_log_var": raw.posterior_log_var,
            "lane_mu": data["sd_lane"].posterior_mu,
            "lane_log_var": data["sd_lane"].posterior_log_var,
        }
    return agent
