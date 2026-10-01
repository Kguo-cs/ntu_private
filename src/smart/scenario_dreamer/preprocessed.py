"""Read official Waymo AE pickles through SMART's existing data pipeline."""
from pathlib import Path

import torch


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
    return {
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
