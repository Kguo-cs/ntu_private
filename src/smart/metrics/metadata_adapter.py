"""Keep only batch-safe SD metadata in a SMART/PyG training sample."""
from __future__ import annotations
import torch

from src.smart.scenario_dreamer.preprocessed import read_vectorworld_map_metadata


def attach_sd_metric_metadata(data, saved_sample, *, generation_scene_timestep=None):
    """Call in Dataset.__getitem__ after loading the rebuilt .pt file.

    `saved_sample` is the original dictionary written by the rebuild script.
    Dense variable-size lane adjacency matrices in its nested scenario_dreamer
    dict are not needed in a PyG batch. GT maps/features come from official cache.

    Only pass generation_scene_timestep AFTER the model input pipeline has
    actually aligned its initial state/conditioning to that raw timestep.
    Copying a label alone does not align trajectories or model conditioning.
    """
    info = saved_sample.get("scenario_dreamer")
    if info is None:
        raise KeyError("Rebuild the .pt samples with --save-scene-info.")
    # Keep category labels before removing the nested dense lane graph. Missing
    # labels have an explicit false validity mask, so their placeholder 0 cannot
    # silently become the official map category in a mixed PyG batch.
    map_ids, map_valid, map_sources = read_vectorworld_map_metadata(saved_sample, 1)
    data["vectorworld_map_id"] = map_ids
    data["vectorworld_map_valid_mask"] = map_valid
    data["vectorworld_map_source"] = map_sources[0]
    data["scenario_dreamer_cache_file"] = saved_sample["scenario_dreamer_cache_file"]
    data["scene_timestep"] = int(saved_sample["scene_timestep"])
    data["sd_center_world"] = torch.as_tensor(info["center_world"], dtype=torch.float64).reshape(1, 2)
    data["sd_rotation_angle"] = torch.as_tensor(info["rotation_angle"], dtype=torch.float64).reshape(1)
    if generation_scene_timestep is not None:
        value = int(generation_scene_timestep)
        if value != data["scene_timestep"]:
            raise ValueError("Model input frame does not equal the official reference frame.")
        data["generation_scene_timestep"] = value
    # Avoid trying to concatenate variable-size dense [L,L] graphs in PyG.
    if "scenario_dreamer" in data or "scenario_dreamer" in getattr(data, "node_types", ()):
        del data["scenario_dreamer"]
    return data
