"""Scenario Dreamer-compatible vehicle metrics with persistent real statistics.

The numerical helpers are sibling modules. No external repository or AST loading.
Before validating, run precompute_real.py once on the official test cache.
World-coordinate SMART predictions / ego-last rows are left unchanged; only the
chosen metric snapshot is converted, using the saved reference transform.
"""
from __future__ import annotations
import copy
import pickle
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

if __package__:
    from .official_backend import load_official_backend
    from .metric_core import SPECS, DistributionAccumulator, finalize_metrics, validate_unified
    from .real_cache import CachedReferenceStore, prepare_real_cache, DEFAULT_CACHE_NAME
else:
    from official_backend import load_official_backend
    from metric_core import SPECS, DistributionAccumulator, finalize_metrics, validate_unified
    from real_cache import CachedReferenceStore, prepare_real_cache, DEFAULT_CACHE_NAME

def as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def scalar(value: Any, name: str) -> int:
    array = as_numpy(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    item = array.item()
    if isinstance(item, bool) or not np.isfinite(item) or int(item) != item:
        raise ValueError(f"{name} must be an integer, got {item!r}")
    return int(item)

def field_of(data: Any, name: str, default: Any = None) -> Any:
    try:
        return data[name] if name in data else default
    except TypeError:
        return getattr(data, name, default)

def _metric_metadata_field(data, name):
    """Read a field without creating a missing HeteroData node store."""
    if isinstance(data, Mapping):
        return data.get(name)
    global_store = getattr(data, "_global_store", None)
    if global_store is not None:
        if name in global_store:
            return global_store[name]
        # Older samples may store scenario_dreamer as a node-like container.
        return getattr(data, "_node_store_dict", {}).get(name)
    store = getattr(data, "_store", None)
    if store is not None:
        return store.get(name)
    return field_of(data, name)

def _split_metric_metadata(value, name, count, width=1):
    """Split fixed-width graph metadata; reject ambiguous/malformed layouts.

    Scalar fields: [B] or [B,1]. XY centers: [B,2], [B,1,2], or
    concatenated [2*B]. Values and dtypes are preserved, not normalized.
    None means the field is absent; the evaluator checks required fields.
    """
    if value is None:
        return [None] * count
    if name == "scenario_dreamer_cache_file":
        if isinstance(value, (str, Path)):
            if count != 1:
                raise ValueError(f"{name}: one filename cannot describe {count} scenes")
            return [str(value)]
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != count:
            raise ValueError(f"{name} must contain exactly {count} filenames")
        result = []
        for item in value:
            # An unbatched PyG example sometimes retains a one-element list.
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 1:
                item = item[0]
            if not isinstance(item, (str, Path)):
                raise TypeError(f"{name}: expected a filename, got {type(item).__name__}")
            result.append(str(item))
        return result

    if isinstance(value, (list, tuple)) and any(torch.is_tensor(item) for item in value):
        raise ValueError(
            f"{name}: a list of tensors has ambiguous batch/coordinate axes. "
            "Store each sample's metadata as a Tensor before PyG collation "
            "using metadata_adapter.attach_sd_metric_metadata."
        )
    array = as_numpy(value)
    allowed = {(count * width,), (count, width), (count, 1, width)}
    if count == 1 and width == 1:
        allowed.add(())
    if tuple(array.shape) not in allowed:
        raise ValueError(
            f"{name}: got shape {array.shape}; expected {width} value(s) per "
            f"scene for B={count}. Use metadata_adapter.attach_sd_metric_metadata "
            "before batching; do not broadcast one scene's transform to a batch."
        )
    rows = array.reshape(count, width)
    return [row.copy() if width != 1 else row[0] for row in rows]

def _scene_records(data: Any, count: int) -> list[dict[str, Any]]:
    """Read ONLY the per-scene metadata consumed by ScenarioDreamerEvaluator.

    Model forward passes may add node/edge stores or change tensor lengths.
    Full Batch.to_data_list() would use stale collation slices for those stores.
    We instead split the explicit fixed-width metadata contract and leave all
    model stores, their tensors and PyG's internal slice/inc dictionaries alone.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(data, (list, tuple)):
        if len(data) != count:
            raise ValueError("Scene-list length disagrees with generated batch IDs")
        return [_scene_records(item, 1)[0] for item in data]

    num_graphs = getattr(data, "num_graphs", None)
    if num_graphs is not None and int(num_graphs) != count:
        raise ValueError(
            f"PyG scene count {int(num_graphs)} disagrees with generated batch count {count}"
        )

    records = [{} for _ in range(count)]
    fields = (
        ("scenario_dreamer_cache_file", 1),
        ("scene_timestep", 1),
        ("generation_scene_timestep", 1),
        ("sd_center_world", 2),
        ("sd_rotation_angle", 1),
    )
    for name, width in fields:
        values = _split_metric_metadata(_metric_metadata_field(data, name), name, count, width)
        for record, value in zip(records, values):
            if value is not None:
                record[name] = value

    # Legacy fallback: read only the transform and reference time, never the
    # nested road graphs, agent tensors, source_index arrays, or adjacencies.
    info = _metric_metadata_field(data, "scenario_dreamer")
    if info is not None:
        if isinstance(info, (list, tuple)) and len(info) != count:
            raise ValueError("scenario_dreamer metadata list must have one item per scene")
        for name, width in (("center_world", 2), ("rotation_angle", 1), ("scene_timestep", 1)):
            if isinstance(info, (list, tuple)):
                values = [
                    _split_metric_metadata(_metric_metadata_field(item, name), name, 1, width)[0]
                    for item in info
                ]
            else:
                values = _split_metric_metadata(_metric_metadata_field(info, name), name, count, width)
            for record, value in zip(records, values):
                if value is not None:
                    record.setdefault("scenario_dreamer", {})[name] = value
    return records

def _generated_arrays(out: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    pos, head, size = (as_numpy(out[k]) for k in ("traj", "head", "size"))
    vel = out.get("vel")
    if vel is None:
        raise ValueError("Instantaneous velocity is required; do not substitute finite-difference trajectory speed.")
    if isinstance(vel, (list, tuple)):
        vel = np.stack([as_numpy(v) for v in vel], axis=1)
    else:
        vel = as_numpy(vel)
    if pos.ndim != 4 or pos.shape[-1] not in (2, 3):
        raise ValueError(f"traj must be [N,R,T,2/3], got {pos.shape}")
    n, r, t, _ = pos.shape
    if head.shape != (n, r, t):
        raise ValueError(f"head must be {(n,r,t)}, got {head.shape}")
    if size.ndim != 4 or size.shape[:3] != (n, r, t) or size.shape[-1] < 2:
        raise ValueError(f"size must be [N,R,T,2/3], got {size.shape}")
    if vel.shape not in ((n, r, 2), (n, r, t, 2)):
        raise ValueError(f"vel must be [N,R,2] or [N,R,T,2], got {vel.shape}")
    return pos, head, size, vel

def make_generated_scene(out, batch, types, graph_index, timestep, record, gt,
                         *, prediction_frame="world") -> tuple[dict, np.ndarray, np.ndarray]:
    pos, head, size, vel = _generated_arrays(out)
    if pos.shape[1] != 1:
        raise ValueError("Use exactly one rollout per official entry: n_rollout_closed_val=1.")
    if not 0 <= timestep < pos.shape[2]:
        raise IndexError(f"sd_gen_timestep={timestep} outside generated T={pos.shape[2]}")
    mask = batch == graph_index
    xy = pos[mask, 0, timestep, :2].astype(np.float64)
    yaw = head[mask, 0, timestep].astype(np.float64)
    # Initial velocity [N,R,2] and per-timestep velocity [N,R,T,2] are distinct.
    velocity = vel[mask, 0] if vel.ndim == 3 else vel[mask, 0, timestep]
    speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
    dimensions = size[mask, 0, timestep, :2]
    local_types = types[mask]
    if prediction_frame == "world":
        info = field_of(record, "scenario_dreamer")
        center_value = field_of(record, "sd_center_world")
        angle_value = field_of(record, "sd_rotation_angle")
        if center_value is None and info is not None:
            center_value = info.get("center_world")
            angle_value = info.get("rotation_angle")
        if center_value is None or angle_value is None:
            raise ValueError(
                "World-coordinate predictions require scenario_dreamer.center_world and "
                "rotation_angle. Rebuild with --save-scene-info and preserve that metadata "
                "through the dataset loader. Do not infer the transform from float32 GT positions."
            )
        center = as_numpy(center_value).reshape(2).astype(np.float64)
        angle = float(as_numpy(angle_value).reshape(()))
        offset = xy - center
        c, s = np.cos(angle), np.sin(angle)
        xy = np.column_stack((c * offset[:, 0] - s * offset[:, 1],
                              s * offset[:, 0] + c * offset[:, 1]))
        yaw = yaw + angle
    elif prediction_frame != "sd_local":
        raise ValueError("sd_prediction_frame must be 'world' or 'sd_local' (ego +Y).")
    states = np.column_stack((xy, speed, np.cos(yaw), np.sin(yaw), dimensions))
    # Do not reapply GT validity, TrafficGen filtering, FOV clipping or off-road
    # deletion to generated agents. Optional model-emitted existence is allowed.
    existence = out.get("initial_valid")
    if existence is not None:
        existence = as_numpy(existence)
        if existence.shape == (len(batch), 1):
            keep = existence[mask, 0].astype(bool)
        elif existence.shape == (len(batch), 1, pos.shape[2]):
            keep = existence[mask, 0, timestep].astype(bool)
        else:
            raise ValueError("initial_valid must be [N,R] or [N,R,T].")
        states, local_types = states[keep], local_types[keep]
    if not np.isin(local_types, (0, 1, 2)).all():
        raise ValueError("Generated types must use SMART encoding vehicle=0, pedestrian=1, cyclist=2.")
    scene = {"vehicles": states[local_types == 0], "lanes": gt["lanes"], "G": gt["G"]}
    if "metric_lanes" in gt:
        scene["metric_lanes"] = gt["metric_lanes"]
    validate_unified(scene)
    return scene, states, local_types


def compute_agent_metrics(samples, gt_samples, gt_dist=None, vis=False, *, official_repo=None):
    """Small-list compatibility path; no repo needed. Use the evaluator for 50k.

    A previously returned DistributionAccumulator may be passed as gt_dist when
    it describes exactly the same reference list. Persistent identity-checked
    caching is implemented by ScenarioDreamerEvaluator, not by this helper.
    """
    del vis, official_repo
    official = load_official_backend()
    generated = DistributionAccumulator()
    for scene in samples:
        generated.update(scene, official, collision=True)
    if gt_dist is None:
        if len(samples) != len(gt_samples):
            raise ValueError("Generated and GT scene counts must match")
        real = DistributionAccumulator()
        for scene in gt_samples:
            real.update(scene, official)
    elif isinstance(gt_dist, DistributionAccumulator):
        real = gt_dist
        if real.num_scenes != len(samples):
            raise ValueError("Cached GT scene count disagrees with generated count")
    else:
        raise TypeError("gt_dist must be None or the returned DistributionAccumulator")
    return finalize_metrics(generated, real), real


class OfficialReferenceStore(CachedReferenceStore):
    """Backward-compatible class name; now reads prepared local SQLite data."""
    def __init__(self, official_repo=None, cache_root=None, eval_set=None, *,
                 expected_scenes=50_000, real_cache=None, auto_precompute=False):
        del official_repo  # old caller argument, not an external dependency
        if real_cache is None:
            if cache_root is None:
                raise ValueError("Provide real_cache=... or cache_root=...")
            real_cache = Path(cache_root)/DEFAULT_CACHE_NAME
        if not Path(real_cache).expanduser().is_file() and auto_precompute:
            if cache_root is None or eval_set is None:
                raise ValueError("auto_precompute requires cache_root and eval_set")
            prepare_real_cache(cache_root, eval_set, real_cache, expected_scenes=expected_scenes)
        super().__init__(real_cache, eval_set=eval_set, expected_scenes=expected_scenes)


class ScenarioDreamerEvaluator:
    """Single-process evaluator; GT statistics are computed once, not per epoch.

    reference_mode='matched': sum cached histograms for this epoch's filenames
    (preserves your edited partial-evaluation behavior). 'full': compare against
    the full prepared reference, even for debug runs with fewer generated scenes.
    require_full_set=True enforces full official membership in either mode.
    """
    def __init__(self, official_repo=None, cache_root=None, eval_set=None, *, expected_scenes=50_000,
                 gen_timestep=5, prediction_frame="world", require_generation_timestep=True,
                 export_dir=None, real_cache=None, auto_precompute=False,
                 require_full_set=False, reference_mode="matched"):
        if reference_mode not in ("matched", "full"):
            raise ValueError("reference_mode must be matched or full")
        self.store = OfficialReferenceStore(official_repo, cache_root, eval_set,
            expected_scenes=expected_scenes, real_cache=real_cache, auto_precompute=auto_precompute)
        self.gen_timestep = int(gen_timestep)
        self.prediction_frame = prediction_frame
        self.require_generation_timestep = bool(require_generation_timestep)
        self.require_full_set = bool(require_full_set)
        self.reference_mode = reference_mode
        self.export_dir = Path(export_dir) if export_dir else None
        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        self.generated = DistributionAccumulator()
        self.real = DistributionAccumulator()
        self.seen = set()
        self.unverified_frame_count = 0

    def update(self, data, tokenized_agent, out) -> None:
        batch = as_numpy(tokenized_agent["batch"]).astype(np.int64)
        types = as_numpy(tokenized_agent["type"]).reshape(-1)
        if batch.ndim != 1 or len(batch) != len(types):
            raise ValueError("Generated batch/type arrays must align with generated agents.")
        if len(batch) != as_numpy(out["traj"]).shape[0] or len(batch) == 0:
            raise ValueError("Generated row count does not match tokenized_agent['batch']")
        count = int(batch.max()) + 1
        if not np.array_equal(np.unique(batch), np.arange(count)):
            raise ValueError("Generated batch IDs must be contiguous 0..B-1")
        pos, head, size, vel = _generated_arrays(out)
        prepared_out = {"traj": pos, "head": head, "size": size, "vel": vel}
        if out.get("initial_valid") is not None:
            prepared_out["initial_valid"] = as_numpy(out["initial_valid"])
        records = _scene_records(data, count)
        for b, record in enumerate(records):
            filename = field_of(record, "scenario_dreamer_cache_file")
            if filename is None:
                raise KeyError("Dataset must retain scenario_dreamer_cache_file for every sample.")
            if isinstance(filename, (list, tuple)) and len(filename) == 1:
                filename = filename[0]
            name = Path(str(filename)).name
            if name in self.seen:
                raise ValueError(f"Duplicate evaluation sample in this epoch: {name}")
            cache, gt = self.store.get(name) #'scenario_dreamer_cache_file'
            ref_t = scalar(cache["scene_timestep"], "official scene_timestep")
            sample_t = field_of(record, "scene_timestep")
            if sample_t is None or scalar(sample_t, "scene_timestep") != ref_t:
                raise ValueError(f"Input reference timestep does not match official cache: {name}, expected {ref_t}")
            # This field is a contract supplied by the generation input pipeline,
            # not a guess from the trajectory slot or the ego pose.
            generation_t = field_of(record, "generation_scene_timestep")
            if generation_t is None:
                if self.require_generation_timestep:
                    raise ValueError(
                        f"Missing generation_scene_timestep for {name}. Confirm the model's generated "
                        f"initial snapshot is conditioned at raw frame {ref_t}, then record that frame. "
                        "sd_gen_timestep is an output-array index, NOT a raw Waymo timestep."
                    )
                self.unverified_frame_count += 1
            elif scalar(generation_t, "generation_scene_timestep") != ref_t:
                raise ValueError(f"Model/reference frame mismatch for {name}: generated frame {generation_t}, GT frame {ref_t}")
            info = field_of(record, "scenario_dreamer")
            if info is not None and "scene_timestep" in info:
                if scalar(info["scene_timestep"], "scene_info timestep") != ref_t:
                    raise ValueError(f"Saved coordinate transform comes from another timestep: {name}")
            gen, states, gen_types = make_generated_scene(
                prepared_out, batch, types, b, self.gen_timestep, record, gt,
                prediction_frame=self.prediction_frame,
            )
            self.generated.update(gen, self.store.official, collision=True)
            if self.reference_mode == "matched":
                self.store.add_real(self.real, name)
            self.seen.add(name)
            if self.export_dir:
                # Use official cached map + connections, and our generated agents.
                # This is directly readable by the official Waymo conversion path.
                output = {k: copy.deepcopy(cache[k]) for k in (
                    "lg_type", "num_lanes", "road_points", "road_connection_types", "edge_index_lane_to_lane"
                ) if k in cache}
                output.update(agent_states=states,
                              agent_types=np.eye(3)[gen_types.astype(np.int64)],
                              num_agents=len(states))
                path = self.export_dir / f"{self.store.index[name]:05d}.pkl"
                with path.open("wb") as handle:
                    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute(self):
        if not self.seen:
            raise ValueError("No generated scenes were evaluated")
        if self.require_full_set and self.seen != set(self.store.files):
            missing = sorted(set(self.store.files) - self.seen)[:3]
            raise ValueError(f"Incomplete official evaluation: {len(self.seen)}/{len(self.store.files)}; "
                             f"missing examples={missing}. Check sampler/drop_last/limit_val_batches.")
        real = self.store.full_real if self.reference_mode == "full" else self.real
        return finalize_metrics(self.generated, real)

    def report(self):
        real = self.store.full_real if self.reference_mode == "full" else self.real
        return {
            "metric_scope": "Scenario Dreamer initial-scene vehicle metrics",
            "generation_task": "map-conditioned; generated agents on official reference maps",
            "num_samples": self.generated.num_scenes, "num_gt_samples": real.num_scenes,
            "num_generated_vehicles": self.generated.num_vehicles, "num_gt_vehicles": real.num_vehicles,
            "num_colliding_vehicles": self.generated.num_colliding,
            "feature_counts_generated": self.generated.totals.tolist(),
            "feature_counts_gt": real.totals.tolist(),
            "unverified_generation_timestep_count": self.unverified_frame_count,
            "full_membership": self.seen == set(self.store.files),
            "require_full_set": self.require_full_set, "reference_mode": self.reference_mode,
            "eval_set_sha256": self.store.metadata["eval_set_sha256"],
            "official_source_sha256": self.store.official.source_sha256,
            "backend_scope": "local Waymo helper files; see SOURCES.md",
            "real_cache": str(self.store.database),
            "reference_source_manifest_sha256": self.store.metadata["source_manifest_sha256"],
            "gt_feature_recomputations_this_epoch": 0,
        }