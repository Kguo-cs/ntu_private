"""Scenario Dreamer initial-scene AGENT metrics for SMART.

Scope: six vehicle-distribution JSDs + instantaneous vehicle collision rate.
The official Python functions are loaded from ``sd_repo``. No TrafficGen filter,
no raw-map fallback, no finite-difference speed, and no per-batch JSD averaging.

SMART may retain ego-last/world-coordinate trajectories. Only a COPY of the
chosen generated snapshot is converted to the saved SD reference coordinate
frame. Ground truth comes directly from the official eval cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.spatial import distance

try:
    from .official_backend import load_official_backend
except ImportError:  # standalone use
    from official_backend import load_official_backend

# (metric key, clipping lower bound, upper bound, bin width, displayed multiplier)
SPECS = (
    ("nearest_dist_jsd", 0.0, 50.0, 1.0, 10.0),
    ("lat_dev_jsd", 0.0, 1.5, 0.1, 10.0),
    ("ang_dev_jsd", -200.0, 200.0, 5.0, 100.0),
    ("length_jsd", 0.0, 25.0, 0.1, 100.0),
    ("width_jsd", 0.0, 5.0, 0.1, 100.0),
    ("speed_jsd", 0.0, 50.0, 1.0, 100.0),
)


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


def validate_unified(scene: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vehicles = as_numpy(scene["vehicles"])
    lanes = as_numpy(scene["lanes"])
    if vehicles.ndim != 2 or vehicles.shape[1] != 7:
        raise ValueError(f"vehicles must be [N,7], not {vehicles.shape}; flatten rollouts into scenes.")
    if lanes.ndim != 3 or lanes.shape[-1] != 2 or lanes.shape[0] == 0 or lanes.shape[1] < 2:
        raise ValueError(f"Expected nonempty unified lanes [L,P,2], got {lanes.shape}")
    if not np.isfinite(vehicles).all() or not np.isfinite(lanes).all():
        raise ValueError("Nonfinite metric input; do not silently delete bad generated agents.")
    return vehicles, lanes


@dataclass
class DistributionAccumulator:
    """Pool integer histogram counts; memory does not grow with scene count."""
    histograms: list[np.ndarray] = field(default_factory=lambda: [
        np.zeros(len(np.arange(lo, hi + step, step)) - 1, dtype=np.int64)
        for _, lo, hi, step, _ in SPECS
    ])
    totals: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))
    num_scenes: int = 0
    num_vehicles: int = 0
    num_colliding: int = 0

    def update(self, scene: Mapping[str, Any], official, *, collision: bool = False) -> None:
        vehicles, lane_points = validate_unified(scene)
        # Exact official order: metric compact lanes -> resample to 100 -> onroad.
        lanes = official.resample_lanes(lane_points, num_points=100)
        onroad = official.get_onroad_vehicles(vehicles, lanes)
        empty = np.empty(0, dtype=np.float64)
        values = (
            official.get_nearest_dists(vehicles) if len(vehicles) > 1 else empty,
            official.get_lateral_devs(onroad, lanes) if len(onroad) else empty,
            official.get_angular_devs(onroad, lanes) if len(onroad) else empty,
            official.get_lengths(vehicles), official.get_widths(vehicles),
            official.get_speeds(vehicles),
        )
        for i, (value, (_, lo, hi, step, _)) in enumerate(zip(values, SPECS)):
            value = np.asarray(value).reshape(-1)
            bins = np.arange(lo, hi + step, step)
            counts = np.histogram(np.clip(value, lo, hi), bins=bins)[0]
            self.histograms[i] += counts
            self.totals[i] += len(value)
        if collision and len(vehicles):
            # Recover the integer count from the official per-scene fraction.
            # Averaging per-scene fractions would give the wrong global rate.
            rate = float(official.compute_collision_rate([{"vehicles": vehicles}]))
            self.num_colliding += int(round(rate * len(vehicles)))
        self.num_vehicles += len(vehicles)
        self.num_scenes += 1


def finalize_metrics(generated: DistributionAccumulator, real: DistributionAccumulator) -> dict[str, float]:
    metrics = {}
    for i, (name, _, _, _, multiplier) in enumerate(SPECS):
        if generated.totals[i] == 0 or real.totals[i] == 0:
            raise ValueError(
                f"{name} is undefined: generated count={generated.totals[i]}, "
                f"GT count={real.totals[i]}. An empty distribution must not be reported as zero JSD."
            )
        p = generated.histograms[i] / generated.totals[i]
        q = real.histograms[i] / real.totals[i]
        metrics[name] = float(distance.jensenshannon(p, q) ** 2 * multiplier)
    if generated.num_vehicles == 0:
        raise ValueError("collision_rate is undefined: no generated vehicles")
    metrics["collision_rate"] = generated.num_colliding / generated.num_vehicles * 100.0
    return metrics


def compute_agent_metrics(samples, gt_samples, gt_dist=None, vis=False, *, official_repo=None):
    """Compatibility wrapper for unified SINGLE-SCENE dictionaries.

    Returns (metrics, None), retaining the SMART return convention. No stale GT
    cache is retained. For 50k evaluation, use ScenarioDreamerEvaluator instead
    of accumulating all lane arrays in lists.
    """
    del vis
    if gt_dist is not None:
        raise ValueError("Old gt_dist caches are incompatible; reset gt_dist=None.")
    if len(samples) != len(gt_samples):
        raise ValueError("Generated and GT scene counts must match.")
    repo = official_repo or os.environ.get("SCENARIO_DREAMER_ROOT")
    if not repo:
        raise ValueError("Pass official_repo=... or set SCENARIO_DREAMER_ROOT.")
    official = load_official_backend(str(repo))
    gen, real = DistributionAccumulator(), DistributionAccumulator()
    for sample, truth in zip(samples, gt_samples):
        gen.update(sample, official, collision=True)
        real.update(truth, official)
    return finalize_metrics(gen, real), None


class OfficialReferenceStore:
    """Read exact GT cache files; membership is keyed by cache filename, not batch order."""
    def __init__(self, official_repo, cache_root, eval_set, *, expected_scenes=50_000):
        self.official = load_official_backend(str(official_repo))
        self.root = Path(cache_root).expanduser().resolve()
        self.eval_set = Path(eval_set).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Official GT cache directory does not exist: {self.root}")
        # Only load trusted, author-provided pickle files.
        with self.eval_set.open("rb") as handle:
            content = pickle.load(handle)
        if not isinstance(content, dict) or "files" not in content:
            raise ValueError("eval_set must contain a dict with key 'files'")
        self.files = [Path(str(name)).name for name in content["files"]]
        if not self.files or len(self.files) != len(set(self.files)):
            raise ValueError("Expected a nonempty eval list with unique cache filenames.")
        if expected_scenes and len(self.files) != expected_scenes:
            raise ValueError(f"Expected {expected_scenes} official entries, got {len(self.files)}")
        self.index = {name: i for i, name in enumerate(self.files)}
        self.cache: OrderedDict[str, tuple[dict, dict]] = OrderedDict()
        self.expected_scenes = int(expected_scenes)

    def get(self, filename: str) -> tuple[dict, dict]:
        name = Path(str(filename)).name
        if name not in self.index:
            raise ValueError(f"{name} is not in the official evaluation list")
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"Official reference cache missing: {path}")
        with path.open("rb") as handle:
            data = pickle.load(handle)
        data = {k: as_numpy(v) if torch.is_tensor(v) else v for k, v in data.items()}
        if scalar(data["lg_type"], "lg_type") != 0:
            raise ValueError(f"Initial-scene metrics require regular lg_type=0: {name}")
        if int(data.get("num_lanes", 0)) <= 0:
            raise ValueError(f"Official reference has no lanes: {name}")
        # Includes the SECOND metric-level lane graph compaction. Do not replace
        # this with {'lanes': data['road_points']}.
        unified = self.official.convert_data_to_unified_format(copy.deepcopy(data), dataset_name="waymo_gt")
        validate_unified(unified)
        self.cache[name] = (data, unified)
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return data, unified


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
    validate_unified(scene)
    return scene, states, local_types


class ScenarioDreamerEvaluator:
    """Streaming, single-process evaluator for this map-conditioned SMART model."""
    def __init__(self, official_repo, cache_root, eval_set, *, expected_scenes=50_000,
                 gen_timestep=5, prediction_frame="world", require_generation_timestep=True,
                 export_dir=None):
        self.store = OfficialReferenceStore(official_repo, cache_root, eval_set,
                                            expected_scenes=expected_scenes)
        self.gen_timestep = int(gen_timestep)
        self.prediction_frame = prediction_frame
        self.require_generation_timestep = bool(require_generation_timestep)
        self.export_dir = Path(export_dir) if export_dir else None
        if self.export_dir:
            self.export_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.generated = DistributionAccumulator()
        self.real = DistributionAccumulator()
        self.seen: set[str] = set()
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
            self.real.update(gt, self.store.official)
            self.seen.add(name)
            if self.export_dir:
                # Use official cached map + connections, and our generated agents.
                # This is directly readable by the official Waymo conversion path.
                output = {k: copy.deepcopy(cache[k]) for k in (
                    "lg_type", "num_lanes", "road_points", "road_connection_types"
                )}
                output.update(agent_states=states,
                              agent_types=np.eye(3)[gen_types.astype(np.int64)],
                              num_agents=len(states))
                path = self.export_dir / f"{self.store.index[name]:05d}.pkl"
                with path.open("wb") as handle:
                    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute(self) -> dict[str, float]:
        # if self.store.expected_scenes:
        #     missing = set(self.store.files) - self.seen
        #     if missing or len(self.seen) != self.store.expected_scenes:
        #         example = sorted(missing)[:3]
        #         raise ValueError(
        #             f"Incomplete official evaluation: {len(self.seen)}/{self.store.expected_scenes}; "
        #             f"missing examples={example}. Check limit_val_batches, drop_last and the sampler."
        #         )
        return finalize_metrics(self.generated, self.real)

    def report(self) -> dict[str, Any]:
        return {
            "metric_scope": "Scenario Dreamer initial-scene vehicle metrics",
            "generation_task": "map-conditioned; generated agents on official reference maps",
            "num_samples": self.generated.num_scenes,
            "num_gt_samples": self.real.num_scenes,
            "num_generated_vehicles": self.generated.num_vehicles,
            "num_gt_vehicles": self.real.num_vehicles,
            "num_colliding_vehicles": self.generated.num_colliding,
            "feature_counts_generated": self.generated.totals.tolist(),
            "feature_counts_gt": self.real.totals.tolist(),
            "unverified_generation_timestep_count": self.unverified_frame_count,
            "full_membership": self.seen == set(self.store.files),
            "eval_set_sha256": hashlib.sha256(self.store.eval_set.read_bytes()).hexdigest(),
            "official_source_sha256": self.store.official.source_sha256,
        }
