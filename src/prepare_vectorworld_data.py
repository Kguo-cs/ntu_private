#!/usr/bin/env python3
"""Add real VectorWorld history labels to local Scenario Dreamer snapshots.

Native SD .pkl basenames identify a Waymo TFRecord, record index and frame. Full
SMART .pt scenes with saved scenario_dreamer metadata are an alternative when
those raw records are unavailable. Source snapshots and trajectories are never
modified. Output preserves native cache basenames and explicit sample-list order.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import pickle
import re
import sys
import tempfile

import numpy as np
import torch

SIM_ROOT = Path(__file__).resolve().parents[1]
SIM_DATA = SIM_ROOT / "src/waymo_data"
# The existing preprocessing helpers use sibling imports (sd_reference, etc.).
if str(SIM_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SIM_ROOT / "src"))
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

SOURCE_NAME = re.compile(
    r"^(?P<tfrecord>(?P<split>training|validation|testing)\.tfrecord-\d+-of-\d+)"
    r"_(?P<record>\d+)_(?P<kind>[01])_(?P<timestep>\d+)\.(?:pkl|pt)$"
)
MOTION_FIELDS = ("agent_motion_raw", "agent_motion_is_static", "agent_motion_valid_mask")


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def parse_source_name(filename):
    """Recover the actual source, never infer a scenario ID from a basename."""
    name = Path(filename).name
    match = SOURCE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Cannot locate raw Waymo source for {name!r}; expected "
            "<training|validation|testing>.tfrecord-<shard>-of-<count>_<record>_<lg_type>_<timestep>.pkl. "
            "Use a full SMART .pt scene with saved scenario_dreamer metadata instead."
        )
    row = match.groupdict()
    return dict(tfrecord=row["tfrecord"], split=row["split"], record_index=int(row["record"]),
                lg_type=int(row["kind"]), scene_timestep=int(row["timestep"]))


def strict_state_order(target, source, *, atol=1e-5):
    """Return source rows for target states; reject approximate/ambiguous matches."""
    first, second = as_numpy(target["agent_states"]), as_numpy(source["agent_states"])
    types_first, types_second = as_numpy(target["agent_types"]), as_numpy(source["agent_types"])
    if first.shape != second.shape or types_first.shape != types_second.shape:
        raise ValueError("Reconstructed source selection differs from the cached agent counts; cannot align real history")
    equal = np.isclose(first[:, None], second[None], rtol=0, atol=atol).all(-1)
    equal &= (types_first[:, None] == types_second[None]).all(-1)
    counts = equal.sum(-1)
    if not (counts == 1).all():
        bad = np.flatnonzero(counts != 1).tolist()
        raise ValueError(f"Cached agent states lack unique exact source matches at rows {bad}; do not approximate source IDs")
    order = equal.argmax(-1)
    if len(np.unique(order)) != len(order):
        raise ValueError("Multiple cached agents map to the same source trajectory")
    return order


def full_trajectory_rows(data, scene):
    """Map native ego-first source IDs to the SMART trajectory row order."""
    if "source_index" not in scene or "output_source_index" not in scene:
        raise ValueError("Full .pt fallback requires scenario_dreamer.source_index and output_source_index; rebuild with --save-scene-info")
    native = as_numpy(scene["source_index"]).reshape(-1)
    output = as_numpy(scene["output_source_index"]).reshape(-1)
    n = int(scene["num_agents"])
    if (native.shape != (n,) or output.shape != (n,) or len(np.unique(native)) != n
            or len(np.unique(output)) != n or set(native.tolist()) != set(output.tolist())):
        raise ValueError("Invalid native-to-SMART source-agent mapping in full .pt scene")
    if len(data["agent"]["position"]) != n:
        raise ValueError("Full .pt trajectory count disagrees with saved native scene")
    lookup = {int(source): row for row, source in enumerate(output)}
    return np.asarray([lookup[int(source)] for source in native], dtype=np.int64)


def native_scene_from_full(data, filename):
    """Export exact saved static states/topology, retaining SD channel ordering."""
    info = data.get("scenario_dreamer")
    if not isinstance(info, Mapping) or not info.get("valid_scene", True):
        raise ValueError(f"{filename}: full trajectory cache must contain valid saved scenario_dreamer metadata")
    if "agent" not in data or "position" not in data["agent"]:
        raise ValueError(f"{filename}: tokenized-only caches have no real trajectory history")
    from src.smart.scenario_dreamer.data import attach_model_map
    from src.smart.scenario_dreamer.core.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite
    map_data = {}
    attach_model_map(map_data, info)
    edge = map_data[("sd_lane", "to", "sd_lane")]
    n, l = int(info["num_agents"]), len(map_data["sd_lane"]["x"])
    scene = {
        "idx": Path(filename).stem, "lg_type": int(info["lg_type"]),
        "scene_timestep": int(info["scene_timestep"]), "num_agents": n, "num_lanes": l,
        "agent_states": as_numpy(info["agent_states"]).copy(),
        "agent_types": as_numpy(info["agent_types"]).copy(),
        "road_points": as_numpy(info["graphs"]["regular" if int(info["lg_type"]) == 0 else "partitioned"]["road_points"]).copy(),
        "edge_index_lane_to_lane": as_numpy(edge["edge_index"]).copy(),
        "road_connection_types": np.eye(6)[as_numpy(edge["type"]).astype(np.int64)],
        "edge_index_agent_to_agent": get_edge_index_complete_graph(n).numpy(),
        "edge_index_lane_to_agent": get_edge_index_bipartite(l, n).numpy(),
    }
    return scene, full_trajectory_rows(data, info)


def add_motion(scene, position, heading, velocity, valid_mask, *, provenance, **motion_options):
    """Write physical body-frame history in the snapshot's unchanged agent order."""
    from src.smart.vectorworld.data import compute_motion_code
    from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
    adapt_preprocessed_scene(scene, provenance.get("source_file", "scene.pkl"))
    raw, static, available = compute_motion_code(
        position, heading, velocity, valid_mask, int(scene["scene_timestep"]), **motion_options,
    )
    if len(raw) != int(scene["num_agents"]) or not bool(available.all()):
        raise ValueError("Every cached agent needs a valid real source observation at the snapshot frame")
    result = dict(scene)
    result.update(zip(MOTION_FIELDS, (as_numpy(raw), as_numpy(static), as_numpy(available))))
    result["vectorworld_motion_metadata"] = dict(provenance, scene_timestep=int(scene["scene_timestep"]), **motion_options)
    return result


def convert_full_scene(data, filename, *, target=None, state_atol=1e-5, **motion_options):
    native, rows = native_scene_from_full(data, filename)
    if target is not None:
        rows = rows[strict_state_order(target, native, atol=state_atol)]
        native = target
    agent = data["agent"]
    required = ("position", "heading", "velocity", "valid_mask")
    missing = set(required) - set(agent)
    if missing:
        raise ValueError(f"{filename}: real full trajectories are missing {sorted(missing)}")
    values = [as_numpy(agent[key])[rows] for key in required]
    return add_motion(native, *values, provenance={"source": "full_smart_trajectories", "source_file": str(filename)}, **motion_options)


class WaymoSourceReader:
    """Read requested uncompressed TFRecord rows once in monotonically sorted order."""
    def __init__(self, root):
        self.root = Path(root)
        self.path, self.iterator, self.index, self.scenario = None, None, -1, None

    def get(self, request):
        path = self.root / request["split"] / request["tfrecord"]
        if not path.is_file():
            flat = self.root / request["tfrecord"]
            if not flat.is_file():
                raise FileNotFoundError(
                    f"Raw Waymo source is missing: {path}. Supply --waymo-root containing this split, "
                    "or --input-dir/--trajectory-dir containing full SMART .pt trajectories with saved scene metadata. "
                    "Initial/tokenized-only caches cannot supply VectorWorld motion supervision."
                )
            path = flat
        if path != self.path or request["record_index"] < self.index:
            import tensorflow as tf
            self.iterator = iter(tf.compat.v1.io.tf_record_iterator(str(path)))
            self.path, self.index = path, -1
        while self.index < request["record_index"]:
            from waymo_open_dataset.protos import scenario_pb2
            try:
                record = next(self.iterator)
            except StopIteration as error:
                raise IndexError(f"{path} has no requested record {request['record_index']}") from error
            self.index += 1
            self.scenario = scenario_pb2.Scenario()
            self.scenario.ParseFromString(record)
        return self.scenario


def convert_raw_scene(scene, filename, scenario, request, *, state_atol=1e-5, **motion_options):
    """Reapply exact SD source selection and preserve cached partition reordering."""
    from scenario_dreamer_filter import decode_tracks_from_proto, get_agent_features
    if int(scene["scene_timestep"]) != request["scene_timestep"] or int(scene["lg_type"]) != request["lg_type"]:
        raise ValueError(f"{filename}: cached frame/graph type disagrees with source basename")
    tracks = decode_tracks_from_proto(scenario)
    _, rebuilt = get_agent_features(
        tracks, split=request["split"], num_historical_steps=int(scenario.current_time_index) + 1,
        num_steps=tracks["states"].shape[1], scenario=scenario,
        scene_timestep=request["scene_timestep"], return_scene_info=True,
    )
    if not rebuilt["valid_scene"]:
        raise ValueError(f"{filename}: requested source scene is invalid ({rebuilt.get('reason')})")
    rows = as_numpy(rebuilt["source_index"])[strict_state_order(scene, rebuilt, atol=state_atol)].astype(np.int64)
    states, valid = tracks["states"][rows], tracks["valid"][rows]
    return add_motion(
        scene, states[..., :2], states[..., 6], states[..., 7:9], valid,
        provenance={"source": "waymo_tfrecord", "source_file": str(filename),
                    "source_tfrecord": request["tfrecord"], "source_record_index": request["record_index"],
                    "scenario_id": scenario.scenario_id, "source_agent_indices": rows.tolist()},
        **motion_options,
    )


def _load(path):
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            return pickle.load(handle)
    return torch.load(path, map_location="cpu", weights_only=False)


def select_inputs(directory, sample_list=None, limit=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory is missing: {directory}")
    if sample_list is not None:
        with Path(sample_list).open("rb") as handle:
            names = pickle.load(handle)["files"]
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or Path(name).name != name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("--sample-list must contain unique plain basenames in {'files': [...]}")
        paths = [directory / name for name in names]
    else:
        paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in (".pkl", ".pt"))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No .pkl/.pt scenes selected in {directory}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Selected source scene is missing: {path}")
    return paths


def _atomic_pickle(value, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def _trajectory_index(directory):
    if directory is None:
        return {}
    paths = select_inputs(directory)
    result = {}
    for path in paths:
        if path.suffix != ".pt":
            continue
        stem = re.sub(r"^\d{5}_", "", path.stem)
        name = stem + ".pkl"
        if name in result:
            raise ValueError(f"Duplicate trajectory fallback for {name}: {directory}")
        result[name] = path
    return result


def prepare(*, input_dir, output_dir, split="train", sample_list=None, waymo_root=SIM_DATA / "waymo110",
            trajectory_dir=None, limit=None, overwrite=False, state_atol=1e-5, **motion_options):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Output must be a new directory; source data is never modified")
    if state_atol <= 0 or not np.isfinite(state_atol):
        raise ValueError("--state-atol must be finite and positive")
    paths = select_inputs(input_dir, sample_list, limit)
    fallback = _trajectory_index(trajectory_dir)
    reader = WaymoSourceReader(waymo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Source TFRecords are traversed in record order; manifest order stays unchanged.
    work = []
    for index, path in enumerate(paths):
        request = parse_source_name(path.name) if path.suffix == ".pkl" else None
        key = (request["tfrecord"], request["record_index"], request["scene_timestep"]) if request else (str(path), 0, 0)
        work.append((key, index, path, request))
    output_names = [None] * len(paths)
    seen = set()
    manifest_path = output_dir.parent / f"{split}_motion_manifest.jsonl"
    sample_path = output_dir.parent / f"{split}_files.pkl"
    source_list = Path(sample_list) if sample_list is not None else input_dir.parent / f"{split}_files.pkl"
    if sample_path.resolve() == source_list.resolve():
        raise ValueError("Generated file list would replace the source sample list; use a separate output root")
    temporary_manifest = None
    try:
        with tempfile.NamedTemporaryFile(dir=manifest_path.parent, prefix=manifest_path.name + ".",
                                         suffix=".tmp", mode="w", encoding="utf-8", delete=False) as manifest:
            temporary_manifest = Path(manifest.name)
            for _, index, path, request in sorted(work):
                data = _load(path)
                if request is None:
                    if "scenario_dreamer_cache_file" in data:
                        name = Path(data["scenario_dreamer_cache_file"]).name
                    else:
                        name = path.with_suffix(".pkl").name
                    result = convert_full_scene(data, path.name, state_atol=state_atol, **motion_options)
                else:
                    name = path.name
                    if name in fallback:
                        result = convert_full_scene(_load(fallback[name]), str(fallback[name]), target=data,
                                                    state_atol=state_atol, **motion_options)
                    else:
                        result = convert_raw_scene(data, path.name, reader.get(request), request,
                                                   state_atol=state_atol, **motion_options)
                if Path(name).name != name or Path(name).suffix != ".pkl" or name in seen:
                    raise ValueError(f"Invalid or duplicate output snapshot basename: {name}")
                target = output_dir / name
                if target.exists() and not overwrite:
                    raise FileExistsError(f"Output scene already exists: {target}; pass --overwrite to replace only generated data")
                _atomic_pickle(result, target)
                seen.add(name)
                output_names[index] = name
                manifest.write(json.dumps(dict(index=index, output=name, input=path.name,
                                               num_agents=int(result["num_agents"]),
                                               static_agents=int(result["agent_motion_is_static"].sum()),
                                               **result["vectorworld_motion_metadata"])) + "\n")
                print(f"[{len(seen)}/{len(paths)}] {name}", flush=True)
    except BaseException:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
        raise
    temporary_manifest.replace(manifest_path)
    _atomic_pickle({"files": output_names}, sample_path)
    report = dict(scenes=len(paths), output_dir=str(output_dir), sample_list=str(sample_path),
                  manifest=str(manifest_path), motion_options=motion_options)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--input-dir", type=Path, help="Native SD .pkl or full SMART .pt directory")
    parser.add_argument("--output-dir", type=Path, help="New motion-labelled directory; default src/waymo_data/vectorworld/vae/<split>")
    parser.add_argument("--sample-list", type=Path, help="{'files': [...]} pickle; selection/order is retained exactly")
    parser.add_argument("--waymo-root", type=Path, default=SIM_DATA / "waymo110")
    parser.add_argument("--trajectory-dir", type=Path, help="Optional matching full .pt fallback for native SD basenames")
    parser.add_argument("--limit", type=int, help="Process only the first selected N scenes")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated output scenes, never input data")
    parser.add_argument("--state-atol", type=float, default=1e-5, help="Absolute physical-state tolerance for unique source alignment")
    parser.add_argument("--num-points", type=int, default=6)
    parser.add_argument("--history-max-m", type=float, default=12.)
    parser.add_argument("--d-static", type=float, default=.5)
    parser.add_argument("--v-static", type=float, default=.2)
    parser.add_argument("--t-hist-max", type=int, default=8)
    args = parser.parse_args(argv)
    input_dir = args.input_dir or SIM_DATA / "scenario_dreamer_ae_preprocess_waymo" / args.split
    output_dir = args.output_dir or SIM_DATA / "vectorworld/vae" / args.split
    sample_list = args.sample_list
    automatic_list = input_dir.parent / f"{args.split}_files.pkl"
    if sample_list is None and automatic_list.is_file():
        sample_list = automatic_list
    try:
        prepare(input_dir=input_dir, output_dir=output_dir, split=args.split, sample_list=sample_list,
                waymo_root=args.waymo_root, trajectory_dir=args.trajectory_dir, limit=args.limit,
                overwrite=args.overwrite, state_atol=args.state_atol, num_points=args.num_points,
                history_max_m=args.history_max_m, d_static=args.d_static, v_static=args.v_static,
                t_hist_max=args.t_hist_max)
    except (ValueError, FileNotFoundError, FileExistsError, IndexError) as error:
        parser.exit(1, f"VectorWorld preprocessing failed: {error}\n")


if __name__ == "__main__":
    main()
