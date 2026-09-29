#!/usr/bin/env python3
"""Rebuild Scenario Dreamer's official Waymo 50k real evaluation set.

This script uses the official ``metadata/waymo_eval_set.pkl`` membership list.
Each Scenario Dreamer cache filename has the form

    <raw_file_name>_<lg_type>_<scene_timestep>.pkl

and the raw extracted file name has the form

    <waymo_tfrecord_filename>_<record_index>.pkl

so the exact source TFRecord record and reference timestep can be recovered
without relying on a random seed or on scenario-id guessing.

The actual agent selection is delegated to ``scenario_dreamer_filter.py``.
In the supplied adaptation that means:
  * selection follows Scenario Dreamer's ego-first local-frame rules;
  * returned SMART ``data['agent']`` is reordered to ego-last;
  * returned SMART trajectories remain in original Waymo/world coordinates.

Typical usage
-------------
python rebuild_scenario_dreamer_eval50k.py \
    --eval-set /path/to/scenario-dreamer/metadata/waymo_eval_set.pkl \
    --waymo-root /path/to/waymo_open_dataset_motion_v_1_1_0 \
    --output-dir /path/to/sd_eval50k_smart \
    --manifest-out /path/to/sd_eval50k_manifest.jsonl \
    --official-cache-root /path/to/scenario_dreamer_ae_preprocess_waymo/test

Use ``--manifest-only`` to only decode the 50k cache filenames.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scenario_dreamer_filter import decode_tracks_from_proto, get_agent_features
from data_preprocess_scenario_dreamer import (
    decode_dynamic_map_states_from_proto,
    decode_map_features_from_proto,
    get_map_features,
    process_dynamic_map,
)
import sys

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/sim')
sys.path.append('/home/ke/code/sim')
sys.path.append('/home/users/ntu/ke.guo/scratch/sim')
sys.path.append('/home/zs/code/sim')
sys.path.append('/mnt/d/code/sim')
sys.path.append('/home/ke/keguo/sim')
sys.path.append('/home/guoke/sim')


# Official raw extraction creates, e.g.:
# testing.tfrecord-00000-of-00150_123.pkl
RAW_RE = re.compile(
    r"^(?P<tfrecord>.+\.tfrecord-\d+-of-\d+)_(?P<record_index>\d+)$"
)


@dataclass(frozen=True)
class EvalRequest:
    eval_index: int
    cache_file: str
    raw_file_name: str
    lg_type: int
    scene_timestep: int
    source_split: str
    source_tfrecord: str
    record_index: int


def load_eval_files(eval_set_path: Path) -> list[str]:
    with eval_set_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "files" not in obj:
        raise ValueError(f"{eval_set_path} must contain a dict with key 'files'")
    files = list(obj["files"])
    if not all(isinstance(x, (str, Path)) for x in files):
        raise TypeError("eval_set['files'] must be a list of filenames")
    return [str(x) for x in files]


def infer_waymo_split(tfrecord_name: str) -> str:
    # Scenario Dreamer's test pool can contain copied validation scenarios.
    if tfrecord_name.startswith("training.tfrecord-"):
        return "training"
    if tfrecord_name.startswith("validation.tfrecord-"):
        return "validation"
    if tfrecord_name.startswith("testing.tfrecord-"):
        return "testing"
    raise ValueError(f"Cannot infer WOMD split from TFRecord name: {tfrecord_name}")


def parse_eval_cache_name(cache_file: str, eval_index: int) -> EvalRequest:
    """Invert Scenario Dreamer's two filename-construction stages."""
    basename = Path(cache_file).name
    stem = Path(basename).stem
    try:
        raw_file_name, lg_text, timestep_text = stem.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"Unexpected Scenario Dreamer cache filename: {basename}") from exc

    try:
        lg_type = int(lg_text)
        scene_timestep = int(timestep_text)
    except ValueError as exc:
        raise ValueError(f"Cannot parse lg_type/timestep from {basename}") from exc
    if lg_type not in (0, 1):
        raise ValueError(f"Unexpected lg_type={lg_type} in {basename}")

    match = RAW_RE.fullmatch(raw_file_name)
    if match is None:
        raise ValueError(
            f"Cannot recover TFRecord + record index from raw file name {raw_file_name!r} "
            f"(from cache {basename!r})"
        )
    source_tfrecord = match.group("tfrecord")
    record_index = int(match.group("record_index"))
    source_split = infer_waymo_split(source_tfrecord)

    return EvalRequest(
        eval_index=eval_index,
        cache_file=cache_file,
        raw_file_name=raw_file_name,
        lg_type=lg_type,
        scene_timestep=scene_timestep,
        source_split=source_split,
        source_tfrecord=source_tfrecord,
        record_index=record_index,
    )


def build_requests(eval_set_path: Path, expected_count: int = 50_000) -> list[EvalRequest]:
    files = load_eval_files(eval_set_path)
    if expected_count > 0 and len(files) != expected_count:
        raise ValueError(
            f"Expected {expected_count} eval files, got {len(files)}. "
            "Use --expected-count 0 if intentionally using another list."
        )
    requests = [parse_eval_cache_name(name, i) for i, name in enumerate(files)]
    return requests


def write_manifest(path: Path, requests: Iterable[EvalRequest], extra_by_index=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_by_index = extra_by_index or {}
    with path.open("w", encoding="utf-8") as f:
        for req in requests:
            row = asdict(req)
            row.update(extra_by_index.get(req.eval_index, {}))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_official_cache(req: EvalRequest, cache_root: Path) -> None:
    """Optional sanity check against the downloaded SD preprocessed test cache."""
    path = cache_root / req.cache_file
    if not path.exists():
        # eval_set entries are normally basenames; accept nested entries too.
        path = cache_root / Path(req.cache_file).name
    if not path.exists():
        raise FileNotFoundError(f"Official cache listed by eval set not found: {req.cache_file}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if "scene_timestep" in data and int(data["scene_timestep"]) != req.scene_timestep:
        raise ValueError(f"Timestep mismatch in {path}")
    if "lg_type" in data and int(data["lg_type"]) != req.lg_type:
        raise ValueError(f"lg_type mismatch in {path}")


def _empty_lights() -> pd.DataFrame:
    return pd.DataFrame(columns=["lane_id", "time_step", "state"])


def save_rebuilt_sample(
    *,
    scenario,
    req: EvalRequest,
    output_dir: Path,
    save_scene_info: bool,
) -> dict:
    """Rebuild one exact (raw scenario, timestep, lg_type) reference sample."""
    # Import here so --manifest-only does not require the SMART package.
    from src.smart.utils.preprocess import preprocess_map

    track_infos = decode_tracks_from_proto(scenario)
    agents, scene = get_agent_features(
        track_infos,
        split=req.source_split,
        num_historical_steps=int(scenario.current_time_index) + 1,
        num_steps=max(91, track_infos["states"].shape[1]),
        scenario=scenario,
        scene_timestep=req.scene_timestep,
        return_scene_info=True,
    )
    if not bool(scene["valid_scene"]):
        raise ValueError(
            f"Official eval request became invalid: {req.cache_file}; "
            f"reason={scene.get('reason', 'unknown')}"
        )

    # Select the SAME Scenario Dreamer map variant encoded by the official cache filename.
    graph_key = "regular" if req.lg_type == 0 else "partitioned"
    graph = scene["graphs"][graph_key]
    scene["lg_type"] = req.lg_type
    scene["road_points"] = graph["road_points"]
    scene["num_lanes"] = graph["num_lanes"]

    # SMART map output remains global/world coordinates. Traffic-light state is taken
    # at the exact Scenario Dreamer reference frame, not current_time_index.
    map_infos = decode_map_features_from_proto(scenario.map_features)
    dynamic = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)
    lights = process_dynamic_map(dynamic) if len(dynamic["lane_id"]) else _empty_lights()
    current_lights = lights.loc[lights["time_step"] == req.scene_timestep]
    data = preprocess_map(get_map_features(map_infos, current_lights))

    data["agent"] = agents                         # ego-last, WORLD coordinates
    data["scenario_id"] = scenario.scenario_id
    data["scene_timestep"] = int(req.scene_timestep)
    data["scenario_dreamer_lg_type"] = int(req.lg_type)
    data["scenario_dreamer_eval_index"] = int(req.eval_index)
    data["scenario_dreamer_cache_file"] = req.cache_file
    data["source_split"] = req.source_split
    data["source_tfrecord"] = req.source_tfrecord
    data["source_record_index"] = int(req.record_index)
    # IMPORTANT: output_source_index matches the ego-last SMART agent rows.
    data["agent_source_index"] = scene["output_source_index"]
    if save_scene_info:
        data["scenario_dreamer"] = scene

    # Prefix with eval index to preserve the exact 50k list order and avoid collisions.
    output_name = f"{req.eval_index:05d}_{Path(req.cache_file).stem}.pt"
    output_path = output_dir / output_name
    torch.save(data, output_path)

    return {
        "scenario_id": scenario.scenario_id,
        "output_file": output_name,
        "num_agents": int(agents["num_nodes"]),
        "selected_track_ids": [int(x) for x in agents["id"].tolist()],
        "ego_track_id": int(agents["id"][-1]) if agents["num_nodes"] else None,
    }


def rebuild(
    requests: list[EvalRequest],
    *,
    waymo_root: Path,
    output_dir: Path,
    official_cache_root: Path | None,
    save_scene_info: bool,
) -> dict[int, dict]:
    """Read every needed TFRecord once and rebuild all requested samples."""
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:
        raise RuntimeError(
            "Rebuild mode requires tensorflow and waymo_open_dataset. "
            "Manifest-only mode does not."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[EvalRequest]] = defaultdict(list)
    for req in requests:
        grouped[(req.source_split, req.source_tfrecord)].append(req)

   # grouped=grouped[79:]#[79:]

    resolved: dict[int, dict] = {}
    for (split, tfrecord_name), group in tqdm(
        sorted(grouped.items()), desc="TFRecord files"
    ):
        source_path = waymo_root / split / tfrecord_name
        if not source_path.exists():
            raise FileNotFoundError(
                f"Required source TFRecord not found: {source_path}\n"
                "Note: Scenario Dreamer's test pool may contain validation scenarios, "
                "so both validation/ and testing/ must be present."
            )

        by_record: dict[int, list[EvalRequest]] = defaultdict(list)
        for req in group:
            by_record[req.record_index].append(req)
            if official_cache_root is not None:
                verify_official_cache(req, official_cache_root)
        needed = set(by_record)
        max_needed = max(needed)
        found = set()

        dataset = tf.data.TFRecordDataset(str(source_path), compression_type="")
        for record_index, record in enumerate(dataset):
            if record_index > max_needed and found == needed:
                break
            if record_index not in needed:
                continue

            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytes(record.numpy()))
            found.add(record_index)

            # A single raw scenario may appear multiple times in the 50k list at
            # different timesteps / graph variants. Decode it once, rebuild each request.
            for req in by_record[record_index]:
                info = save_rebuilt_sample(
                    scenario=scenario,
                    req=req,
                    output_dir=output_dir,
                    save_scene_info=save_scene_info,
                )
                resolved[req.eval_index] = info

            if found == needed:
                break

        missing = needed - found
        if missing:
            raise IndexError(
                f"TFRecord {source_path} does not contain requested record indices: "
                f"{sorted(missing)[:10]}"
            )

    if len(resolved) != len(requests):
        missing = sorted(set(range(len(requests))) - set(resolved))
        raise RuntimeError(f"Only rebuilt {len(resolved)}/{len(requests)}; missing {missing[:10]}")
    return resolved


def summarize(requests: list[EvalRequest]) -> None:
    split_counts = defaultdict(int)
    lg_counts = defaultdict(int)
    unique_raw = set()
    unique_tfrecords = set()
    for r in requests:
        split_counts[r.source_split] += 1
        lg_counts[r.lg_type] += 1
        unique_raw.add((r.source_split, r.source_tfrecord, r.record_index))
        unique_tfrecords.add((r.source_split, r.source_tfrecord))
    print(f"eval entries      : {len(requests)}")
    print(f"unique raw scenes : {len(unique_raw)}")
    print(f"unique TFRecords  : {len(unique_tfrecords)}")
    print(f"source splits     : {dict(sorted(split_counts.items()))}")
    print(f"lg_type counts    : {dict(sorted(lg_counts.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default='./waymo_data/waymo_eval_set.pkl',
                        help="Official scenario-dreamer metadata/waymo_eval_set.pkl")
    parser.add_argument("--waymo-root", type=Path,default=Path("./waymo_data/waymo110"),
                        help="WOMD v1.1.0 root containing training/ validation/ testing/")
    parser.add_argument("--output-dir", type=Path,default=Path("./waymo_data/full/scenario_dreamer_val"),
                        help="Directory for rebuilt ego-last/global SMART .pt samples")
    parser.add_argument("--manifest-out", type=Path, default=Path("./waymo_data/sd_eval50k_manifest.jsonl"))
    parser.add_argument("--official-cache-root", type=Path,
                        help="Optional Scenario Dreamer preprocessed test dir; verifies cache metadata")
    parser.add_argument("--expected-count", type=int, default=50_000,
                        help="Expected eval-set size; use 0 to disable")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only decode cache filenames; do not read Waymo TFRecords")
    parser.add_argument("--save-scene-info", action="store_true",default=True,
                        help="Also save exact Scenario Dreamer local features/map graph in each .pt")
    args = parser.parse_args()

    requests = build_requests(args.eval_set, expected_count=args.expected_count)
    summarize(requests)

    # Step ③ + first half of ④: this manifest already records exact source
    # TFRecord, record index, lg_type and scene_timestep.
    write_manifest(args.manifest_out, requests)
    print(f"Wrote source manifest: {args.manifest_out}")

    if args.manifest_only:
        return
    if args.waymo_root is None or args.output_dir is None:
        parser.error("Rebuild mode requires --waymo-root and --output-dir")

    # Step ④ + ⑤: load exact raw Scenario protos and rebuild with fixed timesteps.
    resolved = rebuild(
        requests,
        waymo_root=args.waymo_root,
        output_dir=args.output_dir,
        official_cache_root=args.official_cache_root,
        save_scene_info=args.save_scene_info,
    )
    resolved_manifest = args.manifest_out.with_name(
        args.manifest_out.stem + "_resolved.jsonl"
    )
    write_manifest(resolved_manifest, requests, resolved)
    print(f"Rebuilt {len(resolved)} samples into: {args.output_dir}")
    print(f"Wrote resolved manifest: {resolved_manifest}")


if __name__ == "__main__":
    main()
