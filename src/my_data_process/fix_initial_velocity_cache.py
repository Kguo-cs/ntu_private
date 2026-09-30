"""Repair non-SD initial-velocity caches without changing the originals.

By default this only validates and reports. ``--write --output-dir NEW_DIR``
creates a separate cache, changing only fallback rows of ``local_vel``.
Reports and temporary files are kept outside the dataset directory because
MultiDataset attempts to load every directory entry.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


SRC_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = SRC_ROOT / "waymo_data" / "full"
AGENT_NAMES = ("veh", "ped", "cyc")
TOKEN_SHIFT = 5
TOKEN_DT = TOKEN_SHIFT * 0.1


def load_token_endpoints(path: Path) -> torch.Tensor:
    """Load only the endpoint contours needed to recover token velocities."""
    with path.open("rb") as handle:
        token_all = pickle.load(handle)["token_all"]
    endpoints = []
    for name in AGENT_NAMES:
        token = torch.as_tensor(token_all[name], dtype=torch.float32)
        if token.ndim != 4 or token.shape[1] <= TOKEN_SHIFT or token.shape[2:] != (4, 2):
            raise ValueError(f"Invalid {name} agent token library: {tuple(token.shape)}")
        endpoints.append(token[:, TOKEN_SHIFT].clone())
    if any(value.shape != endpoints[0].shape for value in endpoints):
        raise ValueError("Agent token libraries must have the same shape")
    return torch.stack(endpoints)


def _tensor(agent: Mapping[str, Any], key: str) -> torch.Tensor:
    value = agent.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Missing or non-tensor agent field {key!r}")
    return value


def _equal_field(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if actual.shape != expected.shape or not torch.equal(actual, expected):
        raise ValueError(f"Cache/source {name} mismatch; refusing to infer agent correspondence")


def repair_cache_data(
    cache_data: Mapping[str, Any],
    source_data: Mapping[str, Any],
    endpoints: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, int | float]]:
    """Recompute eligible velocities from source tokens, never from old velocity.

The source and cache must retain identical agent ordering and initial poses.
Other cache fields are preserved. Running this again gives identical tensors.
"""
    cache = cache_data.get("tokenized_agent")
    source = source_data.get("tokenized_agent")
    if not isinstance(cache, Mapping) or not isinstance(source, Mapping):
        raise ValueError("Both inputs must contain a tokenized_agent mapping")
    agent_type = _tensor(cache, "type")
    _equal_field(agent_type, _tensor(source, "type"), "type")
    if agent_type.ndim != 1:
        raise ValueError("Agent type must be one-dimensional")
    n = len(agent_type)
    if bool(((agent_type < 0) | (agent_type > 2)).any()):
        raise ValueError("Agent types must be vehicle=0, pedestrian=1, cyclist=2")
    for label, agent in (("cache", cache), ("source", source)):
        if "num_nodes" in agent and int(agent["num_nodes"]) != n:
            raise ValueError(f"{label} num_nodes does not match agent count")

    pos = _tensor(source, "sampled_pos")
    heading = _tensor(source, "sampled_heading")
    token_index = _tensor(source, "sampled_idx")
    token_mask = _tensor(source, "token_mask")
    if (pos.ndim != 3 or pos.shape[0] != n or pos.shape[1] < 2 or pos.shape[2] != 2
            or heading.shape != pos.shape[:2] or token_index.shape != heading.shape
            or token_mask.shape != heading.shape or token_mask.dtype != torch.bool):
        raise ValueError("Source requires aligned [N,T>=2] token fields and [N,T,2] positions")
    if token_index.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("sampled_idx must contain integer token indices")
    _equal_field(_tensor(cache, "initial_pos"), pos[:, 0], "initial_pos")
    _equal_field(_tensor(cache, "initial_heading"), heading[:, 0], "initial_heading")
    shape_keys = [name for name in ("initial_shape", "shape") if name in cache]
    if not shape_keys:
        raise ValueError("Cache contains neither initial_shape nor shape")
    source_shape = _tensor(source, "shape")
    for key in shape_keys:
        _equal_field(_tensor(cache, key), source_shape, key)
    if source_shape.ndim != 2 or source_shape.shape[0] != n:
        raise ValueError("Source shape must contain one row per agent")
    old_velocity = _tensor(cache, "local_vel")
    if old_velocity.shape != (n, 2) or not old_velocity.is_floating_point():
        raise ValueError("Cache local_vel must be floating point [N,2]")
    if not bool(torch.isfinite(old_velocity).all()):
        raise ValueError("Cache local_vel contains non-finite values")
    if endpoints.ndim != 4 or endpoints.shape[0] != 3 or endpoints.shape[2:] != (4, 2):
        raise ValueError("Token endpoints must have shape [3,K,4,2]")

    fallback = ~token_mask[:, 0] & token_mask[:, 1]
    index = token_index[fallback, 1].long()
    if bool(((index < 0) | (index >= endpoints.shape[1])).any()):
        raise ValueError("Fallback token index outside the selected token library")
    new_velocity = old_velocity.clone()
    # The chosen next token is expressed at its START (the initial frame).
    # Its endpoint displacement / duration therefore already has the desired
    # coordinate frame; do not rotate it into the next token's ending heading.
    selected = endpoints[agent_type[fallback].long(), index]
    new_velocity[fallback] = (selected.mean(-2) / TOKEN_DT).to(old_velocity)
    if not bool(torch.isfinite(new_velocity).all()):
        raise ValueError("Selected token library produces non-finite velocities")
    changed = (new_velocity != old_velocity).any(-1)
    delta = torch.linalg.vector_norm(new_velocity - old_velocity, dim=-1)
    updated_agent = dict(cache)
    updated_agent["local_vel"] = new_velocity
    updated_data = dict(cache_data)
    updated_data["tokenized_agent"] = updated_agent
    stats = {
        "agents": n,
        "eligible_fallback_agents": int(fallback.sum()),
        "unavailable_next_token_agents": int((~token_mask[:, 0] & ~token_mask[:, 1]).sum()),
        "changed_agents": int(changed.sum()),
        "max_velocity_change_mps": float(delta.max()) if n else 0.0,
    }
    return updated_data, stats


def _same_data(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and torch.equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and left.keys() == right.keys()
                and all(_same_data(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same_data(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _validate_paths(cache_dir: Path, source_dir: Path, output_dir: Path | None,
                    report: Path | None, write: bool) -> None:
    if not cache_dir.is_dir() or not source_dir.is_dir():
        raise ValueError("cache-dir and source-dir must be existing directories")
    if write and output_dir is None:
        raise ValueError("--write requires a separate --output-dir")
    if output_dir is not None:
        for original in (cache_dir, source_dir):
            if _inside(output_dir, original) or _inside(original, output_dir):
                raise ValueError("output-dir must be separate from cache-dir and source-dir")
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("output-dir already exists and is not a directory")
    if report is not None:
        for data_dir in (cache_dir, source_dir, output_dir):
            if data_dir is not None and _inside(report, data_dir):
                raise ValueError("Report must be outside all dataset directories")


def _atomic_output(data: dict, destination: Path, unchanged_source: Path | None) -> None:
    # Keep temporary files beside, not inside, the output dataset directory.
    fd, name = tempfile.mkstemp(prefix=".velocity-cache-", suffix=".tmp", dir=destination.parent.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if unchanged_source is None:
            torch.save(data, temporary)
        else:
            shutil.copy2(unchanged_source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_report(summary: dict[str, Any], report: Path | None) -> None:
    if report is None:
        return
    report.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".velocity-report-", suffix=".tmp", dir=report.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, report)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_directory(
    cache_dir: Path,
    source_dir: Path,
    token_file: Path,
    *,
    output_dir: Path | None = None,
    write: bool = False,
    limit: int | None = None,
    report: Path | None = None,
    progress_every: int = 1000,
) -> dict[str, Any]:
    started = time.monotonic()
    cache_dir, source_dir = cache_dir.resolve(), source_dir.resolve()
    output_dir = output_dir.resolve() if output_dir is not None else None
    report = report.resolve() if report is not None else None
    if report is None and output_dir is not None:
        report = output_dir.with_name(output_dir.name + ".velocity-fix-report.json")
    _validate_paths(cache_dir, source_dir, output_dir, report, write)
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    if progress_every < 0:
        raise ValueError("progress-every must be non-negative")
    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt caches under {cache_dir}")
    total_files = len(files)
    if write and output_dir is not None and output_dir.exists():
        expected_names = {path.name for path in files}
        unexpected = [path.name for path in output_dir.iterdir()
                      if not path.is_file() or path.name not in expected_names]
        if unexpected:
            raise ValueError(f"Unexpected entries in output dataset: {unexpected[:5]}")
    if limit is not None:
        files = files[:limit]
    endpoints = load_token_endpoints(token_file.resolve())
    summary: dict[str, Any] = {
        "mode": "write" if write else "dry-run",
        "cache_dir": str(cache_dir), "source_dir": str(source_dir),
        "output_dir": str(output_dir) if output_dir else None,
        "total_input_files": total_files, "selected_files": len(files),
        "files_checked": 0, "files_written": 0, "existing_outputs_verified": 0,
        "agents": 0, "eligible_fallback_agents": 0, "changed_agents": 0,
        "unavailable_next_token_agents": 0, "max_velocity_change_mps": 0.0,
        "complete_dataset": len(files) == total_files,
        "completed": False,
    }
    _write_report(summary, report)
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
    for cache_path in files:
        source_path = source_dir / cache_path.name
        if not source_path.is_file():
            raise ValueError(f"Missing paired source: {source_path}")
        try:
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
            source_data = torch.load(source_path, map_location="cpu", weights_only=False)
            repaired, stats = repair_cache_data(cache_data, source_data, endpoints)
        except (ValueError, KeyError, RuntimeError) as error:
            raise ValueError(f"{cache_path.name}: {error}") from error
        if write:
            destination = output_dir / cache_path.name
            if destination.exists():
                existing = torch.load(destination, map_location="cpu", weights_only=False)
                if not _same_data(existing, repaired):
                    raise ValueError(f"Existing output differs; refusing to overwrite: {destination}")
                summary["existing_outputs_verified"] += 1
            else:
                _atomic_output(repaired, destination, cache_path if stats["changed_agents"] == 0 else None)
                summary["files_written"] += 1
        summary["files_checked"] += 1
        for key in ("agents", "eligible_fallback_agents", "changed_agents", "unavailable_next_token_agents"):
            summary[key] += stats[key]
        summary["max_velocity_change_mps"] = max(summary["max_velocity_change_mps"], stats["max_velocity_change_mps"])
        if progress_every and (summary["files_checked"] % progress_every == 0 or summary["files_checked"] == len(files)):
            print(f"{summary['mode']}: {summary['files_checked']}/{len(files)} files; "
                  f"changed_agents={summary['changed_agents']}; written={summary['files_written']}; "
                  f"resumed={summary['existing_outputs_verified']}; elapsed={time.monotonic() - started:.1f}s",
                  file=sys.stderr, flush=True)
    summary["completed"] = True
    summary["elapsed_seconds"] = time.monotonic() - started
    _write_report(summary, report)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DATA_ROOT / "training_map2_init5v")
    parser.add_argument("--source-dir", type=Path, default=DATA_ROOT / "training_map2_03_light")
    parser.add_argument("--token-file", type=Path, default=SRC_ROOT / "smart/tokens/agent_vocab_555_s2.pkl")
    parser.add_argument("--output-dir", type=Path, help="Separate corrected dataset directory; originals are never overwritten")
    parser.add_argument("--write", action="store_true", help="Write corrected copies (default: dry-run)")
    parser.add_argument("--limit", type=int, help="Inspect only the first N sorted files; such output is an incomplete dataset")
    parser.add_argument("--report", type=Path, help="JSON report outside dataset directories; otherwise beside output-dir")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N files; 0 disables it")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = migrate_directory(args.cache_dir, args.source_dir, args.token_file,
                                   output_dir=args.output_dir, write=args.write,
                                   limit=args.limit, report=args.report, progress_every=args.progress_every)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Migration stopped: {error}. Original caches were not modified.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
