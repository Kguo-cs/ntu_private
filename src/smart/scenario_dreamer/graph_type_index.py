"""Reuse actual pickle lg_type labels without reopening every training scene."""
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import torch


INDEX_VERSION = 1
DEFAULT_INDEX_NAME = ".scenario_dreamer_graph_types.npz"


def scene_graph_type(scene, path):
    """Require the explicit scalar field; filenames never define graph type."""
    if not isinstance(scene, Mapping) or "lg_type" not in scene:
        raise ValueError(f"Missing lg_type in Scenario Dreamer scene: {path}")
    value = scene["lg_type"]
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"lg_type must be scalar in {path}")
        value = value.item()
    elif isinstance(value, (np.ndarray, np.generic)):
        if np.asarray(value).size != 1:
            raise ValueError(f"lg_type must be scalar in {path}")
        value = value.item()
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or value not in (0, 1)):
        raise ValueError(f"Expected lg_type=0 or 1 in {path}, got {value!r}")
    return int(value)


def _inspect(path):
    try:
        before = path.stat()
        if path.suffix == ".pkl":
            with path.open("rb") as handle:
                scene = pickle.load(handle)
        else:
            scene = torch.load(path, map_location="cpu", weights_only=False)
        kind = scene_graph_type(scene, path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source scene changed while reading its graph type")
        return path.name, (after.st_size, after.st_mtime_ns, kind)
    except Exception as exc:
        raise ValueError(f"Cannot inspect Scenario Dreamer lg_type in {path}: {exc}") from exc


def _load_index(path, root):
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if archive["version"].item() != INDEX_VERSION or archive["root"].item() != str(root):
                raise ValueError("Index version or source directory does not match")
            rows = archive["records"]
        if rows.ndim != 1 or rows.dtype.names != ("name", "size", "mtime_ns", "kind"):
            raise ValueError("Invalid index record layout")
        if (rows["name"].dtype.kind != "S" or rows["size"].dtype.kind != "i"
                or rows["mtime_ns"].dtype.kind != "i" or rows["kind"].dtype.kind != "i"
                or (rows["size"] < 0).any() or not np.isin(rows["kind"], (0, 1)).all()):
            raise ValueError("Invalid index fields")
        result = {}
        for row in rows:
            name = row["name"].decode("utf-8")
            if not name or Path(name).name != name or name in result:
                raise ValueError("Index filenames must be unique basenames")
            result[name] = (int(row["size"]), int(row["mtime_ns"]), int(row["kind"]))
        return result
    except Exception as exc:
        raise ValueError(f"Invalid graph-type index {path}; remove it to rebuild: {exc}") from exc


def _save_index(path, root, records):
    width = max((len(name.encode("utf-8")) for name in records), default=1)
    rows = np.empty(len(records), dtype=[("name", f"S{width}"), ("size", "<i8"),
                                          ("mtime_ns", "<i8"), ("kind", "i1")])
    for i, (name, (size, mtime, kind)) in enumerate(records.items()):
        rows[i] = (name.encode("utf-8"), size, mtime, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, version=np.asarray(INDEX_VERSION),
                                root=np.asarray(str(root)), records=rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_non_partitioned(paths, root, index_path=None):
    """Return full scenes in original order, validating source size/mtime on reuse.

    Changed/new files are read again. Dataset.get also checks actual lg_type,
    including when reading posterior caches, so the index is never the final
    authorization to feed a partitioned sample into lane-conditioned training.
    The small thread pool avoids forking a process after CUDA initialization.
    """
    root = Path(root).expanduser().resolve()
    index_path = Path(index_path).expanduser().resolve() if index_path else root / DEFAULT_INDEX_NAME
    records = _load_index(index_path, root)
    changed = []
    reused = 0
    for path in paths:
        stat = path.stat()
        record = records.get(path.name)
        if record is None or record[:2] != (stat.st_size, stat.st_mtime_ns):
            changed.append(path)
        else:
            reused += 1
    # Bound queued work: Executor.map on a million paths would retain a million futures.
    if changed:
        with ThreadPoolExecutor(max_workers=4) as workers:
            for start in range(0, len(changed), 512):
                records.update(workers.map(_inspect, changed[start:start + 512]))
                done = min(start + 512, len(changed))
                if done % 65536 == 0 or done == len(changed):
                    print(f"Scenario Dreamer lg_type index: inspected {done}/{len(changed)} scenes", flush=True)
        _save_index(index_path, root, records)
    selected = [path for path in paths if records[path.name][2] == 0]
    report = dict(total=len(paths), selected=len(selected), partitioned=len(paths) - len(selected),
                  inspected=len(changed), reused=reused, index_path=str(index_path))
    if not selected:
        raise ValueError(f"No non-partitioned lg_type=0 scenes in {root}; checked {len(paths)} files")
    return selected, report
