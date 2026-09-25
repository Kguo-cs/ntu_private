"""Load the numerical agent-metric functions from a local Scenario Dreamer repo.

Only selected function definitions are loaded. The repository's top-level
imports are NOT executed, avoiding its unrelated nuPlan/torchaudio/PyG imports
and the name clash between its ``utils`` package and SMART's ``utils`` package.
Function bodies, including lane compaction and angular-deviation rules, are not
rewritten. Use only a trusted repository: loading Python functions executes code.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import torch
from scipy.spatial import distance

LANE_FUNCTIONS = (
    "find_lane_groups", "find_lane_group_id", "resample_polyline", "resample_lanes",
)
METRIC_FUNCTIONS = (
    "jsd", "compute_vehicle_circles", "compute_collision_rate",
    "get_compact_lane_graph", "get_networkx_lane_graph",
    "convert_data_to_unified_format", "get_onroad_vehicles", "get_nearest_dists",
    "get_lateral_devs", "get_angular_devs", "get_lengths", "get_widths",
    "get_speeds", "compute_jsd_metrics", "compute_agent_metrics",
)


def _load_functions(path: Path, names: tuple[str, ...], ns: dict[str, Any]) -> None:
    """Compile selected functions and their same-file function dependencies."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = set(names) - functions.keys()
    if missing:
        raise RuntimeError(f"Official source {path} lacks functions: {sorted(missing)}")
    wanted = set(names)
    # e.g. a future complete-graph helper might call another helper in its file.
    while True:
        required = {
            n.id for name in wanted for n in ast.walk(functions[name])
            if isinstance(n, ast.Name) and n.id in functions
        }
        if required <= wanted:
            break
        wanted |= required
    referenced = {n.id for name in wanted for n in ast.walk(functions[name])
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    # Load only numerical dependencies actually referenced by these functions.
    # Never import the repository's global `utils`/`cfgs` packages into SMART.
    safe_roots = {"numpy", "torch", "torch_geometric", "typing", "itertools",
                  "functools", "collections", "math", "scipy"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] not in safe_roots:
                continue
            aliases = [a for a in node.names if a.name != "*"
                       and (a.asname or a.name) in referenced
                       and (a.asname or a.name) not in ns]
            if aliases:
                subset = ast.ImportFrom(module=node.module, names=aliases, level=0)
                imports = ast.fix_missing_locations(ast.Module(body=[subset], type_ignores=[]))
                exec(compile(imports, str(path), "exec"), ns)
        elif isinstance(node, ast.Import):
            aliases = [a for a in node.names if a.name.split(".")[0] in safe_roots
                       and (a.asname or a.name.split(".")[0]) in referenced
                       and (a.asname or a.name.split(".")[0]) not in ns]
            if aliases:
                imports = ast.fix_missing_locations(ast.Module(body=[ast.Import(names=aliases)], type_ignores=[]))
                exec(compile(imports, str(path), "exec"), ns)
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), ns)


@lru_cache(maxsize=4)
def load_official_backend(repo: str) -> SimpleNamespace:
    root = Path(repo).expanduser().resolve()
    paths = {
        "metrics": root / "utils/metrics_helpers.py",
        "lanes": root / "utils/lane_graph_helpers.py",
        "pyg": root / "utils/pyg_helpers.py",
        "config": root / "cfgs/config.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing official source: {path}. Set sd_repo to the Scenario Dreamer repo, "
                "not to the SMART project or to the preprocessed data directory."
            )
    ns: dict[str, Any] = {
        "np": np, "numpy": np, "torch": torch, "nx": nx, "math": math,
        "itertools": itertools, "product": itertools.product,
        "combinations": itertools.combinations, "distance": distance,
        "tqdm": lambda values, *args, **kwargs: values,
        "print": lambda *args, **kwargs: None,
    }
    # These are literal public constants; do not execute the config module.
    constants = {"NUPLAN_VEHICLE", "NON_PARTITIONED", "UNIFIED_FORMAT_INDICES"}
    cfg = ast.parse(paths["config"].read_text(encoding="utf-8"))
    for node in cfg.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    ns[target.id] = ast.literal_eval(node.value)
    if not constants <= ns.keys():
        raise RuntimeError("Cannot extract official metric constants from cfgs/config.py")
    _load_functions(paths["lanes"], LANE_FUNCTIONS, ns)
    _load_functions(paths["pyg"], ("get_edge_index_complete_graph",), ns)
    _load_functions(paths["metrics"], METRIC_FUNCTIONS, ns)
    # Resolve common helper imports used by some complete-graph revisions.
    # The numerical functions still come from the user-selected official source.
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths.values()}
    return SimpleNamespace(**{key: value for key, value in ns.items() if not key.startswith("__")},
                           source_root=str(root), source_sha256=hashes)
