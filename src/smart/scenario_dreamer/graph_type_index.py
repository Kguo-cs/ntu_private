"""Select Scenario Dreamer graph types from the standard scene filename."""
from collections.abc import Mapping
import os
import re

import numpy as np
import torch


_FILENAME = re.compile(r".+_\d+_([01])_\d+\.(?:pkl|pt)")


def scene_graph_type(scene, path):
    """Validate the actual scalar field when a selected scene is loaded."""
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



def filename_graph_type(path):
    """Read lg_type from <prefix>_<scene_index>_<lg_type>_<timestep>.pkl/.pt."""
    name = os.path.basename(os.fspath(path))
    match = _FILENAME.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Cannot determine lg_type from filename {name!r}; expected "
            "<prefix>_<scene_index>_<lg_type:0|1>_<timestep>.pkl or .pt"
        )
    return int(match.group(1))


def select_non_partitioned(paths, root, index_path=None):
    """Select lg_type=0 by filename only, retaining input order without scene I/O.

    index_path is accepted for compatibility with old configurations and ignored.
    Dataset.get validates the actual lg_type after loading each selected scene.
    """
    selected = [path for path in paths if filename_graph_type(path) == 0]
    if not selected:
        raise ValueError(f"No non-partitioned lg_type=0 scenes in {root}; checked {len(paths)} filenames")
    report = dict(total=len(paths), selected=len(selected), partitioned=len(paths) - len(selected),
                  selection_source="filename")
    return selected, report
