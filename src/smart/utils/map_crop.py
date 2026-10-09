"""Spatial selection for initialization map tokens."""

import math
from numbers import Real

from torch import Tensor


def validate_init_map_crop(shape: str, half_extent: float) -> float:
    if not isinstance(shape, str) or shape not in ("circle", "square"):
        raise ValueError("init_map_crop must be circle or square")
    if isinstance(half_extent, bool) or not isinstance(half_extent, Real):
        raise ValueError("init_map_half_extent must be finite and positive")
    extent = float(half_extent)
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("init_map_half_extent must be finite and positive")
    return extent


def square_map_mask(position: Tensor, batch: Tensor, scene_pos: Tensor,
                    scene_heading: Tensor, half_extent: float) -> Tensor:
    """Strict square bounds on token positions in each scene's ego frame."""
    delta = position[..., :2] - scene_pos[batch]
    heading = scene_heading[batch]
    c, s = heading.cos(), heading.sin()
    local_x = delta[:, 0] * c + delta[:, 1] * s
    local_y = -delta[:, 0] * s + delta[:, 1] * c
    return (local_x.abs() < half_extent) & (local_y.abs() < half_extent)
