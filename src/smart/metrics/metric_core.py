"""Shared histogram aggregation, independent of any external checkout."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
import torch
from scipy.spatial import distance

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
        lanes = scene.get("metric_lanes")
        if lanes is None:
            lanes = official.resample_lanes(lane_points, num_points=100)
        elif (lanes.ndim != 3 or lanes.shape != (len(lane_points), 100, 2)
              or not np.isfinite(lanes).all()):
            raise ValueError("metric_lanes must be the cached [L,100,2] resampling")
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
