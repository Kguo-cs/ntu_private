"""Fixed-reference score for native Scenario Dreamer agent metrics."""
import json
import math
from pathlib import Path

R_METRICS = (
    "nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
    "width_jsd", "speed_jsd", "collision_rate",
)
# Sub-scores over the same per-metric ratios: the six distribution JSDs, and the
# three agent-attribute JSDs (length, width, speed).
R_JSD_METRICS = R_METRICS[:6]
R_ATTR_METRICS = ("length_jsd", "width_jsd", "speed_jsd")


def load_r_reference(path):
    """Accept a reference JSON or an existing sd_agent_metrics.json report."""
    if path is None:
        return None
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = reference["agent_metrics"]
    for name in R_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"R reference {name} must be finite and strictly positive")
    return reference


def relative_score(metrics, reference, report):
    """Score aggregated metrics, never average per-scene or per-batch ratios."""
    ratios = {}
    for name in R_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Cannot compute R: invalid metric {name}={value}")
        ratios[name] = value / float(reference["agent_metrics"][name])
    fields = ("num_samples", "num_gt_samples", "metric_scope", "map_source",
              "eval_set_sha256", "official_source_sha256",
              "reference_source_manifest_sha256")
    mismatches = [name for name in fields
                  if name not in reference or reference[name] != report.get(name)]
    if not report.get("full_membership", False) or not reference.get("full_membership", False):
        mismatches.append("full_membership")
    sub_scores = {
        "R_jsd": math.fsum(ratios[name] for name in R_JSD_METRICS) / len(R_JSD_METRICS),
        "R_attr": math.fsum(ratios[name] for name in R_ATTR_METRICS) / len(R_ATTR_METRICS),
    }
    return math.fsum(ratios.values()) / len(R_METRICS), {
        "name": reference.get("name", "configured reference"),
        "definition": "mean of seven native candidate/reference metric ratios; lower is better",
        "sub_scores": sub_scores,
        "sub_score_definitions": {
            "R_jsd": "mean ratio over the six JSD metrics (collision excluded)",
            "R_attr": "mean ratio over length, width and speed JSD",
        },
        "ratios": ratios,
        "scope_matches_reference": not mismatches,
        "scope_mismatches": mismatches,
        "interpretation": "single evaluation against a fixed baseline; not a pooled or paired multi-seed score",
        "reference_agent_metrics": {name: reference["agent_metrics"][name] for name in R_METRICS},
    }
