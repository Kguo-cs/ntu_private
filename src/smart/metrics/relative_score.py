"""Fixed-reference score for native Scenario Dreamer agent metrics."""
import json
import math
from pathlib import Path

R_METRICS = (
    "nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
    "width_jsd", "speed_jsd", "collision_rate",
)
JSD_METRICS = R_METRICS[:6]
ATTR_METRICS = ("length_jsd", "width_jsd", "speed_jsd")


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
    scores = {
        "R": math.fsum(ratios.values()) / len(R_METRICS),
        "R_jsd": math.fsum(ratios[name] for name in JSD_METRICS) / len(JSD_METRICS),
        "R_attr": math.fsum(ratios[name] for name in ATTR_METRICS) / len(ATTR_METRICS),
    }
    return scores["R"], {
        "name": reference.get("name", "configured reference"),
        "definition": "mean of seven native candidate/reference metric ratios; lower is better",
        "ratios": ratios,
        "scores": scores,
        "score_definitions": {
            "R": "mean of seven native candidate/reference ratios, including collision",
            "R_jsd": "mean of six JSD candidate/reference ratios, excluding collision",
            "R_attr": "mean of length, width and speed JSD candidate/reference ratios",
        },
        "scope_matches_reference": not mismatches,
        "scope_mismatches": mismatches,
        "interpretation": "single evaluation against a fixed baseline; not a pooled or paired multi-seed score",
        "reference_agent_metrics": {name: reference["agent_metrics"][name] for name in R_METRICS},
    }
