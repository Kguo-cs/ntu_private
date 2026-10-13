"""Independently recompute reported W&B ratios from unsampled source histories."""
import collections
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "output/reports/wandb_cc_20261013"
METRICS = ["nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
           "width_jsd", "speed_jsd", "collision_rate"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a = json.loads((ROOT / "refresh_analysis.json").read_text())
    histories, checks = {}, []
    for path in sorted((ROOT / "histories").glob("*.json")):
        snapshot = json.loads(path.read_text())
        bins = collections.defaultdict(list)
        for row in snapshot["rows"]:
            assert row.get("_step", -1) < snapshot["snapshot_max_step_exclusive"]
            if row.get("epoch") is not None and all(isinstance(row.get(k), (int, float))
                    and math.isfinite(row[k]) for k in METRICS):
                bins[int(row["epoch"])].append(row)
        histories[path.stem] = {e: {k: statistics.fmean(r[k] for r in rows)
                                    for k in METRICS} for e, rows in bins.items()}
        checks.append(path.stem + " source step cap")
    for comp in a["comparisons"]:
        ratios = []
        for key in METRICS:
            c = statistics.fmean(histories[comp["candidate"]][e][key] for e in comp["epochs"])
            b = statistics.fmean(histories[comp["baseline"]][e][key] for e in comp["epochs"])
            value = c / b
            assert math.isclose(value, comp["ratios"][key], rel_tol=1e-12, abs_tol=1e-12)
            ratios.append(value)
        assert math.isclose(statistics.fmean(ratios), comp["observed_R7"], rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(statistics.fmean(ratios[3:6]), comp["observed_R_attr"], rel_tol=1e-12, abs_tol=1e-12)
        assert comp["conditions"]["val_batch_size"]["equal"] and comp["paired_multiseed_R"] is False
        checks.append(comp["candidate"] + " " + comp["baseline"] + " " + str(comp["epochs"]) + " numeric/configured-cohort proof")
    for file, expected in a["source_files"].items():
        assert sha(ROOT / file) == expected
        checks.append(file + " source sha")
    assert sha(Path(__file__).with_name("analyze_wandb_cc_snapshot.py")) == a["analysis_script_sha256"]
    checks.append("analysis generator source sha")
    assert set(a["online_R_reference_audit"]) == {"arpfw0fr", "az3mgqvh"}
    assert all(not r["used_for_same_cohort_ranking"] for r in a["online_R_reference_audit"].values())
    checks.append("online cross-scope R exclusion")
    qa = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "PASS",
          "numeric_checks": checks, "check_count": len(checks), "analysis_sha256": sha(ROOT / "refresh_analysis.json"),
          "verification_script_sha256": sha(Path(__file__)), "no_gpu": True,
          "remote_dataset_membership_proven": False, "remote_gen_pairing_proven": False}
    (ROOT / "numeric_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    files = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name != "source_manifest.json"]
    manifest = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only_remote_access": True, "no_checkpoint_payloads": True,
        "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files],
        "scripts": {str(p.relative_to(ROOT.parents[2])): sha(p) for p in [Path(__file__),
            Path(__file__).with_name("analyze_wandb_cc_snapshot.py"), Path(__file__).with_name("refresh_wandb_cc_readonly.py")]}}
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PASS", len(checks), "checks", "analysisSHA", qa["analysis_sha256"])


if __name__ == "__main__":
    main()
