"""Independent histogram and pairing checks for completed native JSD results."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "output/reports/wandb_cc_20261013"
KEYS = ("nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd", "width_jsd", "speed_jsd")


def read(p):
    return json.loads(p.read_text())


def main():
    scopes = [(ART / "jsd_priority/observations.json", ART / "jsd_priority/comparisons.json", False)]
    if (ART / "expanded/verified_completion.json").exists():
        scopes.append((ART / "expanded/observations.json", ART / "expanded/result.json", True))
    checks = []
    source = {}
    for obsfile, comparisonfile, expanded in scopes:
        if not comparisonfile.exists():
            continue
        rows = read(obsfile)
        source[str(obsfile)] = hashlib.sha256(obsfile.read_bytes()).hexdigest()
        source[str(comparisonfile)] = hashlib.sha256(comparisonfile.read_bytes()).hexdigest()
        comparisons = [read(comparisonfile)] if expanded else read(comparisonfile)["comparisons"]
        controls = {(r["seed"],r["budget"],r["eval_seed"]):r for r in rows if r["variant"]=="fresh657"}
        for row in rows:
            path = Path(row["metrics_path"])
            report = read(path)
            hist = read(path.parent / "feature_histograms.json")
            source[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            source[str(path.parent / "feature_histograms.json")] = hashlib.sha256((path.parent / "feature_histograms.json").read_bytes()).hexdigest()
            for index,key in enumerate(KEYS):
                value = jensenshannon(hist[key]["generated"], hist[key]["matched_gt"])**2 * (10 if index<2 else 100)
                assert math.isclose(value,row[key],rel_tol=0,abs_tol=1e-10)
                checks.append("histogram_reconstruction")
            control = controls[(row["seed"],row["budget"],row["eval_seed"])]
            base_hist = read(Path(control["metrics_path"]).parent / "feature_histograms.json")
            assert all(hist[key]["matched_gt"] == base_hist[key]["matched_gt"] for key in KEYS)
            assert row["num_gt_vehicles"]==control["num_gt_vehicles"]
            for key in ("train_order_sha256","eval_manifest_sha256","initial_generator_sha256","initial_map_sha256","calibration_sha256","map_checkpoint_sha256","seen_training_examples"):
                assert row[key]==control[key], key
            assert report["decoder"]["sampling_steps"]==20
            assert report["decoder"]["ema_num_updates"]==row["budget"]
            checks.append("paired_input_and_GT_histogram_identity")
        for comp in comparisons:
            candidate = [r for r in rows if r["variant"]==comp["arm"] and r["budget"]==comp["budget"]]
            base = [controls[(r["seed"],r["budget"],r["eval_seed"])] for r in candidate]
            seeds = sorted({r["seed"] for r in candidate})
            def mean(values,key):
                return statistics.fmean(statistics.fmean(r[key] for r in values if r["seed"]==s) for s in seeds)
            ratios = {key:mean(candidate,key)/mean(base,key) for key in KEYS}
            for key in KEYS:
                assert math.isclose(ratios[key],comp["ratios"][key],rel_tol=0,abs_tol=1e-12)
            for key,metrics in (("R_jsd",KEYS),("R_geometry",KEYS[:3]),("R_attr",KEYS[3:])):
                assert math.isclose(statistics.fmean(ratios[k] for k in metrics),comp[key],rel_tol=0,abs_tol=1e-12)
                checks.append("independent_score_reconstruction")
    assert checks, "No completed observations to verify"
    out = {"all_checks_pass":True,"check_count":len(checks),
           "counts":{k:checks.count(k) for k in sorted(set(checks))},"source_sha256":source,
           "does_not_establish_128epoch_convergence":True}
    (ART / "jsd_numeric_qa.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="source_sha256"}))


if __name__ == "__main__":
    main()
