"""Review lane-query runs without treating short training as a final ranking."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "output/reports/lane_only_wandb_review_20261014"
JSD = ("speed_jsd", "ang_dev_jsd", "length_jsd", "width_jsd",
       "lat_dev_jsd", "nearest_dist_jsd")
RUNS = ("myqhg1rq", "8bblh5do", "0ub06wyq", "0vftb3cc", "cchgfuhr")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model(record):
    value = record["config"].get("model", {}).get("model_config", {})
    return ast.literal_eval(value) if isinstance(value, str) else value


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = item
    return result


def main():
    inventory_file = ART / "inventory.json"
    inventory = read(inventory_file)
    records = {r["id"]: r for r in inventory["runs"]}
    status = read(ART / "history_fetch_status.json")
    if status["errors"] or set(status["requested"]) != set(RUNS):
        raise RuntimeError("Review requires all five fresh unsampled histories")
    histories, manifest = {}, []
    for rid in RUNS:
        record = records[rid]
        path = ART / "histories" / (rid + ".json")
        history = read(path)
        groups = {}
        for row in history["rows"]:
            if row.get("epoch") is not None and all(
                isinstance(row.get(k), (int, float)) and math.isfinite(row[k])
                for k in JSD
            ):
                groups.setdefault(int(row["epoch"]), []).append(row)
        histories[rid] = {
            e: {k: statistics.fmean(row[k] for row in rows) for k in JSD}
            for e, rows in groups.items()
        }
        m = model(record)
        c = record["config"]
        limit = c.get("trainer", {}).get("limit_val_batches")
        scope = (dict(status="full_membership_enforced", scenes=m.get("sd_expected_scenes"))
                 if m.get("sd_require_full_set") else
                 dict(status="configured_batch_cap", scenes=limit * c["data"]["val_batch_size"])
                 if isinstance(limit, (int, float)) and limit > 1 else
                 dict(status="unverified", scenes=None))
        manifest.append(dict(
            id=rid, name=record["name"], state=record["state"], commit=record["commit"],
            url="https://forge.coreweave.com/wandb/u3006612/bc/runs/" + rid,
            map_condition_types=m["decoder"]["init_diffusion"]["map_condition_types"],
            scope=scope, training_epoch=record["summary"].get("epoch"),
            validation_history=histories[rid], history_sha256=sha(path),
            history_fetched_at_utc=history["fetched_at_utc"],
        ))
    comparisons = []
    for candidate, reference in (("0ub06wyq", "cchgfuhr"), ("myqhg1rq", "0vftb3cc")):
        ma, mb = model(records[candidate]), model(records[reference])
        a, b = flatten(ma["decoder"]), flatten(mb["decoder"])
        diff = {k: dict(candidate=a.get(k), reference=b.get(k))
                for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}
        for epoch in sorted(histories[candidate].keys() & histories[reference].keys()):
            ratios = {k: histories[candidate][epoch][k] / histories[reference][epoch][k]
                      for k in JSD}
            comparable_scope = bool(ma.get("sd_require_full_set") and mb.get("sd_require_full_set")
                                    and ma.get("sd_expected_scenes") == mb.get("sd_expected_scenes"))
            comparisons.append(dict(
                candidate=candidate, reference=reference, epoch=epoch,
                ratios=ratios, R_jsd=statistics.fmean(ratios.values()),
                R_focus=statistics.fmean(ratios[k] for k in JSD[:3]),
                comparable_declared_validation_scope=comparable_scope,
                matched_causal_ablation=False,
                decoder_config_differences=diff,
                source_commits_identical=records[candidate]["commit"] == records[reference]["commit"],
                caveat=("Observational: remote cohort hashes and GEN seeds not downloaded; source commits differ."
                        if comparable_scope else "4096 versus 50000 scenes; ratios are descriptive, not a matched ranking."),
            ))
    new_mode = [r["id"] for r in inventory["runs"]
                if (model(r).get("decoder", {}).get("init_diffusion", {}) or {}).get("map_condition_types") == "lanes_only"]
    result = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(), read_only=True,
        inventory_fetched_at_utc=inventory["fetched_at_utc"], inventory_sha256=sha(inventory_file),
        analyzer_sha256=sha(Path(__file__)), runs=manifest, comparisons=comparisons,
        genuine_lanes_only_remote_runs=new_mode,
        conclusion="Existing lanes_and_edges runs do not support long-term improvement. New lanes_only has no remote long-term evidence; withdraw its first-choice full-training recommendation.",
        recommendation=dict(config="init_diffusion_657_n20_road_k8_full", status="KEEP_PROVEN_FULL_REFERENCE",
                            lanes_only="EXPLORATORY_ONLY_REQUIRE_MID_TERM_VALIDATION"),
        limits=["Lane runs end at validation epoch23/31/47, not128 epochs.",
                "No new training, remote run changes or sampler changes were performed.",
                "Early map alignment advantage is not a reliable predictor of final speed/angle/shape JSD.",
                "The mechanism is unproven; frozen-feature shift and neighbor competition are hypotheses."])
    output = ART / "review.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for comparison in comparisons:
        if comparison["candidate"] == "0ub06wyq":
            print(json.dumps(comparison, ensure_ascii=False))
    print("genuine_lanes_only_remote_runs", new_mode)
    print("saved", output)


if __name__ == "__main__":
    main()
