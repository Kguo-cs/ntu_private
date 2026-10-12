"""Recover the frozen overnight long matrix with deferred cyclic GC and no manual GC.

Default is a CPU-only dry run. Only --execute submits isolated GPU subprocesses.
The frozen model/runner/controller and final exposure budget are never rewritten.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import continue_overnight_init_diffusion as base

ART = base.ART
RUNS = base.RUNS
REC = ART / "runtime_recovery"
GIB_KIB = 1024 ** 2
RUNTIME_LOADER_WRAPPER = None
RUNTIME_LOADER_DEPENDENCIES = []


def runtime_sources():
    paths = ([RUNTIME_LOADER_WRAPPER] if RUNTIME_LOADER_WRAPPER else []) + RUNTIME_LOADER_DEPENDENCIES
    return {str(path): sha(path) for path in sorted(set(paths))}


def interrupted(signum, unused_frame):
    raise KeyboardInterrupt(f"Recovery supervisor received signal {signum}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_sources():
    expected = base.read(ART / "source_freeze.json")["source_hashes"]
    current = base.source_hashes()
    if current != expected:
        raise RuntimeError("Frozen source hashes differ; recovery cannot silently refreeze")
    return current


def fingerprints(arm, command, inputs, checkpoint=None):
    result = {"design_inputs": {path: base.digest(value) for path, value in inputs.items()},
              "source_hashes": frozen_sources(),
              "external_inputs": {str(path): sha(path) for path in (arm["map"], arm["normalizer"])}}
    if "--eval-manifest" in command:
        manifest = command[command.index("--eval-manifest") + 1]
        if manifest not in inputs:
            result["external_inputs"][manifest] = sha(manifest)
    if checkpoint:
        result["external_inputs"][str(checkpoint)] = sha(checkpoint)
    return result


def describe(label, arm, steps, seed, *, checkpoint=None):
    control_steps = base.read(ART / "frozen_long_protocol.json")["final_budget_B512"]
    checks = [] if checkpoint else [step * 512 // arm["batch"]
                                   for step in (control_steps // 3, control_steps * 2 // 3)]
    scenes = 50000 if checkpoint else 4096
    original, inputs = base.make_command(label, arm, steps=steps, scenes=scenes, seed=seed,
                                        checkpoints=checks, extra_seed=True,
                                        evaluation_checkpoint=checkpoint)
    command = list(original)
    command[command.index("--young-gc-interval") + 1] = "0"
    if RUNTIME_LOADER_WRAPPER is not None:
        command[1] = str(RUNTIME_LOADER_WRAPPER)
    spec = {"name": arm["name"], "seed": seed, "steps": steps, "scenes": scenes,
            "batch": arm["batch"], "generations": [2817, 3817],
            "budgets": [steps] if checkpoint else sorted(set([steps, *checks]))}
    return {"label": label, "arm": arm, "command": command, "original_command": original,
            "inputs": inputs, "spec": spec, "checkpoint": str(checkpoint) if checkpoint else None}


def existing_success(stage):
    folder = RUNS / stage["label"]
    if not (folder / "execution.json").exists():
        return None
    execution = base.read(folder / "execution.json")
    if execution.get("status") != "success" or execution.get("returncode") != 0:
        return None
    actual_command = execution.get("command")
    if actual_command not in (stage["command"], stage["original_command"]):
        raise RuntimeError(stage["label"] + ": successful stage command is incompatible")
    fp = fingerprints(stage["arm"], actual_command, stage["inputs"], stage["checkpoint"])
    if execution.get("fingerprints") != fp or execution.get("arm") != stage["arm"]:
        raise RuntimeError(stage["label"] + ": successful stage source/design fingerprint differs")
    if actual_command == stage["command"]:
        runtime = execution.get("runtime_recovery", {})
        loader_sha = sha(RUNTIME_LOADER_WRAPPER) if RUNTIME_LOADER_WRAPPER else None
        if (runtime.get("wrapper_sha256") != sha(__file__) or runtime.get("loader_wrapper_sha256") != loader_sha
                or runtime.get("runtime_loader_source_hashes", {}) != runtime_sources()):
            raise RuntimeError(stage["label"] + ": recovery wrapper source changed")
    rows, artifacts = base.verify_completion(folder, RUNS / (stage["label"] + ".log"), stage["spec"])
    if execution.get("artifacts") != artifacts:
        raise RuntimeError(stage["label"] + ": successful stage artifact bytes differ")
    return rows


def verify_scope(stage, rows):
    for row in rows:
        report = base.read(row["metrics_path"])
        full = stage["spec"]["scenes"] == 50000
        if (report.get("require_full_set") is not full or report.get("full_membership") is not full
                or report.get("map_source") != "reference"
                or report.get("decoder", {}).get("mode") != "lane_conditioned"):
            raise RuntimeError(stage["label"] + ": actual full membership/map conditioning differs")
        if full:
            if (row["num_gt_vehicles"] != 362309 or row["num_gt_onroad_vehicles"] != 360099
                    or report.get("feature_counts_gt") != [358867, 360099, 360099, 362309, 362309, 362309]):
                raise RuntimeError(stage["label"] + ": full50000 physical GT counts differ")


def memory_limit_reason(rss_kib, available_kib, max_rss_gib=18., min_available_gib=1.5):
    if rss_kib > max_rss_gib * GIB_KIB:
        return "child_RSS_exceeded_limit"
    if available_kib < min_available_gib * GIB_KIB:
        return "host_MemAvailable_below_limit"
    return None


def event(name, **details):
    path = REC / "events.json"
    rows = base.read(path) if path.exists() else []
    rows.append({"event": name, "unix": time.time(), **details})
    base.save(path, rows)


def prepare_archive(stages):
    """Only the known original manual-GC failure is eligible for this fresh retry."""
    failed = []
    for stage in stages:
        folder = RUNS / stage["label"]
        if not folder.exists() or existing_success(stage) is not None:
            continue
        p = folder / "execution.json"
        ex = base.read(p) if p.exists() else {}
        if (stage["label"] != "long_seed4817_separate_speed" or ex.get("status") != "failed"
                or ex.get("returncode") != -11 or ex.get("command") != stage["original_command"]):
            raise RuntimeError(stage["label"] + ": unverified partial output; manual recovery required")
        failed.append(stage["label"])
    return failed


def archive_attempt(failed):
    archive = REC / "attempt1"
    if archive.exists():
        if failed:
            raise RuntimeError("Recovery archive already exists; refusing replay")
        return
    archive.mkdir(parents=True)
    records = []
    for name in failed:
        for source in (RUNS / name, RUNS / (name + ".log")):
            if not source.exists():
                raise RuntimeError("Missing known failed stage/log while archiving")
            target = archive / source.name
            records.append({"original": str(source), "archive": str(target),
                            "sha256": sha(source) if source.is_file() else None})
            source.rename(target)
    for name in ("queue_complete.json", "final_selection.json", "long_completed_rows.json", "queue_active.json"):
        source = ART / name
        if source.exists():
            target = archive / name
            records.append({"original": str(source), "archive": str(target), "sha256": sha(source)})
            source.rename(target)
    base.save(archive / "archive_ledger.json", {"records": records, "status": "ARCHIVED_FAILED_MANUAL_GC_ATTEMPT",
              "actual_failure": "SIGSEGV during manual collect(0) after TRAIN step5760; not deadline exhaustion"})
    event("RECOVERY_STARTED", archive=str(archive), failed_stages=failed,
          frozen_long_sha256=sha(ART / "frozen_long_protocol.json"), wrapper_sha256=sha(__file__))
    base.save(ART / "queue_complete.json", {"status": "RECOVERING", "complete_final_matrix": False,
              "complete_full50000_audit": False, "recovery_ledger": str(REC / "protocol.json")})


def stop_owned(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_stage(stage, args):
    recorded = base.read(REC / "protocol.json")
    if recorded.get("runtime_loader_source_hashes", {}) != runtime_sources() or recorded["wrapper_sha256"] != sha(__file__):
        raise RuntimeError("Runtime recovery executable source changed after freeze")
    rows = existing_success(stage)
    if rows is not None:
        verify_scope(stage, rows)
        event("INHERIT_VERIFIED_SUCCESS", label=stage["label"],
              original_young_gc_interval=base.read(RUNS / stage["label"] / "protocol.json")["young_gc_interval"])
        print("SKIP VERIFIED COMPLETE " + stage["label"], flush=True)
        return rows
    folder = RUNS / stage["label"]
    if folder.exists():
        raise RuntimeError(stage["label"] + ": partial recovery output; no silent replay")
    if time.time() >= base.DEADLINE:
        return []
    fp = fingerprints(stage["arm"], stage["command"], stage["inputs"], stage["checkpoint"])
    folder.mkdir(parents=True, exist_ok=False)
    for path, value in stage["inputs"].items():
        base.save(path, value)
    execution = {"label": stage["label"], "arm": stage["arm"], "command": stage["command"],
        "seed": stage["spec"]["seed"], "started_unix": time.time(), "status": "running",
        "source_hashes": fp["source_hashes"], "fingerprints": fp, "completion_spec": stage["spec"],
        "runtime_recovery": {"wrapper_sha256": sha(__file__), "ledger": str(REC / "protocol.json"),
                             "loader_wrapper_sha256": sha(RUNTIME_LOADER_WRAPPER) if RUNTIME_LOADER_WRAPPER else None,
                             "runtime_loader_source_hashes": runtime_sources(),
                             "young_gc_interval": 0, "automatic_gc": "disabled", "fresh_training": not bool(stage["checkpoint"])}}
    base.save(folder / "execution.json", execution)
    log_path = RUNS / (stage["label"] + ".log")
    print("START " + stage["label"], flush=True)
    with log_path.open("w") as log:
        process = None
        try:
            process = subprocess.Popen(stage["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            execution["child_pid"] = process.pid
            base.save(folder / "execution.json", execution)
            base.save(ART / "queue_active.json", {"stage": stage["label"], "child_pid": process.pid,
                      "controller_pid": os.getpid(), "log": str(log_path), "recovery_ledger": str(REC / "protocol.json")})
            while process.poll() is None:
                try:
                    lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                    rss = int(next(x for x in lines if x.startswith("VmRSS:")).split()[1])
                    avail = int(next(x for x in Path("/proc/meminfo").read_text().splitlines()
                                     if x.startswith("MemAvailable:")).split()[1])
                    execution.update(last_rss_kib=rss, last_host_memavailable_kib=avail,
                                     peak_observed_rss_kib=max(rss, execution.get("peak_observed_rss_kib", 0)))
                    reason = memory_limit_reason(rss, avail, args.max_rss_gib, args.min_available_gib)
                    if reason:
                        execution.update(status="memory_interrupted", stop_reason=reason)
                        stop_owned(process)
                        break
                except (OSError, StopIteration, ValueError):
                    pass
                if time.time() >= base.DEADLINE:
                    execution.update(status="deadline_interrupted", stop_reason="UTC01:35 GPU cutoff")
                    stop_owned(process)
                    break
                base.save(folder / "execution.json", execution)
                time.sleep(10)
        except BaseException:
            if process is not None:
                stop_owned(process)
            execution.update(status="supervisor_interrupted", ended_unix=time.time(), returncode=process.returncode if process else None)
            base.save(folder / "execution.json", execution)
            raise
    execution.update(ended_unix=time.time(), returncode=process.returncode)
    if execution["status"] == "running":
        execution["status"] = "completed" if process.returncode == 0 else "failed"
    base.save(folder / "execution.json", execution)
    event("STAGE_EXIT", label=stage["label"], status=execution["status"], returncode=process.returncode,
          peak_observed_rss_kib=execution.get("peak_observed_rss_kib"), stop_reason=execution.get("stop_reason"))
    if execution["status"] != "completed":
        return []
    rows, artifacts = base.verify_completion(folder, log_path, stage["spec"])
    verify_scope(stage, rows)
    execution.update(status="success", artifacts=artifacts, last_verified_unix=time.time())
    base.save(folder / "execution.json", execution)
    return rows


def select_final(final, proto, arms):
    matrix = [row for rows in final.values() for row in rows]
    required = {(name, seed, gen) for name in final for seed in (4817, 5817) for gen in (2817, 3817)}
    actual = {(r["variant"], r["seed"], r["eval_seed"]) for r in matrix}
    complete = actual == required and len(matrix) == 8
    decision = {"complete_matrix": complete, "rows": matrix, "frozen_challenger": proto["challenger"],
                "actual_final_exposure_per_TRAIN": proto["actual_final_exposure_per_TRAIN"],
                "final_budget_rule": proto["final_budget_rule"], "runtime_recovery_ledger": str(REC / "protocol.json"),
                "runtime_comparability": "Inherited TRAIN4817 control manual-GC128; remaining runs no manual GC. Same objective/model/data/seed/budget, runtime differs; explicitly disclose.",
                "independent_5817_no_reselection": True}
    if not complete:
        return decision
    if len({base.digest(base.row_cohort(row)) for row in matrix}) != 1:
        raise ValueError("Final physical/model cohort differs")
    if {r["seen_training_examples"] for r in matrix} != {proto["actual_final_exposure_per_TRAIN"]}:
        raise ValueError("Final exposure differs from frozen budget")
    baseline = base.aggregate(final["control"])
    ranked = []
    for name, rows in final.items():
        comparison = dict(base.compare(base.aggregate(rows), baseline), name=name)
        per_seed = {str(seed): base.compare(base.aggregate([r for r in rows if r["seed"] == seed]),
                    base.aggregate([r for r in final["control"] if r["seed"] == seed])) for seed in (4817, 5817)}
        comparison.update(per_training_seed=per_seed, aggregate_gates_pass=comparison["eligible"],
                          independent_5817=per_seed["5817"],
                          eligible=comparison["eligible"] and all(x["eligible"] for x in per_seed.values()))
        ranked.append(comparison)
    ranked.sort(key=lambda r: (not r["eligible"], r["R"], r["R_attr"], r["name"]))
    better = [r for r in ranked if r["eligible"] and r["R"] < 1.]
    selected = better[0]["name"] if better else "control"
    decision.update(selected=selected, ranking=ranked, selected_arm=arms[selected])
    return decision


def main():
    global RUNTIME_LOADER_WRAPPER, RUNTIME_LOADER_DEPENDENCIES
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--max-rss-gib", type=float, default=18.)
    parser.add_argument("--min-available-gib", type=float, default=1.5)
    parser.add_argument("--runtime-loader-wrapper", type=Path,
                        help="Optional tools-only subprocess entrypoint; must accept the original sweep CLI")
    parser.add_argument("--runtime-loader-dependency", type=Path, action="append", default=[],
                        help="Additional executable helper to freeze in the recovery ledger; repeatable")
    args = parser.parse_args()
    if args.max_rss_gib <= 0 or args.min_available_gib <= 0:
        parser.error("Memory bounds must be positive")
    if args.runtime_loader_wrapper:
        RUNTIME_LOADER_WRAPPER = args.runtime_loader_wrapper.resolve(strict=True)
        if not RUNTIME_LOADER_WRAPPER.is_relative_to(ROOT / "tools"):
            parser.error("Runtime loader wrapper must be an explicit repository tools file")
        # This loader's metaclass implementation is executable, so hash it too.
        helper = ROOT / "tools/torch_pickle_class_cache.py"
        if helper.exists():
            RUNTIME_LOADER_DEPENDENCIES.append(helper.resolve())
    for path in args.runtime_loader_dependency:
        path = path.resolve(strict=True)
        if not path.is_relative_to(ROOT / "tools") or not path.is_file():
            parser.error("Runtime helper must be an explicit repository tools file")
        RUNTIME_LOADER_DEPENDENCIES.append(path)
    if args.execute:
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
    frozen_sources()
    proto = base.read(ART / "frozen_long_protocol.json")
    if proto["train_seeds"] != [4817, 5817] or proto["generation_seeds"] != [2817, 3817]:
        raise RuntimeError("Unexpected frozen seed matrix")
    arms = {x["name"]: x for x in base.read(ART / "search_protocol.json")["arms"]}
    stages = [describe(f"long_seed{seed}_{name}", arms[name],
                       proto["final_budget_B512"] if arms[name]["batch"] == 512 else proto["final_budget_B256"], seed)
              for seed in proto["train_seeds"] for name in ("control", proto["challenger"])]
    failed = prepare_archive(stages)
    plan = {"status": "CPU_DRY_RUN_PLAN" if not args.execute else "RECOVERY_EXECUTION_PROTOCOL",
        "wrapper_sha256": sha(__file__), "wrapper": str(Path(__file__).resolve()),
        "loader_wrapper": str(RUNTIME_LOADER_WRAPPER) if RUNTIME_LOADER_WRAPPER else None,
        "loader_wrapper_sha256": sha(RUNTIME_LOADER_WRAPPER) if RUNTIME_LOADER_WRAPPER else None,
        "runtime_loader_source_hashes": runtime_sources(),
        "frozen_long_protocol_sha256": sha(ART / "frozen_long_protocol.json"),
        "source_freeze_sha256": sha(ART / "source_freeze.json"), "source_hashes": frozen_sources(),
        "unchanged_model_objective_data_seeds_and_budget": True, "checkpoint_resume": False,
        "runtime_change": {"inherited_success_young_gc_interval": 128, "remaining_young_gc_interval": 0,
                           "automatic_cyclic_gc_disabled": True, "manual_collection_disabled": True,
                           "imports_frozen": False, "reference_counting_active": True},
        "memory_guard": {"child_RSS_limit_GiB": args.max_rss_gib, "host_available_min_GiB": args.min_available_gib,
                         "poll_seconds": 10, "only_self_submitted_child_process_groups": True},
        "gpu_cutoff_unix": base.DEADLINE, "no_budget_recalculation": True,
        "failed_stages_to_archive": failed,
        "stages": [{"label": s["label"], "command": s["command"], "spec": s["spec"],
                    "inherit_verified_success": existing_success(s) is not None} for s in stages],
        "full50000": {"arms": ["control", proto["challenger"]], "TRAIN": 4817, "GEN": [2817, 3817],
                      "require_full_set": True, "reference_mode": "matched", "no_reselection": True},
        "runtime_limitations": ["No-GC may exhaust host memory; the guard stops and preserves partial diagnostics.",
                                "Runtime cohorts differ for inherited control4817; do not claim exact runtime equality."]}
    plan["full50000"]["planned_stages"] = []
    for name in ("control", proto["challenger"]):
        budget = proto["final_budget_B512"] if arms[name]["batch"] == 512 else proto["final_budget_B256"]
        stage = describe("full50000_seed4817_" + name, arms[name], budget, 4817,
                         checkpoint=RUNS / f"long_seed4817_{name}" / name / "last.ckpt")
        plan["full50000"]["planned_stages"].append({"label": stage["label"], "command": stage["command"],
                                                     "spec": stage["spec"], "inputs": stage["inputs"]})
    base.save(ART / "runtime_recovery_plan.json", plan)
    if not args.execute:
        print(json.dumps({"status": plan["status"], "plan": str(ART / "runtime_recovery_plan.json"),
                          "verified_inherited": [s["label"] for s in plan["stages"] if s["inherit_verified_success"]]}, indent=2))
        return
    REC.mkdir(parents=True, exist_ok=True)
    lock = (REC / "execution.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (REC / "protocol.json").exists():
        # Completion flags and the initial failed-stage archive are live state,
        # not immutable command/design/runtime inputs. Verified successes reuse.
        previous = base.read(REC / "protocol.json")
        def immutable(value):
            value = dict(value)
            value.pop("failed_stages_to_archive", None)
            value["stages"] = [{k: v for k, v in stage.items() if k != "inherit_verified_success"}
                               for stage in value["stages"]]
            return value
        if immutable(previous) != immutable(plan):
            raise RuntimeError("Existing recovery protocol differs; no silent restart")
    else:
        base.save(REC / "protocol.json", plan)
    active = base.read(ART / "queue_active.json") if (ART / "queue_active.json").exists() else {}
    for key in ("child_pid", "controller_pid"):
        pid = active.get(key)
        if pid and pid != os.getpid() and base.process_running(pid):
            raise RuntimeError("Prior queue process still alive; refusing GPU overlap")
    archive_attempt(failed)
    final = {"control": [], proto["challenger"]: []}
    failed_stage = None
    for stage in stages:
        rows = run_stage(stage, args)
        final[stage["arm"]["name"]].extend(base.endpoint(rows))
        base.save(ART / "long_completed_rows.json", final)
        if not rows:
            failed_stage = stage["label"]
            break
    decision = select_final(final, proto, arms)
    base.save(ART / "final_selection.json", decision)
    audits = {}
    if decision["complete_matrix"]:
        for name in final:
            origin = next(r for r in final[name] if r["seed"] == 4817 and r["eval_seed"] == 2817)
            stage = describe("full50000_seed4817_" + name, arms[name], origin["budget"], 4817,
                             checkpoint=origin["checkpoint"])
            audits[name] = run_stage(stage, args)
            if not audits[name]:
                failed_stage = stage["label"]
                break
        base.save(ART / "full50000_audit.json", {"selection_is_frozen": True, "no_reselection": True,
                  "train_seed": 4817, "generation_seeds": [2817, 3817], "rows_by_arm": audits,
                  "complete": len(audits) == 2 and all(len(r) == 2 for r in audits.values()),
                  "physical_matched_reference": True, "require_full_set": True,
                  "runtime_recovery_ledger": str(REC / "protocol.json")})
    full = len(audits) == 2 and all(len(r) == 2 for r in audits.values())
    status = "COMPLETE" if decision["complete_matrix"] and full else "PARTIAL_RECOVERY"
    base.save(ART / "queue_complete.json", {"status": status, "complete_final_matrix": decision["complete_matrix"],
              "complete_full50000_audit": full, "finished_unix": time.time(), "selected": decision.get("selected"),
              "failed_or_interrupted_stage": failed_stage, "runtime_recovery_ledger": str(REC / "protocol.json")})
    base.save(ART / "queue_active.json", {"stage": status, "controller_pid": os.getpid(),
                                         "runtime_recovery_ledger": str(REC / "protocol.json")})
    event("RECOVERY_FINISHED", status=status, failed_or_interrupted_stage=failed_stage)
    print("RECOVERY " + status, flush=True)


if __name__ == "__main__":
    main()
