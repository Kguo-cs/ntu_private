"""Certify a completed real runtime-loader stage, then continue frozen recovery.

This only repairs the terminal-log verifier. The first recovery supervisor and
loader/model sources remain frozen; no missing training/evaluation is synthesized.
Default is CPU dry run. --execute requires both prior processes to have exited.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import recover_overnight_init_diffusion_runtime as v1
from tools import runtime_completion


def setup():
    protocol = v1.base.read(v1.REC / "protocol.json")
    if protocol["wrapper_sha256"] != v1.sha(v1.__file__):
        raise RuntimeError("First recovery supervisor changed")
    v1.frozen_sources()
    if protocol["frozen_long_protocol_sha256"] != v1.sha(v1.ART / "frozen_long_protocol.json"):
        raise RuntimeError("Frozen long budget changed")
    v1.RUNTIME_LOADER_WRAPPER = Path(protocol["loader_wrapper"])
    v1.RUNTIME_LOADER_DEPENDENCIES = [Path(p) for p in protocol["runtime_loader_source_hashes"]
                                   if p != protocol["loader_wrapper"]]
    if v1.runtime_sources() != protocol["runtime_loader_source_hashes"]:
        raise RuntimeError("Frozen loader sources changed")
    return protocol


def actual_completed(stage):
    folder = v1.RUNS / stage["label"]
    path = folder / "execution.json"
    if not path.exists():
        return None
    execution = v1.base.read(path)
    if execution.get("status") != "completed" or execution.get("returncode") != 0:
        return None
    if (execution.get("command") != stage["command"] or execution.get("arm") != stage["arm"]
            or execution.get("completion_spec") != stage["spec"]
            or execution.get("fingerprints") != v1.fingerprints(stage["arm"], stage["command"], stage["inputs"], stage["checkpoint"])):
        raise RuntimeError("Completed stage command/source/design differs from its frozen plan")
    runtime = execution.get("runtime_recovery", {})
    contract = v1.base.read(v1.REC / "protocol.json")
    if (runtime.get("wrapper_sha256") != contract["wrapper_sha256"]
            or runtime.get("runtime_loader_source_hashes") != contract["runtime_loader_source_hashes"]):
        raise RuntimeError("Completed stage runtime sources differ from frozen recovery contract")
    rows, artifacts = runtime_completion.verify_completion(folder, v1.RUNS / (stage["label"] + ".log"), stage["spec"])
    v1.verify_scope(stage, rows)
    return execution, rows, artifacts


def stages_for_review():
    frozen = v1.base.read(v1.ART / "frozen_long_protocol.json")
    arms = {a["name"]: a for a in v1.base.read(v1.ART / "search_protocol.json")["arms"]}
    stages = [v1.describe(f"long_seed{seed}_{name}", arms[name],
                         frozen["final_budget_B512"] if name == "control" else frozen["final_budget_B256"], seed)
              for seed in frozen["train_seeds"] for name in ("control", frozen["challenger"])]
    for name in ("control", frozen["challenger"]):
        budget = frozen["final_budget_B512"] if name == "control" else frozen["final_budget_B256"]
        checkpoint = v1.RUNS / f"long_seed4817_{name}" / name / "last.ckpt"
        stages.append(v1.describe("full50000_seed4817_" + name, arms[name], budget, 4817, checkpoint=checkpoint))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol = setup()
    # Patch this process's read-only verifier, never the frozen file. The helper
    # retained the original function before this assignment.
    runtime_completion.bind_legacy_verifier(v1.base.verify_completion)
    v1.base.verify_completion = runtime_completion.verify_completion
    candidates = []
    live_status = []
    for stage in stages_for_review():
        execution_path = v1.RUNS / stage["label"] / "execution.json"
        status = v1.base.read(execution_path).get("status") if execution_path.exists() else "not_started"
        completed = actual_completed(stage)
        if completed:
            candidates.append((stage, completed))
        elif status == "success":
            # Verify existing success with actual raw log/artifact hashes.
            if v1.existing_success(stage) is None:
                raise RuntimeError("Success could not be strictly verified")
        live_status.append({"stage": stage["label"], "status": status,
                            "actual_completion_certifiable": completed is not None})
    bridge = {"status": "CPU_DRY_RUN" if not args.execute else "VERIFICATION_BRIDGE_EXECUTION",
        "source_hashes": {str(Path(__file__).resolve()): v1.sha(__file__),
                          str(Path(runtime_completion.__file__).resolve()): v1.sha(runtime_completion.__file__)},
        "original_recovery_protocol_sha256": v1.sha(v1.REC / "protocol.json"),
        "original_recovery_controller_sha256": protocol["wrapper_sha256"],
        "frozen_long_protocol_sha256": protocol["frozen_long_protocol_sha256"],
        "repair": "Authenticate DONE + exact runtime success suffix/policy; preserve raw log hash. No objective/runtime/data/budget changes.",
        "no_training_or_evaluation_fabricated": True, "fresh_remaining_training": True,
        "actual_stages": live_status,
        "certifiable_stages": [stage["label"] for stage, _ in candidates]}
    v1.base.save(v1.REC / "verification_bridge_plan.json", bridge)
    if not args.execute:
        print(json.dumps(bridge, indent=2))
        return
    lock = (v1.REC / "bridge.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = v1.base.read(v1.ART / "queue_active.json")
    for key in ("controller_pid", "child_pid"):
        pid = active.get(key)
        if pid and pid != os.getpid() and v1.base.process_running(pid):
            raise RuntimeError("First recovery supervisor/child still lives; do not interrupt it")
    stored = v1.REC / "verification_bridge.json"
    if stored.exists():
        prior = v1.base.read(stored)
        if prior["source_hashes"] != bridge["source_hashes"] or prior["original_recovery_protocol_sha256"] != bridge["original_recovery_protocol_sha256"]:
            raise RuntimeError("Verification bridge source/protocol changed")
    else:
        v1.base.save(stored, bridge)
    v1.event("VERIFICATION_BRIDGE_STARTED", source_hashes=bridge["source_hashes"],
             original_recovery_protocol_sha256=bridge["original_recovery_protocol_sha256"])
    for stage, (execution, rows, artifacts) in candidates:
        path = v1.RUNS / stage["label"] / "execution.json"
        previous_sha = v1.sha(path)
        v1.base.save(v1.REC / (stage["label"] + ".completed_before_certification.json"), execution)
        execution.update(status="success", artifacts=artifacts, last_verified_unix=time.time(),
            completion_verification_bridge={"ledger": str(stored), "source_hashes": bridge["source_hashes"],
                "prior_execution_sha256": previous_sha,
                "actual_complete_rows": len(rows), "actual_returncode": 0})
        v1.base.save(path, execution)
        v1.event("CERTIFY_ACTUAL_COMPLETED_STAGE", label=stage["label"], prior_execution_sha256=previous_sha,
                 actual_rows=len(rows), raw_log_sha256=artifacts["log"]["sha256"], source_hashes=bridge["source_hashes"])
    # Continue exactly the original recovery logic with the certified verifier.
    sys.argv = [v1.__file__, "--execute", "--runtime-loader-wrapper", protocol["loader_wrapper"],
                "--max-rss-gib", str(protocol["memory_guard"]["child_RSS_limit_GiB"]),
                "--min-available-gib", str(protocol["memory_guard"]["host_available_min_GiB"])]
    v1.main()


if __name__ == "__main__":
    main()
