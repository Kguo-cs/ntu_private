"""Build an actual-only Markdown report of the overnight InitDiffusion search.

This CPU utility re-verifies completed stages and renders compact actual-only
figures without importing torch, running training, generating PDFs, writing
selection, or creating a completion marker. Missing or interrupted experiments
remain missing. Native commands are published only after an actual materializer
artifact passes provenance checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

if __name__ == "__main__":
    # CPU report CLI only. Avoid collector activity during optional plotting
    # imports; actual transient exit139 events remain diagnostic observations.
    import gc
    gc.disable()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import overnight_search_analysis as analysis
from tools import continue_overnight_init_diffusion as controller

ART = ROOT / "docs/experiments/init_diffusion_overnight_search_20261011"
HISTORICAL_ART = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
OUTPUT = ROOT / "docs/init_diffusion_overnight_search_20261011.md"
METRICS = controller.METRICS
SHORT = ("Near", "Lat", "Ang", "Length", "Width", "Speed", "Col%")
MULTIPLIERS = {key: (1. if key == "collision_rate" else 10. if key in METRICS[:2] else 100.)
               for key in METRICS}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(text)
        temporary = f.name
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = handle.name
    try:
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def link(path, label=None, *, output=OUTPUT):
    path = Path(path).resolve()
    relative = os.path.relpath(path, Path(output).resolve().parent).replace(os.sep, "/")
    return f"[{label or path.name}](<{relative}>)"


def image_link(path, caption, *, output=OUTPUT):
    """Bare image targets also work with the repository's PDF Markdown parser."""
    relative = os.path.relpath(Path(path).resolve(), Path(output).resolve().parent).replace(os.sep, "/")
    if any(character in relative for character in "()\n\r "):
        raise ValueError("Embedded figure target needs a simple path: " + relative)
    return f"![{caption.replace(']', '')}]({relative})"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def number(value, digits=6):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def stat(value):
    if not value:
        return "—"
    mean = number(value.get("mean"))
    if value.get("n", 0) < 2 or value.get("sd") is None:
        return mean
    return mean + " ± " + number(value["sd"])


def table(headers, rows):
    return ["| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(map(cell, row)) + " |" for row in rows], ""]


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        result.update(flatten(item, prefix + "." + key if prefix else key))
    return result


def config_delta(arm, control):
    baseline = control.get("config", {})
    changed = {key: value for key, value in arm.get("config", {}).items() if baseline.get(key) != value}
    names = []
    for key, value in sorted(changed.items()):
        name = key.removeprefix(controller.PREFIX).removeprefix("model.model_config.")
        names.append(name + "=" + json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    if arm.get("lr", .0005) != control.get("lr", .0005):
        names.append("effective_lr=" + str(arm["lr"]))
    if arm.get("normalizer") != control.get("normalizer"):
        names.append("normalizer=" + Path(arm["normalizer"]).name)
    return "; ".join(names) or "与 control 相同"


def loss_delta(arm, control):
    base, current = flatten(control.get("recipe", {})), flatten(arm.get("recipe", {}))
    return "; ".join(key.removeprefix("components.") + "=" + str(value)
                     for key, value in sorted(current.items()) if base.get(key) != value) or "与 control 相同"


def verified_raw_rows(stages):
    """Only stage roots already accepted by the atomic completion verifier."""
    result = []
    for stage in stages:
        if stage.get("included"):
            result.extend(read(Path(stage["run"]) / "summary.json", []))
    return result


def clipping_evidence(folder, *, label, scope, expected_steps, expected_clip=None):
    """Cross-check diagnostic summaries against every actual pre-update trace.

    These sidecars support a clipping observation, not a causal interpretation
    or the completion/selection decision. They are never treated as endpoints.
    """
    folder = Path(folder)
    summary_path, trace_path = folder / "train_summary.json", folder / "train_trace.json"
    summary, trace = read(summary_path), read(trace_path)
    if not isinstance(summary, dict) or not isinstance(trace, list) or not trace:
        raise ValueError("Missing actual clipping diagnostic summary/trace: " + str(folder))
    steps, clip = summary.get("steps"), summary.get("gradient_clip_val")
    if steps != expected_steps or len(trace) != steps or [r.get("step") for r in trace] != list(range(steps)):
        raise ValueError("Clipping trace is incomplete, resumed, or has repeated steps: " + str(folder))
    if not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip < 0:
        raise ValueError("Invalid actual gradient_clip_val")
    if expected_clip is not None and clip != expected_clip:
        raise ValueError("Clipping diagnostic/config threshold mismatch")
    norms = [r.get("grad_norm_before_clip") for r in trace]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in norms):
        raise ValueError("Nonfinite clipping diagnostic gradients")
    fraction = sum(v > clip for v in norms) / len(norms)
    if not math.isclose(summary.get("clip_fraction", math.nan), fraction, abs_tol=1e-12):
        raise ValueError("Saved clip_fraction disagrees with actual trace")
    window = min(32, len(norms))
    first, last = sum(norms[:window]) / window, sum(norms[-window:]) / window
    for key, value in (("first_window", first), ("last_window", last)):
        if not math.isclose(summary[key]["grad_norm_before_clip"], value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Saved gradient norm window disagrees with actual trace")
    return {"label": label, "scope": scope, "steps": steps, "gradient_clip_val": clip,
            "clip_fraction": fraction, "first32_mean_grad_norm_before_clip": first,
            "last32_mean_grad_norm_before_clip": last,
            "summary": {"path": str(summary_path), "sha256": sha(summary_path)},
            "trace": {"path": str(trace_path), "sha256": sha(trace_path)},
            "interpretation": "observed clipping frequency; does not establish causal metric influence"}


def historical_state(artifact_dir=HISTORICAL_ART):
    """Verify the previous search matrix separately, without loading weights."""
    artifact_dir = Path(artifact_dir)
    decision_path = artifact_dir / "final_selection.json"
    decision = read(decision_path)
    if not decision:
        return {"verified": False, "reason": "旧 final_selection 尚不存在"}
    try:
        from tools import audit_frozen_map_final_artifacts as historical_audit
        frozen_path = artifact_dir / "frozen_confirmation_protocol.json"
        frozen = read(frozen_path)
        final_rows = decision["rows"]
        if len(final_rows) != 12:
            raise ValueError("Historical twelve-cell matrix incomplete")
        folders = {Path(row["checkpoint"]).resolve().parent.parent for row in final_rows}
        all_rows, sources, hash_cache = [], [], {}
        def file_hash(path):
            path = Path(path).resolve()
            if str(path) not in hash_cache:
                hash_cache[str(path)] = sha(path)
            return hash_cache[str(path)]
        for folder in sorted(folders):
            execution_path = artifact_dir / "stage_executions" / (folder.name + ".json")
            execution = read(execution_path)
            if not execution or execution.get("status") != "success" or execution.get("returncode") != 0:
                raise ValueError("Historical stage is not successfully completed: " + folder.name)
            if Path(execution.get("output", "")).resolve() != folder:
                raise ValueError("Historical execution output mismatch")
            artifacts = execution.get("artifacts", {})
            for key in ("protocol", "summary", "log"):
                verify_record(artifacts.get(key))
            if Path(artifacts["protocol"]["path"]).resolve() != folder / "protocol.json" or Path(artifacts["summary"]["path"]).resolve() != folder / "summary.json":
                raise ValueError("Historical manifest points to another protocol/summary")
            log_path = Path(artifacts["log"]["path"])
            if log_path.read_text().splitlines()[-1].strip() != "DONE " + str(folder):
                raise ValueError("Historical stage terminal DONE mismatch")
            protocol, stage_rows = read(folder / "protocol.json"), read(folder / "summary.json")
            selected_rows = [row for row in final_rows if Path(row["checkpoint"]).resolve().parent.parent == folder]
            variant, seed = selected_rows[0]["variant"], selected_rows[0]["seed"]
            for row in selected_rows:
                if sum(row == actual for actual in stage_rows) != 1:
                    raise ValueError("Historical final row is not exactly one completed stage row")
                historical_audit.verify_metric_report(row)
                if row.get("resumed_from") is not None or row.get("resume_global_step") != 0 or row.get("optimizer_continued") is not False:
                    raise ValueError("Historical endpoint is not fresh")
            init_path = folder / variant / "initialization.json"
            init = read(init_path)
            manifests = ((folder / "train_order.txt", "train_order_sha256"),
                         (folder / variant / "train_files.txt", "train_manifest_sha256"),
                         (folder / "eval_files.txt", "eval_manifest_sha256"))
            manifest_sources = []
            for path, key in manifests:
                actual_hash = file_hash(path)
                if init.get(key) != actual_hash or any(row.get(key) != actual_hash for row in selected_rows):
                    raise ValueError("Historical actual filename manifest SHA mismatch: " + key)
                manifest_sources.append({"path": str(path), "sha256": actual_hash, "field": key})
            if file_hash(protocol["normalizer_json"]) != init["calibration_sha256"] or file_hash(protocol["map_checkpoint"]) != init["map_checkpoint_sha256"]:
                raise ValueError("Historical map/normalizer source fingerprint mismatch")
            steps = protocol["steps"]
            trace_path = folder / variant / "train_trace.json"
            trace = read(trace_path)
            if len(trace) != steps or [row.get("step") for row in trace] != list(range(steps)) or any(row.get("map_grad_norm_before_clip") != 0. for row in trace):
                raise ValueError("Historical full update trace / frozen map diagnostic mismatch")
            all_rows.extend(stage_rows)
            sources.append({"stage": folder.name, "TRAIN_seed": seed, "variant": variant,
                            "execution": {"path": str(execution_path), "sha256": file_hash(execution_path)},
                            "artifacts": artifacts, "actual_filename_manifests": manifest_sources,
                            "trace": {"path": str(trace_path), "sha256": file_hash(trace_path)}})
        facts = historical_audit.audit_final_matrix(decision, frozen, all_rows)
        names = (facts["control"], facts["candidate"])
        arm_rows = {name: [r for r in final_rows if r["variant"] == name] for name in names}
        seed_results = {str(seed): controller.compare(controller.aggregate([r for r in arm_rows[names[1]] if r["seed"] == seed]),
                       controller.aggregate([r for r in arm_rows[names[0]] if r["seed"] == seed])) for seed in (817, 1817, 2817)}
        pooled = controller.compare(controller.aggregate(arm_rows[names[1]]), controller.aggregate(arm_rows[names[0]]))
        return {"verified": True, "facts": facts, "pooled_under_current_gate": pooled,
                "per_training_seed_current_gate": seed_results,
                "would_pass_new_per_TRAIN_gate": pooled["eligible"] and all(r["eligible"] for r in seed_results.values()),
                "decision": {"path": str(decision_path), "sha256": file_hash(decision_path)},
                "frozen_protocol": {"path": str(frozen_path), "sha256": file_hash(frozen_path)},
                "stage_sources": sources, "excluded_from_overnight_inventory": True,
                "checkpoint_tensors_reloaded": False}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        return {"verified": False, "reason": str(error), "excluded_from_overnight_inventory": True}


def runtime_state(artifact_dir):
    """Verify recorded CPU/GPU memory diagnostics; never perform the probes."""
    artifact_dir = Path(artifact_dir)
    state = {"CPU": {"verified": False, "reason": "暂无已记录 CPU lifecycle 诊断"},
             "GPU": {"verified": False, "reason": "暂无已记录 GPU runtime pilot"},
             "excluded_from_performance_inventory": True}
    try:
        review_path = artifact_dir / "gc_object_lifecycle_cpu_review.json"
        review = read(review_path)
        if review:
            probes, sources = {}, []
            for mode, filename in (("disabled", "gc_disabled_two_scene_cpu_probe.json"), ("young_after_batch", "gc_young_two_scene_cpu_probe.json")):
                path, probe = artifact_dir / filename, read(artifact_dir / filename)
                if probe["mode"] != mode or probe["cuda_initialized"] or probe["tensorflow_imported"]:
                    raise ValueError("CPU lifecycle probe scope/initialization mismatch")
                final = probe["before_final_full_gc"]
                declared = review["probe"][mode]
                if (probe["loops"] * probe["scenes_per_loop"] != declared["actual_scene_loads"]
                        or final["unpickler_classes"] != declared["unreachable_local_unpickler_classes_final"]
                        or final["confirmed_super_class_cycles"] != declared["confirmed_class_closure_cycles_final"]):
                    raise ValueError("CPU lifecycle review disagrees with actual probe")
                probes[mode] = {"actual_scene_loads": declared["actual_scene_loads"], "unreachable_classes": final["unpickler_classes"],
                                "confirmed_class_closure_cycles": final["confirmed_super_class_cycles"]}
                sources.append({"path": str(path), "sha256": sha(path)})
            state["CPU"] = {"verified": True, "probes": probes, "sources": sources,
                            "review": {"path": str(review_path), "sha256": sha(review_path)}}
    except (OSError, ValueError, TypeError, KeyError) as error:
        state["CPU"] = {"verified": False, "reason": str(error)}
    try:
        review_path = artifact_dir / "runtime_gc_gpu_pilot_review.json"
        review = read(review_path)
        if review:
            if review.get("excluded_from_model_selection") is not True or review.get("status") != "ACTUAL_384_STEP_GPU_PILOT_PASS":
                raise ValueError("GPU pilot lacks successful diagnostic-only scope")
            done = review["actual_terminal_DONE"]
            folder = Path(done.removeprefix("DONE "))
            log_path = folder.with_suffix(".log")
            expected_sources = {log_path.resolve(), (folder / "summary.json").resolve(), (folder / "protocol.json").resolve(),
                                (artifact_dir / "runtime_gc_pilot_execution.json").resolve()}
            if {(ROOT / name).resolve() for name in review["artifact_sha256"]} != expected_sources:
                raise ValueError("GPU pilot lacks the four actual artifact fingerprints")
            for name, expected_sha in review["artifact_sha256"].items():
                verify_record({"path": str(ROOT / name), "sha256": expected_sha})
            lines = log_path.read_text().splitlines()
            events = [line for line in lines if line.startswith("RUNTIME_YOUNG_GC ")]
            memory = [json.loads(line.removeprefix("RUNTIME_MEMORY ")) for line in lines if line.startswith("RUNTIME_MEMORY ")]
            if lines[-1].strip() != done or events != review["generation0_actual_events"] or memory != review["actual_memory_samples"]:
                raise ValueError("GPU pilot review disagrees with actual log events/memory/DONE")
            rows = read(folder / "summary.json")
            if len(rows) != 3 or {r["budget"] for r in rows} != {128, 256, 384} or any(r["evaluation_weights"] != "ema" for r in rows):
                raise ValueError("GPU pilot does not have three completed EMA checkpoint evaluations")
            from tools import audit_frozen_map_final_artifacts as historical_audit
            for row in rows:
                historical_audit.verify_metric_report(row)
                if row.get("map_frozen") is not True or row.get("resumed_from") is not None or row.get("optimizer_continued") is not False:
                    raise ValueError("GPU runtime pilot is not fresh G1/frozen map")
            protocol = read(folder / "protocol.json")
            if protocol.get("steps") != 384 or protocol.get("young_gc_interval") != 128 or protocol.get("defer_cyclic_gc") is not True:
                raise ValueError("GPU runtime pilot actual protocol differs from review")
            execution = read(artifact_dir / "runtime_gc_pilot_execution.json")
            if execution.get("returncode") != 0 or execution.get("status") != "completed":
                raise ValueError("GPU pilot process did not complete successfully")
            state["GPU"] = {"verified": True, "steps": 384, "generation0_actual_events": events,
                            "actual_memory_samples": memory, "review": {"path": str(review_path), "sha256": sha(review_path)},
                            "actual_artifact_sha256": review["artifact_sha256"], "three_EMA_eval_cells_completed": True}
    except (OSError, ValueError, TypeError, KeyError) as error:
        state["GPU"] = {"verified": False, "reason": str(error)}
    return state


def intentional_restart_state(artifact_dir):
    """Classify an explicitly terminated diagnostic, never a model failure."""
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / "screen_control_pre_deadline_revision_interruption.json"
    marker = read(path)
    frozen = read(artifact_dir / "source_freeze.json", {})
    archive = frozen.get("controller_forecast_revision_archive")
    if not marker or not archive:
        return None
    archive = Path(archive)
    execution_path, log_path = archive / "execution.json", archive.with_suffix(".log")
    execution = read(execution_path)
    if (marker.get("status") != "INTENTIONAL_RUNTIME_CONTROLLER_REVISION_NOT_MODEL_RESULT"
            or marker.get("excluded_from_search_results") is not True
            or not execution or execution.get("excluded_from_model_selection") is not True):
        return {"verified": False, "reason": "Archived interruption does not explicitly certify diagnostic exclusion"}
    if not log_path.is_file():
        return {"verified": False, "reason": "Archived diagnostic log is missing"}
    observed_steps = [int(match.group(1)) for line in log_path.read_text().splitlines()
                      if (match := re.match(r"STEP (\d+):", line))]
    return {"verified": True, "reason": marker["reason"], "returncode": execution.get("returncode"),
            "last_logged_update": max(observed_steps) if observed_steps else None,
            "excluded_from_model_failure_or_performance": True,
            "sources": [{"path": str(source), "sha256": sha(source)} for source in (path, execution_path, log_path)]}


def corpus_scope_state(historical_artifact_dir=HISTORICAL_ART):
    """Count native filename raw keys and verify the matched manifest, no loads."""
    historical_artifact_dir = Path(historical_artifact_dir)
    audit_path = historical_artifact_dir / "manifest_overlap_audit.json"
    audit = read(audit_path)
    if not audit:
        return {"verified": False, "reason": "暂无 corpus/overlap 审计"}
    try:
        pattern = re.compile(r"\d+_((?:testing|validation)\.tfrecord-\d+-of-\d+)_(\d+)_(\d+)_(\d+)\.pt$")
        def raw_key(name):
            match = pattern.fullmatch(name)
            if not match:
                raise ValueError("Unparsed native cache filename: " + name)
            return match.group(1), int(match.group(2))
        names = sorted(path.name for path in (ROOT / "src/waymo_data/full/scenario_dreamer_val").glob("*.pt"))
        unique = {raw_key(name) for name in names}
        manifest_path = ROOT / audit["manifest"]
        selected = read(manifest_path)
        files = selected["files"] if isinstance(selected, dict) else selected
        actual_lines_sha = hashlib.sha256(("\n".join(files) + "\n").encode()).hexdigest()
        confirmation_unique = len({raw_key(name) for name in files})
        if (len(names) != audit["cache_pool_files"] or len(unique) != audit["cache_pool_unique_raw_keys"]
                or sha(manifest_path) != audit["manifest_sha256"]
                or actual_lines_sha != audit["ordered_basename_lines_sha256"]
                or len(files) != audit["confirmation_files"] or confirmation_unique != audit["confirmation_unique_raw_keys"]
                or not set(files) <= set(names)):
            raise ValueError("Current filename corpus/confirmation differs from actual overlap audit")
        return {"verified": True, "cache_entries": len(names), "unique_raw_keys": len(unique),
                "matched_confirmation_entries": len(files), "matched_confirmation_unique_raw_keys": confirmation_unique,
                "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
                "ordered_basename_lines_sha256": actual_lines_sha,
                "corpus_ordered_basename_lines_sha256": hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest(),
                "prior_overlap_audit": {"path": str(audit_path), "sha256": sha(audit_path)},
                "not_unseen_or_independent_test": True, "cache_tensors_loaded": False}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"verified": False, "reason": str(error)}


def final_state(decision, frozen, planned, actual_raw):
    if not decision:
        return {"status": "missing", "verified": False, "reason": "尚无 final_selection.json"}
    if decision.get("complete_matrix") is not True:
        return {"status": "incomplete", "verified": False, "reason": "最终两 arm ×两 TRAIN ×两 GEN 矩阵未齐",
                "declared_cells": len(decision.get("rows", []))}
    try:
        from tools import materialize_overnight_init_diffusion_best as materializer
        materializer.validate(decision, frozen, planned)
        for row in decision["rows"]:
            matches = [r for r in actual_raw if r == row]
            if len(matches) != 1:
                raise ValueError("Final decision row is not exactly one verified successful stage observation")
        expected_cohort = {controller.digest(controller.row_cohort(row)) for row in decision["rows"]}
        if len(expected_cohort) != 1:
            raise ValueError("Final physical metric/source cohorts differ")
        names = ("control", frozen["challenger"])
        rows = {name: [r for r in decision["rows"] if r["variant"] == name] for name in names}
        results = {name: materializer.assessment(rows[name], rows["control"]) for name in names}
        return {"status": "verified_complete_matrix", "verified": True, "selected": decision["selected"],
                "results": results, "actual_exposure": frozen["actual_final_exposure_per_TRAIN"],
                "TRAIN_seeds": list(materializer.TRAIN_SEEDS), "GEN_seeds": list(materializer.GEN_SEEDS),
                "cells": len(decision["rows"])}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        return {"status": "invalid_or_unverified", "verified": False, "reason": str(error)}


def verify_record(record):
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError("Missing durable materialization evidence fingerprint")
    path = Path(record["path"])
    if not path.is_file() or sha(path) != record["sha256"]:
        raise ValueError("Materialization evidence changed: " + str(path))


def native_state(artifact, final, decision, frozen, planned, *, decision_path, frozen_path, search_path):
    if not artifact:
        return {"verified": False, "reason": "尚无通过实际 materializer 的 native_best_materialization.json"}
    if not final.get("verified"):
        return {"verified": False, "reason": "实际最终矩阵未核验，已有 native artifact 不作为推荐依据"}
    try:
        import yaml
        from tools import materialize_overnight_init_diffusion_best as materializer
        if (artifact.get("selected") != final["selected"] or artifact.get("Hydra_composition_verified") is not True
                or artifact.get("fit_resumes_evidence_checkpoint") is not False
                or artifact.get("full128epoch_optimum_proven") is not False):
            raise ValueError("Materializer does not certify fresh native fit / matching actual selection")
        for key, actual_path in (("decision", decision_path), ("frozen_protocol", frozen_path), ("search_protocol", search_path)):
            verify_record(artifact.get(key))
            if Path(artifact[key]["path"]).resolve() != Path(actual_path).resolve():
                raise ValueError("Materializer points to a different search decision/protocol")
        audits = artifact.get("source_audits", [])
        expected = {(name, seed) for name in ("control", frozen["challenger"]) for seed in (4817, 5817)}
        if len(audits) != 4 or {(a.get("variant"), a.get("TRAIN_seed")) for a in audits} != expected:
            raise ValueError("Missing actual four-checkpoint CPU materializer audit")
        selected_row = materializer.validate(decision, frozen, planned)
        selected_source = None
        by_name = {arm["name"]: arm for arm in planned}
        for audit in audits:
            for key in ("checkpoint", "source_config", "execution"):
                verify_record(audit.get(key))
            arm = by_name[audit["variant"]]
            row = next(r for r in decision["rows"] if r["variant"] == audit["variant"]
                       and r["seed"] == audit["TRAIN_seed"] and r["eval_seed"] == 2817)
            source = yaml.safe_load(Path(audit["source_config"]["path"]).read_text())
            materializer.validate_checkpoint_metadata(audit["checkpoint_metadata"], row, arm, source)
            if row == selected_row:
                selected_source = source
        if selected_source is None or artifact.get("evidence_checkpoint") != selected_row["checkpoint"]:
            raise ValueError("Materializer source checkpoint does not match the selected verified endpoint")
        fit_path, eval_path = Path(artifact["fit_config"]), Path(artifact["eval_config"])
        if not fit_path.is_file() or not eval_path.is_file():
            raise ValueError("Materialized native experiment configs do not exist")
        actual_fit, actual_eval = yaml.safe_load(fit_path.read_text()), yaml.safe_load(eval_path.read_text())
        deployed = None
        if not artifact.get("evaluation_requires_explicit_checkpoint"):
            deployed = Path(str(actual_eval["ckpt_path"]).replace("${sim_root:}", str(ROOT)))
        expected_fit, expected_eval = materializer.build_configs(selected_source, decision["selected_arm"], eval_checkpoint=deployed)
        if actual_fit != expected_fit or actual_eval != expected_eval:
            raise ValueError("Native config content differs from actual materializer selected recipe")
        return {"verified": True, "fit_config": str(fit_path), "eval_config": str(eval_path),
                "fit_config_sha256": sha(fit_path), "eval_config_sha256": sha(eval_path),
                "evidence_checkpoint": selected_row["checkpoint"], "selected": final["selected"],
                "native_shuffle": artifact.get("native_shuffle"), "quick_shuffle": artifact.get("quick_search_shuffle"),
                "source_audit_count": len(audits), "full128epoch_completed": False,
                "evaluation_requires_explicit_checkpoint": artifact["evaluation_requires_explicit_checkpoint"]}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        return {"verified": False, "reason": str(error)}


def figure_state(metadata, rows, groups, directory):
    """Reject stale figure metadata that does not describe the current inventory."""
    approved, reasons = [], []
    if not metadata:
        return {"files": [], "reasons": ["暂无实际图形 metadata"]}
    available = {Path(p).resolve() for p in metadata.get("files", []) if Path(p).is_file()}
    for heatmap in metadata.get("heatmaps", []):
        group = next((g for g in groups if g["phase"] == heatmap["phase"]
                      and g["cohort_sha256"] == heatmap["cohort_sha256"]
                      and g["actual_exposure"] == heatmap["actual_exposure"]), None)
        expected = {a["variant"]: a for a in group["arms"]} if group else {}
        names = heatmap.get("arms", [])
        valid = bool(names) and all(name in expected and expected[name].get("comparison")
                                    and "ratios" in expected[name]["comparison"] for name in names)
        if valid:
            matrix = [[expected[name]["comparison"]["ratios"][key] for key in METRICS]
                      + [expected[name]["comparison"]["R"]] for name in names]
            valid = (matrix == heatmap.get("actual_ratios")
                     and heatmap.get("independent_training_N") == [expected[n]["independent_training_N"] for n in names]
                     and heatmap.get("scope_scenes") == group["cohort"]["num_samples"])
        if not valid:
            reasons.append("排除 stale/unverified heatmap: " + str(heatmap.get("phase")))
            continue
        prefix = f'{heatmap["phase"]}_ratios_'
        for p in sorted(available):
            if p.stem.startswith(prefix) and p.stem.endswith(f'_p{heatmap["page"]}'):
                # Matching cohort index is encoded in the figure filename.
                cohorts = sorted({g["cohort_sha256"] for g in groups if g["phase"] == heatmap["phase"] and g["control_present"]})
                if p.stem == f'{heatmap["phase"]}_ratios_c{cohorts.index(group["cohort_sha256"]) + 1}_p{heatmap["page"]}':
                    approved.append({"phase": heatmap["phase"], "path": str(p), "sha256": sha(p)})
    for curve in metadata.get("long_curves", []):
        cohort_groups = [g for g in groups if g["phase"] == "long" and g["cohort_sha256"] == curve["cohort_sha256"]]
        valid = bool(cohort_groups)
        for item in curve.get("curves", []):
            points = sorted((a for g in cohort_groups for a in g["arms"] if a["arm_id"] == item["arm_id"]), key=lambda a: a["actual_exposure"])
            metric = curve["metric"]
            means = [a["independent_training_N"] if metric == "independent_training_N" else a["metrics_raw_jsd_collision_percent"][metric]["mean"] for a in points]
            sd = [None if metric == "independent_training_N" else a["metrics_raw_jsd_collision_percent"][metric]["sd"] for a in points]
            valid &= (bool(points) and [a["actual_exposure"] for a in points] == item.get("actual_exposures")
                      and means == item.get("mean") and sd == item.get("sample_sd_across_TRAIN")
                      and [a["independent_training_N"] for a in points] == item.get("independent_TRAIN_N")
                      and [a["actual_full_data_epochs"] for a in points] == item.get("x_actual_epochs")
                      and [a["training_seed_ids"] for a in points] == item.get("TRAIN_membership_by_exposure")
                      and json.loads(json.dumps([a["batch_update_pairs"] for a in points])) == item.get("batch_update_pairs"))
        if valid and curve.get("curves"):
            approved.extend({"phase": "long", "path": str(Path(p).resolve()), "sha256": sha(p)}
                            for p in curve.get("files", []) if Path(p).resolve() in available)
        else:
            reasons.append("排除 stale/unverified long curve: " + str(curve.get("metric")))
    full = metadata.get("full50000", {})
    full_rows = [r for r in rows if r["phase"] == "full50000"]
    if full.get("status") == "actual_complete" and len(full_rows) == 4:
        expected = {(r["variant"], r["eval_seed"]): Path(r["report_source"]).parent / "feature_histograms.json" for r in full_rows}
        sources = full.get("histogram_sources", [])
        valid = len(sources) == 4 and {(s["arm"], s["GEN_seed"]) for s in sources} == set(expected)
        try:
            for source in sources:
                verify_record(source)
                valid &= Path(source["path"]).resolve() == expected[(source["arm"], source["GEN_seed"])].resolve()
        except (OSError, ValueError, KeyError):
            valid = False
        if valid:
            approved.extend({"phase": "full50000", "path": str(Path(p).resolve()), "sha256": sha(p)}
                            for p in full.get("files", []) if Path(p).resolve() in available)
        else:
            reasons.append("排除未核验 full50000 histogram 图")
    elif full.get("files"):
        reasons.append("排除不完整 full50000 scope 图")
    return {"files": list({entry["path"]: entry for entry in approved}.values()), "reasons": reasons}


def compact_plot_payload(groups):
    """Pure actual data for endpoint heatmaps and long curves; no extrapolation."""
    heatmaps, long = [], []
    for phase in ("screen", "medium"):
        cohorts = sorted({g["cohort_sha256"] for g in groups if g["phase"] == phase and g["control_present"]})
        for cohort in cohorts:
            group = max((g for g in groups if g["phase"] == phase and g["cohort_sha256"] == cohort
                         and g["control_present"]), key=lambda g: g["actual_exposure"])
            arms = [a for a in group["arms"] if a.get("comparison") and "ratios" in a["comparison"]]
            arms.sort(key=lambda a: (a["variant"] != "control", not a["comparison"]["ranking_eligible"],
                                     a["comparison"]["R"], a["variant"]))
            if arms:
                heatmaps.append({"phase": phase, "cohort_sha256": cohort,
                    "actual_exposure": group["actual_exposure"], "actual_full_data_epochs": group["actual_full_data_epochs"],
                    "scope_entries": group["cohort"]["num_samples"], "arms": [a["variant"] for a in arms],
                    "ranking_eligible": [a["comparison"]["ranking_eligible"] for a in arms],
                    "independent_TRAIN_N": [a["independent_training_N"] for a in arms],
                    "TRAIN_membership": [a["training_seed_ids"] for a in arms],
                    "GEN_seeds_by_TRAIN": [{seed: values["generation_seeds"] for seed, values in a.get("per_training_seed", {}).items()} for a in arms],
                    "ratios": [[a["comparison"]["ratios"][k] for k in METRICS] + [a["comparison"]["R"]] for a in arms]})
    for cohort in sorted({g["cohort_sha256"] for g in groups if g["phase"] == "long"}):
        actual = [a for g in groups if g["phase"] == "long" and g["cohort_sha256"] == cohort for a in g["arms"]]
        curves = []
        for arm_id in sorted({a["arm_id"] for a in actual}):
            points = sorted((a for a in actual if a["arm_id"] == arm_id), key=lambda a: a["actual_exposure"])
            for point in points:
                for metric in METRICS:
                    value = point["metrics_raw_jsd_collision_percent"][metric]
                    if point["independent_training_N"] == 1 and value["sd"] is not None:
                        raise ValueError("N=1 long figure must not invent SD")
            curves.append({"arm_id": arm_id, "variant": points[0]["variant"],
                "actual_exposures": [a["actual_exposure"] for a in points],
                "actual_epochs": [a["actual_full_data_epochs"] for a in points],
                "independent_TRAIN_N": [a["independent_training_N"] for a in points],
                "TRAIN_membership": [a["training_seed_ids"] for a in points],
                "batch_update_pairs": [a["batch_update_pairs"] for a in points],
                "mean": {k: [a["metrics_raw_jsd_collision_percent"][k]["mean"] for a in points] for k in METRICS},
                "sample_sd_across_TRAIN": {k: [a["metrics_raw_jsd_collision_percent"][k]["sd"] for a in points] for k in METRICS}})
        long.append({"phase": "long", "cohort_sha256": cohort, "curves": curves, "missing_points_padded": False})
    return {"heatmaps": heatmaps, "long": long}


def compact_full_payload(rows, approved):
    """Require complete approved full scope and recheck every physical histogram."""
    observed = [r for r in rows if r["phase"] == "full50000"]
    if not any(f["phase"] == "full50000" for f in approved.get("files", [])) or len(observed) != 4:
        return None
    if len({r["cohort_sha256"] for r in observed}) != 1 or {r["seed"] for r in observed} != {4817}:
        raise ValueError("Compact full plot needs same physical cohort and TRAIN4817")
    from tools.plot_component_loss_distributions import FEATURES, validate_counts
    import numpy as np
    arms = sorted({r["variant"] for r in observed}, key=lambda n: (n != "control", n))
    if len(arms) != 2:
        raise ValueError("Compact full plot needs exactly two actual arms")
    loaded, sources = {}, []
    for arm in arms:
        arm_rows = [r for r in observed if r["variant"] == arm]
        analysis.seed_means(arm_rows)
        loaded[arm] = []
        for row in sorted(arm_rows, key=lambda r: r["eval_seed"]):
            path = Path(row["report_source"]).parent / "feature_histograms.json"
            loaded[arm].append(read(path))
            sources.append({"arm": arm, "GEN_seed": row["eval_seed"], "path": str(path), "sha256": sha(path)})
    features = []
    for feature, spec in FEATURES.items():
        edges, gt_count, gt_probability = validate_counts(loaded[arms[0]][0][feature], "matched_gt", "Full GT")
        series = []
        for arm in arms:
            probabilities = []
            for histogram in loaded[arm]:
                ref_edges, ref_count, _ = validate_counts(histogram[feature], "matched_gt", arm + " GT")
                generated_edges, _, probability = validate_counts(histogram[feature], "generated", arm)
                if not (np.array_equal(edges, ref_edges) and np.array_equal(gt_count, ref_count)
                        and np.array_equal(edges, generated_edges)):
                    raise ValueError("Compact full histogram exact GT/bin comparison failed")
                probabilities.append(probability)
            series.append({"arm": arm, "mean_normalized_probability": np.mean(probabilities, axis=0).tolist()})
        features.append({"feature": feature, "spec": spec, "edges": edges.tolist(),
                         "matched_gt_counts": gt_count.tolist(), "matched_gt_probability": gt_probability.tolist(), "series": series})
    if len({(r["num_gt_vehicles"], r["num_gt_onroad_vehicles"]) for r in observed}) != 1:
        raise ValueError("Compact full GT counts disagree")
    return {"phase": "full50000", "scope_cache_entries": 50000, "TRAIN_seed": 4817,
        "GEN_seeds": [2817, 3817], "histogram_sources": sources, "features": features, "arms": arms,
        "coverage": [observed[0]["gt_onroad_fraction"]] + [sum(r["generated_onroad_fraction"] for r in observed if r["variant"] == arm) / 2 for arm in arms],
        "vehicle_counts": [observed[0]["num_gt_vehicles"]] + [sum(r["num_generated_vehicles"] for r in observed if r["variant"] == arm) / 2 for arm in arms],
        "adds_independent_TRAIN": False, "reselection": False}


def compact_report_figures(rows, groups, approved, directory):
    """Create compact CPU PNG/SVG panels from freshly verified observations."""
    payload = compact_plot_payload(groups)
    full = compact_full_payload(rows, approved)
    if not payload["heatmaps"] and not payload["long"] and full is None:
        return {"embedded": [], "payload": payload, "full": full}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "svg.fonttype": "none"})
    embedded = []
    def save(fig, name, phase, caption, provenance):
        files = []
        for suffix in ("png", "svg"):
            path = directory / (name + "." + suffix)
            with tempfile.NamedTemporaryFile(dir=directory, suffix="." + suffix, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                fig.savefig(temporary, dpi=180, format=suffix, facecolor="white")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            files.append({"path": str(path), "sha256": sha(path)})
        plt.close(fig)
        embedded.append({"phase": phase, "path": files[0]["path"], "sha256": files[0]["sha256"],
                         "exports": files, "caption": caption, "actual_payload_sha256": digest(provenance)})
    phase_indices = {}
    for heatmap in payload["heatmaps"]:
        phase = heatmap["phase"]
        phase_indices[phase] = phase_indices.get(phase, 0) + 1
        n = len(heatmap["arms"])
        fig, ax = plt.subplots(figsize=(6.8, 2.25 + .255 * n))
        matrix = np.asarray(heatmap["ratios"])
        ax.imshow(matrix, cmap="RdYlGn_r", norm=TwoSlopeNorm(vmin=.5, vcenter=1., vmax=1.5), aspect="auto")
        ax.set_xticks(range(8), (*SHORT[:-1], "Col", "R"))
        ax.set_yticks(range(n), [name + (" *" if not gate else "") for name, gate in zip(heatmap["arms"], heatmap["ranking_eligible"])], fontsize=8)
        for y in range(n):
            for x in range(8):
                ax.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(heatmap["phase"].capitalize() + ": verified endpoint candidate/control ratios")
        independent_n = ",".join(map(str, sorted(set(heatmap["independent_TRAIN_N"]))))
        gen_sets = {tuple(seeds) for item in heatmap["GEN_seeds_by_TRAIN"] for seeds in item.values()}
        gen_note = "; GEN " + "/".join(map(str, next(iter(gen_sets)))) if len(gen_sets) == 1 else ""
        fig.text(.5, .025, f'{heatmap["actual_exposure"]:,} actual TRAIN cache exposures / {heatmap["actual_full_data_epochs"]:.3f} epochs; '
                 f'{heatmap["scope_entries"]:,} evaluation entries.\nTRAIN N={independent_n}{gen_note}; GEN averaged within TRAIN.\n'
                 'Lower is better; * = gate failure. Color capped at0.5..1.5; printed values unchanged.', ha="center", fontsize=8)
        fig.subplots_adjust(left=.30, right=.98, top=.90, bottom=.26 if n < 5 else .15)
        caption = f'{heatmap["phase"]} 已完成终点：六项 JSD 与 collision 的 candidate/control 比值及 R；星号表示 gates FAIL。'
        save(fig, f'{phase}_endpoint_c{phase_indices[phase]}', phase, caption, heatmap)
    for index, panel in enumerate(payload["long"]):
        for page, panel_metrics in enumerate((METRICS[:4], (*METRICS[4:], "independent_training_N")), start=1):
            fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.5))
            legend = []
            for ax, metric in zip(axes.flat, panel_metrics):
                for curve_index, curve in enumerate(panel["curves"]):
                    n = curve["independent_TRAIN_N"]
                    y = n if metric == "independent_training_N" else curve["mean"][metric]
                    color = palette[curve_index % len(palette)]
                    batches = sorted({pair[0] for pairs in curve["batch_update_pairs"] for pair in pairs})
                    label = curve["variant"] + f' (B{batches[0]})'
                    line, = ax.plot(curve["actual_epochs"], y, "-o", color=color, markersize=3, linewidth=1.4, label=label)
                    if metric == panel_metrics[0]:
                        legend.append((line, label))
                    if metric != "independent_training_N":
                        sd = curve["sample_sd_across_TRAIN"][metric]
                        defined = [i for i, value in enumerate(sd) if value is not None]
                        if defined:
                            ax.errorbar([curve["actual_epochs"][i] for i in defined], [y[i] for i in defined],
                                        yerr=[sd[i] for i in defined], fmt="none", color=color, capsize=2, linewidth=1.)
                    else:
                        ax.set_ylim(.5, max(2.5, max(n) + .5))
                        ax.set_yticks((1, 2))
                ax.set_title("Independent TRAIN N" if metric == "independent_training_N" else SHORT[METRICS.index(metric)])
                ax.set_ylabel("TRAIN N" if metric == "independent_training_N" else "Collision (%)" if metric == "collision_rate" else "Raw JSD")
                ax.set_xlabel("Actual cache exposures /493,542")
                ax.grid(alpha=.2)
            fig.suptitle(f"Long training: actual full-data equivalent epochs ({page}/2)", fontsize=11)
            fig.legend([v[0] for v in legend], [v[1] for v in legend], loc="lower center", bbox_to_anchor=(.5, .045), ncol=2, frameon=False)
            fig.text(.5, .008, "GEN averaged within TRAIN; bars = sample SD across TRAIN, absent at N=1.\nOnly complete stages; N/membership may differ. Unmatched TRAIN sets are descriptive, not paired selection.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .16, 1, .95), h_pad=.8, w_pad=.8)
            save(fig, f"long_actual_epochs_c{index + 1}_p{page}", "long", f"long实际epoch曲线（{page}/2）；第二张下右图为TRAIN N，N=1不画SD。每一budget的实际TRAIN membership和相同source在metadata保留；未配齐TRAIN只作原始均值描述。", panel)
    if full is not None:
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.6))
        for ax, feature in zip(axes.flat, full["features"]):
            spec, edges = feature["spec"], np.asarray(feature["edges"])
            ax.stairs(feature["matched_gt_probability"], edges, label="Matched GT", color="black", linewidth=1.5)
            for index, series in enumerate(feature["series"]):
                ax.stairs(series["mean_normalized_probability"], edges, label=series["arm"], color=palette[index], linewidth=1.3)
            ax.set_title(spec["label"])
            ax.set_xlim(*(spec["zoom"] or (edges[0], edges[-1])))
            ax.set_xlabel(spec["label"] + " (" + spec["unit"] + ")")
            ax.set_ylabel("Probability per official bin")
            ax.grid(alpha=.2, axis="y")
        ax = axes.flat[3]
        values = np.asarray(full["coverage"]) * 100
        ax.bar(range(3), values, color=("black", *palette[:2]), width=.55)
        ax.set_xticks(range(3), ["GT", "control", full["arms"][1]], rotation=10)
        ax.set_ylim(0, 120); ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel("On-road vehicles (%)"); ax.set_title("Coverage / vehicle counts")
        for index, (value, count) in enumerate(zip(values, full["vehicle_counts"])):
            ax.text(index, value + 2, f"{value:.2f}%\n{count:,.0f} veh", ha="center", fontsize=8)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
        fig.suptitle("Main N20 full50k: attributes / coverage | TRAIN4817, 2 GEN", fontsize=11)
        fig.text(.5, .008, "50,000 cache entries /43,601 unique raw keys; TRAIN4817 only, two GEN means.\nSize axes zoom without renormalization; this audit adds no TRAIN seed and does not reselect.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .94), h_pad=1.3)
        save(fig, "full50000_attributes_coverage", "full50000", "main N20训练recipe比较的full50k speed/length/width histogram与onroad/vehicle count；TRAIN4817×两GEN，只作scope audit。不是native40或hybrid40的full直方图。", full)
    result = {"embedded": embedded, "payload": payload, "full": full,
              "verified_rows_sha256": digest(rows), "verified_groups_sha256": digest(groups),
              "GPU_started": False, "torch_imported": "torch" in sys.modules,
              "source_tool_sha256": sha(Path(__file__))}
    path = directory / "metadata.json"
    atomic_text(path, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    result["metadata"] = {"path": str(path), "sha256": sha(path)}
    return result


def runtime_failure_state(artifact_dir, stages):
    """Seal actual failed-process evidence before a root-controlled fresh retry.

    These audit copies are not execution roots and are never consumed by the
    search inventory. They retain observed diagnostics after its run is moved.
    """
    artifact_dir = Path(artifact_dir)
    index = artifact_dir / "report_runtime_failures.json"
    records = read(index, [])
    seen = {r["original_execution_sha256"] for r in records}
    for stage in stages:
        if stage.get("status") != "failed":
            continue
        folder = Path(stage["run"])
        execution_path, log_path = folder / "execution.json", folder.with_suffix(".log")
        execution = read(execution_path, {})
        if execution.get("returncode") != -11 or not log_path.is_file():
            continue
        execution_sha = sha(execution_path)
        if execution_sha in seen:
            continue
        log_bytes = log_path.read_bytes()
        text = log_bytes.decode("utf-8", errors="replace")
        steps = re.findall(r"^STEP (\d+):", text, re.M)
        memory = re.findall(r"^RUNTIME_MEMORY (\{[^\n]+\})", text, re.M)
        young_steps = re.findall(r"^RUNTIME_YOUNG_GC step=(\d+) collected=(\d+)", text, re.M)
        if "Fatal Python error: Segmentation fault" not in text:
            continue
        name = stage["label"] + "_" + execution_sha[:12]
        copy_execution = artifact_dir / "report_runtime_failures" / (name + ".execution.json")
        copy_log = artifact_dir / "report_runtime_failures" / (name + ".log")
        atomic_bytes(copy_execution, execution_path.read_bytes())
        atomic_bytes(copy_log, log_bytes)
        if sha(copy_execution) != execution_sha or sha(copy_log) != sha(log_path):
            raise ValueError("Runtime diagnostic changed while its exact audit copy was sealed")
        record = {"stage": stage["label"], "returncode": -11,
            "elapsed_process_seconds": execution["ended_unix"] - execution["started_unix"]
                if isinstance(execution.get("ended_unix"), (int, float)) and isinstance(execution.get("started_unix"), (int, float)) else None,
            "original_execution_path": str(execution_path), "original_execution_sha256": execution_sha,
            "original_log_path": str(log_path), "original_log_sha256": sha(log_path),
            "execution_copy": {"path": str(copy_execution), "sha256": sha(copy_execution)},
            "log_copy": {"path": str(copy_log), "sha256": sha(copy_log)},
            "last_logged_update": int(steps[-1]) if steps else None,
            "last_successful_young_gc": {"step": int(young_steps[-1][0]), "collected": int(young_steps[-1][1])} if young_steps else None,
            "last_actual_memory": json.loads(memory[-1]) if memory else None,
            "faulthandler_garbage_collecting": "Garbage-collecting" in text,
            "root_cause_proven": False, "included_in_performance_or_TRAIN_N": False,
            "complete_training": False, "audit_copy_created_utc": datetime.now(timezone.utc).isoformat()}
        records.append(record)
        seen.add(execution_sha)
    valid, reasons = [], []
    for record in records:
        try:
            verify_record(record["execution_copy"])
            verify_record(record["log_copy"])
            execution = read(record["execution_copy"]["path"])
            if execution.get("status") != "failed" or execution.get("returncode") != -11:
                raise ValueError("Runtime audit copy is not an actual failed -11 execution")
            if isinstance(execution.get("ended_unix"), (int, float)) and isinstance(execution.get("started_unix"), (int, float)):
                record["elapsed_process_seconds"] = execution["ended_unix"] - execution["started_unix"]
            valid.append(record)
        except (OSError, ValueError, KeyError) as error:
            reasons.append(str(error))
    if records:
        atomic_text(index, json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return {"verified_failures": valid, "reasons": reasons, "GPU_started": False,
            "index": {"path": str(index), "sha256": sha(index)} if index.is_file() else None}


def recovery_execution_role(runtime, rows):
    """Fresh fit and complete checkpoint-only evaluation have distinct ledgers."""
    if runtime.get("fresh_training") is True:
        if any(row.get("evaluation_only") is True for row in rows):
            raise ValueError("Fresh training runtime contains evaluation-only rows")
        return "fresh_training"
    if (runtime.get("fresh_training") is False and rows
            and all(row.get("evaluation_only") is True and row.get("train_seconds") == 0.
                    and row.get("optimizer_continued") is False for row in rows)):
        return "evaluation_only"
    raise ValueError("Runtime role is neither fresh fit nor verified checkpoint-only TEST")


def runtime_recovery_state(artifact_dir, stages):
    """Verify actual recovery metadata and bounded CPU probes without Torch."""
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / "runtime_recovery/protocol.json"
    protocol = read(path)
    if not protocol:
        return {"verified": False, "reason": "尚无实际 recovery execution protocol"}
    try:
        if (protocol.get("status") != "RECOVERY_EXECUTION_PROTOCOL"
                or protocol.get("checkpoint_resume") is not False
                or protocol.get("unchanged_model_objective_data_seeds_and_budget") is not True):
            raise ValueError("Recovery is only a plan or does not certify fresh unchanged-model execution")
        freeze_path = artifact_dir / "source_freeze.json"
        frozen_long = artifact_dir / "frozen_long_protocol.json"
        freeze = read(freeze_path)
        if (sha(freeze_path) != protocol["source_freeze_sha256"]
                or sha(frozen_long) != protocol["frozen_long_protocol_sha256"]
                or freeze["source_hashes"] != protocol["source_hashes"]):
            raise ValueError("Recovery model/source/budget freeze differs")
        for name, expected in protocol["source_hashes"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Frozen model/data/runner source changed: " + name)
        added = {protocol["wrapper"]: protocol["wrapper_sha256"], **protocol["runtime_loader_source_hashes"]}
        for source, expected in added.items():
            if sha(source) != expected:
                raise ValueError("Separate recovery wrapper/helper source changed: " + source)
        if (protocol["runtime_change"]["remaining_young_gc_interval"] != 0
                or protocol["runtime_change"]["automatic_cyclic_gc_disabled"] is not True
                or protocol["runtime_change"]["manual_collection_disabled"] is not True):
            raise ValueError("Recovery runtime policy differs from explicit no-GC cached loader")
        proof_path = artifact_dir / "cached_torch_loader_100_real_scene_equivalence.json"
        proof = read(proof_path)
        helper = next(name for name in added if Path(name).name == "torch_pickle_class_cache.py")
        if (proof.get("status") != "PASS" or proof.get("unique_real_scenes") != 100
                or len(set(proof.get("filenames", []))) != 100
                or proof.get("helper_sha256") != added[helper]
                or proof.get("CUDA_initialized") is not False or proof.get("TensorFlow_imported") is not False
                or proof.get("gc_collect_called") is not False or proof.get("GC_enabled") is not False
                or proof["cache"]["classes_created"] != 1 or proof["cache"]["class_reuses"] != 99):
            raise ValueError("Cached loader actual100-scene CPU equivalence metadata incomplete")
        probe_path = artifact_dir / "gc_off_20k_cached_actual_scene_load_probe.json"
        probe = read(probe_path)
        final = probe["records"][-1]
        if (probe.get("actual_loads") != 20000 or len(set(probe.get("paths", []))) != 2
                or probe.get("CUDA_initialized") is not False or probe.get("TensorFlow_imported") is not False
                or probe.get("GC_enabled") is not False or probe.get("gc_collect_called") is not False
                or final["loads"] != 20000 or final["local_classes"] != 1 or final["torch_tensors"] != 0
                or final["class_cache"]["classes_created"] != 1 or final["class_cache"]["class_reuses"] != 19999):
            raise ValueError("Cached loader bounded20k CPU probe metadata incomplete")
        events_path = artifact_dir / "runtime_recovery/events.json"
        events = read(events_path, [])
        if not any(e.get("event") == "RECOVERY_STARTED" and e.get("wrapper_sha256") == protocol["wrapper_sha256"]
                   and e.get("frozen_long_sha256") == protocol["frozen_long_protocol_sha256"] for e in events):
            raise ValueError("Actual RECOVERY_STARTED event missing or mismatched")
        inherited = [e["label"] for e in events if e.get("event") == "INHERIT_VERIFIED_SUCCESS"]
        actual_stages = {s["label"]: s for s in stages}
        if any(not actual_stages.get(name, {}).get("included") for name in inherited):
            raise ValueError("Inherited control is not a verified successful stage")
        archive_path = artifact_dir / "runtime_recovery/attempt1/archive_ledger.json"
        archive = read(archive_path, {})
        archives = {r["original"]: r["archive"] for r in archive.get("records", [])}
        fresh, evaluation_only_stages = [], []
        fields = ("seed", "batch_size", "train_order_sha256", "calibration_sha256", "map_checkpoint_sha256",
                  "initial_generator_sha256", "initial_map_sha256", "train_manifest_sha256", "eval_manifest_sha256",
                  "lr", "gradient_clip_val", "initialization_scope", "actual_train_offset", "resumed_from",
                  "resume_global_step", "optimizer_continued")
        for stage in stages:
            folder = Path(stage["run"])
            execution = read(folder / "execution.json", {})
            runtime = execution.get("runtime_recovery")
            if not runtime:
                continue
            if (runtime.get("wrapper_sha256") != protocol["wrapper_sha256"]
                    or runtime.get("runtime_loader_source_hashes") != protocol["runtime_loader_source_hashes"]
                    or execution.get("source_hashes") != protocol["source_hashes"]):
                raise ValueError("Recovered actual execution/source ledger disagrees")
            actual_rows = read(folder / "summary.json", []) if stage.get("included") else []
            role = recovery_execution_role(runtime, actual_rows)
            if role == "evaluation_only":
                evaluation_only_stages.append({"stage": stage["label"], "status": stage["status"],
                    "actual_training_updates_in_this_job": 0, "checkpoints": sorted({row["checkpoint"] for row in actual_rows}),
                    "execution": {"path": str(folder / "execution.json"), "sha256": sha(folder / "execution.json")}})
                continue
            name = execution["arm"]["name"]
            initialization_path = folder / name / "initialization.json"
            if not initialization_path.is_file():
                continue  # It is still initializing, not a completed observation.
            current = read(initialization_path)
            if current.get("resumed_from") is not None or current.get("actual_train_offset") != 0 or current.get("optimizer_continued") is not False:
                raise ValueError("Recovered stage is not truly fresh")
            old_path = Path(archives[str(folder)]) / name / "initialization.json" if str(folder) in archives else None
            previous = read(old_path) if old_path and old_path.is_file() else None
            if previous and any(previous.get(k) != current.get(k) for k in fields):
                raise ValueError("Fresh retry differs from archived original initialization")
            fresh.append({"stage": stage["label"], "status": stage["status"], "metadata_fields_checked": list(fields),
                "matches_archived_initialization": bool(previous), "initial_G1_sha256": current["initial_generator_sha256"],
                "initial_map_sha256": current["initial_map_sha256"], "actual_train_offset": 0, "checkpoint_resume": False,
                "initialization": {"path": str(initialization_path), "sha256": sha(initialization_path)},
                "archived_initialization": {"path": str(old_path), "sha256": sha(old_path)} if previous else None})
        return {"verified": True, "protocol": {"path": str(path), "sha256": sha(path)},
            "original_frozen_source_count": len(protocol["source_hashes"]),
            "additional_runtime_sources": [{"path": name, "sha256": expected} for name, expected in added.items()],
            "inherited_stages": inherited, "original_young_gc_interval": 128, "recovery_young_gc_interval": 0,
            "fresh_stages": fresh, "evaluation_only_stages": evaluation_only_stages, "equivalence_probe": {"path": str(proof_path), "sha256": sha(proof_path), "unique_cache_entries": 100,
                "counters": proof["counters"], "classes_created": 1, "class_reuses": 99},
            "bounded_load_probe": {"path": str(probe_path), "sha256": sha(probe_path), "actual_loads": 20000,
                "unique_cache_entries": 2, "final_local_classes": 1, "final_torch_tensors": 0,
                "first_rss_kib": probe["records"][0]["rss_kib"], "last_rss_kib": final["rss_kib"]},
            "events": {"path": str(events_path), "sha256": sha(events_path)},
            "archive_ledger": {"path": str(archive_path), "sha256": sha(archive_path)},
            "full_GPU_safety_proven": False, "runtime_equals_inherited_control": False,
            "GPU_started_by_report_builder": False}
    except (OSError, KeyError, TypeError, ValueError, StopIteration) as error:
        return {"verified": False, "reason": str(error)}


def result_group_lines(group, *, output):
    lines = [f'### {group["phase"]} / {group["evaluation_weights"]} / exposure {group["actual_exposure"]:,}', "",
             f'实际完整 TRAIN 等价 epoch：{group["actual_full_data_epochs"]:.4f}；评价 {group["cohort"].get("num_samples", "未知")} 场景；cohort `{group["cohort_sha256"][:12]}`。不同 cohort 分开列出，不交叉排名。', ""]
    arms = sorted(group["arms"], key=lambda a: (a["variant"] != "control", a["variant"]))
    control_seeds = next((a["training_seed_ids"] for a in arms if a["variant"] == "control"), [])
    if control_seeds and any(a["training_seed_ids"] != control_seeds for a in arms):
        lines += ["当前各 arm 的完整 TRAIN seed 集合未配齐。下表只展示各自完整模型的原始均值；不将 control N=2 与 candidate N=1 做 paired ratio、R 或选择依据，等待候选同 TRAIN 集合完成。", ""]
    values = [[a["variant"], a["independent_training_N"], *[stat(a["metrics_raw_jsd_collision_percent"][k]) for k in METRICS]] for a in arms]
    lines += table(("arm", "TRAIN N", *SHORT), values)
    lines += ["上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。", ""]
    assessment = []
    for a in arms:
        comparison = a.get("comparison") or {}
        gate = ("PASS" if comparison.get("ranking_eligible") else "FAIL") if "ratios" in comparison else comparison.get("reason", "无同 scope control")
        pairs = ",".join(f"B{b}/{u}upd" for b, u in a["batch_update_pairs"])
        native = a["metrics_native"]
        count_ratio = native["num_generated_vehicles"]["mean"] / max(native["num_gt_vehicles"]["mean"], 1e-12)
        assessment.append([a["variant"], pairs, ",".join(map(str, a["training_seed_ids"])),
                           number(comparison.get("R"), 4), number(comparison.get("R_attr"), 4),
                           number(100 * native["generated_onroad_fraction"]["mean"], 2), number(count_ratio, 4), gate])
    lines += table(("arm", "batch/updates", "TRAIN seeds", "R", "R_attr", "onroad%", "veh/GT", "gates"), assessment)
    if group.get("mixed_execution_runtime_cohorts"):
        lines += ["本组 metric/model/data/precision 条件相同，但存在不同的实际执行 runtime cohort。以下 GC/loader provenance 保留分开；本组比较不宣称运行策略相同，也不单独隔离其因果影响。", ""]
        lines += table(("runtime cohort", "youngGC interval", "entrypoint", "额外 wrapper SHA"),
                       [[key[:12], value.get("young_gc_interval"), Path(value.get("entrypoint") or "未记录").name,
                         (value.get("recovery_wrapper_sha256") or "无额外 wrapper")[:12]]
                        for key, value in sorted(group["runtime_cohorts"].items())])
        lines += ["各 arm / TRAIN 对应的实际 runtime IDs（完整 SHA 保存在机器审计）：" + "; ".join(a["variant"] + "=" + json.dumps({seed: [value[:12] for value in ids] for seed, ids in a.get("runtime_cohorts_by_TRAIN", {}).items()}, separators=(",", ":")) for a in arms) + "。", ""]
    return lines


def sampler_statistics(rows, planned_cells, *, complete, baseline_option="n20_eps005"):
    """Paired inference statistics only after the actual full grid is verified.

    No result from a partial grid enters ranking, and GEN draws never increase
    independent TRAIN N. This is descriptive: it cannot select a sampler.
    """
    identity = lambda row: (row["TRAIN_seed"], row["GEN_seed"], row["option"])
    expected = [identity(row) for row in planned_cells]
    observed = [identity(row) for row in rows]
    if len(set(expected)) != len(expected) or len(set(observed)) != len(observed):
        raise ValueError("Duplicate sampler cells")
    if observed != expected[:len(observed)]:
        raise ValueError("Sampler actual rows do not match planned cell prefix")
    if not complete:
        return []
    if observed != expected:
        raise ValueError("Complete sampler flag lacks actual cells")
    options = sorted({row["option"] for row in rows})
    train_seeds = sorted({row["TRAIN_seed"] for row in rows})
    gen_seeds = sorted({row["GEN_seed"] for row in rows})
    if baseline_option not in options or len(train_seeds) != 2 or len(gen_seeds) != 2:
        raise ValueError("Paired sampler grid needs two TRAIN, two GEN and default20")
    keys = METRICS + ("generated_onroad_fraction", "num_generated_vehicles", "num_gt_vehicles")
    means = {}
    for option in options:
        means[option] = {}
        for seed in train_seeds:
            cell_rows = [row for row in rows if row["option"] == option and row["TRAIN_seed"] == seed]
            if sorted(row["GEN_seed"] for row in cell_rows) != gen_seeds:
                raise ValueError("Incomplete paired sampler TRAIN cell")
            means[option][seed] = {key: sum(row[key] for row in cell_rows) / len(cell_rows) for key in keys}
    baseline = means[baseline_option]
    baseline_mean = {key: sum(value[key] for value in baseline.values()) / len(baseline) for key in keys}
    result = []
    for option in options:
        pooled = {key: sum(value[key] for value in means[option].values()) / len(train_seeds) for key in keys}
        comparison = controller.compare(pooled, baseline_mean)
        per_train = {str(seed): controller.compare(means[option][seed], baseline[seed]) for seed in train_seeds}
        comparison.update(each_training_seed_gates_pass=all(value["eligible"] for value in per_train.values()),
                          all_gates_pass=comparison["eligible"] and all(value["eligible"] for value in per_train.values()))
        result.append({"option": option, "independent_TRAIN_N": len(train_seeds),
                       "reused_training_seeds": train_seeds, "new_training_runs": 0,
                       "generation_seeds": gen_seeds,
                       "metrics_raw": {key: analysis.original.stats(value[key] / MULTIPLIERS[key]
                                           for value in means[option].values()) for key in METRICS},
                       "pooled": comparison, "per_TRAIN": per_train,
                       "sampler_selection_made_by_report": False})
    return result


def sampler_state(artifact_dir, final, native, *, directory=None, fresh_native=False, hybrid=False):
    """Read and re-certify auxiliary TEST artifacts; never launch a process."""
    fresh_native = fresh_native or hybrid
    directory = Path(directory or ROOT / ("logs/init_diffusion_hybrid_sampler_cohort_20261012" if hybrid else "logs/init_diffusion_native_sampler_cohort_20261012" if fresh_native else "logs/init_diffusion_sampler_inference_20261012"))
    result = {"directory": str(directory), "stages": {}, "main_selection_unchanged": True,
              "new_training_runs": 0, "selection_made_by_report": False}
    result["fresh_native_baselines"] = fresh_native
    result["hybrid_position_cohort"] = hybrid
    result["baseline_option"] = "n40_eps005" if hybrid else "n20_eps005"
    protocol_path = Path(artifact_dir) / ("hybrid_sampler_selection_protocol.json" if hybrid else "native_sampler_selection_protocol.json" if fresh_native else "sampler_selection_protocol.json")
    selector = None
    try:
        if not protocol_path.is_file():
            raise ValueError("尚无预注册 sampler selection protocol")
        if hybrid:
            from tools import select_init_diffusion_hybrid_sampler as selector
        elif fresh_native:
            from tools import select_init_diffusion_native_sampler as selector
        else:
            from tools import select_init_diffusion_sampler as selector
        protocol = selector.validate_protocol(protocol_path)
        result["selection_protocol"] = {"verified": True, "path": str(protocol_path),
            "sha256": sha(protocol_path), "created_utc": protocol["created_utc"], "criteria": protocol["criteria"]}
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["selection_protocol"] = {"verified": False, "reason": str(error)}
    for stage in ("grid", "full"):
        folder, summary_path = directory / stage, directory / stage / "summary.json"
        if not summary_path.is_file():
            result["stages"][stage] = {"status": "not_started_or_no_completed_cell", "verified_rows": [],
                                       "reason": "尚无实际 sampler summary"}
            continue
        try:
            if not final.get("verified") or not native.get("verified"):
                raise ValueError("Main final selection/native materialization is not verified")
            if not result["selection_protocol"]["verified"]:
                raise ValueError("Pre-registered sampler selection protocol is not verified")
            from types import SimpleNamespace
            if hybrid:
                from tools import evaluate_init_diffusion_hybrid_sampler as driver
            elif fresh_native:
                from tools import evaluate_init_diffusion_native_sampler as driver
            else:
                from tools import evaluate_init_diffusion_sampler as driver
            summary, plan = read(summary_path), read(folder / "plan.json")
            if summary["plan"] != plan or summary.get("main_selection_unchanged") is not True or summary.get("new_selection_made") is not False:
                raise ValueError("Sampler summary/plan or no-selection guards disagree")
            frozen_sampler = (plan.get("frozen_sampler") or {}).get("path")
            actual_plan = driver.make_plan(SimpleNamespace(main_art=Path(artifact_dir), stage=stage,
                manifest=driver.MANIFEST, frozen_sampler=Path(frozen_sampler) if frozen_sampler else None,
                full_train_seeds=plan["train_seeds"]))
            if actual_plan != plan:
                raise ValueError("Sampler plan differs from actual main/checkpoint/config/scene/source cohort")
            rows = summary["rows"]
            if type(summary.get("complete")) is not bool or len(rows) > len(plan["cells"]):
                raise ValueError("Invalid actual sampler completion declaration")
            if summary["complete"] != (len(rows) == len(plan["cells"])):
                raise ValueError("Sampler completion does not match actual cell count")
            for row, cell in zip(rows, plan["cells"]):
                cell_dir = driver.cell_folder(folder, cell)
                driver.validate_completed_cell(row, cell, cell_dir, plan, require_execution=True)
                execution = read(cell_dir.parent / (cell_dir.name + ".execution.json"))
                if execution["started_unix"] < datetime.fromisoformat(protocol["created_utc"]).timestamp():
                    raise ValueError("Actual sampler cell predates pre-registered selection criterion")
            statistics = sampler_statistics(rows, plan["cells"], complete=summary["complete"], baseline_option=result["baseline_option"]) if stage == "grid" else []
            parity = [row for row in rows if row["option"] == result["baseline_option"]]
            result["stages"][stage] = {"status": "verified_complete" if summary["complete"] else "verified_partial",
                "verified_rows": rows, "planned_cells": len(plan["cells"]), "complete": summary["complete"],
                "plan": {"path": str(folder / "plan.json"), "sha256": sha(folder / "plan.json")},
                "summary": {"path": str(summary_path), "sha256": sha(summary_path)},
                "default20_verified_cells": len(parity), "statistics": statistics,
                "frozen_sampler": plan.get("frozen_sampler"), "full_comparison_or_reselection": False}
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            result["stages"][stage] = {"status": "excluded_invalid", "verified_rows": [], "reason": str(error),
                                       "summary": {"path": str(summary_path), "sha256": sha(summary_path)}}
    decision_path = Path(artifact_dir) / ("frozen_hybrid_sampler.json" if hybrid else "frozen_native_sampler.json" if fresh_native else "frozen_sampler.json")
    result["actual_frozen_selection"] = {"verified": False, "reason": "尚无实际 " + decision_path.name}
    if decision_path.is_file():
        try:
            grid = result["stages"]["grid"]
            if grid["status"] != "verified_complete":
                raise ValueError("Frozen sampler cannot cite partial/unverified grid")
            decision = read(decision_path)
            proof = selector.rank_actual_rows(grid["verified_rows"], complete=True)
            if (decision.get("selection_is_frozen") is not True or decision["selected_option"] != proof["selected_option"]
                    or decision["criterion_proof"] != proof or decision["protocol_sha256"] != sha(protocol_path)
                    or decision["grid_summary_sha256"] != grid["summary"]["sha256"]
                    or Path(decision["grid_summary"]).resolve() != Path(grid["summary"]["path"]).resolve()
                    or decision["selector_sha256"] != sha(selector.__file__)
                    or decision.get("main_selection_unchanged") is not True
                    or decision.get("full50000_cannot_reselect") is not True):
                raise ValueError("Frozen sampler decision differs from actual pre-registered criterion/grid")
            result["actual_frozen_selection"] = {"verified": True, "path": str(decision_path),
                "sha256": sha(decision_path), "created_utc": decision["created_utc"],
                "selected_option": decision["selected_option"], "criterion_proof": proof}
        except (OSError, KeyError, TypeError, ValueError) as error:
            result["actual_frozen_selection"] = {"verified": False, "reason": str(error)}
    full = result["stages"]["full"]
    if full.get("verified_rows"):
        evidence = full.get("frozen_sampler") or {}
        if (not result["actual_frozen_selection"].get("verified")
                or Path(evidence.get("path", "")).resolve() != decision_path.resolve()
                or evidence.get("sha256") != sha(decision_path)):
            result["stages"]["full"] = {"status": "excluded_invalid", "verified_rows": [],
                                         "reason": "Full sampler does not cite the verified actual preregistered frozen decision"}
    if not fresh_native:
        result["strict_replay_failure"] = sampler_replay_failure_state(artifact_dir, directory)
        if result["strict_replay_failure"].get("verified"):
            result["stages"]["grid"] = {"status": "FAILED_DEFAULT20_PARITY", "verified_rows": [],
                "reason": "首个default20实际复现失败，旧1e-6门槛保持不变；该grid永久排除选优"}
    if fresh_native:
        result["native_configuration"] = sampler_configuration_state(artifact_dir, result, hybrid=hybrid)
    return result


def sampler_configuration_state(artifact_dir, state, *, hybrid=False):
    """Publish standalone sampler commands only after actual full-scope proof."""
    path = Path(artifact_dir) / ("selected_hybrid_sampler_native_materialization.json" if hybrid else "selected_sampler_native_materialization.json")
    expected_status = "ACTUAL_SELECTED_HYBRID_NATIVE_EVAL_YAML_MATERIALIZED_CPU_COMPOSE_PASS" if hybrid else "ACTUAL_SELECTED40_NATIVE_EVAL_YAML_MATERIALIZED_CPU_COMPOSE_PASS"
    no_anchor_key = "no_native_full40_fresh_anchor" if hybrid else "no_native_full20_fresh_anchor"
    try:
        value = read(path)
        if not value:
            raise ValueError("尚无实际 standalone sampler YAML materialization")
        if value["status"] != expected_status:
            raise ValueError("Sampler YAML artifact does not certify actual materialization")
        full = state["stages"]["full"]
        decision = state["actual_frozen_selection"]
        if full["status"] != "verified_complete" or not decision.get("verified"):
            raise ValueError("Standalone sampler YAML cannot cite incomplete full or unverified selection")
        if (value["selected_option"] != decision["selected_option"]
                or value["frozen_sampler"] != {"path": decision["path"], "sha256": decision["sha256"]}
                or value["full_actual"]["summary_sha256"] != full["summary"]["sha256"]
                or Path(value["full_actual"]["summary"]).resolve() != Path(full["summary"]["path"]).resolve()
                or value["full_actual"]["complete_2GEN_TRAIN4817"] is not True):
            raise ValueError("Sampler YAML cites a different actual selection/full cohort")
        verify_record(value["configuration"])
        verify_record({"path": value["full_actual"]["scope_audit"], "sha256": value["full_actual"]["scope_audit_sha256"]})
        for source, wanted in value["sources"].items():
            verify_record({"path": ROOT / source, "sha256": wanted})
        written = [check for check in value["CPU_composition_checks"] if check["configuration"] == "actualwritten"]
        if (set(check["GEN_seed"] for check in written) != {2817, 3817}
                or any(check["GPU_run"] is not False or check["model_conditioning_exact"] is not True
                       or check["immutable_native_hook_factory_and_kwargs_exact"] is not True
                       or check["sorted_native_full_manifest_exact"] is not True
                       or check["TEST_data_and_trainer_exact"] is not True
                       or check["sampling_steps"] != 40 or check["inference_t_eps"] != .05
                       or check["EMA_updates"] != 20480 for check in written)
                or value[no_anchor_key] is not True
                or value["full50k_did_not_reselect"] is not True
                or value["GPU_started"] is not False):
            raise ValueError("Sampler native composition/scope guard differs")
        return {"verified": True, "metadata": {"path": str(path), "sha256": sha(path)},
                "configuration": value["configuration"], "selected_option": value["selected_option"],
                "command": value["shell_command"], "alternative_GEN_command_override": value["alternative_GEN_command_override"],
                "no_fresh_full_baseline_anchor": True, "no_reselection": True,
                "experiment": "init_diffusion_lane_conditioned_overnight_best_hybrid_sampler_eval" if hybrid else "init_diffusion_lane_conditioned_overnight_best_sampler_eval"}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"verified": False, "reason": str(error)}


def sampler_figure_state(artifact_dir, native, hybrid):
    """Recheck already generated inference figures against each frozen cohort."""
    path = Path(artifact_dir) / "analysis/report_figures/sampler_tradeoffs_actual.metadata.json"
    try:
        value = read(path)
        if not value:
            raise ValueError("尚无实际 sampler tradeoff 图")
        for state, prefix, expected in ((native, "native", 16), (hybrid, "hybrid", 8)):
            decision = state["actual_frozen_selection"]
            grid = state["stages"]["grid"]
            if (not decision.get("verified") or grid["status"] != "verified_complete"
                    or value[prefix + "_decision_sha256"] != decision["sha256"]
                    or value[prefix + "_grid_rows"] != len(grid["verified_rows"])
                    or len(grid["verified_rows"]) != expected):
                raise ValueError("Figure inference cohort differs from certified complete actual grid")
        if (value["phase_baselines_are_different"] is not True or value["new_training_runs"] != 0
                or value["no_cumulative_ratio_multiplication"] is not True
                or value["torch_imported"] is not False or value["GPU_started"] is not False):
            raise ValueError("Figure scope/CPU guard differs")
        verify_record({"path": ROOT / "tools/build_overnight_sampler_figures.py", "sha256": value["source_tool_sha256"]})
        for figure, wanted in value["figure_paths"].items():
            verify_record({"path": figure, "sha256": wanted})
        for option, changes in value["native_percent_change_vs_fresh20"].items():
            actual = native["actual_frozen_selection"]["criterion_proof"]["comparisons"][option]["ratios"]
            if any(not math.isclose(changes[key], 100 * (actual[key] - 1), rel_tol=0., abs_tol=1e-12) for key in METRICS):
                raise ValueError("Figure native values differ from frozen actual comparisons")
        actual = hybrid["actual_frozen_selection"]["criterion_proof"]["comparisons"]["hybrid_pos0001"]["ratios"]
        if any(not math.isclose(value["hybrid_percent_change_vs_separate_fresh40"][key], 100 * (actual[key] - 1), rel_tol=0., abs_tol=1e-12) for key in METRICS):
            raise ValueError("Figure hybrid values differ from frozen actual comparisons")
        readable_pngs = [str(path.parent / name) for name in ("sampler_steps_actual.png", "sampler_hybrid_actual.png")]
        if not all(name in value["figure_paths"] for name in readable_pngs):
            raise ValueError("Publication-width sampler panels are not certified")
        return {"verified": True, "metadata": {"path": str(path), "sha256": sha(path)},
                "pngs": readable_pngs,
                "svg": next(name for name in value["figure_paths"] if name.endswith(".svg")),
                "caption": "每个面板使用自己的新跑基线，不表示跨阶段复合收益；两套TRAIN权重复用、各两GEN，不新增训练N。负值表示该指标改善。"}
    except (OSError, KeyError, TypeError, ValueError, StopIteration) as error:
        return {"verified": False, "reason": str(error)}


def sampler_replay_failure_state(artifact_dir, directory):
    """Verify existing failed replay and same-entry diagnostics; never rerun."""
    try:
        artifact_dir, directory = Path(artifact_dir), Path(directory)
        path = artifact_dir / "actual_native_default20_parity_failure_review.json"
        review = read(path)
        if not review:
            return {"verified": False, "reason": "暂无实际strict replay失败记录"}
        if review["status"] != "ACTUAL_DEFAULT20_PARITY_FAILED_STOPPED" or review["inference_grid_interpretation_allowed"] is not False:
            raise ValueError("Replay failure review does not certify excluded failed grid")
        for name, value in review["artifact_sha256"].items():
            verify_record({"path": str(ROOT / name), "sha256": value})
        folder = directory / "grid/train4817_gen2817_n20_eps005"
        execution = read(folder.parent / (folder.name + ".execution.json"))
        if execution["status"] != "failed" or execution["returncode"] == 0 or (folder / "cell_summary.json").exists():
            raise ValueError("Failed original default20 was relabelled as successful")
        parity = read(folder / "default20_main_endpoint_comparison.json")
        if parity["within_absolute_tolerance"] is not False or parity["native_absolute_tolerance"] != 1e-6:
            raise ValueError("Original strict parity threshold or failure changed")
        result = {"verified": True, "review": {"path": str(path), "sha256": sha(path)},
            "execution": {"path": str(folder.parent / (folder.name + ".execution.json")), "sha256": sha(folder.parent / (folder.name + ".execution.json"))},
            "parity": parity, "successful_sampler_cells": 0, "main_selection_unchanged": True}
        replay_path = artifact_dir / "sampler_numerical_repeatability_independent_cpu_review.json"
        replay = read(replay_path)
        if replay:
            if replay["status"] != "ACTUAL_REPEATABILITY_FAILURE_IN_BOTH_ENTRYPOINTS_NO_SAMPLER_OPTION_INFERENCE":
                raise ValueError("Numeric diagnostic status differs")
            for record in replay["reports"].values():
                verify_record({"path": str(ROOT / record["path"]), "sha256": record["sha256"]})
            result["repeatability_review"] = {"path": str(replay_path), "sha256": sha(replay_path)}
            result["native_ranges"] = {key: max(record["metrics"][key] for record in replay["reports"].values()) - min(record["metrics"][key] for record in replay["reports"].values()) for key in METRICS}
            result["causal_operator_proven"] = False
        return result
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"verified": False, "reason": str(error)}


def transfer_state(artifact_dir, stages):
    """Metadata/source review of EMA and native memory limits, not a forecast."""
    artifact_dir = Path(artifact_dir)
    result = {}
    review_path = artifact_dir / "ema_optimizer_128epoch_transfer_cpu_review.json"
    try:
        review = read(review_path)
        if not review:
            raise ValueError("尚无 EMA/optimizer CPU review")
        if review["GPU_used"] is not False or review["model_instantiated"] is not False or review["active_sources_changed"] is not False:
            raise ValueError("EMA CPU review scope differs")
        verify_record(review["metadata_artifact"])
        meta = read(review["metadata_artifact"]["path"])
        if meta != review["actual_control_checkpoint_metadata"]:
            raise ValueError("Actual control metadata differs from review")
        verify_record({"path": meta["checkpoint"], "sha256": meta["checkpoint_sha256"]})
        if meta["global_step"] != 10240 or meta["ema_num_updates"] != 10240 or meta["optimizer_state_step_min"] != 10240 or meta["optimizer_state_step_max"] != 10240:
            raise ValueError("Actual control update counters disagree")
        if not any(s.get("included") and s["label"] == "long_seed4817_control" for s in stages):
            raise ValueError("Actual control stage is not verified")
        for source in review["source_evidence"]:
            verify_record(source)
        verify_record(review["TRAIN"]["actual_serialized_order"])
        if review["TRAIN"]["files_per_epoch"] != controller.TRAIN_COUNT or review["source_rule"]["first_cap_update"] != 89990:
            raise ValueError("EMA source rule/TRAIN boundary differs")
        result["ema_optimizer"] = {"verified": True, "review": {"path": str(review_path), "sha256": sha(review_path)},
                                   "actual_control": meta, "cap_boundaries": review["cap_boundaries"],
                                   "kernel_horizons": review["observed_and_projected_horizons"],
                                   "optimizer": review["optimizer"], "candidate_kernel_is_projection": True}
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["ema_optimizer"] = {"verified": False, "reason": str(error)}
    review_path = artifact_dir / "native_guard_long_memory_boundary_cpu_review.json"
    try:
        review = read(review_path)
        if not review:
            raise ValueError("尚无 native guard/memory CPU review")
        if review["GPU_used_by_auditor"] is not False or review["active_source_or_wrapper_modified"] is not False:
            raise ValueError("Native memory review scope differs")
        for path, wanted in review["source_evidence"].items():
            verify_record({"path": path, "sha256": wanted})
        verify_record(review["new_CPU_transform_batch_probe"])
        memory = review["actual_training_memory"]
        actual = [json.loads(line.removeprefix("RUNTIME_MEMORY ")) for line in Path(memory["log_path"]).read_text().splitlines() if line.startswith("RUNTIME_MEMORY ")]
        for key in ("snapshot_first", "snapshot14336", "snapshot_latest"):
            if memory[key] not in actual:
                raise ValueError("Recorded production memory sample not in actual runtime log")
        guard = review["native_guard"]
        verify_record({"path": guard["entrypoint"], "sha256": guard["sha256"]})
        result["native_memory"] = {"verified": True, "review": {"path": str(review_path), "sha256": sha(review_path)},
                                   "guard": guard, "actual_training_memory": memory,
                                   "full128_runtime_feasibility_proven": False}
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["native_memory"] = {"verified": False, "reason": str(error)}
    return result


def sampler_report_lines(state, *, output):
    fresh_native = state.get("fresh_native_baselines", False)
    hybrid = state.get("hybrid_position_cohort", False)
    baseline_name = "fresh N40" if hybrid else "fresh N20" if fresh_native else "default20"
    lines = ["## 追加 position-only hybrid 推断附录（单独8cell cohort）" if hybrid else "## 新 native-to-native sampler 实测附录" if fresh_native else "## 原严格 sampler 复现审计（失败保留）", "",
             "本附录复用主搜索已选 checkpoint，仅改变推理 steps / t_eps；实际 training updates=0，不改变主模型选择。GEN 先在每 TRAIN 内平均，grid最多复用两套训练权重，full以实际列出的TRAIN为准，不新增独立训练 N。TRAIN4817参与自适应模型选择，TRAIN5817是已冻结模型的独立确认；sampler同时使用这两套权重择优，因此并未获得新的独立 sampler 确认。改变 t_eps 会改变末端积分路径，不能都解释成纯 ODE 步长精度改善。", ""]
    failure = state.get("strict_replay_failure", {})
    if failure.get("verified"):
        lines += ["原预注册grid的首个N20实际未通过1e-6原生单位复现门槛，未接受任何successful sampler cell，也没有任何changed-step结果；原driver/selector/protocol保持不变，永久排除选优。主搜索8+4实际结果和选中训练recipe不受此失败重新选择。证据：" + link(failure["review"]["path"], output=output) + "。", ""]
        if failure.get("repeatability_review"):
            lines += ["相同checkpoint/seed的两个native结果与原冻结runner自重放也都不能达到1e-6；CPU记录的map/G1/EMA/normalizer完全一致。该现象没有证明native restoration有误，CUDA scatter非确定性是待验证机制，source noise逐元素一致尚未测，重复同TRAIN/GEN不会增加独立N。来源：" + link(failure["repeatability_review"]["path"], output=output) + "。", ""]
    if hybrid:
        lines += ["本追加实验另行预注册两options×两TRAIN×两GEN=8cells；四个重新执行的N40/eps=.05为唯一配对anchor，hybrid仅在0<t<.05将pos维0:2的velocity分母floor改为1e-4，heading/L/W/speed/type路径保留。完整8cells且R≤.98与pooled AND每TRAIN五gates全通过才选择hybrid，否则保留fresh N40。前native16选择和主训练recipe保持不变。两TRAIN权重都参与这次选择，不新增独立holdout；full只对该次冻结option作scope audit，不能重选。实际40步最后t约.025大于1e-4，因此hybrid position使用真实t；1e-4 floor在此次最后更新并未生效，这里没有单独优化该floor数值。", ""]
    elif fresh_native:
        lines += ["这是一项新建并另行预注册的native-to-native推断实验：四个重新执行的N20是唯一比较基线，不复用旧失败结果、不宣称与主endpoint通过1e-6。四options×两TRAIN×两GEN=16cells；仅eligible且pooled R≤.98才选更低R，否则保留fresh N20。2%是对观察到数值波动设置的描述性margin，不是置信区间或统计显著性保证。", ""]
    protocol = state.get("selection_protocol", {})
    if protocol.get("verified"):
        requirement = "完整8cells、四个fresh N40 anchors均实际完成" if hybrid else "完整16cells、四个fresh N20 anchors均实际完成" if fresh_native else "完整20cells且四个default20全部通过≤1e-6原生parity"
        threshold = "R≤.98，否则保留fresh40步" if hybrid else "R≤.98，否则保留fresh20步" if fresh_native else "R<1，否则保留20步"
        lines += [f'推理选择准则已在实际 UTC {protocol["created_utc"]}、本cohort所有 sampler cell 启动之前冻结，源SHA见 {link(protocol["path"], output=output)}。{requirement}后，计算GEN→TRAIN均值与R7；L/W/speed≤baseline×1.15、onroad≥baseline−.02、vehicle/GT在.9..1.1须同时通过pooled及每TRAIN。只选eligible最小{threshold}；精确R tie依次更少steps、eps更接近训练.05、option名。full50k禁止重选。', ""]
    else:
        lines += ["sampler selection protocol 尚未通过本报告核验：" + protocol.get("reason", "未提供") + "。", ""]
    frozen = state.get("actual_frozen_selection", {})
    if frozen.get("verified"):
        proof = frozen["criterion_proof"]
        cost_ratio = proof["model_evaluations_ratio_to_baseline40"] if hybrid else proof["model_evaluations_ratio_to_default20"]
        lines += [f'实际已冻结推理设置 **{frozen["selected_option"]}**（{proof["steps"]} steps，base eps={proof["eps"]}），由完整实际grid按上述预注册规则选择。相对{baseline_name}的模型求值次数为×{number(cost_ratio, 2)}；这是计算次数比，不是未经实测的wall-time倍率。冻结artifact：{link(frozen["path"], output=output)}。主模型配方选择保持不变。', ""]
    else:
        lines += ["尚未发布已冻结 sampler 设置：" + frozen.get("reason", "未提供") + "。", ""]
    for stage in ("grid", "full"):
        value = state.get("stages", {}).get(stage, {})
        rows = value.get("verified_rows", [])
        lines += [f'### sampler {stage} / {value.get("status", "not_started")}', ""]
        if not rows:
            lines += [value.get("reason", "尚无核验完成的 sampler cell") + "；不报告 sampler 改善或最佳设置。", ""]
            continue
        lines += [f'实际完成 {len(rows)}/{value["planned_cells"]} cells；只有 execution rc0、native marker、checkpoint/source/scene/metric artifact SHA 与 hook metadata 全部通过才纳入。来源：{link(value["summary"]["path"], output=output)}。', ""]
        if stage == "grid":
            if fresh_native:
                lines += [f'{baseline_name} anchors 已核验 {value["default20_verified_cells"]}/4 cells；这是新的配对基线，没有old-main parity成功声明。', ""]
            else:
                lines += [f'default20 已核验 {value["default20_verified_cells"]}/4 cells；其七项 helper 原生单位绝对误差必须全部 ≤1e-6。原生数值复现通过也不宣称 GPU RNG 逐 bit 相同。', ""]
        if not value.get("complete"):
            lines += ["当前只有 partial facts，禁止排序、选优或归因 sampler 效果。", ""]
            lines += table(("TRAIN", "GEN", "option", *SHORT),
                           [[r["TRAIN_seed"], r["GEN_seed"], r["option"], *[number(r[k] / MULTIPLIERS[k]) for k in METRICS]] for r in rows])
            continue
        if stage == "grid":
            stats = value["statistics"]
            lines += table(("option", "TRAIN N", *SHORT),
                           [[r["option"], r["independent_TRAIN_N"], *[stat(r["metrics_raw"][k]) for k in METRICS]] for r in stats])
            lines += ["六项为 raw JSD，Col% 为百分比；± 是两套复用 TRAIN 权重的 GEN 均值间 sample SD，并非各cell都对应独立训练。以下 ratio/R 均对同checkpoint、同TRAIN/GEN、本cohort重新执行的anchor。", ""]
            lines += table(("option", "pooled R", "Speed ratio", "Length ratio", "Width ratio", "Col ratio", "pooled / each TRAIN gates"),
                           [[r["option"], number(r["pooled"]["R"], 4), *[number(r["pooled"]["ratios"][k], 4) for k in ("speed_jsd", "length_jsd", "width_jsd", "collision_rate")],
                             f'{r["pooled"]["eligible"]} / {r["pooled"]["each_training_seed_gates_pass"]}'] for r in stats])
            lines += table(("option", "TRAIN", "R", "Speed ratio", "Col ratio", "gates"),
                           [[r["option"], seed, number(v["R"], 4), number(v["ratios"]["speed_jsd"], 4), number(v["ratios"]["collision_rate"], 4), "PASS" if v["eligible"] else "FAIL"]
                            for r in stats for seed, v in r["per_TRAIN"].items()])
            lines += ["这是完整配对 grid 的描述性比较，本报告不创建 sampler selection。若另行冻结一个推理设置并做 full50k，必须引用实际完整 grid SHA；不能用未完成 grid 或事后 full50k 重选。", ""]
            if fresh_native:
                times = {option: [row["eval_seconds"] for row in rows if row["option"] == option] for option in sorted({row["option"] for row in rows})}
                baseline_key = state.get("baseline_option", "n20_eps005")
                baseline_time = sum(times[baseline_key]) / len(times[baseline_key])
                baseline_steps = 40 if hybrid else 20
                lines += table(("option", "native run 秒/cell 均值", "范围秒", "实测时间/本次anchor", "model calls/本次anchor"),
                               [[option, number(sum(values)/len(values), 3), f'{min(values):.3f}–{max(values):.3f}',
                                 number((sum(values)/len(values))/baseline_time, 3), number(next(row["steps"] for row in rows if row["option"] == option)/baseline_steps, 2)]
                                for option, values in times.items()])
                if not hybrid:
                    lines += ["时间来自各实际cell的eval_seconds，覆盖native run的模型/restore/data setup和TEST，不包含子进程启动前的Python imports。40步模型求值次数为2倍，实测约1.34倍时间；不把计算次数与wall-time等同。", ""]
                    lines += ["实现含义：eps是低t时Euclidean速度分母的floor，不是solver终止时间；各option均匀时间网格1→0。令最后一次预测为x0_hat，v=(x_t−x0_hat)/max(t,eps)，x_next=x_t+(t_next−t)v。N20/eps=.05最后t=.05、t_next=0，最终x=x0_hat；N40/eps=.05最后t=.025，最终x=.5x_t+.5x0_hat；N40/eps=.0001最后分母为真实.025，最终x=x0_hat。该公式适用于pos/L/W/speed。圆周heading使用真实t与机器epsilon，separate type最终argmax最后的clean logits；不能把上述半混合公式套到heading或最终type。训练t由U(0,1)抽样，t<.05并非未训练范围。来源：" + link(ROOT / "src/smart/diffusion/scale_flow.py", "scale_flow.py（681–697、1565、1696、1803）", output=output) + "。", "",
                              "因此40/.05的结果同时涉及更多模型预测和最后一次Euclidean低t damping；额外残余变化可能影响宽度/速度分布，是符合源码的机制假设，尚未以可控实验隔离因果。不能把收益全部解释成数值截断误差下降，或声称模型训练目标改变。", ""]
                comparisons = {row["option"]: row["pooled"] for row in stats}
                if "n40_eps005" in comparisons and "n40_eps0001" in comparisons:
                    gap = comparisons["n40_eps0001"]["R"] - comparisons["n40_eps005"]["R"]
                    lines += [f'两40步选项的实际R：eps=.05为{comparisons["n40_eps005"]["R"]:.6f}，eps=.0001为{comparisons["n40_eps0001"]["R"]:.6f}，差{gap:+.6f}。eps=.0001的碰撞相对fresh20为{100*(comparisons["n40_eps0001"]["ratios"]["collision_rate"]-1):+.2f}%，而eps=.05有更强的尺寸收益。差距小于2%择优margin的量级，该margin针对相对N20而非两40选项之间；结合数值重放波动，不能称eps=.05是稳健最优。预注册选择保持不变。', ""]
        else:
            lines += ["本native sampler driver未导出feature histogram bins；这里只能报告实际七项metrics/counts/coverage，不能据此分析新40步生成结果的stationary或尺寸尾部分布。", ""]
            lines += table(("TRAIN", "GEN", "frozen option", *SHORT, "onroad%", "veh/GT"),
                           [[r["TRAIN_seed"], r["GEN_seed"], r["option"], *[number(r[k] / MULTIPLIERS[k]) for k in METRICS],
                             number(100 * r["generated_onroad_fraction"], 2), f'{r["num_generated_vehicles"]}/{r["num_gt_vehicles"]}'] for r in rows])
            by_training = {}
            for row in rows:
                by_training.setdefault(row["TRAIN_seed"], []).append(row)
            lines += table(("TRAIN", "GEN repeats", *SHORT),
                           [[seed, len(values), *[number(sum(row[key] for row in values) / len(values) / MULTIPLIERS[key]) for key in METRICS]]
                            for seed, values in by_training.items()])
            lines += ["上表是各TRAIN内两GEN的raw JSD均值；同一checkpoint的生成重复不增加独立训练N，不提供TRAIN间SD。full50k仅展示已冻结sampler的实际原生TEST metrics/counts；50,000 cache entries不是50,000个未见独立原始场景。不同GT population的4096 grid不混算ratio，也不依据full结果重新选sampler。", ""]
            if fresh_native:
                lines += ["这次full没有重新执行本cohort的baseline full anchor，不能报告native full配对R或把历史main20当新的fresh full对照。", ""]
    configuration = state.get("native_configuration", {})
    if configuration.get("verified"):
        lines += [("实际hybrid40步" if hybrid else "实际40步") + "standalone评价配置已通过两GEN的native builder/hook/数据/EMA纯CPU compose核验：" + link(configuration["configuration"]["path"], output=output) + "；metadata：" + link(configuration["metadata"]["path"], output=output) + "。以下命令采用已落地YAML的默认4817checkpoint/EMA20480，不改变训练20步设置：", "", "```bash",
                  "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                  "export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS",
                  "export SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC=1",
                  "python tools/run_init_diffusion_safe.py " + chr(92),
                  "  experiment=" + configuration["experiment"], "```", "",
                  "第二GEN可追加sampler_eval_seed=3817。未来更换checkpoint时应同步实际saved EMA update计数；没有128epoch已训练权重或bitwise replay保证。", ""]
    return lines


def render(snapshot, *, output=OUTPUT):
    rows, stages, groups = snapshot["rows"], snapshot["stages"], snapshot["groups"]
    plan, final, native = snapshot["plan"], snapshot["final"], snapshot["native"]
    count = sum(s.get("included", False) for s in stages)
    try:
        generated_local = datetime.fromisoformat(snapshot["generated_utc"]).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8")
    except ValueError:
        generated_local = "fixture（仅CPU测试）"
    lines = ["# InitDiffusion 固定地图夜间搜索分析", "", f'生成时间（上海/北京时间）：{generated_local}；UTC：{snapshot["generated_utc"]}。', "",
             "本轮授权截止：2026-10-12 10:00 UTC+8（02:00 UTC）；GPU 队列 cutoff：09:35 UTC+8（01:35 UTC）。恢复没有改变截止时间。目录名20261011表示开始时的UTC日期。", ""]
    if final.get("verified"):
        result = final["results"][final["selected"]]
        lines += [f'实际最终训练recipe选择：**{final["selected"]}**；完整 2 arms ×2 TRAIN ×2 GEN 的 8 个 endpoint 已核验；R={result["R"]:.4f}，R_attr={result["R_attr"]:.4f}。训练recipe比较均固定20步推断 / t_eps=.05；后续40步与hybrid是另行预注册的推断选择。', ""]
        selected_arm = next(a for a in plan["arms"] if a["name"] == final["selected"])
        control_arm = next(a for a in plan["arms"] if a["name"] == "control")
        lines += [f'选中 recipe：B{selected_arm["batch"]}，effective LR={selected_arm.get("lr", .0005)}；loss 相对 control：{loss_delta(selected_arm, control_arm)}；模型/表示相对 control：{config_delta(selected_arm, control_arm)}。完整分量配置在后文保留。', "",
                  f'每个模型实际训练 {final["actual_exposure"]:,} TRAIN cache exposures（{final["actual_exposure"] / controller.TRAIN_COUNT:.4f} 等价全数据 epoch）；map encoder 来自固定 AIRL32 source，G1 从头训练。此预算结论不等于 128 epoch 的最终收敛效果。', ""]
        lines += ["这是完整loss recipe、模型架构与batch设置组合的相等曝光对比，不能把指标变化单独归因于独立speed head。各分量、batch和模型改动在后文完整列出。", ""]
        challenger = snapshot.get("frozen", {}).get("challenger")
        assessed_names = ["control"] if challenger == "control" else ["control", challenger]
        lines += table(("final arm", "pooled R", "R_attr", "speed变化%", "collision变化%", "pooled / all TRAIN gates"),
                       [[name, number(final["results"][name]["R"], 4), number(final["results"][name]["R_attr"], 4),
                         number(100 * (final["results"][name]["ratios"]["speed_jsd"] - 1), 2),
                         number(100 * (final["results"][name]["ratios"]["collision_rate"] - 1), 2),
                         ("PASS" if final["results"][name]["aggregate_gates_pass"] else "FAIL") + " / " +
                         ("PASS" if final["results"][name]["eligible"] else "FAIL")] for name in assessed_names])
        if challenger:
            lines += table(("final metric", "control raw", "challenger raw", "变化%"),
                           [[short, number(final["results"]["control"]["metrics"][key] / MULTIPLIERS[key]),
                             number(final["results"][challenger]["metrics"][key] / MULTIPLIERS[key]),
                             number(100 * (final["results"][challenger]["ratios"][key] - 1), 2)]
                            for key, short in zip(METRICS, SHORT)])
            lines += ["此表为 matched4096 的完整两 TRAIN 配对均值，先平均每 TRAIN 的两 GEN；六项 raw JSD 无量纲、Col% 为 vehicle-weighted 百分比，均较小为好。helper 原生值为 Near/Lat ×10、Ang/Length/Width/Speed ×100、Col% 不缩放。负变化代表改善，整体 R 不能替代逐项取舍。", ""]
            lines += table(("challenger TRAIN", "角色", "R", "speed变化%", "collision变化%", "onroad%", "gates"),
                           [[seed, "adaptive" if seed == "4817" else "independent confirmation", number(v["R"], 4),
                             number(100 * (v["ratios"]["speed_jsd"] - 1), 2), number(100 * (v["ratios"]["collision_rate"] - 1), 2),
                             number(100 * v["metrics"]["generated_onroad_fraction"], 2), "PASS" if v["eligible"] else "FAIL"]
                            for seed, v in final["results"][challenger]["per_training_seed"].items()])
        if final["selected"] == "control":
            lines += ["候选没有同时满足预定门槛和整体改进条件，保留 control；不能报告新方法已经提升。", ""]
        elif result["per_training_seed"]["5817"]["R"] >= 1.:
            lines += ["独立 TRAIN5817 的 R 未改善，尽管整体选择符合冻结规则；需要保留这项不一致性。", ""]
        full_groups = [group for group in groups if group["phase"] == "full50000"]
        if len(full_groups) == 1 and challenger:
            full_candidate = next((arm for arm in full_groups[0]["arms"] if arm["variant"] == challenger), None)
            if full_candidate and full_candidate.get("comparison") and "ratios" in full_candidate["comparison"]:
                full_comparison = full_candidate["comparison"]
                lines += [f'已完成full50k scope audit：仅TRAIN4817×两GEN（NTRAIN=1），R={full_comparison["R"]:.4f}；L/W变化{100*(full_comparison["ratios"]["length_jsd"]-1):+.2f}% / {100*(full_comparison["ratios"]["width_jsd"]-1):+.2f}%，speed{100*(full_comparison["ratios"]["speed_jsd"]-1):+.2f}%，collision{100*(full_comparison["ratios"]["collision_rate"]-1):+.2f}%。full速度变化幅度很小，基本相当，不能从单次数值变化声称稳定改善；full碰撞退步与上表两TRAIN pooled变化须分别保留。full没有新增独立TRAIN，也未重新选模型。', ""]
        new_sampler = snapshot.get("native_sampler", {})
        new_choice = new_sampler.get("actual_frozen_selection", {})
        if new_choice.get("verified"):
            proof = new_choice["criterion_proof"]
            change = proof["comparisons"][new_choice["selected_option"]]
            lines += [f'单独的native-to-native推断cohort已完成16cells并冻结{new_choice["selected_option"]}：同已选权重、四个fresh N20基线，R={change["R"]:.4f}；L/W变化{100*(change["ratios"]["length_jsd"]-1):+.2f}% / {100*(change["ratios"]["width_jsd"]-1):+.2f}%，speed{100*(change["ratios"]["speed_jsd"]-1):+.2f}%，collision{100*(change["ratios"]["collision_rate"]-1):+.2f}%。这是推断设置的取舍，不改变上面的训练recipe选择，也不能与main历史N20混合算ratio。40步的model calls为2倍；完整指标、时间和小eps选项在独立附录列出。', ""]
        hybrid_sampler = snapshot.get("hybrid_sampler", {})
        hybrid_choice = hybrid_sampler.get("actual_frozen_selection", {})
        if hybrid_choice.get("verified"):
            proof = hybrid_choice["criterion_proof"]
            change = proof["comparisons"][hybrid_choice["selected_option"]]
            lines += [f'另行预注册8cell推断cohort已冻结 **{hybrid_choice["selected_option"]}**：相对本次四个fresh N40 anchors，R={change["R"]:.4f}，Near/Lat/Ang变化{100*(change["ratios"]["nearest_dist_jsd"]-1):+.2f}% / {100*(change["ratios"]["lat_dev_jsd"]-1):+.2f}% / {100*(change["ratios"]["ang_dev_jsd"]-1):+.2f}%，collision{100*(change["ratios"]["collision_rate"]-1):+.2f}%；L/W/speed变化{100*(change["ratios"]["length_jsd"]-1):+.3f}% / {100*(change["ratios"]["width_jsd"]-1):+.3f}% / {100*(change["ratios"]["speed_jsd"]-1):+.3f}%幅度接近数值重放波动，应视为基本相当。仅最后一步position保留直接x0投影，其他特征仍用原40步路径。两TRAIN均复用现有权重，不新增独立训练；这些ratio不能与前16cell或main搜索ratio相乘。full状态与可用配置在单独附录按实际证据列出。', ""]
        if native.get("verified"):
            entry = "tools/run_init_diffusion_safe.py" if snapshot.get("transfer", {}).get("native_memory", {}).get("verified") else "src/run.py"
            final_inference = hybrid_sampler.get("native_configuration", {})
            if final_inference.get("verified"):
                lines += ["当前建议采用已选训练recipe **separate_speed** 与单独验证的 **hybrid40推断**；训练配置：" + link(native["fit_config"], output=output) + "；hybrid评价配置：" + link(final_inference["configuration"]["path"], output=output) + "。两GEN full50k实际完成与两GEN CPU compose/source审计均已通过。原main20训练recipe比较、新native16和hybrid8各自保留自己的基线，不合成跨cohort改善比例。下一次128epoch仍从固定预训练map、fresh G1开始，是后续任务；本次没有完成128epoch。", "", "```bash",
                          "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                          "export SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC=1",
                          f"python {entry} " + chr(92),
                          "  experiment=init_diffusion_lane_conditioned_overnight_best",
                          f"python {entry} " + chr(92),
                          "  experiment=" + final_inference["experiment"], "```", ""]
            else:
                lines += [f'已通过实际 native materializer：{link(native["fit_config"], "训练配置", output=output)}、{link(native["eval_config"], "20步评价配置", output=output)}。下一次128epoch从固定预训练map、fresh G1开始；以下是后续命令，尚未完成128epoch训练。', "", "```bash",
                          f"python {entry} experiment=init_diffusion_lane_conditioned_overnight_best",
                          f"python {entry} experiment=init_diffusion_lane_conditioned_overnight_best_eval" +
                          (" ckpt_path=" + str(native["evidence_checkpoint"]) if native.get("evaluation_requires_explicit_checkpoint", True) else ""), "```", ""]
            lines += ["评价命令恢复上述实际selected checkpoint的完整模型/EMA/buffers；其预算结果和未来128epoch收敛效果须分别理解。", ""]
            if native.get("evaluation_requires_explicit_checkpoint") is False:
                lines += ["评价配置已默认指向本轮选中TRAIN4817的10.6224epoch预算checkpoint；完整保存路径和SHA留在来源证据中，不是未来128epoch的权重。", ""]
            if snapshot.get("sampler", {}).get("strict_replay_failure", {}).get("verified"):
                lines += ["完整权重/EMA/buffers的CPU恢复已核验，但原N20 TEST重放没有通过1e-6数值门槛；下列结果是已保存的真实评价，native命令不承诺重放逐指标相同。失败与新native推断cohort在附录明确分开。", ""]
    else:
        lines += [f'**当前为部分报告，尚无可认可的最终最佳配置。** {final["reason"]}。', ""]
        completed_pairs = [(group, arm) for group in groups if group["phase"] == "long"
                           for arm in group["arms"] if arm["variant"] != "control"
                           and arm.get("comparison") and "ratios" in arm["comparison"]]
        if completed_pairs:
            group, arm = max(completed_pairs, key=lambda pair: pair[0]["actual_exposure"])
            observed = arm["comparison"]
            tradeoff = ("这是尺寸收益与速度/碰撞代价的实际取舍，不能说所有指标都改善" if
                        all(observed["ratios"][k] < 1 for k in ("length_jsd", "width_jsd")) and
                        all(observed["ratios"][k] > 1 for k in ("speed_jsd", "collision_rate")) else
                        "这是已完整配对的实际变化，不能把R变化等同于每项指标同向改善")
            lines += [f'最新已完整配对的 long 对照为 {arm["variant"]}，TRAIN N={arm["independent_training_N"]}，同 {group["actual_exposure"]:,} 场景曝光，R={observed["R"]:.4f}。相对 control 的 length/width 变化为 {100 * (observed["ratios"]["length_jsd"] - 1):+.2f}% / {100 * (observed["ratios"]["width_jsd"] - 1):+.2f}%，speed/collision 为 {100 * (observed["ratios"]["speed_jsd"] - 1):+.2f}% / {100 * (observed["ratios"]["collision_rate"] - 1):+.2f}%；JSD及collision较小为好。{tradeoff}；TRAIN4817参与自适应选择，不能代替独立TRAIN5817确认。', ""]
    lines += [f'本轮只计 {count} 个已核验完成 stage、{len(rows)} 个实际预算/TRAIN/GEN/EMA observation。CPU 预检、未完成 checkpoint、失败尝试与计划候选不计入指标。', "",
              "这次预算内结果不证明 128 epoch 的最优训练效果；128 epoch 的 native fit 是后续建议，当前没有完成该训练。", "",
              "## 实际完成与缺失状态", ""]
    queue = snapshot.get("queue_complete") or {}
    active = snapshot.get("queue_active") or {}
    lines += [f'队列文件状态：`{queue.get("status", "尚无 queue_complete")}`；活动项：`{active.get("stage", "未记录")}`。文件声明与实际完成核验分别列出。', ""]
    recovery = snapshot.get("runtime", {}).get("recovery", {})
    if recovery.get("verified"):
        lines += [f'实际 runtime recovery 已启动，ledger 为 {link(recovery["protocol"]["path"], output=output)}。继承已完整完成的 {", ".join(sorted(set(recovery["inherited_stages"])))}；余下模型按相同 recipe/seed/数据次序/预算从 exposure=0 训练，未恢复失败 checkpoint。'
                  f'原 {recovery["original_frozen_source_count"]} 个 frozen source 文件保持一致；新增 recovery entrypoint、loader wrapper 和 helper 有独立 SHA，原18-file freeze 并不覆盖这些新文件。'
                  '执行策略已从 manual youngGC128 改为 cached public pickle_module、automatic/manual cyclic GC 均禁用、young interval0；继承 control 的运行策略不同，不能宣称相同 runtime cohort 或把差异归因于 model loss/math 变化。', ""]
        for item in recovery["fresh_stages"]:
            if item["matches_archived_initialization"]:
                completion_note = ("初始化记录本身不作为评价证据；已列成绩来自完整stage核验。" if
                                   any(s.get("included") and s["label"] == item["stage"] for s in stages) else "尚未完成的成绩不纳入。")
                lines += [f'实际 fresh retry `{item["stage"]}` 的 {len(item["metadata_fields_checked"])} 项初始化元数据与 archived attempt 一致，包括 G1/map hash、seed/B/LR/clip、normalizer 和 TRAIN/eval order；offset=0、optimizer 不延续。此初始化一致性不保证随后 GPU 轨迹逐 bit 相同；{completion_note}', ""]
    elif recovery.get("reason") and (queue.get("status") == "RECOVERING"):
        lines += ["恢复正在进行，但本报告尚未通过其 metadata 核验：" + recovery["reason"] + "。", ""]
    restart = snapshot.get("intentional_restart")
    if restart and restart.get("verified"):
        lines += [f'此前 control 尝试在最后已记录 update={restart["last_logged_update"]} 后被主动终止（returncode={restart["returncode"]}），用于统一最后 deadline forecast 的已加载 controller 版本，并把全部 18 arms fresh restart。归档是有意中止的运行诊断，不计为候选模型失败、完成训练量或成绩；来源见机器审计。', ""]
    arms = plan.get("arms", [])
    if arms:
        statuses = {s["label"]: s for s in stages}
        lines += table(("screen arm", "stage 状态", "纳入结果", "原因"),
                       [[arm["name"], statuses.get("screen_" + arm["name"], {}).get("status", "not_started / 无 execution"),
                         "是" if statuses.get("screen_" + arm["name"], {}).get("included") else "否",
                         statuses.get("screen_" + arm["name"], {}).get("reason", "—")] for arm in arms])
    other = [s for s in stages if s["phase"] != "screen" or not s.get("included")]
    if other:
        lines += table(("stage", "状态", "纳入结果", "原因"),
                       [[s["label"], s.get("status"), "是" if s.get("included") else "否", s.get("reason", "—")] for s in other])
    frozen = snapshot.get("frozen") or {}
    if frozen.get("actual_final_exposure_per_TRAIN") and frozen.get("train_seeds"):
        endpoint_rows = [r for r in rows if r["phase"] == "long"
                         and r["seen_training_examples"] == frozen["actual_final_exposure_per_TRAIN"]]
        cells = []
        statuses = {s["label"]: s.get("status") for s in stages}
        actual_count = 0
        for seed in frozen["train_seeds"]:
            for name in ("control", frozen["challenger"]):
                values = []
                for gen in frozen["generation_seeds"]:
                    matches = [r for r in endpoint_rows if r["variant"] == name and r["seed"] == seed and r["eval_seed"] == gen]
                    values.append("已核验" if len(matches) == 1 else "缺失" if not matches else "重复，不认可")
                    actual_count += len(matches) == 1
                cells.append([name, seed, *values, statuses.get(f"long_seed{seed}_{name}", "not_started")])
        lines += [f'最终预算 endpoint readiness：{actual_count}/8 个已核验 cell。单个已完成 cell 不能替代两 TRAIN 配对和 final selection 核验。', ""]
        lines += table(("final arm", "TRAIN", *[f"GEN{g}" for g in frozen["generation_seeds"]], "stage 状态"), cells)
    for phase in ("medium", "long", "full50000"):
        observed = [s for s in stages if s.get("included") and s["phase"] == phase]
        if not observed:
            lines += [f"{phase}：尚无通过完整核验的完成 stage；不补数值、不把缺失项填成 0。", ""]
    lines += ["## 数据、训练来源和评价条件", "",
              "TRAIN 使用 `src/waymo_data/full/training_map2_sd` 的 493,542 场景，全 iterator 按 seeded full order 连续遍历、每 epoch 重复该完整次序，保留最后的不足 batch。各 promotion 都 fresh G1/optimizer/scheduler/EMA，从 exposure=0 重新训练。", "",
              "仅加载 AIRL32 checkpoint 的预训练 map encoder 并永久冻结；不恢复它的 G1、optimizer 或 EMA。训练-only 8192 场景 normalizer 固定。screen 为确认 cohort 前 1024 场景；medium/long 为同一 matched4096；full50000 为整个 scenario_dreamer_val 的 scope audit，不再择优。", ""]
    scope = snapshot.get("corpus_scope", {})
    if scope.get("verified"):
        lines += [f'本报告重新核对 native filename corpus 与已有 overlap audit：full50000 的单位是 **{scope["cache_entries"]:,} 个 cache entries**，对应 {scope["unique_raw_keys"]:,} 个 unique `(source_tfrecord, record_index)` raw keys；matched4096 是 {scope["matched_confirmation_entries"]:,} entries / {scope["matched_confirmation_unique_raw_keys"]:,} unique raw keys。medium/long 反复使用同一 4096 selection，full50k 包含之前 aggregate audit 已见条目；这些 scope 不能称为 unseen independent test，GEN repeats、不同 cache timestep 也不能当独立原场景。', ""]
    for key, value in snapshot.get("input_files", {}).items():
        lines += [f'{key}：{link(value["path"], output=output)}；实际文件 SHA256 `{value["sha256"]}`。', ""]
    conditions = [g["cohort"].get("conditioning") for g in groups if g["cohort"].get("conditioning")]
    if conditions:
        lines += ["实际报告中的条件：`" + cell(conditions[0]) + "`。", ""]
    else:
        lines += ["预定条件：参考地图、GT ego frame、固定 ego position/heading/shape/type、输入总 agent 数和 ego 槽位；ego speed 与非 ego 属性生成。尚无本轮 verified evaluator 输出来替代这些计划说明。", ""]
    lines += ["对应固定 ego 的 L1=0/type accuracy=1 是施加的条件，不能当作 ego 生成精度。非 ego GT type 是训练标签；推理使用生成 type，所有 vehicle metrics 都由生成 type0 的 population 决定。", "",
              "精度保持 32-true、float32 matmul medium、EMA 评价、20 步采样；具体每个实际 group 的 cohort/source fingerprint 留在机器审计中。不存在通过调整 evaluator 或筛掉 offroad 生成 vehicle 来改善主指标。", "",
              "## 七个指标、单位和选择规则", ""]
    lines += table(("helper 原名", "表中短名", "原生值 / raw 倍数", "population / 含义"),
                   [[k, label, "原生/raw=" + str(int(MULTIPLIERS[k])) if k != "collision_rate" else "保持百分比，不除 100",
                     "onroad vehicle 的 unsigned centerline 距离" if k == "lat_dev_jsd" else
                     "onroad vehicle 的相对 lane heading（degree histogram）" if k == "ang_dev_jsd" else
                     "碰撞 vehicle 数 / 全部 vehicle 数 ×100" if k == "collision_rate" else
                     "全部 vehicle 的最近 center 距离（singleton 无贡献）" if k == "nearest_dist_jsd" else
                     "全部生成 vehicle 的物理属性 histogram"] for k, label in zip(METRICS, SHORT)])
    lines += ["JSD 是 SciPy Jensen–Shannon distance 的平方（自然对数、无量纲），不是米或 m/s。官方 onroad 判定为到 resampled compact centerline points 距离 ≤1.5 m；lat/angle 仅比较 onroad 子集，offroad vehicle 仍进入 speed/shape/collision。更低 lat JSD 不能替代道路覆盖检查。", "",
              "matched4096 与 full50000 使用相同 GT 匹配算法和固定 histogram bins，但 GT population 的范围不同，不能混合排名。局部生成报告的 full_reference_agent_metrics 另以全50k GT 对比生成子集，只作辅助诊断，不替代 matched 主指标；全量 matched scope 才与 full-real scope 一致。", "",
              "先在每个 TRAIN 模型内平均配对 GEN2817/3817，再对 TRAIN 模型等权平均；screen 计划仅 GEN2817。七个 candidate/control arm 均值比值的算术平均为 R，Length/Width/Speed 三项比值平均为 R_attr。R 是 dimensionless ratio-of-arm-means，不是各场景 JSD 的平均、平均逐种子 R 或物理误差。raw/native 固定缩放在比值中抵消；collision 保持按 vehicle 总数加权的百分比。", "",
              "预定 gates：Length、Width、Speed 每项 ≤control×1.15；generated onroad fraction 不比 control 低超过 0.02；生成 vehicle 数/匹配 GT 数在 [0.9,1.1]。最终 pooled 和每个 TRAIN 都需通过 gates，且 challenger pooled R<1 才替换 control。这里允许单指标最多 15% 退化，须逐项报告。", "",
              "TRAIN4817 用于 adaptive screen/medium/long 选择；TRAIN5817 是冻结 recipe 后独立确认。两 GEN 不增加独立训练 N，full50000 也不增加 TRAIN N。N=1 不报告 SD；N=2 的 sample SD 仅描述两个训练模型，不能证明稳健最优，也不能把 4817 当未参与选择的 holdout。", "",
              "## 实际指标与每阶段比较", ""]
    if not groups:
        lines += ["本轮尚无 actual-only 结果表。以下实验配置仅是计划，性能结论等待完整 stage。", ""]
    for group in sorted(groups, key=lambda g: (("screen", "medium", "long", "full50000", "other").index(g["phase"]), g["actual_exposure"], g["cohort_sha256"])):
        lines += result_group_lines(group, output=output)
    if final.get("verified"):
        lines += ["### 冻结最终选择的逐指标结果", ""]
        selected = final["results"][final["selected"]]
        baseline = final["results"]["control"]
        lines += table(("metric", "control raw / Col%", "selected raw / Col%", "control native", "selected native", "变化%"),
                       [[k, number(baseline["metrics"][k] / MULTIPLIERS[k]), number(selected["metrics"][k] / MULTIPLIERS[k]),
                         number(baseline["metrics"][k]), number(selected["metrics"][k]), number(100 * (selected["ratios"][k] - 1), 2)] for k in METRICS])
        lines += table(("TRAIN", "角色", "selected R", "R_attr", "speed变化%", "L变化%", "W变化%", "gates"),
                       [[seed, "adaptive selection" if seed == "4817" else "independent confirmation", number(v["R"], 4), number(v["R_attr"], 4),
                         *[number(100 * (v["ratios"][k] - 1), 2) for k in ("speed_jsd", "length_jsd", "width_jsd")],
                         "PASS" if v["eligible"] else "FAIL"] for seed, v in selected["per_training_seed"].items()])
    historical = snapshot.get("historical", {})
    lines += ["## 旧搜索的种子敏感性（独立历史证据）", ""]
    if historical.get("verified"):
        facts, pooled = historical["facts"], historical["pooled_under_current_gate"]
        control, candidate = facts["control"], facts["candidate"]
        means = facts["arm_means_native"]
        lines += [f'旧 {link(historical["decision"]["path"], "final_selection", output=output)} 的 2 arms ×3 TRAIN ×2 GEN 共 12 cells 已按六个 actual stage 的 execution manifest SHA、terminal DONE、实际 train/eval filename manifest SHA 和指标报告核验。各 TRAIN 内平均两 GEN，再平均三个 TRAIN；独立训练 N=3，817 是 adaptive discovery，1817/2817 是冻结后确认。', "",
                  f'旧 pooled 规则选择 **{facts["selected"]}**：combo R={pooled["R"]:.5f}、speed ratio={pooled["ratios"]["speed_jsd"]:.5f}，pooled gates={"PASS" if pooled["eligible"] else "FAIL"}。本节不重新加载 checkpoint tensors，不把这些旧结果混入本轮 TRAIN4817/5817。', ""]
        lines += table(("metric", "旧 control raw / Col%", "旧 combo raw / Col%", "旧 control native", "旧 combo native", "变化%"),
                       [[key, number(means[control][key] / MULTIPLIERS[key]), number(means[candidate][key] / MULTIPLIERS[key]),
                         number(means[control][key]), number(means[candidate][key]), number(100 * (pooled["ratios"][key] - 1.), 2)] for key in METRICS])
        lines += table(("旧 TRAIN", "角色", "R", "speed ratio", "speed变化%", "按本轮逐TRAIN gates"),
                       [[seed, "adaptive discovery" if seed == "817" else "post-freeze confirmation", number(result["R"], 5),
                         number(result["ratios"]["speed_jsd"], 5), number(100 * (result["ratios"]["speed_jsd"] - 1.), 2),
                         "PASS" if result["eligible"] else "FAIL"] for seed, result in historical["per_training_seed_current_gate"].items()])
        lines += [f'若把本轮更严格的 pooled+每 TRAIN gates 用于诊断这份旧矩阵，结论为 {"PASS" if historical["would_pass_new_per_TRAIN_gate"] else "FAIL"}。TRAIN1817 的 speed ratio 超过 1.15，而 TRAIN2817 的 speed 改善；这说明均值会遮住失败种子，支持本轮增加逐 TRAIN 门槛。它不改写旧冻结选择规则，也不证明某个单独参数导致种子差异。', ""]
    else:
        lines += ["旧最终矩阵尚未通过本报告核验：" + historical.get("reason", "未读取") + "。只保留此前设计依据，不冒充完整历史结果。", ""]
    lines += ["## Loss 和模型配置的对照", ""]
    control_arm = next((a for a in arms if a["name"] == "control"), {})
    if control_arm:
        loss = control_arm["recipe"]
        lines += ["control 的计划 loss；同一个分量仍可独立选择 kind、时间 power、系数和 TRAIN scale：", ""]
        lines += table(("分量", "kind", "time power", "weight", "scale"),
                       [[k, v.get("kind"), v.get("time_power"), v.get("weight"), v.get("scale")] for k, v in loss["components"].items()])
        lines += [f'计划全局设置：normalize_time_weights={loss.get("normalize_time_weights")}，std_floor={loss.get("std_floor")}，heading_space={loss.get("heading_space")}。实际权重为 max(t,0.05)^(-p)/C_p；t≈0 接近 clean data、t≈1 接近 source noise，生成从 1 到 0。因此较大的 p 更强调低噪声/接近 clean 的误差。C_p 是固定 Uniform(t) 解析均值，p1=3.995732、p2=39、p3=599.5；不是逐 batch 或改后的抽样密度归一化。', ""]
        lines += table(("计划 arm", "B", "loss 相对 control", "模型 / transport 相对 control"),
                       [[a["name"], a["batch"], loss_delta(a, control_arm), config_delta(a, control_arm)] for a in arms])
        lines += ["上表来自实际保存的 search_protocol，但尚未完成的 arm 不能据此宣称这些选项已产生性能变化；容量/梯度/CPU 一步预检也不是收敛验证。", ""]
        by_name = {arm["name"]: arm for arm in arms}
        if "combo_reference" in by_name and "speed_p0" in by_name:
            base_speed = by_name["combo_reference"]["recipe"]["components"]["velocity"]
            p0_speed = by_name["speed_p0"]["recipe"]["components"]["velocity"]
            lines += [f'具体地，speed_p0 相对 combo_reference 同时把 velocity time power 从 {base_speed["time_power"]} 改为 {p0_speed["time_power"]}、weight 从 {base_speed["weight"]} 改为 {p0_speed["weight"]}；不能把它的结果归因于纯 power 改动。normalized_speed_p0 在这组联合 loss 改动上再加入输入归一化，需作为另一组联合干预解释。', ""]
    effective = snapshot.get("effective_configs", [])
    if effective:
        lines += ["以下为 verified stage 真正保存的配置（不是仅解析计划）：", ""]
        lines += table(("arm / config", "loss hash", "optimizer / LR / clip", "G1 hidden/layers", "输入 / head / x0", "map / t sampler"), effective)
    clipping = snapshot.get("clipping_evidence", [])
    if clipping:
        lines += ["以下裁剪记录由实际 train_summary 和完整 pre-update train_trace 重算交叉检查。旧实验仅提供机制线索，不计入本轮 overnight 指标或 TRAIN N：", ""]
        lines += table(("阶段 / 范围", "updates", "clip", "clip_fraction", "首32步 grad norm", "末32步 grad norm", "实际来源"),
                       [[r["label"] + " / " + r["scope"], r["steps"], number(r["gradient_clip_val"], 2),
                         number(r["clip_fraction"], 4), number(r["first32_mean_grad_norm_before_clip"], 3),
                         number(r["last32_mean_grad_norm_before_clip"], 3), link(r["summary"]["path"], r["summary"]["sha256"][:8], output=output)] for r in clipping])
        lines += ["强全局裁剪会把总梯度缩放到阈值附近，因此增大一个分量的 weight 不等于等比例增大参数更新，也会改变各分量的相对梯度贡献。上述触发率说明这可能影响权重干预的效果；它不证明裁剪造成任何已观测的 JSD 改善或恶化。", ""]
    lines += ["所有 speed scalar arm 使用 `[pos,cos/sin,L,W,speed]`，沿生成 heading 重建 velocity；speed norm 已直接监督，没有再添加 vector magnitude loss。time_residual 维持 x0 目标但 head 梯度带 t 因子；高-noise t=U^0.5 改变抽样密度且未 importance-correct，不能描述成仅降低方差的等权实验。", "",
              "默认 road_edges 保留旧权重与数值路径；lanes_and_edges opt-in 只增加 type0/1/2 中心线 query，attention source/crop/radius/k 不变。训练缓存中心线是稀疏片段，naive lane-distance 辅助项未加入；不能用 road edge 的 position 当 centerline。", "",
              ]
    time_weights = ((snapshot.get("prior_review") or {}).get("nonuniform_time_sampling_effect", {})
                    .get("expected_normalized_weights", {}).get("0.5", {}))
    if "1" in time_weights and "2" in time_weights:
        lines += [f'已有 CPU 权重积分复核：保持 Uniform(t) 解析归一化常数时，t=U^0.5 下 p1 的平均有效权重约 {number(time_weights["1"], 3)}，p2 约 {number(time_weights["2"], 3)}（Uniform 下均为 1）。high_noise 同时干预时间密度及分量平均权重；这些是机制计算，不是新实验的性能观察。', ""]
    lines += ["## 真实图形、运行成本与不足", ""]
    for phase in ("screen", "medium", "long", "full50000"):
        files = [f for f in snapshot["figures"].get("embedded", []) if f["phase"] == phase]
        if files:
            for figure in files:
                lines += [image_link(figure["path"], figure["caption"], output=output), "", figure["caption"], ""]
                lines += ["可缩放原图：" + ", ".join(link(f["path"], output=output) for f in figure["exports"] if Path(f["path"]).suffix == ".svg") + "。", ""]
        else:
            lines += [phase + "：暂无与当前 actual-only inventory 一致的已生成图；不绘制缺失点或 N=1 SD。", ""]
    sampler_figure = snapshot["figures"].get("sampler", {})
    if sampler_figure.get("verified"):
        for number_in_figure, figure_path in enumerate(sampler_figure["pngs"], start=1):
            panel_caption = ("sampler16：相对fresh N20 anchors。" if number_in_figure == 1 else "hybrid8：相对另行重新执行的fresh N40 anchors。") + sampler_figure["caption"]
            lines += [image_link(figure_path, panel_caption, output=output), "", panel_caption, ""]
        lines += ["可缩放原图：" + link(sampler_figure["svg"], output=output) + "；实际grid/decision/source/file SHA：" + link(sampler_figure["metadata"]["path"], output=output) + "。", ""]
    if snapshot["figures"].get("compact_metadata"):
        lines += ["嵌入图的 actual 数据、独立 TRAIN membership、source SHA 与文件 SHA：" + link(snapshot["figures"]["compact_metadata"]["path"], output=output) + "。", ""]
    timing = snapshot.get("timing", {})
    if timing.get("per_completed_stage"):
        lines += table(("stage", "fit秒（一次）", "eval秒", "B / updates", "秒/update"),
                       [[r["stage"], number(r["fit_seconds_counted_once"], 1), number(r["eval_seconds_all_actual_GEN_budget_cells"], 1),
                         f'B{r["batch"]}/{r["fit_update_budget"]}', number(r["fit_seconds_per_update"], 4)] for r in timing["per_completed_stage"]])
        lines += [f'已完成主搜索 fit 合计 {number(timing.get("total_fit_seconds"), 1)} 秒，所有实际 eval cell 合计 {number(timing.get("total_actual_eval_seconds"), 1)} 秒。fit 时间对每个 stage 只计一次。失败尝试成本不冒充这些成功 fit 的训练量。', ""]
    frozen = snapshot.get("frozen") or {}
    if frozen:
        exposure = frozen.get("actual_final_exposure_per_TRAIN")
        epochs = number(exposure / controller.TRAIN_COUNT, 4) if isinstance(exposure, int) else "未记录"
        lines += [f'最终预算实际冻结为每 TRAIN {exposure if exposure is not None else "未记录"} TRAIN cache exposures（{epochs} 等价全数据 epoch），B512={frozen.get("final_budget_B512", "未记录")} updates，B256={frozen.get("final_budget_B256", "未记录")} updates；这是选择的预算，完成情况以上面的 verified stage 为准。', ""]
    else:
        lines += ["最终曝光预算尚未冻结。计划由 medium 的实测 throughput 选择能在截止前完成两 TRAIN pair 的最高共同 exposure，并预留 full50000 和报告时间；不把计划 8.50–12.75 epoch 写成实际训练量。", ""]
    runtime_appendix_start = len(lines)
    lines += ["## 附录A：恢复与内存诊断（不参与性能排名）", "",
              "控制台异常和 incomplete stage 保留为诊断，不能把已有部分 checkpoint 当完成 endpoint；恢复/重跑需单独记录，不将失败尝试加到模型已消费样本数。局部 GC/隔离进程是运行环境 workaround，正常独立读取失败附近文件也不能证明 SIGSEGV 的根因。", ""]
    runtime = snapshot.get("runtime", {})
    cpu, gpu = runtime.get("CPU", {}), runtime.get("GPU", {})
    if cpu.get("verified"):
        lines += ["已记录 CPU probe 的真实 JSON 与 lifecycle review 已交叉核验；没有在生成报告时重跑 probe：", ""]
        lines += table(("CPU policy", "实际 scene loads", "未回收 local unpickler classes", "确认 class/closure cycles"),
                       [[mode, probe["actual_scene_loads"], probe["unreachable_classes"], probe["confirmed_class_closure_cycles"]] for mode, probe in cpu["probes"].items()])
        lines += ["400 次真实 scene load 中，禁用自动 GC 时残留 400 个局部 unpickler class；在 batch 后只收集 generation0 的 probe 中为 0。该 Python 对象生命周期证据不证明原训练 SIGSEGV 的根因；CPU probe 不包含完整 Lightning/GPU 训练，也不能外推为百万次 load 的 RSS 预测。", ""]
    else:
        lines += ["CPU lifecycle 诊断未通过本报告核验：" + cpu.get("reason", "未读取") + "。", ""]
    if gpu.get("verified"):
        lines += [f'另一个实际已执行的 {gpu["steps"]}-update GPU runtime pilot 已通过 source artifact SHA、terminal DONE、三次 EMA eval、实际 GC event 与 memory log 核验；它不是主搜索 candidate，也不增加 TRAIN N。记录到 {len(gpu["generation0_actual_events"])} 次 generation0 collection（step128/256/384），不代表长任务已经安全。', ""]
        lines += table(("GPU pilot step", "allocated MiB", "reserved MiB / GiB", "peak RSS KiB / GiB"),
                       [[memory["step"], number(memory["cuda_allocated_mib"], 2),
                         f'{memory["cuda_reserved_mib"]:,} / {number(memory["cuda_reserved_mib"] / 1024, 2)}',
                         f'{memory["peak_rss_kib"]:,} / {number(memory["peak_rss_kib"] / 1024**2, 2)}'] for memory in gpu["actual_memory_samples"]])
        lines += ["这两个实际采样点的 allocated 约 201 MiB，reserved 上升到 13,854 MiB（13.53 GiB），peak RSS 约 2.71 GiB。reserved 包括 PyTorch allocator cache，不能直接当作 live tensor 量或泄漏；两点和短 pilot 不能证明长训练内存有界，更不能证明 GC 修复了 native crash 的根因。阶段间保留的旧 framework/model cycles 仍是单独风险，后续需继续记录。", ""]
    else:
        lines += ["GPU runtime pilot 未通过本报告核验：" + gpu.get("reason", "未读取") + "。", ""]
    failures = runtime.get("actual_long_failures", {})
    for failure in failures.get("verified_failures", []):
        lines += [f'后续真实主搜索失败：`{failure["stage"]}`，returncode={failure["returncode"]}，最后已记录 update={failure["last_logged_update"]}；faulthandler 是否处于 Garbage-collecting：{failure["faulthandler_garbage_collecting"]}。'
                  '该失败尝试没有完成训练或最终评价，不计成绩、exposure 或 TRAIN N；实际失败说明短期 GC pilot 不能保证长训安全，也没有证明 native crash 的根因。'
                  f'完整原始 execution/log 的 exact SHA 审计副本保留于 {link(failure["execution_copy"]["path"], output=output)} 和 {link(failure["log_copy"]["path"], output=output)}。恢复策略和 fresh retry 必须另行记录，不把原未完成 checkpoint 当作最终 endpoint。', ""]
        if failure.get("elapsed_process_seconds") is not None:
            lines += [f'这次失败独立子进程实际耗时 {number(failure["elapsed_process_seconds"], 1)} 秒；它是额外运行成本，没有加进成功 fit 的模型训练曝光量。', ""]
    if recovery.get("verified"):
        equivalent, bounded = recovery["equivalence_probe"], recovery["bounded_load_probe"]
        lines += [f'新 loader 的已有实际 CPU 记录已按 source SHA 和输出字段复核：{equivalent["unique_cache_entries"]} 个不同 native cache entries 中，Python 类型/key顺序/metadata 及 {equivalent["counters"]["tensor"]} 个 tensor 的 dtype/shape/stride/storage-offset/bytes（含 signed zero/NaN bits）一致；wrapper class 创建1次、复用99次。来源：{link(equivalent["path"], output=output)}。报告自身没有重跑或导入 Torch。', "",
                  f'另一个 no-GC CPU probe 反复读取2个实际 native caches 共 {bounded["actual_loads"]:,} 次，末端 local class=1、保留 tensor=0，RSS {bounded["first_rss_kib"]:,}→{bounded["last_rss_kib"]:,} KiB。来源：{link(bounded["path"], output=output)}。这不包含 PyG/Lightning/model/TensorFlow forward，既不是20,000个独立场景，也不证明完整 GPU 长训内存有界或解决 SIGSEGV 根因。', ""]
    runtime_appendix = lines[runtime_appendix_start:] + ["上述 CPU/GPU memory diagnostics 全部排除在性能表、R、gates、训练种子统计及模型选择之外。报告 builder 只读已有记录，自身不运行 GPU。", ""]
    del lines[runtime_appendix_start:]
    lines += ["## 已有证据给出的机制与边界", "",
              "以下是本轮之前 CPU 数据排查与源码设计依据，不能与夜间 TRAIN4817/5817 的新结果混合计 N：", ""]
    transfer = snapshot.get("transfer", {})
    transfer_appendix_start = len(lines)
    ema = transfer.get("ema_optimizer", {})
    if ema.get("verified"):
        meta, optimizer = ema["actual_control"], ema["optimizer"]
        lines += [f'实际已完成 control checkpoint 的 global step / EMA n / AdamW step 均为 {meta["global_step"]:,}；EMA nominal cap=.9999，当前解析 decay=.9991219512，尚未 cap。实际保存 LR={number(meta["optimizer_groups"][0]["lr"], 10)}，是 initial LR 的 {number(100 * optimizer["actual_lr_fraction_of_initial"], 4)}%。128epoch 的低 LR 阶段尚未观测。证据：{link(ema["review"]["path"], output=output)}。', ""]
        lines += table(("EMA kernel 状态", "B", "n", "平均迭代年龄（场景曝光）", "最近95%系数质量（场景曝光）"),
                       [[("实际 control n，解析 kernel" if h["EMA_update_n"] == 10240 else "解析 long B256 kernel" if h["EMA_update_n"] == 20480 else "未来128epoch解析 kernel"),
                         h["batch"], h["EMA_update_n"], number(h["exact_varying_decay_kernel_mean_iterate_age_scene_exposures"], 0),
                         number(h["recent_kernel_weight_mass_windows"]["0.95"]["scene_exposures_during_those_updates"], 0)] for h in ema["kernel_horizons"]])
        lines += ["当前 long 等曝光下 adaptive EMA 的场景单位平均迭代年龄几乎相同，不能仅凭 B256 的更新数翻倍断言窗口更长。cap 首次在 n=89,990 生效，B512≈93.35场景epoch、B256≈46.68；cap后同 per-update decay 对应B256约半个场景窗口。表中 kernel 是相关参数迭代的解析系数历史，既不是原始数据权重/独立N，也不是128epoch指标预测；B256 long 项未由这份 CPU review 读取候选 checkpoint。", ""]
    memory = transfer.get("native_memory", {})
    if memory.get("verified"):
        actual, guard = memory["actual_training_memory"], memory["guard"]
        lines += ["cached loader 消除了已测的 Unpickler 类增长源，但生产训练 RSS 并未因此得到完整有界证明。以下是日志中确实存在的 peak RSS 样本；当前训练次数/内存诊断不增加性能 evidence N。", ""]
        lines += table(("实际 step", "peak RSS GiB"),
                       [[actual[key]["step"], number(actual[key]["peak_rss_kib"] / 1024**2, 3)] for key in ("snapshot_first", "snapshot14336", "snapshot_latest")])
        limits = guard["default_limits"]
        lines += [f'128epoch运行可行性尚未验证，不做线性内存外推。已通过 CPU 合同测试的 native launcher 默认监控自有 child RSS {limits["child_RSS_GiB"]} GiB / host MemAvailable {limits["host_MemAvailable_GiB"]} GiB，每{limits["poll_seconds"]}秒采样；超限停止并记录实际原因，但不保证checkpoint已保存或精确续训。RSS不汇总worker子树，短峰可能发生在采样之间。证据：{link(memory["review"]["path"], output=output)}。', ""]
    if len(lines) > transfer_appendix_start:
        runtime_appendix += ["## 附录B：EMA与长训练运行边界", ""] + lines[transfer_appendix_start:]
        del lines[transfer_appendix_start:]
    prior = snapshot.get("prior_review")
    if prior:
        lines += ["旧 TRAIN817/1817 配对中尺寸和 collision 可改善而 speed gate 失效；短期优势可能反转。标量 speed 已直接监督，低速 [0,1) 过多而高速尾部不足；1 m/s bins 无法识别 exact stationary 或最终 clamp 前负值。", "",
                  "4096 native cache 的真实初始化 speed/L/W bins 与 matched GT 的 29,698 vehicle counts 完全一致，未发现这些属性的评价时刻或单位错位；8192 TRAIN cache scan 只校验校准样本，历史原始 track/timestep 来源仍受缓存缺字段限制。", ""]
    lines += ["更大 loss weight、独立 head、输入归一化、time sampling 和 matching 都是待因果验证的干预。MSE 的条件均值在精确学习/积分下可对应正确概率流，不能只因 MSE 就断言分布必然坍缩。生成 type 会改变 vehicle histogram population，需一起看 counts 与 onroad。", "",
              "比较只在同 metric/helper、物理场景 manifest、source、conditioning、实际 exposure 的组内进行。B256 与 B512 的 optimizer/EMA updates 不同，same exposure 不是 pure batch 因果隔离。GT 地图条件实验也不能直接与 SD/VW 的联合地图生成论文数值宣称优劣。", "",
              "## Native 配置与后续执行", ""]
    if native.get("verified"):
        native_entry = "tools/run_init_diffusion_safe.py" if memory.get("verified") else "src/run.py"
        lines += [f'实际 materializer 已通过：{link(native["fit_config"], output=output)}、{link(native["eval_config"], output=output)}；四个来源 checkpoint 的 metadata/EMA/normalizer/frozen-map 核验与 Hydra composition 记录有效。', "",
                  "后续fresh训练与当前建议推断设置的精确命令见报告开头；128epoch尚未执行/完成。历史main20训练recipe比较的native重放配置仍保留，默认同一本次TRAIN4817权重：", "", "```bash",
                  "python tools/run_init_diffusion_safe.py " + chr(92),
                  "  experiment=init_diffusion_lane_conditioned_overnight_best_eval", "```", "",
                  "上述20步重放仍受已记录的1e-6数值复现失败限制；未将其重标为新sampler基线。", "",
                  f'Native full fit shuffle={native.get("native_shuffle")}，quick fixed-order search shuffle={native.get("quick_shuffle")}；full fit 的 epoch-based128epoch cosine schedule 与此处预算结果不能视为最终效果保证。native eval 恢复 map/G1/normalizer/完整 EMA；fit 只加载 map、fresh G1，不 resume evidence checkpoint。', ""]
    else:
        lines += ["尚不发布 native best 训练/评价命令：" + native["reason"] + "。不会提前生成最佳配置或用未确认 recipe 启动 128 epoch。", ""]
    lines += sampler_report_lines(snapshot.get("sampler", {}), output=output)
    if snapshot.get("native_sampler"):
        lines += sampler_report_lines(snapshot["native_sampler"], output=output)
    if snapshot.get("hybrid_sampler"):
        lines += sampler_report_lines(snapshot["hybrid_sampler"], output=output)
    lines += ["重新生成实际分析和本报告：", "", "```bash",
              "python tools/overnight_search_analysis.py --figures",
              "python tools/build_overnight_sampler_figures.py",
              "python tools/build_overnight_search_report.py", "```", "",
              "该 builder 不启动 GPU、不生成 PDF、不写 selection 或 completion marker。PDF 和视觉核验由单独流程完成。", ""]
    lines += runtime_appendix + ["## 证据索引", ""]
    for path in snapshot["supporting_files"]:
        lines += ["- " + link(path, output=output)]
    lines += [""]
    for warning in (*snapshot.get("warnings", []), *snapshot["figures"].get("reasons", [])):
        lines += ["审计提示：" + cell(warning), ""]
    return "\n".join(lines)


def collect(*, artifact_dir=ART, historical_artifact_dir=HISTORICAL_ART, runs=analysis.DEFAULT_RUNS, output=OUTPUT):
    artifact_dir, runs = Path(artifact_dir), Path(runs)
    rows, stages, warnings = analysis.discover(runs)
    groups = analysis.make_groups(rows)
    plan = read(artifact_dir / "search_protocol.json", {})
    frozen = read(artifact_dir / "frozen_long_protocol.json", {})
    decision = read(artifact_dir / "final_selection.json", {})
    final = final_state(decision, frozen, plan.get("arms", []), verified_raw_rows(stages))
    native = native_state(read(artifact_dir / "native_best_materialization.json"), final, decision, frozen, plan.get("arms", []),
                          decision_path=artifact_dir / "final_selection.json", frozen_path=artifact_dir / "frozen_long_protocol.json", search_path=artifact_dir / "search_protocol.json")
    figures = figure_state(read(artifact_dir / "analysis/figures/metadata.json"), rows, groups, artifact_dir / "analysis/figures")
    compact = compact_report_figures(rows, groups, figures, artifact_dir / "analysis/report_figures")
    figures["embedded"] = compact["embedded"]
    if compact.get("metadata"):
        figures["compact_metadata"] = compact["metadata"]
        figures["compact_verified_rows_sha256"] = compact["verified_rows_sha256"]
        figures["compact_verified_groups_sha256"] = compact["verified_groups_sha256"]
    historical = historical_state(historical_artifact_dir)
    runtime = runtime_state(artifact_dir)
    runtime["actual_long_failures"] = runtime_failure_state(artifact_dir, stages)
    runtime["recovery"] = runtime_recovery_state(artifact_dir, stages)
    intentional_restart = intentional_restart_state(artifact_dir)
    corpus_scope = corpus_scope_state(historical_artifact_dir)
    transfer = transfer_state(artifact_dir, stages)
    sampler = sampler_state(artifact_dir, final, native)
    native_sampler = sampler_state(artifact_dir, final, native, fresh_native=True)
    hybrid_sampler = sampler_state(artifact_dir, final, native, hybrid=True)
    figures["sampler"] = sampler_figure_state(artifact_dir, native_sampler, hybrid_sampler)
    input_files = {}
    control = next((a for a in plan.get("arms", []) if a["name"] == "control"), None)
    if control:
        for label, key in (("source map checkpoint", "map"), ("TRAIN normalizer", "normalizer")):
            path = Path(control[key])
            if path.is_file():
                input_files[label] = {"path": str(path), "sha256": sha(path)}
    supporting_names = ("search_protocol.json", "selection_protocol_review.json", "source_freeze.json", "current_protocol_units_scope_cpu_review.json", "current_protocol_units_scope_cpu_review.md", "time_weighting_direction_cpu_audit.json", "current_screen_endpoint_independent_review.json", "current_long_budget_independent_review.json", "screen_control_pre_deadline_revision_interruption.json", "data_metric_review.md", "data_metric_review.json", "design_model.md", "lane_distance_design.md", "map_condition_types_cpu_audit.json", "default_compatibility_cpu_audit.json", "controller_cpu_audit.json", "runtime_failure_handoff_cpu_audit.json", "gc_object_lifecycle_cpu_review.md", "gc_object_lifecycle_cpu_review.json", "gc_disabled_two_scene_cpu_probe.json", "gc_young_two_scene_cpu_probe.json", "runtime_gc_gpu_pilot_review.json", "runtime_gc_pilot_execution.json", "report_cpu_transient_exit139.json", "report_watcher_session.json", "report_watcher.json", "medium_protocol.json", "frozen_long_protocol.json", "final_selection.json", "full50000_audit.json", "queue_complete.json", "native_best_materialization.json", "analysis/summary.json", "analysis/actual_observations.csv", "analysis/figures/metadata.json", "analysis/report_figures/metadata.json")
    supporting_names += ("report_runtime_failures.json", "sampler_report_appendix_plan.md", "runtime_recovery/protocol.json",
                         "runtime_recovery/events.json", "runtime_recovery/attempt1/archive_ledger.json",
                         "runtime_recovery/first_paired_TRAIN4817_cpu_review.json", "runtime_recovery/verification_bridge.json",
                         "cached_torch_loader_100_real_scene_equivalence.json", "gc_off_20k_cached_actual_scene_load_probe.json",
                         "runtime_young_gc_failure_cached_loader_cpu_review.json", "analysis_runtime_cohort_cpu_review.json",
                         "ema_optimizer_128epoch_transfer_cpu_review.json", "ema_optimizer_128epoch_transfer_cpu_review.md",
                         "long_control_10240_optimizer_ema_metadata_cpu.json", "native_guard_long_memory_boundary_cpu_review.json",
                         "native_guard_long_memory_boundary_cpu_review.md", "sampler_runtime_parity_cpu_review.json", "sampler_inference_driver_review.json",
                         "sampler_pre_run_parity_diagnostic_review.json", "sampler_pre_run_parity_diagnostic_review.md",
                         "sampler_selection_protocol.json", "frozen_sampler.json", "native_sampler_selection_protocol.json", "frozen_native_sampler.json",
                         "actual_native_default20_parity_failure_review.json", "sampler_numerical_repeatability_independent_cpu_review.json",
                         "final_actual_tradeoff_independent_cpu.json", "actual_final_population_scope_review.json", "actual_final_histogram_coverage_review.json",
                         "hybrid_sampler_selection_protocol.json", "frozen_hybrid_sampler.json", "native_sampler_protocol_independent_cpu_review.json",
                         "selected_sampler_native_materialization.json", "actual_selected_native_full_scope_review.json",
                         "selected_hybrid_sampler_native_materialization.json", "actual_selected_hybrid_native_full_scope_review.json",
                         "analysis/report_figures/sampler_tradeoffs_actual.metadata.json")
    supporting = [str(artifact_dir / name) for name in supporting_names if (artifact_dir / name).is_file()]
    effective = []
    clipping = []
    visited = set()
    diagnostic_folders = {}
    import yaml
    for row in rows:
        path = Path(row["run_path"]) / row["variant"] / "config.yaml"
        if str(path) in visited:
            continue
        visited.add(str(path))
        cfg = yaml.safe_load(path.read_text())
        model, init = cfg["model"]["model_config"], cfg["model"]["model_config"]["decoder"]["init_diffusion"]
        if row["phase"] != "full50000":
            diagnostic_folders[str(path.parent)] = (row["stage"] + " / " + row["variant"],
                                                   max(r["budget"] for r in rows if r["run_path"] == row["run_path"]
                                                       and r["variant"] == row["variant"]),
                                                   cfg["trainer"].get("gradient_clip_val"))
        effective.append([row["variant"] + " / " + link(path, sha(path)[:8], output=output), row["recipe_sha256"][:8],
                          f'{model.get("optimizer")}/{model.get("lr")}/{cfg["trainer"].get("gradient_clip_val")}',
                          f'{init.get("denoiser_hidden_dim") or 256}/{init.get("denoiser_num_layers") if init.get("denoiser_num_layers") is not None else 3}',
                          f'{init.get("state_feature_normalization", "none")}/{init.get("speed_head_kind", "shared")}/{init.get("speed_x0_parameterization", "identity")}',
                          f'{init.get("map_condition_types", "road_edges")}/{init.get("time_sample_power", 1.)}'])
    for folder, (label, budget, clip) in diagnostic_folders.items():
        try:
            clipping.append(clipping_evidence(folder, label=label, scope="本轮 verified stage",
                                              expected_steps=budget, expected_clip=clip))
        except (OSError, ValueError, TypeError, KeyError) as error:
            warnings.append("Clipping diagnostic excluded (does not alter endpoint selection): " + str(error))
    prior = read(artifact_dir / "data_metric_review.json")
    for previous in (prior or {}).get("completed_input_files", []):
        try:
            verify_record(previous["summary"])
            summary_path = Path(previous["summary"]["path"])
            prior_rows = read(summary_path, [])
            arm_rows = [r for r in prior_rows if r["variant"] == previous["arm"] and r["seed"] == previous["TRAIN_seed"]]
            if not arm_rows:
                raise ValueError("Historical clipping summary has no audited actual endpoint")
            folder = summary_path.parent / previous["arm"]
            clip = yaml.safe_load((folder / "config.yaml").read_text())["trainer"]["gradient_clip_val"]
            clipping.append(clipping_evidence(folder, label=f'TRAIN{previous["TRAIN_seed"]} / {previous["arm"]}',
                                              scope="旧实验历史证据", expected_steps=max(r["budget"] for r in arm_rows),
                                              expected_clip=clip))
        except (OSError, ValueError, TypeError, KeyError) as error:
            warnings.append("Historical clipping diagnostic excluded: " + str(error))
    stored_rows = read(artifact_dir / "analysis/actual_observations.json")
    if stored_rows is not None and stored_rows != rows:
        warnings.append("Stored analysis inventory differs from freshly verified stages; tables use fresh verification. Rerun overnight_search_analysis before final publication.")
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "rows": rows, "stages": stages, "groups": groups,
            "plan": plan, "frozen": frozen, "decision": decision, "final": final, "native": native, "figures": figures,
            "queue_complete": read(artifact_dir / "queue_complete.json"), "queue_active": read(artifact_dir / "queue_active.json"),
            "timing": analysis.timing(rows, stages), "warnings": warnings, "supporting_files": supporting,
            "prior_review": prior, "input_files": input_files, "effective_configs": effective,
            "clipping_evidence": clipping, "historical": historical, "runtime": runtime,
            "intentional_restart": intentional_restart, "corpus_scope": corpus_scope,
            "transfer": transfer, "sampler": sampler, "native_sampler": native_sampler, "hybrid_sampler": hybrid_sampler}


class ReportUnitTests(unittest.TestCase):
    """In-memory fixtures only; never create a real final_selection artifact."""
    @staticmethod
    def sampler_fixture():
        cells = [{"TRAIN_seed": s, "GEN_seed": g, "option": option} for option in ("n20_eps005", "n10_eps005")
                 for s in (4817, 5817) for g in (2817, 3817)]
        rows = [{**c, **{key: 1. for key in METRICS}, "generated_onroad_fraction": .85,
                 "num_generated_vehicles": 100, "num_gt_vehicles": 100} for c in cells]
        return rows, cells

    def test_sampler_missing_stays_missing_without_main_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            state = sampler_state(Path(folder), {"verified": False}, {"verified": False}, directory=folder)
            self.assertEqual(state["stages"]["grid"]["verified_rows"], [])
            self.assertFalse(state["selection_made_by_report"])
            self.assertIn("不报告 sampler 改善", "\n".join(sampler_report_lines(state, output=OUTPUT)))

    def test_inference_figure_rejects_incomplete_or_different_frozen_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(sampler_figure_state(folder, {}, {})["verified"])
            path = Path(folder) / "analysis/report_figures/sampler_tradeoffs_actual.metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"native_decision_sha256": "native", "hybrid_decision_sha256": "hybrid",
                                        "native_grid_rows": 16, "hybrid_grid_rows": 8}))
            native = {"actual_frozen_selection": {"verified": True, "sha256": "native"},
                      "stages": {"grid": {"status": "verified_complete", "verified_rows": [{}] * 16}}}
            hybrid = {"actual_frozen_selection": {"verified": True, "sha256": "hybrid"},
                      "stages": {"grid": {"status": "verified_complete", "verified_rows": [{}] * 7}}}
            value = sampler_figure_state(folder, native, hybrid)
            self.assertFalse(value["verified"])
            self.assertIn("cohort differs", value["reason"])

    def test_hybrid_standalone_config_is_not_published_while_full_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            state = {"stages": {"full": {"status": "verified_partial"}},
                     "actual_frozen_selection": {"verified": True}}
            path = Path(folder) / "selected_hybrid_sampler_native_materialization.json"
            path.write_text(json.dumps({"status": "ACTUAL_SELECTED_HYBRID_NATIVE_EVAL_YAML_MATERIALIZED_CPU_COMPOSE_PASS"}))
            self.assertFalse(sampler_configuration_state(folder, state, hybrid=True)["verified"])

    def test_standalone_sampler_yaml_requires_complete_actual_full(self):
        with tempfile.TemporaryDirectory() as folder:
            state = {"stages": {"full": {"status": "verified_partial"}},
                     "actual_frozen_selection": {"verified": True}}
            path = Path(folder) / "selected_sampler_native_materialization.json"
            path.write_text(json.dumps({"status": "ACTUAL_SELECTED40_NATIVE_EVAL_YAML_MATERIALIZED_CPU_COMPOSE_PASS"}))
            result = sampler_configuration_state(folder, state)
            self.assertFalse(result["verified"])
            self.assertIn("incomplete full", result["reason"])

    def test_native_full_gen_mean_does_not_create_paired_r_or_train_sd(self):
        rows = [{"TRAIN_seed": 4817, "GEN_seed": gen, "option": "n40_eps005",
                 **{key: value for key in METRICS}, "generated_onroad_fraction": .85,
                 "num_generated_vehicles": 100, "num_gt_vehicles": 100}
                for gen, value in ((2817, 1.), (3817, 3.))]
        state = {"fresh_native_baselines": True, "stages": {"full": {"status": "verified_complete",
                 "verified_rows": rows, "planned_cells": 2, "complete": True,
                 "summary": {"path": "/tmp/full.json"}}},
                 "selection_protocol": {"verified": False, "reason": "CPU fixture"}}
        text = "\n".join(sampler_report_lines(state, output=OUTPUT))
        self.assertIn("| 4817 | 2 | 0.200000", text)
        self.assertIn("不能报告native full配对R", text)
        self.assertIn("不提供TRAIN间SD", text)
        self.assertNotIn("pooled R", text.split("### sampler full", 1)[1])

    def test_hybrid_cohort_has_fresh40_anchor_not_n20(self):
        cells = [{"TRAIN_seed": seed, "GEN_seed": gen, "option": option} for option in ("n40_eps005", "hybrid_pos0001") for seed in (4817, 5817) for gen in (2817, 3817)]
        rows = [{**cell, **{key: 1. for key in METRICS}, "generated_onroad_fraction": .85,
                 "num_generated_vehicles": 100, "num_gt_vehicles": 100} for cell in cells]
        stats = sampler_statistics(rows, cells, complete=True, baseline_option="n40_eps005")
        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0]["independent_TRAIN_N"], 2)
        self.assertEqual(stats[0]["pooled"]["R"], 1.)
        state = {"fresh_native_baselines": True, "hybrid_position_cohort": True, "stages": {},
                 "selection_protocol": {"verified": False, "reason": "CPU fixture"}}
        text = "\n".join(sampler_report_lines(state, output=OUTPUT))
        self.assertIn("单独8cell cohort", text)
        self.assertIn("fresh N40", text)
        self.assertNotIn("四options", text)

    def test_runtime_full_evaluation_is_not_fresh_fit_or_added_training(self):
        row = {"evaluation_only": True, "train_seconds": 0., "optimizer_continued": False}
        self.assertEqual(recovery_execution_role({"fresh_training": False}, [row]), "evaluation_only")
        self.assertEqual(recovery_execution_role({"fresh_training": True}, []), "fresh_training")
        with self.assertRaises(ValueError): recovery_execution_role({"fresh_training": True}, [row])
        with self.assertRaises(ValueError): recovery_execution_role({"fresh_training": False}, [{**row, "train_seconds": 1.}])
        with self.assertRaises(ValueError): recovery_execution_role({"fresh_training": False}, [])

    def test_new_native_sampler_scope_never_claims_old_parity(self):
        state = {"fresh_native_baselines": True, "selection_protocol": {"verified": True,
                 "created_utc": "CPU fixture", "path": "/tmp/protocol.json"}, "stages": {},
                 "actual_frozen_selection": {"verified": False, "reason": "incomplete"}}
        text = "\n".join(sampler_report_lines(state, output=OUTPUT))
        self.assertIn("16cells", text)
        self.assertIn("R≤.98", text)
        self.assertIn("不宣称与主endpoint通过1e-6", text)
        self.assertNotIn("完整20cells且四个default20全部通过", text)
        self.assertIn("不是置信区间", text)

    def test_partial_sampler_facts_cannot_rank_or_select(self):
        rows, cells = self.sampler_fixture()
        self.assertEqual(sampler_statistics(rows[:5], cells, complete=False), [])
        with self.assertRaises(ValueError):
            sampler_statistics(rows[:5], cells, complete=True)

    def test_sampler_gen_averaging_keeps_train_n_and_per_seed_gate(self):
        rows, cells = self.sampler_fixture()
        for row in rows:
            if row["option"] == "n10_eps005":
                row["speed_jsd"] = 1.4 if row["TRAIN_seed"] == 5817 else .8
        stats = sampler_statistics(rows, cells, complete=True)
        arm = next(r for r in stats if r["option"] == "n10_eps005")
        self.assertEqual(arm["independent_TRAIN_N"], 2)
        self.assertEqual(arm["new_training_runs"], 0)
        self.assertTrue(arm["pooled"]["eligible"])
        self.assertFalse(arm["pooled"]["each_training_seed_gates_pass"])
        self.assertFalse(arm["sampler_selection_made_by_report"])
        self.assertAlmostEqual(arm["metrics_raw"]["speed_jsd"]["mean"], .011)

    def test_sampler_duplicate_cells_never_become_complete(self):
        rows, cells = self.sampler_fixture()
        rows[0] = rows[1]
        with self.assertRaises(ValueError):
            sampler_statistics(rows, cells, complete=True)

    def test_transfer_without_real_artifacts_does_not_prove_128(self):
        with tempfile.TemporaryDirectory() as folder:
            state = transfer_state(folder, [])
            self.assertFalse(state["ema_optimizer"]["verified"])
            self.assertFalse(state["native_memory"]["verified"])

    def test_raw_metric_units_keep_collision_percentage(self):
        self.assertEqual(MULTIPLIERS["nearest_dist_jsd"], 10.)
        self.assertEqual(MULTIPLIERS["speed_jsd"], 100.)
        self.assertEqual(MULTIPLIERS["collision_rate"], 1.)

    def test_embedded_image_target_matches_pdf_parser(self):
        value = image_link(ROOT / "docs/experiments/fixture/actual.png", "actual verified image", output=OUTPUT)
        match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", value)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(2), "experiments/fixture/actual.png")
        self.assertNotIn("<", match.group(2))

    def test_compact_endpoint_is_actual_maximum_and_long_n1_has_no_sd(self):
        def point(exposure):
            return {"arm_id": "control-sha", "variant": "control", "actual_exposure": exposure,
                    "actual_full_data_epochs": exposure / controller.TRAIN_COUNT,
                    "independent_training_N": 1, "training_seed_ids": [4817], "batch_update_pairs": [(512, exposure // 512)],
                    "metrics_raw_jsd_collision_percent": {k: {"mean": .1, "sd": None} for k in METRICS},
                    "comparison": {"R": 1., "ratios": {k: 1. for k in METRICS}, "ranking_eligible": True}}
        groups = [{"phase": "screen", "cohort_sha256": "fixture", "control_present": True,
                   "actual_exposure": e, "actual_full_data_epochs": e / controller.TRAIN_COUNT,
                   "cohort": {"num_samples": 1024}, "arms": [point(e)]} for e in (512, 1024)]
        long = {"phase": "long", "cohort_sha256": "fixture-long", "arms": [point(2048)]}
        groups.append(long)
        payload = compact_plot_payload(groups)
        self.assertEqual(payload["heatmaps"][0]["actual_exposure"], 1024)
        curve = payload["long"][0]["curves"][0]
        self.assertEqual(curve["actual_exposures"], [2048])
        self.assertEqual(curve["TRAIN_membership"], [[4817]])
        self.assertEqual(curve["sample_sd_across_TRAIN"]["speed_jsd"], [None])
        self.assertFalse(payload["long"][0]["missing_points_padded"])
        long["arms"][0]["metrics_raw_jsd_collision_percent"]["speed_jsd"]["sd"] = 0.
        with self.assertRaisesRegex(ValueError, "invent SD"):
            compact_plot_payload(groups)

    def test_unmatched_train_membership_is_descriptive_not_paired(self):
        control1 = analysis.CohortConsumerTests.row("control", "gc128")
        control2 = analysis.CohortConsumerTests.row("control", "gc0-cached")
        control2["seed"] = 5817
        candidate = analysis.CohortConsumerTests.row("candidate", "gc0-cached")
        group, = analysis.make_groups([control1, control2, candidate])
        arm = next(a for a in group["arms"] if a["variant"] == "candidate")
        self.assertEqual(arm["independent_training_N"], 1)
        self.assertFalse(arm["comparison"]["ranking_eligible"])
        self.assertNotIn("R", arm["comparison"])
        self.assertNotIn("ratios", arm["comparison"])
        text = "\n".join(result_group_lines(group, output=OUTPUT))
        self.assertIn("TRAIN seed 集合未配齐", text)
        self.assertIn("control N=2 与 candidate N=1", text)
        self.assertIn("No complete matched TRAIN seed set", text)
        line = next(line for line in text.splitlines() if line.startswith("| candidate | B"))
        self.assertIn("| — | — |", line)
        curves = compact_plot_payload([group])["long"][0]["curves"]
        memberships = {curve["variant"]: curve["TRAIN_membership"] for curve in curves}
        self.assertEqual(memberships["control"], [[4817, 5817]])
        self.assertEqual(memberships["candidate"], [[4817]])

    def test_compact_full_requires_actual_approved_complete_scope(self):
        self.assertIsNone(compact_full_payload([], {"files": []}))
        self.assertIsNone(compact_full_payload([{"phase": "full50000"}], {"files": [{"phase": "full50000"}]}))

    def test_compact_full_rechecks_bins_and_averages_gen_probabilities(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_full_", dir="/tmp") as temporary:
            rows = []
            for arm in ("control", "fixture"):
                for gen, generated in ((2817, [9, 1]), (3817, [1, 1])):
                    folder = Path(temporary) / (arm + str(gen))
                    folder.mkdir()
                    histograms = {k: {"minimum": 0, "maximum": 2, "bin_width": 1,
                                       "matched_gt": [1, 1], "generated": generated}
                                  for k in ("speed_jsd", "length_jsd", "width_jsd")}
                    (folder / "feature_histograms.json").write_text(json.dumps(histograms))
                    row = {k: .1 for k in METRICS}
                    row.update(phase="full50000", variant=arm, cohort_sha256="fixture", seed=4817,
                               eval_seed=gen, report_source=str(folder / "sd_agent_metrics.json"), budget=1,
                               train_batch_size=512, planned_generation_seeds=[2817, 3817],
                               generated_onroad_fraction=.8, gt_onroad_fraction=.9,
                               num_gt_vehicles=2, num_gt_onroad_vehicles=1, num_generated_vehicles=sum(generated))
                    rows.append(row)
            payload = compact_full_payload(rows, {"files": [{"phase": "full50000"}]})
            probability = payload["features"][0]["series"][0]["mean_normalized_probability"]
            self.assertAlmostEqual(probability[0], .7)
            self.assertNotAlmostEqual(probability[0], 10 / 12)
            self.assertEqual(len(payload["histogram_sources"]), 4)
            self.assertFalse(payload["adds_independent_TRAIN"])
            changed = Path(payload["histogram_sources"][-1]["path"])
            histogram = read(changed)
            histogram["speed_jsd"]["matched_gt"] = [2, 1]
            changed.write_text(json.dumps(histogram))
            with self.assertRaisesRegex(ValueError, "exact GT/bin"):
                compact_full_payload(rows, {"files": [{"phase": "full50000"}]})

    def test_partial_render_embeds_only_verified_compact_figures(self):
        snapshot = {"generated_utc": "fixture", "rows": [], "stages": [], "groups": [], "plan": {},
                    "final": {"verified": False, "reason": "fixture incomplete"}, "native": {"verified": False, "reason": "fixture missing"},
                    "figures": {"files": [{"phase": "long", "path": "/tmp/stale.png"}], "reasons": [],
                                "embedded": [{"phase": "screen", "path": ROOT / "docs/experiments/fixture/verified.png",
                                              "caption": "actual fixture", "exports": []}]}, "supporting_files": []}
        value = render(snapshot)
        self.assertIn("![actual fixture](experiments/fixture/verified.png)", value)
        self.assertNotIn("stale.png", value)
        self.assertIn("尚无可认可的最终最佳配置", value)

    def test_n1_never_prints_zero_sd(self):
        self.assertEqual(stat({"n": 1, "mean": .3, "sd": 0.}), "0.300000")
        self.assertEqual(stat({"n": 2, "mean": .3, "sd": .1}), "0.300000 ± 0.100000")

    def test_missing_and_partial_final_do_not_become_native_best(self):
        self.assertFalse(final_state(None, {}, [], [])["verified"])
        self.assertFalse(final_state({"complete_matrix": False, "rows": []}, {}, [], [])["verified"])
        self.assertFalse(native_state({"Hydra_composition_verified": True}, {"verified": False}, {}, {}, [], decision_path="/tmp/no", frozen_path="/tmp/no", search_path="/tmp/no")["verified"])

    def test_loss_and_model_interventions_are_not_hidden(self):
        baseline = {"recipe": {"components": {"velocity": {"kind": "mse", "time_power": 1}}}, "config": {controller.PREFIX + "speed_head_kind": "shared"}}
        candidate = {"recipe": {"components": {"velocity": {"kind": "l1", "time_power": 0}}}, "config": {controller.PREFIX + "speed_head_kind": "separate"}}
        self.assertIn("velocity.kind=l1", loss_delta(candidate, baseline))
        self.assertIn("velocity.time_power=0", loss_delta(candidate, baseline))
        self.assertIn('speed_head_kind="separate"', config_delta(candidate, baseline))

    def test_stale_missing_scope_figures_are_not_linked(self):
        metadata = {"files": [], "heatmaps": [{"phase": "screen", "cohort_sha256": "fixture", "actual_exposure": 1, "arms": ["fixture"], "page": 1}], "long_curves": [], "full50000": {"status": "missing"}}
        self.assertFalse(figure_state(metadata, [], [], Path("/tmp"))["files"])

    def test_clipping_uses_actual_threshold_and_rejects_stale_summary(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_clipping_", dir="/tmp") as temporary:
            folder = Path(temporary)
            trace = [{"step": i, "grad_norm_before_clip": value} for i, value in enumerate((.6, .8, 1.2))]
            summary = {"steps": 3, "gradient_clip_val": 1., "clip_fraction": 1. / 3.,
                       "first_window": {"grad_norm_before_clip": sum((.6, .8, 1.2)) / 3.},
                       "last_window": {"grad_norm_before_clip": sum((.6, .8, 1.2)) / 3.}}
            (folder / "train_trace.json").write_text(json.dumps(trace))
            (folder / "train_summary.json").write_text(json.dumps(summary))
            evidence = clipping_evidence(folder, label="fixture", scope="CPU fixture", expected_steps=3, expected_clip=1.)
            self.assertEqual(evidence["clip_fraction"], 1. / 3.)
            summary["clip_fraction"] = 1.
            (folder / "train_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "clip_fraction"):
                clipping_evidence(folder, label="fixture", scope="CPU fixture", expected_steps=3)
            summary["clip_fraction"] = 1. / 3.
            (folder / "train_summary.json").write_text(json.dumps(summary))
            trace[-1]["step"] = 1
            (folder / "train_trace.json").write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "repeated steps"):
                clipping_evidence(folder, label="fixture", scope="CPU fixture", expected_steps=3)

    def test_long_curve_rejects_invented_sd_at_n1(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_figures_", dir="/tmp") as temporary:
            plot = Path(temporary) / "fixture.png"
            plot.write_bytes(b"CPU metadata-only fixture; not rendered")
            point = {"arm_id": "fixture", "actual_exposure": 512, "actual_full_data_epochs": 512 / 493542,
                     "independent_training_N": 1, "training_seed_ids": [4817], "batch_update_pairs": [(512, 1)],
                     "metrics_raw_jsd_collision_percent": {"speed_jsd": {"mean": .1, "sd": None}}}
            curve = {"arm_id": "fixture", "actual_exposures": [512], "x_actual_epochs": [512 / 493542],
                     "mean": [.1], "sample_sd_across_TRAIN": [None], "independent_TRAIN_N": [1],
                     "TRAIN_membership_by_exposure": [[4817]], "batch_update_pairs": [[[512, 1]]]}
            metadata = {"files": [str(plot)], "long_curves": [{"cohort_sha256": "fixture", "metric": "speed_jsd",
                                                                     "curves": [curve], "files": [str(plot)]}]}
            groups = [{"phase": "long", "cohort_sha256": "fixture", "arms": [point]}]
            self.assertEqual(len(figure_state(metadata, [], groups, Path(temporary))["files"]), 1)
            curve["sample_sd_across_TRAIN"] = [0.]
            self.assertFalse(figure_state(metadata, [], groups, Path(temporary))["files"])

    def test_gen_repeats_do_not_increase_training_n_or_hide_seed_gate(self):
        from tools import materialize_overnight_init_diffusion_best as materializer
        controls, candidates = [], []
        for seed, speed in ((4817, .8), (5817, 1.2)):
            for gen in (2817, 3817):
                base = {k: 1. for k in METRICS}
                base.update(seed=seed, eval_seed=gen, generated_onroad_fraction=.85,
                            num_generated_vehicles=100, num_gt_vehicles=100,
                            planned_generation_seeds=[2817, 3817], budget=1, train_batch_size=512)
                controls.append(base)
                candidate = dict(base, **{k: .9 for k in METRICS})
                candidate["speed_jsd"] = speed
                candidates.append(candidate)
        self.assertEqual(len(analysis.seed_means(candidates)), 2)
        result = materializer.assessment(candidates, controls)
        self.assertLess(result["R"], 1.)
        self.assertTrue(result["aggregate_gates_pass"])
        self.assertFalse(result["per_training_seed"]["5817"]["eligible"])
        self.assertFalse(result["eligible"])

    def test_final_intro_reports_recipe_and_each_training_tradeoff_before_details(self):
        from tools import materialize_overnight_init_diffusion_best as materializer
        controls, candidates = [], []
        for seed, speed in ((4817, .8), (5817, 1.2)):
            for gen in (2817, 3817):
                row = {k: 1. for k in METRICS}
                row.update(seed=seed, eval_seed=gen, generated_onroad_fraction=.85,
                           num_generated_vehicles=100, num_gt_vehicles=100,
                           planned_generation_seeds=[2817, 3817], budget=1, train_batch_size=512)
                controls.append(row)
                candidate = dict(row, **{k: .9 for k in METRICS})
                candidate["speed_jsd"] = speed
                candidates.append(candidate)
        results = {"control": materializer.assessment(controls, controls),
                   "fixture": materializer.assessment(candidates, controls)}
        plan = {"arms": [{"name": name, "batch": batch, "lr": .0005,
                           "recipe": {"components": {"velocity": {"kind": "mse", "weight": 2, "time_power": 1}}},
                           "config": {}} for name, batch in (("control", 512), ("fixture", 256))]}
        snapshot = {"generated_utc": "fixture", "rows": [], "stages": [], "groups": [], "plan": plan,
                    "frozen": {"challenger": "fixture"}, "final": {"verified": True, "selected": "control",
                          "actual_exposure": 5242620, "results": results},
                    "native": {"verified": False, "reason": "fixture missing"},
                    "figures": {"files": [], "reasons": []}, "supporting_files": []}
        text = render(snapshot)
        self.assertLess(text.index("选中 recipe"), text.index("实际完成与缺失状态"))
        self.assertIn("independent confirmation", text)
        self.assertIn("20.00", text)
        self.assertIn("PASS / FAIL", text)
        self.assertIn("10.6224", text)
        self.assertIn("| final metric | control raw | challenger raw | 变化% |", text)
        self.assertIn("| Near | 0.100000 | 0.090000 | -10.00 |", text)
        self.assertIn("| Length | 0.010000 | 0.009000 | -10.00 |", text)
        self.assertIn("| Col% | 1.000000 | 0.900000 | -10.00 |", text)
        self.assertLess(text.index("| final metric |"), text.index("实际完成与缺失状态"))
        self.assertNotIn("experiment=init_diffusion_lane_conditioned_overnight_best", text)

    def test_cpu_lifecycle_review_rejects_disagreeing_raw_probe(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_lifecycle_", dir="/tmp") as temporary:
            folder = Path(temporary)
            review = {"probe": {}}
            for mode, filename, classes in (("disabled", "gc_disabled_two_scene_cpu_probe.json", 400),
                                            ("young_after_batch", "gc_young_two_scene_cpu_probe.json", 0)):
                probe = {"mode": mode, "cuda_initialized": False, "tensorflow_imported": False,
                         "loops": 200, "scenes_per_loop": 2, "before_final_full_gc": {
                             "unpickler_classes": classes, "confirmed_super_class_cycles": classes}}
                review["probe"][mode] = {"actual_scene_loads": 400, "unreachable_local_unpickler_classes_final": classes,
                                         "confirmed_class_closure_cycles_final": classes}
                (folder / filename).write_text(json.dumps(probe))
            (folder / "gc_object_lifecycle_cpu_review.json").write_text(json.dumps(review))
            self.assertTrue(runtime_state(folder)["CPU"]["verified"])
            path = folder / "gc_young_two_scene_cpu_probe.json"
            probe = read(path)
            probe["before_final_full_gc"]["unpickler_classes"] = 1
            path.write_text(json.dumps(probe))
            self.assertFalse(runtime_state(folder)["CPU"]["verified"])

    def test_missing_historical_matrix_is_not_called_complete(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_history_", dir="/tmp") as temporary:
            self.assertFalse(historical_state(Path(temporary))["verified"])

    def test_intentional_143_restart_remains_diagnostic(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_restart_", dir="/tmp") as temporary:
            folder = Path(temporary)
            archive = folder / "diagnostic"
            archive.mkdir()
            archive.with_suffix(".log").write_text("STEP 256: fixture diagnostic\n")
            (archive / "execution.json").write_text(json.dumps({"returncode": 143, "excluded_from_model_selection": True}))
            (folder / "source_freeze.json").write_text(json.dumps({"controller_forecast_revision_archive": str(archive)}))
            (folder / "screen_control_pre_deadline_revision_interruption.json").write_text(json.dumps({
                "status": "INTENTIONAL_RUNTIME_CONTROLLER_REVISION_NOT_MODEL_RESULT", "excluded_from_search_results": True,
                "reason": "CPU fixture; no training"}))
            result = intentional_restart_state(folder)
            self.assertTrue(result["verified"])
            self.assertEqual(result["returncode"], 143)
            self.assertEqual(result["last_logged_update"], 256)
            self.assertTrue(result["excluded_from_model_failure_or_performance"])

    def test_actual_gc_failure_is_retained_after_run_archive_and_never_a_result(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_actual_failure_", dir="/tmp") as temporary:
            directory = Path(temporary)
            run = directory / "run"
            run.mkdir()
            (run / "execution.json").write_text(json.dumps({"status": "failed", "returncode": -11,
                                                           "started_unix": 1., "ended_unix": 11.}))
            log = run.with_suffix(".log")
            log.write_bytes(b"STEP 5760: fixture\r\nFatal Python error: Segmentation fault\nGarbage-collecting\n")
            artifact = directory / "artifacts"
            result = runtime_failure_state(artifact, [{"label": "fixture", "run": str(run), "status": "failed"}])
            self.assertEqual(len(result["verified_failures"]), 1)
            record = result["verified_failures"][0]
            self.assertEqual(record["last_logged_update"], 5760)
            self.assertEqual(record["elapsed_process_seconds"], 10.)
            self.assertEqual(record["original_log_sha256"], sha(record["log_copy"]["path"]))
            self.assertFalse(record["included_in_performance_or_TRAIN_N"])
            self.assertFalse(record["root_cause_proven"])
            (run / "execution.json").unlink()
            log.unlink()
            self.assertEqual(len(runtime_failure_state(artifact, [])["verified_failures"]), 1)

    def test_runtime_recovery_requires_an_actual_protocol_not_a_dry_run(self):
        with tempfile.TemporaryDirectory(prefix="sim_report_actual_recovery_", dir="/tmp") as temporary:
            folder = Path(temporary)
            self.assertFalse(runtime_recovery_state(folder, [])["verified"])
            (folder / "runtime_recovery").mkdir()
            (folder / "runtime_recovery/protocol.json").write_text(json.dumps({"status": "CPU_DRY_RUN_PLAN",
                "unchanged_model_objective_data_seeds_and_budget": True, "checkpoint_resume": False}))
            self.assertFalse(runtime_recovery_state(folder, [])["verified"])

    def test_partial_render_has_no_executable_native_best_command(self):
        snapshot = {"generated_utc": "fixture", "rows": [], "stages": [], "groups": [], "plan": {},
                    "final": {"verified": False, "reason": "fixture incomplete"}, "native": {"verified": False, "reason": "fixture missing"},
                    "figures": {"files": [], "reasons": []}, "supporting_files": []}
        text = render(snapshot)
        self.assertIn("尚无可认可的最终最佳配置", text)
        self.assertNotIn("experiment=init_diffusion_lane_conditioned_overnight_best", text)
        self.assertNotIn("± 0", text)

    def test_table_cells_escape_parallel_recipe_values(self):
        self.assertEqual(cell("a|b\nc"), "a\\|b c")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ART)
    parser.add_argument("--historical-artifact-dir", type=Path, default=HISTORICAL_ART)
    parser.add_argument("--runs", type=Path, default=analysis.DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReportUnitTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    snapshot = collect(artifact_dir=args.artifact_dir, historical_artifact_dir=args.historical_artifact_dir, runs=args.runs, output=args.output)
    atomic_text(args.output, render(snapshot, output=args.output))
    audit = {"generated_utc": snapshot["generated_utc"], "status": "ACTUAL_FINAL_MATRIX_VERIFIED" if snapshot["final"].get("verified") else "ACTUAL_PARTIAL_REPORT",
             "actual_regenerated": True,
             "report": str(args.output.resolve()), "report_sha256": sha(args.output),
             "verified_stage_count": sum(s.get("included", False) for s in snapshot["stages"]),
             "actual_observations": len(snapshot["rows"]), "verified_rows_sha256": digest(snapshot["rows"]),
             "stages": snapshot["stages"], "final": snapshot["final"], "native": snapshot["native"],
             "figures": snapshot["figures"], "warnings": snapshot["warnings"], "input_files": snapshot["input_files"],
             "clipping_evidence": snapshot["clipping_evidence"],
             "historical": snapshot["historical"], "runtime": snapshot["runtime"],
             "intentional_restart": snapshot["intentional_restart"],
             "corpus_scope": snapshot["corpus_scope"],
             "transfer": snapshot["transfer"], "sampler": snapshot["sampler"], "native_sampler": snapshot["native_sampler"], "hybrid_sampler": snapshot["hybrid_sampler"],
             "GPU_started": False, "torch_imported": "torch" in sys.modules, "PDF_generated": False, "completion_marker_created": False,
             "full128epoch_completed_or_optimum_proven": False, "source_tool_sha256": sha(Path(__file__))}
    atomic_text(args.artifact_dir / "report_build_audit.json", json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: audit[k] for k in ("status", "actual_regenerated", "report", "report_sha256", "verified_stage_count", "actual_observations", "torch_imported")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    # All output is synchronously closed above. Avoid Python shutdown finalizers
    # after the report-only process has finished; prior intermittent exit -11 is
    # kept as a diagnostic, not attributed to an established crash root cause.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
