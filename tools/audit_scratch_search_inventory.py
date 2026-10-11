"""Inventory actual scratch-loss experiments and deduplicate their costs.

CPU only: read JSON/YAML/text and mmap checkpoint metadata on CPU. No model
instantiation, GPU operations, dataset tensors, training or evaluation. The
main search, LR pilots, failed attempts and native smoke are separate scopes.
Pending roots/stages never become completed models or fabricated metrics.

Run again after queued stages finish:
  python tools/audit_scratch_search_inventory.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs/component_loss_scratch"
ART = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011"
MAIN_ROOTS = {
    "core_search": ["search1024_w0", "adaptive1024_w0", "extra1024_w0", "extra1024_recovery",
                    "long4096_w0", "targeted4096_w0", "extend6144_w0"],
    "production": ["batch512_seed817", "batch512_alternative_seed817"],
    "heldout": ["heldout_seed817", "heldout_bridge_seed817", "batch512_seed1817", "batch512_seed2817"],
    "full_set": ["full50000_seed817"],
}
AUX_ROOTS = {"lr_pilots": ["lr256", "lr1024"], "failed_attempts": ["search1024"],
             "implementation_pilots": ["pilot16", "pilot_resume32"]}
METRICS = ("nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd", "width_jsd", "speed_jsd", "collision_rate")
READ_WARNINGS = []


def read_json(path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as error:
        READ_WARNINGS.append(f"Concurrent/incomplete JSON snapshot, rerun audit: {path}: {error}")
        return default


def finite(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def identity(path, *, checksum=True):
    if not path.is_file():
        return {"path": str(path), "exists": False}
    before = path.stat()
    result = {"path": str(path.resolve()), "exists": True, "bytes": before.st_size, "mtime_ns": before.st_mtime_ns}
    if checksum:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["changed_during_read"] = path.stat().st_mtime_ns != before.st_mtime_ns
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def nested(value, *keys):
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def exposure_for_updates(updates, batch_size, loader_examples):
    """Actual exposures when a fixed loader repeats, including partial batches."""
    if updates < 0 or batch_size <= 0 or loader_examples <= 0:
        raise ValueError("Invalid update count, batch size or actual loader length")
    batches = (loader_examples + batch_size - 1) // batch_size
    epochs, remaining = divmod(updates, batches)
    return epochs * loader_examples + min(remaining * batch_size, loader_examples)


def text_line_count(path):
    total, last = 0, b""
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            total += chunk.count(b"\n")
            last = chunk[-1:]
    return total + (last != b"\n" and bool(last))


def trace_audit(path):
    records = read_json(path, [])
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Invalid train trace: {path}")
    steps = [row.get("step") for row in records]
    valid = bool(steps) and all(type(step) is int for step in steps)
    contiguous = valid and steps == list(range(steps[0], steps[0] + len(steps)))
    return {"records": len(records), "first_preupdate_global_step": steps[0] if valid else None,
            "last_preupdate_global_step": steps[-1] if valid else None,
            "attained_global_step": steps[-1] + 1 if contiguous else None,
            "exact_stage_optimizer_updates": len(records) if contiguous else None,
            "contiguous_unique_steps": contiguous,
            "keys": sorted({key for row in records for key in row}),
            "identity": identity(path)}


def failed_csv_evidence(folder):
    """Saved train CSV is diagnostic evidence, not a successful final endpoint."""
    evidence = []
    for path in sorted((folder / "csv").rglob("metrics.csv")):
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        steps = sorted({int(float(row["step"])) for row in rows if finite(row.get("step")) is not None
                        and any(value not in (None, "") for key, value in row.items() if isinstance(key, str) and key.startswith("train/"))})
        if not steps:
            continue
        evidence.append({"identity": identity(path), "confirmed_logged_update_count": len(steps),
            "preupdate_step_minimum": steps[0], "preupdate_step_maximum": steps[-1],
            "contiguous_unique_steps": steps == list(range(steps[0], steps[-1] + 1)),
            "scope": "Failed-stage diagnostic only. Saved CSV confirms logged completed updates; no successful model/metric or proven final crash endpoint."})
    return evidence


def checkpoint_audit(path):
    # mmap maps tensor storage lazily; only scalar training metadata is read.
    import torch
    result = {"identity": identity(path, checksum=False)}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        optimizer_steps = set()
        for optimizer in checkpoint.get("optimizer_states", []):
            for value in optimizer.get("state", {}).values():
                step = value.get("step")
                if step is not None:
                    optimizer_steps.add(float(step.item() if hasattr(step, "item") else step))
        ema = {}
        for key, value in checkpoint.get("state_dict", {}).items():
            if key.endswith("_extra_state") and isinstance(value, dict):
                ema[key] = {label: child.get("num_updates") for label, child in value.items()
                            if isinstance(child, dict) and "num_updates" in child}
        result.update(global_step=checkpoint.get("global_step"), epoch=checkpoint.get("epoch"),
            optimizer_parameter_step_values=sorted(optimizer_steps),
            scheduler_last_epochs=[value.get("last_epoch") for value in checkpoint.get("lr_schedulers", [])],
            scratch_sweep=checkpoint.get("scratch_sweep"), ema_num_updates=ema,
            pytorch_lightning_version=checkpoint.get("pytorch-lightning_version"))
        result["changed_during_read"] = path.stat().st_mtime_ns != result["identity"]["mtime_ns"]
        del checkpoint
    except Exception as error:
        result["read_error"] = f"{type(error).__name__}: {error}"
    return result


def log_stages(path):
    content = path.read_text(errors="replace") if path.is_file() else ""
    result, current = defaultdict(list), None
    for line in content.splitlines():
        match = re.fullmatch(r"START (.+)", line)
        if match:
            current = match[1]
        elif current:
            result[current].append(line)
    return content, {name: "\n".join(lines) for name, lines in result.items()}


def metric_records(run, raw_rows, warnings):
    records, reported = [], set()
    for row in raw_rows:
        if not isinstance(row, dict) or any(finite(row.get(key)) is None or row[key] < 0 for key in METRICS):
            warnings.append(f"Incomplete metric summary row: {run}")
            continue
        report = row.get("metrics_path", row.get("report_path"))
        report = str(Path(report).resolve()) if report else None
        if report and report in reported:
            warnings.append(f"Duplicated metric report row deduplicated: {report}")
            continue
        if report:
            reported.add(report)
        records.append({**row, "report_source": report, "recovered_report_only": False})
    for path in sorted(run.rglob("sd_agent_metrics.json")):
        if str(path.resolve()) in reported:
            continue
        report = read_json(path, {})
        values = report.get("agent_metrics", {})
        if any(finite(values.get(key)) is None for key in METRICS):
            continue
        relative = path.relative_to(run)
        match = next((re.fullmatch(r"eval_step_(\d+)", part) for part in relative.parts if re.fullmatch(r"eval_step_(\d+)", part)), None)
        records.append({"variant": relative.parts[0], "budget": int(match[1]) if match else None,
            "eval_seed": nested(report, "runtime", "torch_initial_seed"),
            "evaluation_weights": "ema" if nested(report, "decoder", "use_ema") else "online",
            "num_samples": report.get("num_samples"), **values,
            "report_source": str(path.resolve()), "recovered_report_only": True})
        warnings.append(f"Complete report recovered before/no summary row: {path}")
    return records


def resolve_loss(config, reports, warnings, folder):
    reported = [nested(read_json(Path(row["report_source"]), {}), "decoder", "component_loss")
                for row in reports if row.get("report_source")]
    # Distinguish a reported null legacy objective from a missing report key.
    if reports and reports[0].get("report_source"):
        report = read_json(Path(reports[0]["report_source"]), {})
        if "component_loss" in report.get("decoder", {}):
            value = report["decoder"]["component_loss"]
            if any(item != value for item in reported):
                warnings.append(f"Resolved loss changed within stage: {folder}")
            return value, "actual_metric_report", None
    init = nested(config, "model", "model_config", "decoder", "init_diffusion") or {}
    if "component_loss" not in init:
        return None, "unknown", None
    value = init["component_loss"]
    if value is None:
        return None, "legacy_null_config", None
    try:
        spec = importlib.util.spec_from_file_location("inventory_component_loss", ROOT / "src/smart/diffusion/component_loss.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = module.ComponentLoss(value, type_kind=init.get("type_objective", "ce"),
            heading_space="angle" if init.get("heading_noise") == "circular" else "vector",
            type_weight=init.get("type_loss_weight", 1.), heading_weight=init.get("heading_flow_loss_weight", 1.),
            speed_weight=init.get("speed_loss_weight", 0.), speed_scale=init.get("speed_loss_scale") or "data").config
        return value, "resolved_config_without_model", None
    except Exception as error:
        return None, "invalid_config", f"{type(error).__name__}: {error}"


def expected_modes(protocol):
    if protocol.get("evaluation_weight_modes"):
        return protocol["evaluation_weight_modes"]
    mode = protocol.get("evaluation_weights")
    if mode in ("ema", "online"):
        return [mode]
    return ["online", "ema"] if mode == "both" or protocol.get("evaluate_ema") else ["online"]


def stage_inventory(folder, run, protocol, records, log, category, full_count, checkpoints, warnings):
    variant = folder.name
    metrics = [row for row in records if row.get("variant") == variant]
    config_path = folder / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    initialization = read_json(folder / "initialization.json", {})
    source = metrics[0] if metrics else initialization
    eval_map = protocol.get("evaluation_checkpoints")
    eval_only = bool(eval_map) or any(row.get("evaluation_only") for row in metrics)
    start = source.get("resume_global_step", initialization.get("resume_global_step", 0)) or 0
    continued = bool(source.get("optimizer_continued", initialization.get("optimizer_continued"))) and not eval_only
    kind = "evaluation_only" if eval_only else "continuation" if continued else "fresh"
    trace = trace_audit(folder / "train_trace.json")
    planned = start if eval_only else protocol.get("steps")
    fit_done = eval_only or (trace["attained_global_step"] is not None and trace["attained_global_step"] == planned)
    milestones = [int(v) for v in str(protocol.get("checkpoint_steps", "")).split(",") if v.strip()]
    budgets = [start] if eval_only else sorted({planned, *milestones} - {None})
    budgets = [budget for budget in budgets if eval_only or budget > start]
    generations = 1 + (protocol.get("extra_eval_seed") is not None)
    expected = len(budgets) * generations * len(expected_modes(protocol))
    failure = ("Traceback (most recent call last)" in log or "segfault" in log.lower()
               or category == "failed_attempts")
    failed = failure and (not fit_done or len(metrics) < expected)
    status = ("failed" if failed else "complete" if fit_done and len(metrics) >= expected else
              "trained_evaluation_pending" if fit_done else "running" if initialization or trace["records"] else "setup_only")
    batch = source.get("batch_size", initialization.get("batch_size", protocol.get("batch_size")))
    actual_offset = source.get("actual_train_offset", initialization.get("actual_train_offset"))
    active_count = full_count - actual_offset if full_count is not None and actual_offset is not None else None
    updates = 0 if eval_only else trace["exact_stage_optimizer_updates"]
    exposure = (0 if eval_only else exposure_for_updates(updates, batch, active_count)
                if updates is not None and batch and active_count else None)
    checkpoint_paths = sorted(folder.glob("step_*.ckpt"), key=lambda p: int(re.search(r"step_(\d+)", p.name)[1]))
    if not checkpoint_paths and (folder / "last.ckpt").is_file():
        checkpoint_paths = [folder / "last.ckpt"]
    terminal = checkpoint_audit(checkpoint_paths[-1]) if checkpoints and checkpoint_paths else None
    if terminal and terminal.get("global_step") is not None and trace["attained_global_step"] is not None and terminal["global_step"] != trace["attained_global_step"]:
        warnings.append(f"Checkpoint/trace endpoint mismatch: {folder}")
    printed = list(re.finditer(r"STEP (\d+): .*elapsed=([\d.]+)s", log))
    last_printed = {"global_step": int(printed[-1][1]), "diagnostics_elapsed_seconds_approx": float(printed[-1][2])} if printed else None
    attained = start if eval_only else trace["attained_global_step"]
    observed = attained if attained is not None else terminal.get("global_step") if terminal else None
    if observed is None and last_printed:
        observed = last_printed["global_step"]
    lower_bound = 0 if eval_only else max(0, observed - start) if observed is not None else None
    lower_exposure = (exposure_for_updates(lower_bound, batch, active_count)
                      if lower_bound is not None and batch and active_count else None)
    timing = [finite(row.get("train_seconds")) for row in metrics]
    timing = [value for value in timing if value is not None]
    fit_seconds = max(timing) if timing and not eval_only else 0. if eval_only else None
    if len(set(timing)) > 1:
        warnings.append(f"Multiple fit elapsed values within stage; max retained: {folder}")
    loss, loss_source, loss_error = resolve_loss(config or {}, metrics, warnings, folder)
    loss_hash = digest(loss) if loss_source != "unknown" and loss_error is None else None
    loss_context = nested(config, "model", "model_config", "decoder", "init_diffusion") or {}
    loss_context = {key: loss_context.get(key) for key in ("velocity_representation", "heading_noise", "heading_path", "heading_objective",
        "heading_x0_loss", "heading_flow_loss_weight", "type_process", "type_objective", "size_representation",
        "speed_loss_weight", "speed_loss_scale", "generate_type", "match_ego_separately")}
    successful_model = kind != "evaluation_only" and fit_done and status == "complete"
    recorded_exposures = []
    for row in metrics:
        if row.get("seen_training_examples") is not None and full_count and batch:
            expected_exposure = exposure_for_updates(row["budget"], batch, full_count - (protocol.get("train_offset") or 0))
            recorded_exposures.append({"budget": row["budget"], "recorded": row["seen_training_examples"], "derived": expected_exposure})
            if row["seen_training_examples"] != expected_exposure:
                warnings.append(f"Recorded/derived cumulative example mismatch: {folder}, budget{row['budget']}")
    failed_csv = failed_csv_evidence(folder) if status == "failed" else []
    return {"stage_path": str(folder.resolve()), "run_path": str(run.resolve()), "category": category,
        "variant": variant, "training_seed": source.get("seed", initialization.get("seed", protocol.get("seed"))),
        "stage_kind": kind, "status": status, "successful_trained_model_or_stage": successful_model,
        "planned_global_endpoint": planned, "resume_global_step": start, "attained_global_step": attained,
        "exact_stage_optimizer_updates": updates, "observed_stage_updates_lower_bound": lower_bound,
        "stage_seen_examples": exposure, "observed_stage_seen_examples_lower_bound": lower_exposure,
        "batch_size": batch, "full_train_manifest_examples": full_count, "active_train_loader_examples": active_count,
        "actual_train_offset": actual_offset, "successful_metric_records": len(metrics), "expected_metric_records": expected,
        "recorded_budgets": sorted({row.get("budget") for row in metrics if row.get("budget") is not None}),
        "evaluation_modes": sorted({row.get("evaluation_weights", "online") for row in metrics}),
        "resolved_loss_sha256": loss_hash, "resolved_loss": loss, "resolved_loss_source": loss_source, "loss_resolution_error": loss_error,
        "loss_context": loss_context, "loss_and_context_sha256": digest({"component_loss": loss, **loss_context}) if loss_hash else None,
        "fresh_initial_generator_sha256": initialization.get("initial_generator_sha256"),
        "fresh_initial_map_sha256": initialization.get("initial_map_sha256"), "initialization_scope": initialization.get("initialization_scope"),
        "resumed_from": initialization.get("resumed_from"),
        "fit_wall_seconds_once": fit_seconds, "fit_timing_source": "max summary train_seconds per stage" if timing else None,
        "evaluation_wall_seconds": sum(finite(row.get("eval_seconds")) or 0. for row in metrics),
        "eval_records_without_timing": sum(finite(row.get("eval_seconds")) is None for row in metrics),
        "last_printed_progress_approx": last_printed, "trace": trace, "terminal_or_latest_checkpoint": terminal,
        "failed_stage_csv_diagnostic_only": failed_csv,
        "failed_csv_confirmed_logged_updates": max((item["confirmed_logged_update_count"] for item in failed_csv), default=0),
        "failed_csv_confirmed_logged_example_exposures": max((exposure_for_updates(item["confirmed_logged_update_count"], batch, active_count)
            for item in failed_csv if batch and active_count), default=0),
        "recorded_cumulative_exposure_crosschecks": recorded_exposures,
        "input_identities": {"config": identity(config_path), "initialization": identity(folder / "initialization.json"),
                             "train_summary": identity(folder / "train_summary.json")},
        "metric_record_identities": [{key: row.get(key) for key in ("budget", "seed", "eval_seed", "evaluation_weights", "num_samples", "report_source", "recovered_report_only")} for row in metrics]}


def totals(stages):
    successful = [row for row in stages if row["successful_trained_model_or_stage"]]
    metric_stages = [row for row in stages if row["successful_metric_records"] > 0]
    fit_stages = [row for row in stages if row["stage_kind"] != "evaluation_only"]
    return {"observed_stage_count": len(stages), "successful_fresh_trained_models": sum(row["stage_kind"] == "fresh" for row in successful),
        "successful_continuation_stages": sum(row["stage_kind"] == "continuation" for row in successful),
        "evaluation_only_stages": sum(row["stage_kind"] == "evaluation_only" for row in stages),
        "failed_stages": sum(row["status"] == "failed" for row in stages),
        "pending_stages": sum(row["status"] not in ("complete", "failed") for row in stages),
        "successful_metric_records": sum(row["successful_metric_records"] for row in stages),
        "distinct_evaluated_variant_names": sorted({row["variant"] for row in metric_stages}),
        "distinct_evaluated_resolved_loss_hashes": sorted({row["resolved_loss_sha256"] for row in metric_stages if row["resolved_loss_sha256"]}),
        "distinct_evaluated_loss_and_context_hashes": sorted({row["loss_and_context_sha256"] for row in metric_stages if row["loss_and_context_sha256"]}),
        "completed_successful_stage_optimizer_updates": sum(row["exact_stage_optimizer_updates"] or 0 for row in successful),
        "completed_successful_stage_seen_examples": sum(row["stage_seen_examples"] or 0 for row in successful),
        "all_exact_known_updates_in_scope": sum(row["exact_stage_optimizer_updates"] or 0 for row in fit_stages),
        "failed_csv_confirmed_updates_diagnostic_only": sum(row["failed_csv_confirmed_logged_updates"] for row in stages if row["status"] == "failed"),
        "fit_wall_seconds_known_once_per_stage": sum(row["fit_wall_seconds_once"] or 0 for row in fit_stages),
        "fit_stages_without_reliable_elapsed": sum(row["fit_wall_seconds_once"] is None for row in fit_stages),
        "evaluation_wall_seconds_known": sum(row["evaluation_wall_seconds"] for row in stages),
        "eval_records_without_timing": sum(row["eval_records_without_timing"] for row in stages)}


def audit_root(run, category, checkpoints, manifest_counts, warnings):
    if not run.is_dir():
        return {"category": category, "path": str(run), "status": "not_started", "stages": [], "summary": totals([])}
    protocol_path = run / "protocol.json"
    protocol = read_json(protocol_path, {})
    log_path = ROOT / ("logs/component_loss_scratch_" + run.name + ".log")
    content, logs = log_stages(log_path)
    raw = read_json(run / "summary.json", [])
    records = metric_records(run, raw if isinstance(raw, list) else [], warnings)
    full_count = None
    manifest = run / "train_order.txt"
    if manifest.is_file():
        key = protocol.get("train_order_sha256") or str(manifest.resolve())
        if key not in manifest_counts:
            manifest_counts[key] = text_line_count(manifest)
        full_count = manifest_counts[key]
    folders = sorted(p for p in run.iterdir() if p.is_dir() and (p / "config.yaml").is_file())
    stages = [stage_inventory(folder, run, protocol, records, logs.get(folder.name, ""), category, full_count, checkpoints, warnings) for folder in folders]
    if stages and all(row["status"] == "complete" for row in stages) and any(line.startswith("DONE ") for line in content.splitlines()):
        status = "complete"
    elif "Traceback (most recent call last)" in content or category == "failed_attempts":
        status = "partial_with_failure" if records else "failed"
    else:
        status = "running_or_pending"
    return {"category": category, "path": str(run.resolve()), "status": status, "stages": stages, "summary": totals(stages),
        "full_train_manifest_examples": full_count, "protocol": protocol,
        "input_identities": {"protocol": identity(protocol_path), "summary": identity(run / "summary.json"),
                             "log": identity(log_path), "train_order": identity(manifest, checksum=False)}}


def native_smoke_inventory(path, checkpoints):
    if not path.is_dir():
        return {"path": str(path), "status": "not_started", "scope": "Native smoke excluded from search/model counts"}
    files = sorted(path.rglob("*.ckpt"))
    candidates = [checkpoint_audit(file) for file in files] if checkpoints else []
    best = max(candidates, key=lambda row: row.get("global_step", -1)) if candidates else None
    return {"path": str(path), "status": "checkpoint_saved" if best else "present_no_verified_checkpoint",
        "scope": "Native smoke excluded from search/model counts; repeated checkpoint copies are not repeated training",
        "max_saved_global_step": best.get("global_step") if best else None,
        "checkpoint_metadata": candidates, "cost": "No monotonic fit elapsed available unless separately recorded; no estimate added"}


def markdown(value):
    lines = ["# 从头训练搜索：实验清单与成本审计", "", f"UTC snapshot: `{value['created_utc']}`。此文件只读取实际文件，可在排队任务完成后重跑。", "",
        "模型数按成功的实际训练生命周期计数：同 seed/初始张量哈希但不同 loss 的独立训练仍分别计一个模型；"
        "续训阶段不重复计新模型；只评价加载的模型不计新训练。失败和运行中阶段不计成功模型。", "",
        "| Scope | Fresh completed models | Continuations | Eval-only stages | Successful metric records | Updates completed | Actual scene exposures | Known fit min | Pending | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in value["category_totals"].items():
        lines.append(f"| {name} | {item['successful_fresh_trained_models']} | {item['successful_continuation_stages']} | {item['evaluation_only_stages']} | {item['successful_metric_records']} | {item['completed_successful_stage_optimizer_updates']:,} | {item['completed_successful_stage_seen_examples']:,} | {item['fit_wall_seconds_known_once_per_stage']/60:.2f} | {item['pending_stages']} | {item['failed_stages']} |")
    main = value["primary_total"]
    lines += ["", f"主比较已完成生成指标记录 **{main['successful_metric_records']}**；实际评价过的 loss 名称 **{len(main['distinct_evaluated_variant_names'])}**、"
        f"resolved loss 哈希 **{len(main['distinct_evaluated_resolved_loss_hashes'])}**。这不是未来预计数量。", "",
        "每个记录是一次生成评价，其完整 GT 参考指标是同次评价的附加结果，不另加一次。online、EMA、不同生成 seed "
        "或 checkpoint 属于多次评价，不增加独立训练 seed N。", "", "## 阶段清单", "",
        "| Category / Run | Variant | Kind | Status | Start→end | Stage updates | Examples | Metric rows | Fit seconds once |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for run in value["roots"]:
        if not run["stages"]:
            lines.append(f"| {run['category']} / {Path(run['path']).name} | — | — | {run['status']} | — | — | — | 0 | — |")
        for row in run["stages"]:
            lines.append(f"| {run['category']} / {Path(run['path']).name} | {row['variant']} | {row['stage_kind']} | {row['status']} | {row['resume_global_step']}→{row['attained_global_step']} | {row['exact_stage_optimizer_updates']} | {row['stage_seen_examples']} | {row['successful_metric_records']} | {row['fit_wall_seconds_once']} |")
    lines += ["", "## 时间、曝光和失败的限制", "",
        "`train_trace.json` 每行记录 train batch 开始时的 pre-update global_step；连续且唯一的 trace 行数等于此阶段 optimizer updates，"
        "末行 step+1 为最终 global endpoint。trace 没有 elapsed/timestamp/实际 batch size；曝光使用真实 manifest 长度、批量和末批大小计算，"
        "并与完成 checkpoint/summary 元数据交叉核对。", "",
        "可靠训练耗时来自 harness 用 monotonic clock 包住 trainer.fit 后写出的 train_seconds，"
        "每个 (run_path,variant) 只计一次（重复评价行取最大相同阶段记录值）。"
        "STEP 日志 elapsed 起点是 Diagnostics 构造，只有近似进度证据；没有 trace/checkpoint 的崩溃最后日志只是已确认更新下界，"
        "不能宣称精确最终步数或加成可靠 fit 成本。若保留 CSV，连续 train step 行的数量是另列的诊断性已完成更新证据，"
        "不会增加成功模型、主比较更新数或生成指标数。", "",
        "已知 fit 时间是 wall time，并非 GPU kernel busy time。未计模型构造、训练/评价之间开销、预处理、统计校准、"
        "CPU 图表/PDF、未知失败耗时和未来任务。评价秒数只加每个唯一实际报告一次，缺少计时的报告另记。"
        "因此这些时间是有范围限制的已知成本，不是完整端到端总时长。", "",
        "native smoke 在 JSON 中单列；LR pilots、implementation pilots 和失败阶段不并入主结果指标/成功模型数。"
        "全部输入 identity、初始化、resolved loss、trace schema、checkpoint global_step/epoch/optimizer/scheduler 和 pending 证据见 search_inventory.json。", ""]
    if value["warnings"]:
        lines += ["## 审计提示", ""] + [f"- {warning}" for warning in value["warnings"]] + [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ART)
    parser.add_argument("--no-checkpoints", action="store_true", help="Read text metadata only; cannot verify saved optimizer/checkpoint state")
    parser.add_argument("--native-roots", nargs="*", type=Path, default=[RUNS / "native_best_train_smoke"])
    args = parser.parse_args()
    warning, counts, roots = [], {}, []
    categories = {**MAIN_ROOTS, **AUX_ROOTS}
    categories["failed_attempts"] = [*categories["failed_attempts"], *(path.name for path in sorted(RUNS.glob("failed_*")) if path.is_dir())]
    for category, names in categories.items():
        for name in names:
            roots.append(audit_root(RUNS / name, category, not args.no_checkpoints, counts, warning))
    grouped = {category: totals([stage for run in roots if run["category"] == category for stage in run["stages"]]) for category in categories}
    for category, total in grouped.items():
        scope = [run for run in roots if run["category"] == category]
        total.update(not_started_root_count=sum(run["status"] == "not_started" for run in scope),
            running_or_pending_root_count=sum(run["status"] == "running_or_pending" for run in scope),
            complete_root_count=sum(run["status"] == "complete" for run in scope))
    primary = [stage for run in roots if run["category"] in MAIN_ROOTS for stage in run["stages"]]
    value = {"created_utc": datetime.now(timezone.utc).isoformat(), "primary_total": totals(primary), "category_totals": grouped,
        "roots": roots, "native_smoke": [native_smoke_inventory(path.resolve(), not args.no_checkpoints) for path in args.native_roots],
        "checkpoint_access": "CPU mmap metadata; no GPU/model/dataset tensor execution" if not args.no_checkpoints else "not audited",
        "counting_unit": "One actual stage directory; completed fresh fit is a new trained model, continuation/evaluation-only is not",
        "timing_unit": "Known monotonic trainer.fit wall seconds per stage, no duplicate checkpoint/gen/EMA-row addition",
        "failure_diagnostic_artifact_identities": [identity(path) for path in sorted(ART.glob("failed_*diagnostic.json"))],
        "warnings": sorted(set(warning + READ_WARNINGS)), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "search_inventory.json").write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output / "search_inventory.md").write_text(markdown(value))
    print(json.dumps({"output": str(args.output.resolve()), "primary": {key: number for key, number in value["primary_total"].items()
        if not key.startswith("distinct_")}, "evaluated_variant_count": len(value["primary_total"]["distinct_evaluated_variant_names"]),
        "resolved_loss_hash_count": len(value["primary_total"]["distinct_evaluated_resolved_loss_hashes"]), "warnings": len(value["warnings"])}))


if __name__ == "__main__":
    main()
