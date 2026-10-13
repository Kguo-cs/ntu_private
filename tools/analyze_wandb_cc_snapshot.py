"""Analyze refreshed W&B evidence without treating unmatched remote runs as trials."""
import ast
import collections
import datetime as dt
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "output/reports/wandb_cc_20261013"
METRICS = ["nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd",
           "width_jsd", "speed_jsd", "collision_rate"]
ATTR = METRICS[3:6]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model(config):
    result = config.get("model", {}).get("model_config", {})
    return ast.literal_eval(result) if isinstance(result, str) else result


def flat(value, prefix=""):
    if isinstance(value, dict):
        return {key: val for k, v in value.items() for key, val in
                flat(v, prefix + ("." if prefix else "") + str(k)).items()}
    return {prefix: value}


def main():
    inventory_path = ROOT / "inventory.json"
    inv = json.loads(inventory_path.read_text())
    records = {r["id"]: r for r in inv["runs"]}
    epoch_values = {}
    manifest = []
    source_rows = {}
    raw_histories = {}
    for path in sorted((ROOT / "histories").glob("*.json")):
        rid = path.stem
        snapshot = json.loads(path.read_text())
        raw_histories[rid] = snapshot["rows"]
        rows = [r for r in snapshot["rows"] if r.get("epoch") is not None and all(
            isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in METRICS)]
        bins = collections.defaultdict(list)
        for row in rows:
            bins[int(row["epoch"])].append(row)
        epoch_values[rid] = {e: {k: statistics.mean(r[k] for r in rr) for k in METRICS}
                             for e, rr in sorted(bins.items())}
        source_rows[rid] = [{k: row.get(k) for k in ["epoch", "trainer/global_step", "_step", "_timestamp"] + METRICS} for row in rows]
        record = records[rid]
        c = record["config"]
        m = model(c)
        init = m.get("decoder", {}).get("init_diffusion", {}) or {}
        data = c.get("data", {})
        trainer = c.get("trainer", {})
        meta_path = ROOT / "metadata" / rid / "selected.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        epochs = sorted(epoch_values[rid])
        manifest.append({"id": rid, "name": record["name"], "state": record["state"],
            "commit": record["commit"], "url": record["url"], "seed": c.get("seed"),
            "config": {"train_batch_size": data.get("train_batch_size"),
                "val_batch_size": data.get("val_batch_size"), "limit_val_batches": trainer.get("limit_val_batches"),
                "train_data": data.get("train_raw_dir"), "val_data": data.get("val_raw_dir"),
                "reference_mode": m.get("sd_reference_mode"), "precision": trainer.get("precision"),
                "float32_matmul_precision": c.get("float32_matmul_precision"),
                "max_epochs": trainer.get("max_epochs"), "lr": m.get("lr"),
                "lr_min_ratio": m.get("lr_min_ratio"), "ckpt_path": c.get("ckpt_path"),
                "initial_map_checkpoint": m.get("initial_map_checkpoint"),
                "sd_r_reference": m.get("sd_r_reference"),
                "map_crop": m.get("token_processor", {}).get("init_map_crop"),
                "map_half_extent": m.get("token_processor", {}).get("init_map_half_extent"),
                "init_diffusion": init}, "metadata": meta.get("selected", {}),
            "summary_training_epoch": record["summary"].get("epoch"),
            "summary_global_step": record["summary"].get("trainer/global_step"),
            "history_rows": snapshot["row_count"], "validation_rows": len(rows),
            "actual_validation_epochs": epochs, "last_validation_epoch": epochs[-1] if epochs else None,
            "last_validation_metrics": epoch_values[rid][epochs[-1]] if epochs else None,
            "history_sha256": digest(path), "history_fetched_at_utc": snapshot["fetched_at_utc"]})
    by_id = {r["id"]: r for r in manifest}

    def compare(candidate, baseline, epochs, purpose):
        a = {k: statistics.mean(epoch_values[candidate][e][k] for e in epochs) for k in METRICS}
        b = {k: statistics.mean(epoch_values[baseline][e][k] for e in epochs) for k in METRICS}
        ratios = {k: a[k] / b[k] for k in METRICS}
        ca, cb = by_id[candidate]["config"], by_id[baseline]["config"]
        # Evaluate configuration conditions explicitly; hash/GEN pairing remain unavailable remotely.
        fields = ["train_batch_size", "val_batch_size", "limit_val_batches", "reference_mode", "precision", "max_epochs"]
        conditions = {k: {"candidate": ca[k], "baseline": cb[k], "equal": ca[k] == cb[k]} for k in fields}
        baseline_flat, candidate_flat = flat(records[baseline]["config"]), flat(records[candidate]["config"])
        changes = {k: {"candidate": candidate_flat.get(k), "baseline": baseline_flat.get(k)}
                   for k in sorted(set(baseline_flat) | set(candidate_flat))
                   if candidate_flat.get(k) != baseline_flat.get(k)
                   and (k.startswith("model.") or k.startswith("data.") or k in ["seed", "ckpt_path", "float32_matmul_precision"])}
        return {"candidate": candidate, "baseline": baseline, "epochs": epochs, "purpose": purpose,
            "evidence_class": "descriptive_same_epoch_remote_single_run_ratios",
            "paired_multiseed_R": False, "remote_data_membership_verified": False,
            "generation_seed_pairing_verified": False, "conditions": conditions,
            "candidate_metric_mean": a, "baseline_metric_mean": b, "ratios": ratios,
            "observed_R7": statistics.mean(ratios.values()),
            "observed_R_attr": statistics.mean(ratios[k] for k in ATTR),
            "config_differences": changes}

    comparisons = [compare(c, b, es, label) for c, b, es, label in [
        ("4bo59qfn", "iu3fa4gx", [127], "N20 cc map64 final vs historical657 final"),
        ("4bo59qfn", "iu3fa4gx", [119, 123, 127], "N20 cc map64 final three-validation window"),
        ("7w5t0hi1", "iu3fa4gx", [107], "N40 cc at latest actual validation; outside N20 constraint"),
        ("7w5t0hi1", "iu3fa4gx", [99, 103, 107], "N40 cc three-validation window; outside N20 constraint"),
        ("4bo59qfn", "7w5t0hi1", [99, 103, 107], "Map64/N20 vs cc N40; mixed changes, not map-K ablation"),
        ("c486qs4q", "iu3fa4gx", [127], "LR5e-4 floor .01 final"),
        ("c486qs4q", "iu3fa4gx", [119, 123, 127], "LR5e-4 floor .01 three-validation window"),
        ("fs0kq6fc", "iu3fa4gx", [79], "LR2e-4 floor .05 latest; not128epoch"),
        ("v62ymyjj", "iu3fa4gx", [91], "LR2e-4 floor .01 latest; not128epoch"),
        ("fs0kq6fc", "iu3fa4gx", [71, 75, 79], "LR2e-4 floor .05 common late window"),
        ("v62ymyjj", "iu3fa4gx", [71, 75, 79], "LR2e-4 floor .01 common late window"),
        ("c486qs4q", "iu3fa4gx", [71, 75, 79], "LR5e-4 floor .01 common late window"),
        ("23ykdick", "iu3fa4gx", [15], "Early A6000 run; initialization/source differences require audit"),
        ("az3mgqvh", "arpfw0fr", [75, 79, 83], "Same128scene configured cohort: depth5/input time vs depth3/every-layer time"),
    ]]
    drift = {}
    training_component_trends = {}
    online_R_reference_audit = {}
    for rid in epoch_values:
        vals = epoch_values[rid]
        if not vals:
            continue
        final = max(vals)
        drift[rid] = {k: {"minimum_epoch": min(vals, key=lambda e: vals[e][k]),
                         "minimum_value": min(a[k] for a in vals.values()), "last_epoch": final,
                         "last_value": vals[final][k],
                         "last_over_minimum": vals[final][k] / min(a[k] for a in vals.values())}
                      for k in METRICS}
        train_keys = ["train/loss"] + ["train/component/" + k + "_loss" for k in
            ["position", "heading", "length", "width", "velocity", "type", "collision"]]
        training_component_trends[rid] = []
        for lo, hi in [(0, 31), (32, 63), (64, 95), (96, 127)]:
            rr = [r for r in raw_histories[rid] if lo <= r.get("epoch", -1) <= hi]
            means, counts = {}, {}
            for key in train_keys:
                data = [r[key] for r in rr if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
                if data:
                    means[key], counts[key] = statistics.mean(data), len(data)
            if means:
                training_component_trends[rid].append({"epoch_min": lo, "epoch_max": hi,
                    "sampled_logged_row_mean": means, "logged_rows_per_component": counts,
                    "full_epoch_batch_weighted_mean": False})
        reference = by_id[rid]["config"]["sd_r_reference"]
        logged_R_rows = [r for r in raw_histories[rid] if isinstance(r.get("R"), (int, float))]
        if logged_R_rows:
            online_R_reference_audit[rid] = {"configured_reference": reference,
                "configured_validation_scenes": by_id[rid]["config"]["val_batch_size"],
                "logged_R_rows": len(logged_R_rows), "used_for_same_cohort_ranking": False,
                "reason": "Configured full50000 reference and128scene monitored validation are different scope; denominator/content/seed/epoch pairing unverified.",
                "logged_R_values": [{"epoch": r.get("epoch"), "R": r["R"], "R_attr": r.get("R_attr")} for r in logged_R_rows]}
    limitations = [
        "Remote checkpoint/data membership/content hashes and paired GEN seeds are not provided by these W&B records.",
        "Observed R7/R_attr are descriptive single-run ratios, not fresh paired TRAIN/GEN pooled search scores.",
        "Norm/time and depth5 validate128 scenes; historical657 and cc validate256; no cross-cohort R is computed.",
        "cc_best uses40 denoiser steps. cc_best_full_map64 uses20 plus exact-x0 terminal update, map64 and iterative collision guidance; improvements cannot be attributed to mapK alone.",
        "finished run status does not imply128epochs: LR2e-4 trials stop actual validation at79/91, cc_best512 at15.",
        "Per-component historical minima are different checkpoints and must not be combined into a generated metric vector.",
        "GPU metadata inventory reports4 A6000 but configured trainer.devices is1; do not infer4-device training.",
        "Most old runs omit matmul precision; 32-true config alone does not prove matching numerical settings.",
        "Remote files/artifact inventories contain no checkpoint weights for the six primary runs queried.",
        "CurrentK8 run has no seven-metric validation yet; no long-training performance claim is supported.",
        "Norm/time and depth5 online loggedR use control657_full50000.json while monitoring128scenes; those loggedR values are excluded from same-cohort ranking.",
        "Training component means are means of logged rows (roughly every100steps), not a full-batch-weighted epoch loss or gradient contribution.",
    ]
    result = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inventory_fetched_at_utc": inv["fetched_at_utc"], "inventory_sha256": digest(inventory_path),
        "inventory_api_host": inv["api_host"], "project": inv["project"], "all_project_run_count": inv["run_count"],
        "selected_runs": len(manifest), "selected_history_rows": sum(r["history_rows"] for r in manifest),
        "complete_seven_metric_rows": sum(r["validation_rows"] for r in manifest),
        "comparisons": comparisons, "drift": drift, "training_component_trends": training_component_trends,
        "online_R_reference_audit": online_R_reference_audit, "limitations": limitations,
        "source_files": {str(path.relative_to(ROOT)): digest(path) for path in sorted(ROOT.rglob("*.json"))
            if path.name in ["inventory.json", "history_fetch_status.json", "remote_artifact_inventory.json"]
            or "histories" in path.relative_to(ROOT).parts
            or "metadata" in path.relative_to(ROOT).parts},
        "analysis_script_sha256": digest(Path(__file__)),
        "no_remote_writes": True, "no_gpu_training_or_evaluation": True}
    for name, value in [("run_manifest.json", manifest), ("validation_history.json", source_rows), ("refresh_analysis.json", result)]:
        (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    cohorts = [("256 scenes, remote monitored", ["iu3fa4gx", "4bo59qfn", "7w5t0hi1", "c486qs4q"]),
               ("128 scenes, remote monitored", ["arpfw0fr", "az3mgqvh"])]
    labels = {"iu3fa4gx": "657 N20", "4bo59qfn": "cc map64 N20 + guidance",
              "7w5t0hi1": "cc N40 + guidance", "c486qs4q": "657 floor .01 N20",
              "arpfw0fr": "norm + every-layer time L3 N20", "az3mgqvh": "norm L5/input time N20"}
    for row, (cohort, ids) in enumerate(cohorts):
        for col, key in enumerate(["lat_dev_jsd", "speed_jsd", "width_jsd", "collision_rate"]):
            ax = axes[row, col]
            for rid in ids:
                values = epoch_values[rid]
                ax.plot(list(values), [a[key] for a in values.values()], label=labels[rid])
            ax.set(title=cohort + "\n" + key, xlabel="Actual validation epoch", ylabel="Native value")
            ax.grid(alpha=.2)
            if key in ["lat_dev_jsd", "collision_rate"]:
                ax.set_yscale("log")
        axes[row, 0].legend(fontsize=8)
    fig.suptitle("W&B refreshed 2026-10-13: separate cohorts; single-run observational evidence")
    fig.savefig(ROOT / "online_curves.png", dpi=150)
    plt.close(fig)

    parts = ["<!doctype html><meta charset='utf-8'><title>W&B + cc refreshed analysis</title>",
        "<style>body{max-width:1300px;margin:30px auto;font:16px system-ui;line-height:1.5;color:#233145}table{border-collapse:collapse;font-size:14px;width:100%;margin:18px 0}td,th{padding:7px;border:1px solid #ccd4dc;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%}code{font-size:13px}</style>",
        "<h1>W&B 与 cc 最新训练结果</h1>",
        f"<p>快照 UTC：{html.escape(inv['fetched_at_utc'])}；141 个 run 中刷新相关 15 个 run。API 只读，不修改远端，不运行 GPU 训练或评价。</p>",
        "<p>cc map64 的 N20 完整训练显示明显降低碰撞；N40 cc 的属性结果更好，但不满足当前 20 步限制。Norm/time 与 depth5 的评价只有 128 场景，不能直接与旧 657/cc 的 256 场景排榜。以下 R 均为同 epoch 的远端单 run 描述性比值，缺少远端数据哈希和 GEN 配对，不能当作严格配对实验结论。</p>",
        "<img src='online_curves.png' alt='Separate256scene and128scene cohorts'><h2>最新实际评价</h2><table><tr><th>run / name</th><th>epoch</th><th>val scenes</th>" + "".join("<th>" + k + "</th>" for k in METRICS) + "</tr>"]
    for item in manifest:
        metrics = item["last_validation_metrics"] or {}
        parts.append("<tr><td><a href='" + html.escape(item["url"], quote=True) + "'>" + item["id"] + "</a> " + html.escape(item["name"]) + "</td><td>" + str(item["last_validation_epoch"]) + "</td><td>" + str(item["config"]["val_batch_size"]) + "</td>" + "".join("<td>" + (f"{metrics[k]:.6g}" if k in metrics else "—") + "</td>" for k in METRICS) + "</tr>")
    parts.append("</table><h2>同 epoch / 同评价窗口的观察性对比</h2><table><tr><th>comparison</th><th>epochs</th><th>R7</th><th>R_attr</th>" + "".join("<th>ratio " + k + "</th>" for k in METRICS) + "</tr>")
    for comp in comparisons:
        parts.append("<tr><td>" + html.escape(comp["purpose"]) + "</td><td>" + html.escape(str(comp["epochs"])) + "</td><td>" + f"{comp['observed_R7']:.4f}" + "</td><td>" + f"{comp['observed_R_attr']:.4f}" + "</td>" + "".join(f"<td>{comp['ratios'][k]:.4f}</td>" for k in METRICS) + "</tr>")
    parts.append("</table><h2>继续探索的重点</h2><p>优先从相同 checkpoint 的 N20 推理消融分解 exact-x0 / collision guidance / map 邻居数；保持 Gaussian 随机源、场景列表、EMA 与精度一致。碰撞指导无需增加网络参数，但必须同时检查位置分布、onroad 与车辆数量。N20 map64 相对 N40 的 shape/speed 明显更弱，不能由混合变更直接归因于地图邻居。旧 657、cc N20、norm/time 都有后期 speed/width 回升现象，应保留中期 checkpoint 并用稳定4096场景选择，而不是仅看最后 epoch。</p>")
    parts.append("<h2>属性漂移与训练 loss</h2><table><tr><th>run</th><th>speed best→last</th><th>width best→last</th><th>velocity loss32–63→96–127</th><th>width loss32–63→96–127</th></tr>")
    for rid in ["iu3fa4gx", "4bo59qfn", "7w5t0hi1", "arpfw0fr", "az3mgqvh", "c486qs4q"]:
        stages = {t["epoch_min"]: t["sampled_logged_row_mean"] for t in training_component_trends.get(rid, [])}
        def loss_pair(key):
            a, b = stages.get(32, {}).get(key), stages.get(96, {}).get(key)
            return (f"{a:.5g} → {b:.5g}" if a is not None and b is not None else "未覆盖完整后期窗口")
        def drift_pair(key):
            d = drift[rid][key]
            return f"e{d['minimum_epoch']} {d['minimum_value']:.5g} → e{d['last_epoch']} {d['last_value']:.5g} ({100 * (d['last_over_minimum'] - 1):+.1f}%)"
        parts.append("<tr><td>" + rid + "</td><td>" + drift_pair("speed_jsd") + "</td><td>" + drift_pair("width_jsd") + "</td><td>" + loss_pair("train/component/velocity_loss") + "</td><td>" + loss_pair("train/component/width_loss") + "</td></tr>")
    parts.append("</table><p>各指标 best 是各自历史最小值，用于描述漂移，不能拼接成一个模型结果。loss 表是已记录训练行的均值，不是梯度占比；重建下降伴随 JSD 回升支持训练目标与自由生成分布存在差距，不足以单独证明过拟合或某一种 loss 必然有害。</p><p>新增 online norm/time、depth5 的 logged R 引用了 control657_full50000.json，但只评价128场景；该 R 不用于公平排名。新报告仅在配置同评价规模、相同实际 epoch 时计算描述性比值。</p>")
    parts.append("<h2>限制</h2><ul>" + "".join("<li>" + html.escape(v) + "</li>" for v in limitations) + "</ul><p>原始证据：inventory.json / histories / run_manifest.json / validation_history.json / remote_artifact_inventory.json。比较配置差异与各分量历史最小值见 refresh_analysis.json；历史最小值仅用于诊断，不组合成 best checkpoint。</p>")
    (ROOT / "refresh_analysis.html").write_text("\n".join(parts))
    print(json.dumps({k: result[k] for k in ["all_project_run_count", "selected_runs", "selected_history_rows", "complete_seven_metric_rows"]}))
    print("Saved read-only refresh_analysis.json/html and cohort-separated curves")


if __name__ == "__main__":
    main()
