"""CPU-only final aggregation after the actual 50k evaluation completes.

With --wait, monitor the active confirmation/full-set logs and stop on a
recorded failure. No model, dataset tensor, training or evaluation is run.
The expected final result count is deliberately not hardcoded: the inventory
and analyzer must agree on their actual completed records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_scratch_search_inventory import MAIN_ROOTS

ART = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011"
RUNS = ROOT / "logs/component_loss_scratch"
LOG = ROOT / "logs/component_loss_scratch_full50000_seed817.log"
WATCH = [ROOT / f"logs/component_loss_scratch_{name}.log" for name in
         ("batch512_seed1817", "batch512_seed2817", "heldout_bridge_seed817", "full50000_seed817")]


def read(path):
    return json.loads(path.read_text())


def identity(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def wait_until_complete(wait):
    while True:
        progress = []
        for path in WATCH:
            text = path.read_text(errors="replace") if path.is_file() else ""
            done = any(line.startswith("DONE ") for line in text.splitlines())
            if not done and any(marker in text for marker in
                                ("Traceback (most recent call last)", "Fatal Python error:", "Segmentation fault")):
                raise RuntimeError(f"Pending pipeline recorded a failure: {path}; no final artifacts were authored")
            last = next((line for line in reversed(text.splitlines()) if line.startswith(("STEP ", "RESULT ", "DONE ", "START "))), "not started")
            progress.append(f"{path.stem}: {last[:160]}")
            if path == LOG and done:
                return
        if not wait:
            raise RuntimeError(f"Successful full-set DONE is not recorded: {LOG}")
        print("WAIT " + " | ".join(progress), flush=True)
        time.sleep(45)


def markdown(value):
    counts = value["actual_main_counts"]
    lines = ["# 从头训练搜索：最终实际文件汇总", "", f"UTC: `{value['created_utc']}`。全量阶段实际 DONE 后才生成；数目由保存文件推导，没有填入预计结果。", "",
             f"主比较：**{counts['successful_metric_records']} 次生成评价，{counts['distinct_variant_names']} 个 loss 名称，"
             f"{counts['distinct_resolved_loss_hashes']} 个 resolved loss 哈希；{counts['successful_fresh_trained_models']} 个成功从头训练模型，"
             f"{counts['successful_continuation_stages']} 个续训阶段，{counts['evaluation_only_stages']} 个只评价阶段**。", "",
             "| Phase | Metric records | Fresh models | Continuations | Eval-only stages | Updates | Actual scene exposures | Known fit min | Known eval min |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data in value["phase_counts"].items():
        lines.append(f"| {name} | {data['successful_metric_records']} | {data['successful_fresh_trained_models']} | "
                     f"{data['successful_continuation_stages']} | {data['evaluation_only_stages']} | "
                     f"{data['completed_successful_stage_optimizer_updates']:,} | {data['completed_successful_stage_seen_examples']:,} | "
                     f"{data['fit_wall_seconds_known_once_per_stage']/60:.2f} | {data['evaluation_wall_seconds_known']/60:.2f} |")
    lines += ["", f"已完成主比较实际 optimizer updates={counts['completed_successful_stage_optimizer_updates']:,}，"
              f"实际样本曝光={counts['completed_successful_stage_seen_examples']:,}；已知去重 fit wall time "
              f"{counts['fit_wall_seconds_known_once_per_stage']/60:.2f} min，已知评价 "
              f"{counts['evaluation_wall_seconds_known']/60:.2f} min。重复 checkpoint、生成 seed、online/EMA 行不重复增加 fit 成本。", "",
              "这些是已知 wall time，不是 GPU busy time或完整端到端成本；没有加入模型构造、数据预处理、CPU 图表/PDF、未知失败时间。"
              "LR/实现 pilots、失败阶段和 native smoke在清单中单列，不并入主比较。保留失败CSV的400次更新仅为诊断证据，不进入成功更新/模型数。", "",
              "全部14个合法主根目录均保留在 final_all_stage；无新增图表，全部实际记录、分组、后期趋势和表格均保存。"
              "各评价 cohort 分开汇总，不将搜索1024、heldout4096和full50000混合排名。full50000不改变此前冻结的heldout部署选择。", "",
              "六项JSD遵循helper输出单位：nearest/lateral×10，其余JSD×100；原始JSD为自然对数，理论上界ln2。"
              "碰撞为百分数。综合S的scale为[1,1,10,20,15,20,5]；训练seed样本SD用ddof=1（N=1为0），生成重复不增加独立训练N。", "",
              "[全部实际指标和表格](final_all_stage/analysis.md)、[observations](final_all_stage/observations.json)、"
              "[inventory](search_inventory.json)、[精确facts及hash](final_search_facts.json)。", "",
              "工具验证：6个CPU inventory单元测试通过；范围为曝光/trace计数、评价矩阵、cost去重、重复报告和失败CSV诊断，"
              "不代表GPU训练测试或完整原生训练验证。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--output", type=Path, default=ART)
    args = parser.parse_args()
    wait_until_complete(args.wait)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": "/tmp/sim_scratch_loss_matplotlib"}
    subprocess.run([sys.executable, "tools/audit_scratch_search_inventory.py", "--output", str(output)],
                   cwd=ROOT, env=env, check=True)
    roots = [RUNS / name for names in MAIN_ROOTS.values() for name in names]
    combined = output / "final_all_stage"
    subprocess.run([sys.executable, "tools/analyze_scratch_loss_search.py", "--runs", *map(str, roots),
                    "--output", str(combined), "--anchor", "warm_winner", "--plot-variants", ""],
                   cwd=ROOT, env=env, check=True)
    test_command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_scratch_search_inventory.py", "-v"]
    test = subprocess.run(test_command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    (output / "final_inventory_tests.txt").write_text(test.stdout + test.stderr)
    inventory = read(output / "search_inventory.json")
    observations = read(combined / "observations.json")
    protocol = read(combined / "analysis_protocol.json")
    total = inventory["primary_total"]
    if total["pending_stages"]:
        raise RuntimeError(f"Actual main stages still pending: {total['pending_stages']}; no final facts claimed")
    if len(observations) != total["successful_metric_records"]:
        raise RuntimeError("Analyzer / inventory actual metric record counts disagree")
    if len({row["report_source"] for row in observations}) != len(observations):
        raise RuntimeError("Actual observation report identities are duplicated")
    phase_counts = {name: inventory["category_totals"][name] for name in MAIN_ROOTS}
    for name, names in MAIN_ROOTS.items():
        actual = sum(row["run"] in names for row in observations)
        if actual != phase_counts[name]["successful_metric_records"]:
            raise RuntimeError(f"Phase actual counts disagree: {name}")
    variants = sorted({row["variant"] for row in observations})
    losses = sorted({row["recipe_sha256"] for row in observations})
    if variants != total["distinct_evaluated_variant_names"] or losses != total["distinct_evaluated_resolved_loss_hashes"]:
        raise RuntimeError("Actual recipe names / resolved hashes disagree")
    files = [output / "search_inventory.json", output / "search_inventory.md", output / "final_inventory_tests.txt",
             *(combined / name for name in ("observations.json", "observations.csv", "aggregate.json", "aggregate.csv",
                                           "late_budget_trends.json", "late_budget_trends.csv", "analysis_protocol.json", "analysis.md"))]
    decisions = [ART / name for name in ("final_deployment_selection_protocol.json", "production_candidate_confirmation_selection.json", "paired_finalists/paired_summary.json")]
    facts = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Actual completed main search only, after successful full50000 DONE",
             "actual_main_counts": {**total, "distinct_variant_names": len(variants), "distinct_resolved_loss_hashes": len(losses)},
             "phase_counts": phase_counts, "main_roots": [str(root) for root in roots],
             "root_inputs": [root["input_identities"] for root in inventory["roots"] if root["category"] in MAIN_ROOTS],
             "variant_names": variants, "resolved_loss_hashes": losses,
             "input_decision_identities": [identity(path) for path in decisions if path.is_file()],
             "output_identities": [identity(path) for path in files], "full_set_done_log": identity(LOG),
             "analyzer_protocol": protocol, "auxiliary_counts": {name: data for name, data in inventory["category_totals"].items() if name not in MAIN_ROOTS},
             "native_smoke": inventory["native_smoke"], "warnings": inventory["warnings"],
             "test_evidence": {"command": test_command, "passing_cpu_inventory_tests": 6,
                               "scope": "Exposure/trace count, eval matrix, report/fit dedup and failed CSV diagnostics; not GPU or native training validation"},
             "cost_limits": "Known stage-deduplicated monotonic fit and unique-report eval wall seconds; excludes setup, preprocess, CPU artifacts and unknown failure time",
             "script_identity": identity(Path(__file__))}
    (output / "final_search_facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "final_search_facts.md").write_text(markdown(facts))
    print(json.dumps({"final_output": str(output), "metrics": len(observations), "variants": len(variants),
                      "fresh_models": total["successful_fresh_trained_models"], "continuations": total["successful_continuation_stages"]}), flush=True)


if __name__ == "__main__":
    main()
