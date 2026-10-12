"""CPU-only source, cache and actual frozen-pair distribution review.

Does not instantiate a model, run inference, mutate datasets or modify metrics.
Reads only completed TRAIN817/1817 final results; GEN is averaged within TRAIN.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

np = None  # Imported only by the isolated sim tensor/cache worker.

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
OUT = ROOT / "docs/experiments/init_diffusion_overnight_search_20261011"
RUNS = ROOT / "logs/component_loss_frozen_map"
METRICS = ("nearest_dist_jsd", "lat_dev_jsd", "ang_dev_jsd", "length_jsd", "width_jsd", "speed_jsd", "collision_rate")
ARMS = ("new_control", "combo_batch256_ego_sincos")
FINAL_STEPS = {"new_control": 4096, "combo_batch256_ego_sincos": 8192}
ATTRS = {"speed_jsd": (0, 1, 2, 5, 10, 15, 20, 25, 50),
         "length_jsd": (0, 3, 4, 4.5, 5, 5.5, 6, 8, 12, 25),
         "width_jsd": (0, 1.8, 2, 2.2, 2.4, 2.6, 3, 5)}


def read(path):
    return json.loads(Path(path).read_text())


def evidence(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source(path, names):
    path = ROOT / path
    tree = ast.parse(path.read_text())
    return dict(evidence(path), functions=[{"name": n.name, "start": n.lineno, "end": n.end_lineno}
                for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names])


def histogram_summary(rows):
    hs = [read(Path(r["metrics_path"]).parent / "feature_histograms.json") for r in rows]
    result = {}
    for key, intervals in ATTRS.items():
        h = hs[0][key]
        assert all(x[key]["matched_gt"] == h["matched_gt"] for x in hs)
        q = [v / sum(h["matched_gt"]) for v in h["matched_gt"]]
        probabilities = [[v / sum(x[key]["generated"]) for v in x[key]["generated"]] for x in hs]
        p = [statistics.mean(xs) for xs in zip(*probabilities)]
        bounds = [h["minimum"] + i * h["bin_width"] for i in range(len(q) + 1)]
        # Confirm exported histograms reproduce the actual native scores.
        scores, contributions = [], []
        for g in probabilities:
            c = [(.5 * a * math.log(2 * a / (a + b)) if a > 0 else 0.) +
                 (.5 * b * math.log(2 * b / (a + b)) if b > 0 else 0.) for a, b in zip(g, q)]
            scores.append(sum(c) * 100)
            contributions.append([v * 100 for v in c])
        assert all(math.isclose(x, r[key], rel_tol=1e-10, abs_tol=1e-12) for x, r in zip(scores, rows))
        def band(lo, hi):
            i = next(i for i, v in enumerate(bounds) if math.isclose(v, lo, rel_tol=0, abs_tol=1e-9))
            j = next(i for i, v in enumerate(bounds) if math.isclose(v, hi, rel_tol=0, abs_tol=1e-9))
            return {"lower": lo, "upper": hi, "generated_fraction": sum(p[i:j]),
                    "GT_fraction": sum(q[i:j]), "difference_percentage_points": 100 * (sum(p[i:j]) - sum(q[i:j]))}
        cmean = [statistics.mean(xs) for xs in zip(*contributions)]
        result[key] = {"unit": "m/s" if key == "speed_jsd" else "m", "bin_width": h["bin_width"],
                       "native_JSD_GEN_mean": statistics.mean(scores),
                       "bands": [band(lo, hi) for lo, hi in zip(intervals[:-1], intervals[1:])],
                       "tails": [band(lo, h["maximum"]) for lo in intervals[-4:-1]],
                       "largest_JSD_bins": [{"lower": float(bounds[i]), "upper": float(bounds[i + 1]),
                           "generated_fraction": float(p[i]), "GT_fraction": float(q[i]),
                           "native_JSD_contribution": float(cmean[i])}
                           for i in sorted(range(len(cmean)), key=lambda i: cmean[i], reverse=True)[:8]],
                       "mean_GEN_probabilities": p, "GT_probabilities": q}
    return result


def cache_values(names, directory, *, cached):
    global np
    import numpy as np
    import torch
    parts = {k: [] for k in ATTRS}
    stats = dict(scenes=0, agents=0, vehicles=0, invalid_size_fields=0,
                 nonfinite_velocity_agents=0, negative_physical_speed_agents=0,
                 cached_initial_scenes=0, raw_scenes=0, metadata_timestep_missing_scenes=0)
    for idx, name in enumerate(names):
        data = torch.load(directory / name, map_location="cpu", weights_only=False)
        if cached:
            a = data["tokenized_agent"]
            velocity = a["local_vel"]
            stats["cached_initial_scenes"] += 1
            stats["metadata_timestep_missing_scenes"] += int("scene_timestep" not in data)
        else:
            a = data["agent"]
            t = int(data["scene_timestep"])
            velocity = a["velocity"][:, t]
            stats["raw_scenes"] += 1
        sizes = a.get("shape", a.get("initial_shape"))[:, :2]
        speed = torch.linalg.vector_norm(velocity.double(), dim=-1)
        valid_vehicle = a["type"].long() == 0
        stats["scenes"] += 1
        stats["agents"] += len(speed)
        stats["vehicles"] += int(valid_vehicle.sum())
        stats["invalid_size_fields"] += int((~torch.isfinite(sizes) | (sizes <= 0)).sum())
        stats["nonfinite_velocity_agents"] += int((~torch.isfinite(velocity).all(-1)).sum())
        stats["negative_physical_speed_agents"] += int((speed < 0).sum())
        parts["speed_jsd"].append(speed[valid_vehicle].numpy())
        parts["length_jsd"].append(sizes[valid_vehicle, 0].double().numpy())
        parts["width_jsd"].append(sizes[valid_vehicle, 1].double().numpy())
        if (idx + 1) % 1024 == 0:
            print(f"CPU cache {directory.name} {idx + 1}/{len(names)}", flush=True)
    return {k: np.concatenate(v) for k, v in parts.items()}, stats


def summarize_cache(values, template):
    result = {}
    for key, xs in values.items():
        h = template[key]
        bins = h["minimum"] + np.arange(len(h["matched_gt"]) + 1) * h["bin_width"]
        counts = np.histogram(np.clip(xs, h["minimum"], h["maximum"]), bins=bins)[0]
        result[key] = {"counts": counts.tolist(), "total": int(counts.sum()),
                       "probabilities": (counts / counts.sum()).tolist(),
                       "mean_physical_feature": float(np.mean(xs)),
                       "exact_zero_fraction": float(np.mean(xs == 0)),
                       "minimum": float(np.min(xs)), "maximum": float(np.max(xs))}
    return result


def toy_sampler(steps):
    latent = 1.
    times = [1. - i / steps for i in range(steps + 1)]
    for t, next_t in zip(times[:-1], times[1:]):
        latent += (next_t - t) * latent / max(t, .05)
    return {"steps": steps, "constant_clean_x0": 0., "source_x1": 1., "Euclidean_terminal_latent": latent,
            "terminal_equals_clean_x0": abs(latent) < 1e-12}


def expected_time_weight(time_sample_power, loss_power, eps=.05):
    """Exact expectation for t=U**a, using existing Uniform(t) normalizer."""
    k = 1 / time_sample_power
    exponent = k - loss_power
    integral = eps ** exponent + (k * math.log(1 / eps) if abs(exponent) < 1e-12
               else k / exponent * (1 - eps ** exponent))
    old_mean = (1. if loss_power == 0 else 1 - math.log(eps) if loss_power == 1
                else eps ** (1 - loss_power) + (eps ** (1 - loss_power) - 1) / (loss_power - 1))
    return integral / old_mean


def main():
    # Keep YAML analysis and tensor deserialization in separate processes.
    # gc.disable alone did not prevent the observed native fault.
    gc.disable()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("frozen_analyzer", ROOT / "tools/analyze_frozen_map_search.py")
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    roots, originals, final = [], {}, []
    for seed in (817, 1817):
        for arm in ARMS:
            run = RUNS / (("long_seed" if seed == 817 else "confirm_seed") + str(seed) + "_" + arm)
            stage = read(OLD / "stage_executions" / (run.name + ".json"))
            assert stage["status"] == "success" and stage["returncode"] == 0
            rows = read(run / "summary.json")
            assert len(rows) == (8 if seed == 817 else 6)
            originals[(seed, arm)] = sorted([r for r in rows if r["budget"] == FINAL_STEPS[arm]], key=lambda r: r["eval_seed"])
            assert len(originals[(seed, arm)]) == 2
            roots.append(run)
    observations, warnings, _ = analyzer.discover(roots)
    assert not warnings, warnings
    final = [r for r in observations if r["budget"] == FINAL_STEPS[r["variant"]]]
    assert len(final) == 8 and len({r["cohort_sha256"] for r in final}) == 1
    assert {r["seen_training_examples"] for r in final} == {2097048}
    comparisons = {str(seed): analyzer.compare([r for r in final if r["seed"] == seed and r["variant"] == ARMS[1]],
                                               [r for r in final if r["seed"] == seed and r["variant"] == ARMS[0]], external=True)
                   for seed in (817, 1817)}
    comparisons["817_and_1817"] = analyzer.compare([r for r in final if r["variant"] == ARMS[1]],
                                                  [r for r in final if r["variant"] == ARMS[0]], external=True)
    distributions = {str(seed): {arm: histogram_summary(originals[(seed, arm)]) for arm in ARMS} for seed in (817, 1817)}
    reference_hist = read(Path(originals[(817, ARMS[0])][0]["metrics_path"]).parent / "feature_histograms.json")
    train_manifest = ROOT / "logs/component_loss_scratch/calibration/calibration_files.pkl"
    train_names = pickle.load(train_manifest.open("rb"))["files"]
    val_manifest = OLD / "confirmation4096_files.pkl"
    val_names = pickle.load(val_manifest.open("rb"))["files"]
    cache_path = OUT / "isolated_cache_scan.json"
    cache_python = (sys.argv[sys.argv.index("--cache-python") + 1] if "--cache-python" in sys.argv else sys.executable)
    child = subprocess.run([cache_python, "-X", "faulthandler", str(Path(__file__).resolve()), "--cache-only", str(cache_path)],
                           cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True)
    cache = read(cache_path)
    train_stats, val_stats = cache["TRAIN_stats"], cache["VAL_stats"]
    train_hist, val_hist = cache["TRAIN_histograms"], cache["VAL_histograms"]
    alignment = {key: {"native_raw_cache_vehicle_count": val_hist[key]["total"],
                       "matched_GT_histogram_count": sum(reference_hist[key]["matched_gt"]),
                       "bins_with_different_counts": [i for i, (a, b) in enumerate(zip(val_hist[key]["counts"], reference_hist[key]["matched_gt"])) if a != b],
                       "absolute_bin_count_difference_sum": sum(abs(a - b) for a, b in zip(val_hist[key]["counts"], reference_hist[key]["matched_gt"]))}
                 for key in ATTRS}
    calibration = read(ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json")
    assert train_stats["scenes"] == calibration["scenes"] == 8192 and train_stats["agents"] == calibration["agents"] == 76027
    mu, sigma = calibration["normal_mean"][0][6], calibration["normal_scale"][0][6]
    negative_source_probability = .5 * math.erfc(mu / sigma / math.sqrt(2))
    source_evidence = [source("src/smart/tokens/token_processor.py", {"get_init", "_load_cached_initial_agent", "process_data"}),
                       source("src/smart/diffusion/denoiser.py", {"get_input", "get_output", "_original_state_embedding", "_parameterize_speed_x0"}),
                       source("src/smart/diffusion/scale_flow.py", {"_sample_noise", "_sample_time", "_prediction_velocity", "_component_supervised_loss", "sample"}),
                       source("src/smart/diffusion/diffusion_utils.py", {"get_closest_sum_idx_fast", "_group_labels"}),
                       source("src/smart/diffusion/component_loss.py", {"time_weight", "regression"}),
                       source("src/my_data_process/data2initmap.py", {"process_file"})]
    report = {"status": "ACTUAL_COMPLETED_817_1817_CPU_DATA_METRIC_REVIEW_PROVISIONAL", "created_utc": datetime.now(timezone.utc).isoformat(),
              "GPU_run": False, "cuda_initialized": cache["cuda_initialized"], "model_or_queue_modified": False,
              "audit_runtime_workaround": "YAML summary analysis and native torch cache loading isolated into separate CPU processes/interpreters. Initial sim-Python attempts SIGSEGV during YAML parsing, and one gc.disable attempt faulted during torch load. Pure torch cache child succeeded. This is runtime evidence, not proof of a specific native fault cause. Parent analysis does not import torch.",
              "analysis_python": sys.executable, "cache_python": cache_python,
              "training_seed_roles": {"817": "adaptive discovery and long selection", "1817": "one held-until-frozen-pair confirmation", "2817": "not included; control first attempt failed before final result"},
              "aggregation": "GEN2817/3817 first mean within TRAIN; TRAIN817/1817 equal weights; primary R is ratio of arm metric means averaged over seven metrics, not mean of seed ratios. Two trained seeds are not four independent trained models. Only TRAIN1817 is an independent confirmation at this snapshot.",
              "comparison": comparisons, "native_units": "natural-log dimensionless JSD, near/lat x10 and angle/L/W/speed x100; collision percentage",
              "completed_input_files": [{"TRAIN_seed": seed, "arm": arm,
                  "summary": evidence(RUNS / (("long_seed" if seed == 817 else "confirm_seed") + str(seed) + "_" + arm) / "summary.json"),
                  "final_evaluations": [{"GEN_seed": r["eval_seed"], "metrics": evidence(r["metrics_path"]),
                    "histograms": evidence(Path(r["metrics_path"]).parent / "feature_histograms.json")} for r in originals[(seed, arm)]]}
                  for seed in (817, 1817) for arm in ARMS],
              "distribution_by_TRAIN_and_arm": distributions,
              "cache_inspection": {"TRAIN8192_manifest": evidence(train_manifest), "TRAIN8192_stats": train_stats, "TRAIN8192_vehicle_histograms": train_hist,
                                   "confirmation4096_manifest": evidence(val_manifest), "confirmation4096_stats": val_stats, "native_GT_histogram_alignment": alignment,
                                   "provenance_limit": "TRAIN caches contain only initial agent fields/map tokens and lack raw timestep/track correspondence. Actual scalar targets are norms of cached local_vel. Current data2initmap.py constructor omits scenario_dreamer_init and would use a token-derived get_init branch; it is not evidence that current stored files were created by that exact source version. Full historical raw-frame provenance cannot be recovered from these caches. This review does not silently rewrite TRAIN labels."},
              "sampler_CPU_toy": {"equation": "x_next=x+(next_t-t)*(x-x0)/max(t,.05), ideal constant x0=0", "results": [toy_sampler(n) for n in (20, 40, 80)],
                                  "interpretation": "Current20 steps has no demonstrated endpoint error. More steps alone changes the near-zero clipped Euclidean equation; even perfect constant x0 leaves source residual for40/80. Keep this overnight screen at20, as authorized by root; a future opt-in terminal-x0/true-time denominator study is a separate generation intervention, not a metric change."},
              "negative_scalar_speed_source_probability": negative_source_probability,
              "negative_source_is_not_measured_negative_final_output_fraction": True,
              "nonuniform_time_sampling_effect": {"density": "t=U**a has density (1/a)*t**(1/a-1)",
                  "normalization": "ComponentLoss still divides by its Uniform(t) analytic expectation, not this changed density.",
                  "expected_normalized_weights": {str(a): {str(p): expected_time_weight(a, p) for p in range(4)} for a in (1., .5, 2.)},
                  "interpretation": "high_noise a=.5 simultaneously changes time density and effective average component weighting; shape p2 is reduced more than speed p1. This is an intentional training intervention, not an evaluator change or a pure equal-average-weight density ablation."},
              "metric_definitions": read(OLD / "long_baseline_distribution_review.json")["metric_population_and_definitions"],
              "source_evidence": source_evidence,
              "default_compatibility_audit": evidence(OUT / "default_compatibility_cpu_audit.json"),
              "matching_semantics_CPU_audit": evidence(OUT / "matching_semantics_cpu_audit.json"),
              "failed2817_cache_neighborhood_audit": evidence(OUT / "seed2817_failed_batch_files_cpu_audit.json"),
              "findings": [
                  "The frozen combo improves L/W and collision on both completed TRAIN seeds, but speed changes reverse: TRAIN817+10.97%, TRAIN1817+36.39%; the latter violates the predeclared15% speed gate. A lower seven-metric average does not certify eligibility.",
                  "Use real speed bins, not an assumed missing-stationary mode. The817 final low-speed[0,1)m/s band is excessive, while high-speed tails are underrepresented. The1817 bands below provide an independent diagnostic rather than extrapolating817's pattern.",
                  "Magnitude mismatch is not an unsupervised target here: the7D scalar speed is already directly supervised by normalized MSE. The vector magnitude auxiliary is deliberately disabled/rejected for scalar speed. Adding the same scalar loss again only changes its effective weight.",
                  "Loss p is inverse time: clamp(t,.05)^(-p) divided by the uniform-time analytic mean. Higher p emphasizes nearly clean t~0 rather than noisy t~1. Scalar-speed weight4 does not fix source matching or make high-noise transport training distribution-aware.",
                  "The frozen combo/control comparison changes batch/update count, ego angle context encoding, shape time powers and speed coefficient together. Equal exposure does not isolate speed weight or context as a causal factor; adaptive EMA effective decay also differs at8192 versus4096 updates. Future ablations must separate these changes.",
                  "Scalar-speed decoding along generated heading preserves the norm used by speed JSD. A bad generated heading does not directly alter the decoded speed norm, though shared training/features can correlate their errors.",
                  "The x0 MSE conditional mean is theoretically compatible with the correct probability-flow velocity for a linear interpolant under exact learning/integration. Therefore MSE alone does not prove unavoidable marginal collapse. Finite shared capacity, low-time emphasis, discrete sampling, source coupling and generated type selection are plausible interacting causes, not measured causal attribution.",
                  "Hungarian matching permutes complete physical source rows within each scene and GT-type group; ego source stays paired to ego. Targets/type labels are not permuted. Normalized seven-coordinate squared costs share an assignment, so position/heading/L/W can overrule speed ordering. Loss coefficient changes do not change this default coupling.",
                  "Generated class0 rows define every attribute metric population. Classification error can alter speed/shape marginals without any direct state-coordinate bug. Keep vehicle coverage and per-GT-type supplementary diagnostics alongside the unchanged native metrics.",
                  "Linear scalar-speed output is clamped only at final decoding; Gaussian endpoint has about21.6% negative physical source mass from shared TRAINmean/std. This is allowed in the internal interpolant, not evidence that21.6% generated vehicles are stationary. Actual histograms cannot identify preclamp negatives or exactzero mass.",
                  "Rare large L/W tails remain sparse even when global shape JSD improves; fixed GT ego dimensions are included in native histograms. Shape JSD improvement can accompany better dominant bins while rarer large vehicles worsen.",
                  "The native4096 validation cache/reference alignment is checked here in physical speed/L/W units. No metric/filter definition is altered. Initial-scene velocity is continuous, not decoded motion-token velocity.",
              ],
              "recommended_experiments": [
                  {"priority": 1, "design": "Keep the established shape recipe, vary speed time_power0/1 and weight2/4 separately; inspect final equal-exposure speed bins and gate before combining.", "reason": "Avoid assuming higher reconstruction weight transports high-noise speed marginals better; isolate demonstrated unstable speed intervention."},
                  {"priority": 2, "design": "TRAIN-normalize continuous L/W/speed input; opt-in separate speed head, then detached head as a distinct ablation.", "reason": "Loss normalization does not normalize raw attributes projected into shared attention. These options address finite capacity/gradient competition while preserving physical outputs and evaluator."},
                  {"priority": 3, "design": "Opt-in minibatch TRAIN-only attribute distribution loss on high-noise predictions, by true type, with small weights; track pointwise/transport losses independently.", "reason": "Directly penalize narrowed prediction marginals where clean reconstruction is ambiguous; it changes training objective and is an experiment, not a guaranteed JSD improvement or evaluator adjustment."},
                  {"priority": 4, "design": "Increase speed cost in source Hungarian matching, or reduce heading cost, one intervention at a time.", "reason": "Improve source coupling without changing endpoint source marginal or GT labels; geometric metrics/gates must stay eligible."},
                  {"priority": 5, "design": "Opt-in positive clean-speed softplus and time-residual x0 parameterization; retain standard20-step generation during screening.", "reason": "Change learnable speed coordinates across complete loss/velocity/output path. Softplus cannot represent exact0 with finite logits; test low-speed bin excess rather than assuming positivity cures it."},
                  {"priority": 6, "design": "Log-shape with the same calibration TRAIN filenames/order and physical evaluator; assess >8mL/>2.6mW tails separately.", "reason": "Positivity/relative-error representation can help optimization but may trade sparse tails against bulk. Compare only same physical evaluation cohort."},
                  {"priority": 7, "design": "Future CPU-verified opt-in terminal-x0/true-time sampler study, separate from this fixed20-step overnight screen.", "reason": "More sampling steps alone changes near-zero Euclidean dynamics because of t_eps; must isolate numerical intervention."},
              ],
              "limits": ["This is a provisional817+1817 result, not the final three-TRAIN conclusion or a128epoch prediction.",
                         "TRAIN8192 cache scan verifies the calibration sample, not all493542 training caches or historical raw-frame origin.",
                         "The4096 confirmation scenes were excluded from old selection manifests, but the encompassing50k pool had already been distribution-audited; do not call it strictly untouched data.",
                         "Independent CPU loading of8704 failed-neighborhood files does not prove GC caused the priorSIGSEGV.",
                         "Speed bins have1m/s resolution; do not infer exact stationary or preclamp-negative fractions from generated histograms."]}
    assert not report["cuda_initialized"]
    (OUT / "data_metric_review.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# CPU 数据与指标排查", "", "本报告仅使用已完成的 TRAIN817 与 TRAIN1817，两者每组先平均两个 GEN。817 参与过自适应选择，1817 是一个独立确认训练种子；2817 尚无可用最终结果。", "",
             "| 指标（原生尺度） | control 均值 | combo 均值 | 相对变化 |", "|---|---:|---:|---:|"]
    c = comparisons["817_and_1817"]
    for key in METRICS:
        v = c["components"][key]
        # Analyzer stores the canonical means in the group statistics.
        b = statistics.mean([statistics.mean([r[key] for r in originals[(s, ARMS[0])]]) for s in (817, 1817)])
        a = statistics.mean([statistics.mean([r[key] for r in originals[(s, ARMS[1])]]) for s in (817, 1817)])
        lines.append(f"| {key} | {b:.6f} | {a:.6f} | {v['relative_change_percent_of_arm_means']:+.2f}% |")
    lines += ["", "原生 JSD 是无量纲自然对数散度：near/lat ×10，其余 JSD ×100；collision 是百分比。R 使用各指标候选均值/control均值后平均。尺寸和 collision 改善，但速度退化；不能用总体平均掩盖速度门槛失败。", "",
              "1817 的真实速度 bin 显示：[0,1)m/s 占比 control44.1993%、combo44.4775%、GT41.8378%；>=15m/s 占比8.0704%→7.8450%，GT9.3003%。这个种子也表现为低速段过多、快速尾部偏少；1m/s的bin无法证明 exact stationary 或 preclamp 负数比例。", "",
              "4096个native验证缓存按其真实scene_timestep直接取velocity与shape，29698辆车辆的speed/L/W每个bin均与matchedGT完全相同，绝对count差为0。没有发现这些属性的评价时刻、单位或标签错位。", "",
              "Combo/control 同时改变了batch/update数、ego角度context、shape时间权重和speed系数；同exposure不能证明其中一项就是原因。8192与4096更新下adaptive EMA有效decay也不同。", "",
              "速度已经是直接监督的标量，不缺 magnitude loss。时间权重 p 越大越偏向接近数据端的低 t；它不会自动提高高噪声端的分布恢复能力。MSE 在理想概率流下可以恢复分布，所以不能简单断言 MSE 本身必然收缩分布。", "",
              "高噪声抽样t=U^.5仍沿用Uniform(t)的时间权重归一化常数，p1平均有效权重约0.488、p2约0.179。因此它同时改变抽样密度与各分量平均权重，不能当作仅改抽t密度的等权试验。", "",
              "建议按既定队列隔离试验 speed 时间权重、输入属性归一化、独立速度头、高噪声分布辅助项与匹配代价。保持20步和原生评价定义；每次继续检查道路覆盖、车辆数量以及 L/W/speed 的15%门槛。", "",
              "缓存扫描检查了 TRAIN 校准8192场景和4096确认场景。TRAIN缓存只保留初始状态，缺原始时刻/track对应关系，现有生成脚本版本不足以证明历史缓存来源；报告保留这个限制，不改训练标签。具体直方图、source函数/SHA、缓存与GT对齐结果见JSON。", "",
              "20步旧采样的欧氏末端没有演示出的误差；若以后单独增加到40/80步，固定t_eps=.05会改变末端动力学，应另设terminal-x0或真t分母实验。当前筛选不混入这一改动。", "",
              "旧配置在新source上的CPU构造通过同seed G1完整tensor hash核验；异常附近8704缓存可独立正常加载，但这不能证明GC是SIGSEGV根因。", ""]
    (OUT / "data_metric_review.md").write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "R": c["R_ratio_of_arm_means"], "ratios": {k: v["ratio_of_arm_means"] for k, v in c["components"].items()},
                      "gates": c["gate_checks"], "alignment": alignment, "TRAIN_stats": train_stats, "VAL_stats": val_stats,
                      "output": str(OUT / "data_metric_review.json")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--cache-only":
        gc.disable()
        train = pickle.load((ROOT / "logs/component_loss_scratch/calibration/calibration_files.pkl").open("rb"))["files"]
        val = pickle.load((OLD / "confirmation4096_files.pkl").open("rb"))["files"]
        rows = read(RUNS / "long_seed817_new_control/summary.json")
        row = next(r for r in rows if r["budget"] == 4096 and r["eval_seed"] == 2817)
        h = read(Path(row["metrics_path"]).parent / "feature_histograms.json")
        tv, ts = cache_values(train, ROOT / "src/waymo_data/full/training_map2_sd", cached=True)
        vv, vs = cache_values(val, ROOT / "src/waymo_data/full/scenario_dreamer_val", cached=False)
        import torch
        Path(sys.argv[2]).write_text(json.dumps({"TRAIN_stats": ts, "VAL_stats": vs,
            "TRAIN_histograms": summarize_cache(tv, h), "VAL_histograms": summarize_cache(vv, h),
            "cuda_initialized": torch.cuda.is_initialized()}, indent=2) + "\n")
    else:
        main()
