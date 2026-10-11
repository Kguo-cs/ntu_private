"""Read-only CPU eligibility audit for an optional map-policy comparison.

Does not load tensors, run evaluators, or change the serialized GPU queue.
The comparison is descriptive and cannot change the new search selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shlex

import yaml

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011"
OUT = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
OLD_RUNS = ROOT / "logs/component_loss_scratch"
NEW_RUNS = ROOT / "logs/component_loss_frozen_map"
MAP = ROOT / "src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt"
SEEDS = (817, 1817, 2817)
ARMS = ("warm_winner", "geo_mse32_col20_heading_l1_shape2")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def canonical_recipe(value):
    value = copy.deepcopy(value)
    # This optional auxiliary term is explicitly disabled in both recipes.
    optional = value.get("components", {}).get("speed_magnitude")
    if optional is not None and optional.get("weight") == 0:
        value["components"].pop("speed_magnitude")
    return value


def shell_command(command):
    chunks = [shlex.join(command[:2])]
    chunks += [shlex.join(command[index:index + 2]) for index in range(2, len(command), 2)]
    return (" " + chr(92) + "\n  ").join(chunks)


def leaf_differences(left, right, prefix=""):
    output = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                output.append({"field": prefix + key, "old": left.get(key), "new": right.get(key)})
            else:
                output += leaf_differences(left[key], right[key], prefix + key + ".")
    elif left != right:
        output.append({"field": prefix.rstrip("."), "old": left, "new": right})
    return output


def metadata(folder):
    config = yaml.safe_load((folder / "config.yaml").read_text())
    init = read(folder / "initialization.json")
    protocol = read(folder.parent / "protocol.json")
    mc = config["model"]["model_config"]
    decoder = mc["decoder"]["init_diffusion"]
    checkpoint = folder / "step_2048.ckpt"
    observations = read(folder.parent / "summary.json") if (folder.parent / "summary.json").is_file() else []
    endpoint = [row for row in observations if row["variant"] == folder.name and row["budget"] == 2048 and row["evaluation_weights"] == "ema"]
    return {
        "folder": relative(folder), "checkpoint2048": relative(checkpoint), "checkpoint2048_exists": checkpoint.is_file(),
        "seed": init["seed"], "train_batch_size": init["batch_size"], "learning_rate": init["lr"],
        "train_order_sha256": init["train_order_sha256"], "train_manifest_sha256": init["train_manifest_sha256"],
        "calibration_sha256": init["calibration_sha256"], "initial_generator_sha256": init["initial_generator_sha256"],
        "initial_map_sha256": init["initial_map_sha256"], "initialization_scope": init["initialization_scope"],
        "map_checkpoint_sha256": init.get("map_checkpoint_sha256"),
        "component_loss": decoder["component_loss"], "canonical_component_loss": canonical_recipe(decoder["component_loss"]),
        "optimizer": mc.get("optimizer"), "optimizer_settings": {key: value for key, value in mc.items() if key.startswith("lr") or "decay" in key},
        "clip": config["trainer"]["gradient_clip_val"], "precision": protocol["precision"],
        "float32_matmul_precision": protocol["float32_matmul_precision"],
        "map_crop": protocol["map_crop"], "map_half_extent": protocol["map_half_extent"],
        "reference_mode": protocol["reference_mode"], "ema_decay": decoder["ema_decay"], "sep_map": mc["decoder"]["sep_map"],
        "num_samples_on_saved_protocol": protocol["scenes"], "eval_manifest_sha256": protocol["eval_manifest_sha256"],
        "actual2048_seen_training_examples": sorted({row["seen_training_examples"] for row in endpoint}),
        "checkpoint2048_ema_metric_rows_in_training_root": len(endpoint),
        "config_sha256": sha(folder / "config.yaml"), "initialization_sha256": sha(folder / "initialization.json"),
        "raw_model_config": mc,
    }


def main():
    current = NEW_RUNS / "long_seed817_new_control/new_control"
    if not (current / "initialization.json").is_file():
        current = NEW_RUNS / "phase1/new_control"
    frozen = metadata(current)
    old_records = [metadata(OLD_RUNS / f"batch512_seed{seed}" / arm) for seed in SEEDS for arm in ARMS]
    model_diffs = {}
    semantic_diffs = {}
    for old in old_records:
        key = str(old["seed"]) + "/" + Path(old["folder"]).name
        differences = leaf_differences(old["raw_model_config"], frozen["raw_model_config"], "model_config.")
        model_diffs[key] = differences
        left, right = copy.deepcopy(old["raw_model_config"]), copy.deepcopy(frozen["raw_model_config"])
        for value in (left, right):
            value.pop("initial_map_checkpoint", None)
            value.pop("finetune", None)
            value["decoder"]["init_diffusion"]["component_loss"] = canonical_recipe(value["decoder"]["init_diffusion"]["component_loss"])
        semantic_diffs[key] = leaf_differences(left, right, "model_config.")
    geo817 = next(row for row in old_records if row["seed"] == 817 and Path(row["folder"]).name == ARMS[1])
    checks = {key: geo817[key] == frozen[key] for key in (
        "seed", "train_batch_size", "learning_rate", "train_order_sha256", "train_manifest_sha256",
        "calibration_sha256", "initial_generator_sha256", "canonical_component_loss", "optimizer",
        "optimizer_settings", "clip", "precision", "float32_matmul_precision", "map_crop",
        "map_half_extent", "reference_mode", "ema_decay", "sep_map")}
    checks["model_config_identical_except_intended_map_policy_and_disabled_optional_term"] = not semantic_diffs["817/" + ARMS[1]]
    old_metric = read(OLD_RUNS / "heldout_seed817" / ARMS[1] / "eval_step_2048/ema/sd_agent_metrics.json")
    new_metric = read(NEW_RUNS / "phase1/new_control/eval_step_512/ema/sd_agent_metrics.json")
    metric_checks = {key: old_metric[key] == new_metric[key] for key in (
        "metric_scope", "generation_task", "map_source", "reference_mode", "official_source_sha256", "reference_source_manifest_sha256")}
    metric_checks["conditioning_identical"] = old_metric["decoder"]["conditioning"] == new_metric["decoder"]["conditioning"]
    old_manifest = OLD / "heldout4096_files.json"
    new_manifest = OUT / "confirmation4096_files.json"
    old_files, new_files = read(old_manifest)["files"], read(new_manifest)["files"]
    old_manifest_lines = hashlib.sha256(("\n".join(old_files) + "\n").encode()).hexdigest()
    new_manifest_lines = hashlib.sha256(("\n".join(new_files) + "\n").encode()).hexdigest()
    expected_future = NEW_RUNS / "long_seed817_new_control/new_control/step_2048.ckpt"
    # These files are templates only.  They are never sent to the GPU queue.
    eval_mapping = OUT / "optional_map_policy_oldheldout_seed817_checkpoints.json"
    eval_mapping.write_text(json.dumps({"frozen_map_new_control": str(expected_future)}, indent=2) + "\n")
    fresh_replay_mapping = OUT / "optional_current_code_random_map_seed817_checkpoints.json"
    fresh_replay_mapping.write_text(json.dumps({"random_map_geo_recipe": str(ROOT / geo817["checkpoint2048"])}, indent=2) + "\n")
    common = ["/home/ke/miniconda3/envs/sim/bin/python", "tools/scratch_component_loss_sweep.py",
              "--steps", "1", "--checkpoint-steps", "", "--variants", "", "--seed", "817",
              "--batch-size", "512", "--scenes", "4096", "--eval-batch-size", "128", "--workers", "0",
              "--eval-seed", "2817", "--extra-eval-seed", "3817", "--evaluation-weights", "ema",
              "--normalizer-json", relative(OLD / "train_normalizer.json"), "--eval-manifest", relative(old_manifest)]
    frozen_command = common + ["--map-checkpoint", str(MAP), "--evaluation-checkpoints", relative(eval_mapping),
                              "--output", "logs/component_loss_optional_map_policy/seed817_frozen_on_oldheldout"]
    replay_command = common + ["--evaluation-checkpoints", relative(fresh_replay_mapping),
                              "--output", "logs/component_loss_optional_map_policy/seed817_random_geo_current_code"]
    for item in old_records + [frozen]:
        item.pop("raw_model_config")
    audit = {
        "status": "ELIGIBLE_CONFIG_AND_METADATA_PENDING_FROZEN_2048_CHECKPOINT",
        "no_gpu_or_modelweights_loaded": True, "comparison_selection_role": "NONE; optional descriptive map-policy audit only",
        "old_random_map_models": old_records, "current_frozen_map_reference_metadata": frozen,
        "old_new_control_same_loss_pair": {"random": geo817["folder"], "frozen_expected": relative(expected_future.parent),
                                           "checks": checks, "all_checks_pass": all(checks.values())},
        "metric_and_conditioning_checks": metric_checks, "all_metric_checks_pass": all(metric_checks.values()),
        "conditioning": old_metric["decoder"]["conditioning"], "metric_source_hashes": old_metric["official_source_sha256"],
        "raw_model_config_differences": model_diffs, "semantic_differences_excluding_intended_map_policy": semantic_diffs,
        "old_heldout_manifest": {"path": relative(old_manifest), "json_sha256": sha(old_manifest), "ordered_lines_sha256": old_manifest_lines, "files": len(old_files)},
        "new_confirmation_manifest": {"path": relative(new_manifest), "json_sha256": sha(new_manifest), "ordered_lines_sha256": new_manifest_lines, "files": len(new_files)},
        "manifest_file_overlap": len(set(old_files) & set(new_files)),
        "proposed_generation_seeds": [2817, 3817], "proposed_training_seed": 817, "proposed_weights": "ema",
        "proposed_budget": 2048, "proposed_actual_exposure_B512": 1048524,
        "frozen_checkpoint_expected_path": str(expected_future), "frozen_checkpoint_exists_now": expected_future.is_file(),
        "commands_templates_not_executed": {"frozen2048_on_oldheldout": frozen_command, "recommended_random_geo2048_current_code_replay": replay_command},
        "limitations_and_required_later_checks": [
            "Current frozen new_control is only trained to512 at this audit. Future long seed817 baseline must retain the exact loss/model/optimizer/normalizer/G1init/train-order metadata checked here before evaluating its2048 checkpoint.",
            "All2048 checkpoint metadata and EMA updates/normalizer must be verified before use; this audit deliberately loads no checkpoint tensors.",
            "Two map policies differ: random map trained jointly vs pretrained map kept fixed. This does not isolate pretraining from freezing, and global clip includes map gradients only for the joint-map policy.",
            "Warm_winner differs in position loss/weight, heading loss/time power, length/width weights/time power and collision weight. It is a historical context arm, not a clean map-only counterpart to frozen new_control.",
            "Old heldout4096 was used for previous selection; the optional comparison is descriptive validation, not untouched final testing, and cannot alter this search selection.",
            "Old reports match official helper hashes and all inspected config semantics, but there is no complete historical hash of every model/training source file. Replaying the random geo2048 checkpoint on current code provides the cleanest numeric code-parity check.",
            "A seed817 comparison has independent TRAIN N=1 even with two GEN seeds. N=3 requires actual frozen new_control2048 metadata checks and old-heldout evaluation for seeds1817/2817 too.",
            "Previously reported old heldout metrics cannot be directly compared with the new confirmation4096 metrics because their scene manifests are disjoint.",
        ],
    }
    assert all(checks.values()) and all(metric_checks.values())
    (OUT / "optional_random_vs_frozen_map_comparison_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = ["# Optional random-map 与 frozen-map 比较：CPU 资格审计", "",
             "状态：配置/metadata 对齐，等待 frozen new_control 的2048 checkpoint；未运行 GPU、未加载 model weights、未改变 queue。该比较不能参与本轮最终配方选择。", "",
             "干净的同 loss 对应项是旧 `geo_mse32_col20_heading_l1_shape2` 与新 `new_control`。seed817 的 G1 初始 hash、TRAIN顺序、TRAIN normalizer、B512、LR5e-4、clip0.5、EMA0.9999、square50、模型参数和 input conditioning 全部对齐。配置中仅有意 map policy（finetune/fixed source）及明确关闭的 speed_magnitude term 不同。", "",
             "旧 `warm_winner` 可作历史参照，但 loss 不同，不能用它声称 map-only ablation。", "",
             "| seed | random geo folder | random control folder | G1 init hash |", "|---:|---|---|---|"]
    for seed in SEEDS:
        geo = next(item for item in old_records if item["seed"] == seed and Path(item["folder"]).name == ARMS[1])
        warm = next(item for item in old_records if item["seed"] == seed and Path(item["folder"]).name == ARMS[0])
        lines.append(f"| {seed} | `{geo['folder']}` | `{warm['folder']}` | `{geo['initial_generator_sha256']}` |")
    lines += ["", "旧 heldout4096 与新 confirmation4096 manifest 不同且文件交集0，不能直接比较两份已有 aggregate。建议只在2048 checkpoint检查通过后，将 frozen baseline **额外评价旧 heldout4096**；GEN2817/3817、EMA、matched reference。", "",
              "这比较的是两种 map 训练策略：random map联合训练 vs pretrained map固定。global clip的参数集合也随策略变化；不能把差异只归因于预训练。", "",
              "历史 metric helper hashes、地图来源和所有 inspected settings 一致，但缺少完整模型源码历史hash。更严格的方法是同时用当前代码重评旧 random geo checkpoint，确认其输出与旧结果一致或报告变化。", "",
              "以下仅是未执行的 command skeleton；路径文件预先写出供审阅。`--steps 1` 不改变实际checkpoint2048：本命令只有 evaluation-checkpoints，没有训练arms。", "",
              "```bash", shell_command(frozen_command), "```", "",
              "建议的 current-code replay：", "", "```bash", shell_command(replay_command), "```", "",
              "完整资格检查、hash、模型差异及限制见 [JSON](optional_random_vs_frozen_map_comparison_audit.json)。seed817只有 TRAIN N=1；若要3seed，必须再检查并评价 frozen1817/2817 的2048 checkpoint，同一旧heldoutmanifest。", ""]
    (OUT / "optional_random_vs_frozen_map_comparison_audit.md").write_text("\n".join(lines))
    print(json.dumps({"config_checks_pass": all(checks.values()), "metric_checks_pass": all(metric_checks.values()),
                      "old_checkpoints": len(old_records), "frozen2048_exists": expected_future.is_file(),
                      "manifest_overlap": audit["manifest_file_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
