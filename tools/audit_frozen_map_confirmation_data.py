"""Freeze a disjoint confirmation manifest and audit actual cache provenance.

This is a CPU-only audit for the fixed map encoder search.  The confirmation
pool remains part of the previously evaluated 50k set; it is not a pristine
unseen test set.  Extra original AE cache filenames are inventoried separately.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/experiments/init_diffusion_scratch_loss_search_20261011"
OUT = ROOT / "docs/experiments/init_diffusion_frozen_map_search_20261011"
CACHE = ROOT / "src/waymo_data/full/scenario_dreamer_val"
TRAIN_MANIFEST = ROOT / "src/waymo_data/full/training_map2_sd_files.pkl"
PATTERN = re.compile(
    r"(?:(?P<eval_index>\d{5})_)?(?P<tfrecord>(?P<split>testing|validation|training)"
    r"\.tfrecord-\d+-of-\d+)_(?P<record>\d+)_(?P<lg>\d+)_(?P<timestep>\d+)\.(?:pt|pkl)"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parts(name):
    match = PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"Unrecognized cache filename: {name}")
    result = match.groupdict()
    for key in ("eval_index", "record", "lg", "timestep"):
        if result[key] is not None:
            result[key] = int(result[key])
    result["key"] = (result["tfrecord"], result["record"])
    return result


def hex_id(value):
    return f"{int(str(value), 16):016x}"


def actual_metadata(files):
    ids, mismatch, split_counts = set(), [], {}
    for name in files:
        expected = parts(name)
        data = torch.load(CACHE / name, map_location="cpu", weights_only=False)
        checks = {
            "source_tfrecord": data["source_tfrecord"] == expected["tfrecord"],
            "source_record_index": int(data["source_record_index"]) == expected["record"],
            "source_split": data["source_split"] == expected["split"],
            "scene_timestep": int(data["scene_timestep"]) == expected["timestep"],
            "scenario_dreamer_lg_type": int(data["scenario_dreamer_lg_type"]) == expected["lg"],
            "scenario_dreamer_eval_index": int(data["scenario_dreamer_eval_index"]) == expected["eval_index"],
            "scenario_dreamer_cache_file": data["scenario_dreamer_cache_file"] == name[6:-3] + ".pkl",
            "embedded_scene_timestep": int(data["scenario_dreamer"]["scene_timestep"]) == expected["timestep"],
            "embedded_lg_type": int(data["scenario_dreamer"]["lg_type"]) == expected["lg"],
        }
        if not all(checks.values()):
            mismatch.append({"file": name, "checks": checks})
        ids.add(str(data["scenario_id"]))
        split_counts[data["source_split"]] = split_counts.get(data["source_split"], 0) + 1
    return ids, mismatch, split_counts


def main():
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        raise RuntimeError("Audit must start without CUDA initialization")
    OUT.mkdir(parents=True, exist_ok=True)
    all_files = sorted(path.name for path in CACHE.glob("*.pt"))
    search = all_files[:1024]
    old_heldout = json.loads((OLD / "heldout4096_files.json").read_text())["files"]
    search_keys = {parts(name)["key"] for name in search}
    old_heldout_keys = {parts(name)["key"] for name in old_heldout}
    excluded_keys = search_keys | old_heldout_keys
    selected, seen, skipped = [], set(excluded_keys), {"excluded_old_raw_key": 0, "duplicate_confirmation_raw_key": 0}
    last_index = None
    for index, name in enumerate(all_files[25000:], 25000):
        key = parts(name)["key"]
        if key in seen:
            skipped["excluded_old_raw_key" if key in excluded_keys else "duplicate_confirmation_raw_key"] += 1
            continue
        selected.append(name)
        seen.add(key)
        if len(selected) == 4096:
            last_index = index
            break
    if len(selected) != 4096:
        raise RuntimeError(f"Only {len(selected)} unique confirmation records")
    payload = {"files": selected}
    manifest = OUT / "confirmation4096_files.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "confirmation4096_files.pkl").open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    (OUT / "confirmation4096_files.txt").write_text("\n".join(selected) + "\n")

    with TRAIN_MANIFEST.open("rb") as handle:
        training = pickle.load(handle)["files"]
    train_ids = set()
    for name in training:
        matched = re.fullmatch(r"(?P<id>[0-9a-fA-F]+)_(?P<lg>\d+)_(?P<timestep>\d+)\.pt", Path(name).name)
        if matched is None:
            raise ValueError(f"Unrecognized TRAIN scenario-id filename: {name}")
        train_ids.add(matched.group("id"))
    search_ids, search_mismatch, _ = actual_metadata(search)
    heldout_ids, heldout_mismatch, _ = actual_metadata(old_heldout)
    selected_ids, selected_mismatch, splits = actual_metadata(selected)
    normalized_train = {hex_id(value) for value in train_ids}
    normalized_selected = {hex_id(value) for value in selected_ids}
    full_keys = {parts(name)["key"] for name in all_files}
    selected_keys = {parts(name)["key"] for name in selected}
    official = [json.loads(line) for line in (ROOT / "src/waymo_data/sd_eval50k_manifest.jsonl").read_text().splitlines() if line]
    official_by_index = {int(row["eval_index"]): row for row in official}
    official_match = all(
        official_by_index[parts(name)["eval_index"]]["cache_file"] == name[6:-3] + ".pkl"
        for name in selected
    )

    # Extra original full-lane caches can support a separately rebuilt future
    # test set.  They currently have no native InitDiffusion evaluation cache.
    inventory = []
    extras = []
    for subdir in ("val", "test"):
        source = ROOT / "src/waymo_data/scenario_dreamer_ae_preprocess_waymo" / subdir
        names = sorted(entry.name for entry in os.scandir(source) if entry.is_file() and entry.name.endswith(".pkl"))
        full_lane = [name for name in names if parts(name)["lg"] == 0]
        full_lane_keys = {parts(name)["key"] for name in full_lane}
        extra_names, extra_keys = [], set()
        for name in full_lane:
            key = parts(name)["key"]
            if key not in full_keys and key not in extra_keys:
                extra_names.append(name)
                extra_keys.add(key)
                extras.append({"directory": str(source.relative_to(ROOT)), "file": name})
        inventory.append({
            "directory": str(source.relative_to(ROOT)), "cache_files": len(names),
            "full_lane_lg0_files": len(full_lane), "full_lane_unique_raw_keys": len(full_lane_keys),
            "full_lane_unique_raw_keys_outside_existing_50k": len(extra_keys),
            "first_outside_existing_50k": extra_names[:3],
        })
    extra_path = OUT / "additional_raw_candidates_outside50k.json"
    extra_path.write_text(json.dumps({"files": extras}, indent=2) + "\n")

    original_checks = []
    for name in selected[::128]:
        original_name = name[6:-3] + ".pkl"
        roots = [ROOT / "src/waymo_data/scenario_dreamer_ae_preprocess_waymo" / split for split in ("test", "val")]
        original_path = next((root / original_name for root in roots if (root / original_name).is_file()), None)
        if original_path is None:
            raise FileNotFoundError(original_name)
        with original_path.open("rb") as handle:
            data = pickle.load(handle)
        expected = parts(name)
        if int(data["scene_timestep"]) != expected["timestep"] or int(data["lg_type"]) != expected["lg"]:
            raise ValueError(f"Original metadata mismatch: {name}")
        original_checks.append({"cache": name, "original": str(original_path.relative_to(ROOT)),
                                "scene_timestep": int(data["scene_timestep"]), "lg_type": int(data["lg_type"]),
                                "original_has_map_id": "map_id" in data})

    overlaps = {
        "confirmation_search_file_overlap": len(set(selected) & set(search)),
        "confirmation_old_heldout_file_overlap": len(set(selected) & set(old_heldout)),
        "confirmation_search_raw_key_overlap": len(selected_keys & search_keys),
        "confirmation_old_heldout_raw_key_overlap": len(selected_keys & old_heldout_keys),
        "confirmation_search_scenario_id_overlap": len(selected_ids & search_ids),
        "confirmation_old_heldout_scenario_id_overlap": len(selected_ids & heldout_ids),
        "train_confirmation_exact_scenario_id_overlap": len(train_ids & selected_ids),
        "train_confirmation_hex_normalized_scenario_id_overlap": len(normalized_train & normalized_selected),
        "confirmation_previous_full50000_file_overlap": len(set(selected) & set(all_files)),
        "confirmation_previous_full50000_raw_key_overlap": len(selected_keys & full_keys),
    }
    assert all(value == 0 for key, value in overlaps.items() if "previous_full50000" not in key)
    assert official_match and not search_mismatch and not heldout_mismatch and not selected_mismatch
    assert len(selected_ids) == len(selected_keys) == len(selected) == 4096
    assert not torch.cuda.is_initialized()
    normalizer = OLD / "train_normalizer.json"
    normalization = json.loads(normalizer.read_text())
    audit = {
        "status": "PASS", "selection": "Sorted native cache index 25000 onward; exclude all old search/heldout raw keys; keep first occurrence of each remaining (source_tfrecord, record_index).",
        "selection_start_index": 25000, "selection_last_index": last_index, "skipped": skipped,
        "cache_pool_files": len(all_files), "cache_pool_unique_raw_keys": len(full_keys),
        "search_files": len(search), "search_unique_raw_keys": len(search_keys),
        "old_heldout_files": len(old_heldout), "old_heldout_unique_raw_keys": len(old_heldout_keys),
        "confirmation_files": len(selected), "confirmation_unique_raw_keys": len(selected_keys),
        "confirmation_unique_scenario_ids": len(selected_ids), "confirmation_source_splits": splits,
        "train_files": len(training), "train_unique_scenario_ids": len(train_ids),
        "train_filename_parse_coverage": len(training), "train_manifest_sha256": sha(TRAIN_MANIFEST),
        "actual_eval_metadata_checked_files": len(search) + len(old_heldout) + len(selected),
        "actual_confirmation_metadata_checked_files": len(selected),
        "filename_raw_key_lg_timestep_and_official_manifest_match": True,
        "metadata_mismatches": [], "overlaps": overlaps,
        "manifest": str(manifest.relative_to(ROOT)), "manifest_sha256": sha(manifest),
        "ordered_basename_lines_sha256": sha(OUT / "confirmation4096_files.txt"),
        "pickle_sha256": sha(OUT / "confirmation4096_files.pkl"),
        "first": selected[0], "last": selected[-1],
        "original_pickle_metadata_checks": original_checks,
        "filename_final_suffix": "scene_timestep; map_id is a separate model/category label and is not inferred from this suffix",
        "train_normalizer_reuse": {
            "path": str(normalizer.relative_to(ROOT)), "sha256": sha(normalizer),
            "normal_mean_width": len(normalization["normal_mean"][0]),
            "normal_scale_width": len(normalization["normal_scale"][0]),
            "scope": "Existing TRAIN-only 7D linear-shape normalizer; suitable only while feature representation stays unchanged.",
        },
        "additional_original_full_lane_cache_inventory": inventory,
        "additional_raw_candidate_manifest": str(extra_path.relative_to(ROOT)),
        "additional_raw_candidate_manifest_sha256": sha(extra_path),
        "additional_raw_candidate_total": len(extras),
        "cpu_seconds": time.perf_counter() - started, "cuda_initialized": torch.cuda.is_initialized(),
        "limitations": [
            "Confirmation scenes are disjoint from prior selection subsets, but all were already included in prior 50k aggregate distribution audit. Do not call them strictly unseen or untouched test data.",
            "Training raw TFRecord and record indices are absent from compact TRAIN cache; leakage check uses preserved filename scenario_id supported by source code, plus actual evaluation metadata.",
            "No claim of geographic or temporal independence is made.",
            "Extra lg_type=0 originals outside 50k are candidate future test sources only. Their scenario IDs, training overlap, native cache validity, and historical use have not been audited. Conversion and exact protocol freeze are needed before using them as strict heldout tests.",
            "The fixed map encoder checkpoint may have previously consumed data outside current TRAIN; its original training provenance is a separate audit and not established here.",
        ],
    }
    (OUT / "manifest_overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    md = f"""# 固定 map encoder 搜索：确认数据审计\n\n状态：PASS。全程 CPU，未初始化 CUDA。\n\n- 从 native validation cache 排序 index 25000 起选择；排除旧 search1024 和 heldout4096 的所有原始 `(TFRecord, record index)`，再按原始场景去重。\n- 新确认集共 **4096 文件 / 4096 原始场景 / 4096 scenario IDs**，最后选择 index {last_index}。\n- 与旧 search/heldout 的文件、原始场景和 scenario ID 交集均为 **0**。与 TRAIN {len(training):,} 文件 / {len(train_ids):,} scenario IDs 的精确及 hex 标准化 ID 交集均为 **0**。\n- 实际读取并核对新 4096 个缓存全部 metadata；连同旧 selection 共核对 {audit['actual_eval_metadata_checked_files']} 文件。另抽查原始 pickle {len(original_checks)} 个。末尾数字是 scene_timestep，**不是 map_id**。\n\nJSON manifest SHA256：`{audit['manifest_sha256']}`。\n\n有序文件名逐行 SHA256：`{audit['ordered_basename_lines_sha256']}`。\n\n[JSON](confirmation4096_files.json)、[pickle](confirmation4096_files.pkl)、[文件名列表](confirmation4096_files.txt) 内容/顺序相同，兼容现有 runner 和 datamodule。\n\n## 评价独立性的边界\n\n这 4096 场景全部来自之前已做过 50k aggregate 评价的数据池，因此是**新的确认子集**，不能称作完全未见的最终 test set。这里只证明它与旧选择子集及当前 TRAIN 的 scenario ID 不重叠；并未证明地理、时间或固定 map encoder 原训练集独立。TRAIN compact cache 未保留 TFRecord/index，训练场景依据生成源码支持的文件名前缀 scenario_id 判断。\n\n## 50k 之外的候选数据\n\n原始 AE 的 val/test 中发现 {len(extras):,} 个去重的 full-lane 原始场景 filename 位于现有 50k 之外。已保存[候选清单](additional_raw_candidates_outside50k.json)。它们尚无本次所需的 native evaluation cache，原始 pickle 不含 scenario_id；必须重建/读取原始 scenario 并审计训练交集、缓存有效性、历史使用，才能冻结为严格最终评价集。当前不夸大为已验证 unseen。\n\n现有 [TRAIN normalizer](../init_diffusion_scratch_loss_search_20261011/train_normalizer.json) 可在保持 7D、linear shape、scalar speed 表示时复用，hash 为 `{sha(normalizer)}`。若换 log shape、velocity 维度或 type 表示，需从 TRAIN 重新计算对应统计量。\n"""
    (OUT / "manifest_overlap_audit.md").write_text(md)
    print(json.dumps({"status": audit["status"], "confirmation": len(selected),
                      "manifest_sha256": audit["manifest_sha256"],
                      "ordered_basename_lines_sha256": audit["ordered_basename_lines_sha256"],
                      "additional_raw_candidates": len(extras), "cpu_seconds": audit["cpu_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
