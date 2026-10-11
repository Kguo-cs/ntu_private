# 固定 map encoder 搜索：确认数据审计

状态：PASS。全程 CPU，未初始化 CUDA。

- 从 native validation cache 排序 index 25000 起选择；排除旧 search1024 和 heldout4096 的所有原始 `(TFRecord, record index)`，再按原始场景去重。
- 新确认集共 **4096 文件 / 4096 原始场景 / 4096 scenario IDs**，最后选择 index 29285。
- 与旧 search/heldout 的文件、原始场景和 scenario ID 交集均为 **0**。与 TRAIN 493,542 文件 / 486,994 scenario IDs 的精确及 hex 标准化 ID 交集均为 **0**。
- 实际读取并核对新 4096 个缓存全部 metadata；连同旧 selection 共核对 9216 文件。另抽查原始 pickle 32 个。末尾数字是 scene_timestep，**不是 map_id**。

JSON manifest SHA256：`9cef282172231663292f3af362bb96c652a44dc10a55e3b5264fd57fa9576e03`。

有序文件名逐行 SHA256：`66e4fb74292d2c5587a6214e70f3ed21f8a2e5d7e26c5687fdc146ca5d623a3b`。

[JSON](confirmation4096_files.json)、[pickle](confirmation4096_files.pkl)、[文件名列表](confirmation4096_files.txt) 内容/顺序相同，兼容现有 runner 和 datamodule。

## 评价独立性的边界

这 4096 场景全部来自之前已做过 50k aggregate 评价的数据池，因此是**新的确认子集**，不能称作完全未见的最终 test set。这里只证明它与旧选择子集及当前 TRAIN 的 scenario ID 不重叠；并未证明地理、时间或固定 map encoder 原训练集独立。TRAIN compact cache 未保留 TFRecord/index，训练场景依据生成源码支持的文件名前缀 scenario_id 判断。

## 50k 之外的候选数据

原始 AE 的 val/test 中发现 45,398 个去重的 full-lane 原始场景 filename 位于现有 50k 之外。已保存[候选清单](additional_raw_candidates_outside50k.json)。它们尚无本次所需的 native evaluation cache，原始 pickle 不含 scenario_id；必须重建/读取原始 scenario 并审计训练交集、缓存有效性、历史使用，才能冻结为严格最终评价集。当前不夸大为已验证 unseen。

现有 [TRAIN normalizer](../init_diffusion_scratch_loss_search_20261011/train_normalizer.json) 可在保持 7D、linear shape、scalar speed 表示时复用，hash 为 `adcc5e5c232b018919b269ec06b7e8b065a49fd9f0219f3d0f432f7caf4cea23`。若换 log shape、velocity 维度或 type 表示，需从 TRAIN 重新计算对应统计量。
