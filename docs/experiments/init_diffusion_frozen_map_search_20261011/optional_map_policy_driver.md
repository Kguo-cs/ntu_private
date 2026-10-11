# Optional random/frozen map comparison driver

`tools/compare_random_frozen_map.py` 默认只执行 CPU 审计并生成命令，绝不等待训练、运行 GPU 或修改主 queue。任一固定地图 `new_control` 的实际 `step_2048.ckpt` 缺失时立即以 exit code 2 退出；不会用512或最终4096 checkpoint替代。

```bash
CUDA_VISIBLE_DEVICES='' /home/ke/miniconda3/envs/sim/bin/python tools/compare_random_frozen_map.py
```

主 queue 全部结束后，若需要额外对比，可显式执行下面命令。该命令重新完成 CPU eligibility 审计，再顺序运行6次 evaluator；默认每次创建新的 `logs/component_loss_frozen_map/map_policy_comparison_<timestamp>/`，不覆盖已有 provenance。

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/compare_random_frozen_map.py --run
```

比较仅包含旧 `geo_mse32_col20_heading_l1_shape2` 与新 `new_control`，TRAIN seeds 为817/1817/2817，GEN seeds 为2817/3817，使用 EMA2048 和同一旧 heldout4096。旧 `warm_winner` 损失不同，已排除。旧 invocation 没有 `--map-checkpoint`，保留 checkpoint 中联合训练的随机地图编码器；新 invocation 使用明确的固定预训练地图源。

审计验证实际493542 TRAIN顺序和每个 arm 的消费清单、随机 G1 初值 hash、TRAIN8192 calibration hash/7D mean/std、完整有效训练配置、ego/type/conditioning/metric provenance、AdamW参数与2048更新、epoch scheduler、EMA2048及实际1,048,524样本 exposure。仅允许地图初始化/冻结和 weight=0 的 speed_magnitude 冗余配置差异；输出路径、evaluation manifest路径和记录的 DataLoader worker数量不作为模型语义差异。源码快照、checkpoint hashes/状态 hashes 和全部生成命令保存到 eligibility.json。

截至本次 CPU 预检，旧3个实际2048 checkpoints全部通过检查，旧seed817与新phase1配置的语义检查通过；新3个2048 checkpoints尚未产生，因此完整 comparison eligibility 尚未成立。`optional_map_policy_driver_cpu_precheck.json` 记录了该边界，CUDA未初始化，未执行GPU命令。19个 analyzer tests 与5个 driver tests覆盖分组、可比性拒绝、缺失文件不执行、命令地图策略、GEN/TRAIN统计层级及独立成本。

生成的 comparison_summary.json 先在每个 TRAIN 内平均两个 GEN，再对三个 TRAIN 等权求 mean/sample SD，同时保留仅1817/2817的敏感性结果。817用于自适应搜索和long候选选择，因此可报告总TRAIN N=3，但独立确认仅2个TRAIN seeds；GEN重复不是N=6。

这项结果独立于主搜索，不参与最终候选选择。主 analyzer 排除 `map_policy_comparison_*` 所有后代，只在 excluded_optional_comparison_diagnostics 保存实际记录与额外evaluation成本。它对比“随机地图联合训练”和“预训练地图冻结”，不能解释为只改变预训练的因果实验。历史完整训练源码hash不可得；两边重评使用相同当前源码，并不证明历史训练代码完全相同。旧heldout已被评价过，也不是严格unseen最终测试。
