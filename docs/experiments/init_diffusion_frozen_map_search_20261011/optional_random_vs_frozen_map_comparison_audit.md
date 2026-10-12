# Optional random-map 与 frozen-map 比较：CPU 资格审计

状态：配置/metadata 对齐，等待 frozen new_control 的2048 checkpoint；未运行 GPU、未加载 model weights、未改变 queue。该比较不能参与本轮最终配方选择。

干净的同 loss 对应项是旧 `geo_mse32_col20_heading_l1_shape2` 与新 `new_control`。seed817 的 G1 初始 hash、TRAIN顺序、TRAIN normalizer、B512、LR5e-4、clip0.5、EMA0.9999、square50、模型参数和 input conditioning 全部对齐。配置中仅有意 map policy（finetune/fixed source）及明确关闭的 speed_magnitude term 不同。

旧 `warm_winner` 可作历史参照，但 loss 不同，不能用它声称 map-only ablation。

| seed | random geo folder | random control folder | G1 init hash |
|---:|---|---|---|
| 817 | `logs/component_loss_scratch/batch512_seed817/geo_mse32_col20_heading_l1_shape2` | `logs/component_loss_scratch/batch512_seed817/warm_winner` | `66ae5d81e82e26af2336043337fff6191b92c0ae010369c314f3fa315c224090` |
| 1817 | `logs/component_loss_scratch/batch512_seed1817/geo_mse32_col20_heading_l1_shape2` | `logs/component_loss_scratch/batch512_seed1817/warm_winner` | `39a36cb6474d1600158349b1c1b59edde64dc2b4df9d0a03656bcfe5e2a969cf` |
| 2817 | `logs/component_loss_scratch/batch512_seed2817/geo_mse32_col20_heading_l1_shape2` | `logs/component_loss_scratch/batch512_seed2817/warm_winner` | `155de9d26b0804f5248cb04ed290f07f1fdc115b2755c1cde8a0313686289d32` |

旧 heldout4096 与新 confirmation4096 manifest 不同且文件交集0，不能直接比较两份已有 aggregate。建议只在2048 checkpoint检查通过后，将 frozen baseline **额外评价旧 heldout4096**；GEN2817/3817、EMA、matched reference。

这比较的是两种 map 训练策略：random map联合训练 vs pretrained map固定。global clip的参数集合也随策略变化；不能把差异只归因于预训练。

历史 metric helper hashes、地图来源和所有 inspected settings 一致，但缺少完整模型源码历史hash。更严格的方法是同时用当前代码重评旧 random geo checkpoint，确认其输出与旧结果一致或报告变化。

以下仅是未执行的 command skeleton；路径文件预先写出供审阅。`--steps 1` 不改变实际checkpoint2048：本命令只有 evaluation-checkpoints，没有训练arms。

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/scratch_component_loss_sweep.py \
  --steps 1 \
  --checkpoint-steps '' \
  --variants '' \
  --seed 817 \
  --batch-size 512 \
  --scenes 4096 \
  --eval-batch-size 128 \
  --workers 0 \
  --eval-seed 2817 \
  --extra-eval-seed 3817 \
  --evaluation-weights ema \
  --normalizer-json docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json \
  --eval-manifest docs/experiments/init_diffusion_scratch_loss_search_20261011/heldout4096_files.json \
  --map-checkpoint /home/ke/code/sim/src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt \
  --evaluation-checkpoints docs/experiments/init_diffusion_frozen_map_search_20261011/optional_map_policy_oldheldout_seed817_checkpoints.json \
  --output logs/component_loss_optional_map_policy/seed817_frozen_on_oldheldout
```

建议的 current-code replay：

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/scratch_component_loss_sweep.py \
  --steps 1 \
  --checkpoint-steps '' \
  --variants '' \
  --seed 817 \
  --batch-size 512 \
  --scenes 4096 \
  --eval-batch-size 128 \
  --workers 0 \
  --eval-seed 2817 \
  --extra-eval-seed 3817 \
  --evaluation-weights ema \
  --normalizer-json docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json \
  --eval-manifest docs/experiments/init_diffusion_scratch_loss_search_20261011/heldout4096_files.json \
  --evaluation-checkpoints docs/experiments/init_diffusion_frozen_map_search_20261011/optional_current_code_random_map_seed817_checkpoints.json \
  --output logs/component_loss_optional_map_policy/seed817_random_geo_current_code
```

完整资格检查、hash、模型差异及限制见 [JSON](optional_random_vs_frozen_map_comparison_audit.json)。seed817只有 TRAIN N=1；若要3seed，必须再检查并评价 frozen1817/2817 的2048 checkpoint，同一旧heldoutmanifest。
