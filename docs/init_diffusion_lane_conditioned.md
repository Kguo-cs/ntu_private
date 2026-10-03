# InitDiffusion 地图条件训练与评价

[训练配置](../configs/experiment/init_diffusion_lane_conditioned.yaml)使用已有的 `init_decoder=flow`（`InitDiffusion`），复用 SMART 的训练器、损失和采样器。训练读取 `src/waymo_data/full/training_map2_sd`，验证和测试读取 `src/waymo_data/full/scenario_dreamer_val`。

## 运行

以下命令均在 `/home/ke/code/sim` 执行，使用现有 `sim` 环境。

训练目录使用预存文件清单，默认路径为 `src/waymo_data/full/training_map2_sd_files.pkl`。首次准备或增删训练数据后，运行以下命令刷新；清单只记录文件名，不读取场景内容。未刷新时，新增文件不会自动进入训练。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.smart.scenario_dreamer.file_list \
  --raw-dir /home/ke/code/sim/src/waymo_data/full/training_map2_sd
```

正式训练默认从配置指定的 SMART 骨干 checkpoint 初始化，并冻结地图编码器。该骨干 checkpoint 不包含 InitDiffusion 参数，初始状态模型从新建参数开始训练。默认 `action=finetune`，batch size 256、64 epochs；每个 epoch 用一个验证 batch 监测指标，checkpoint 按 `collision_rate` 保存。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  experiment=init_diffusion_lane_conditioned
```

恢复本实验的训练状态时使用 `action=fit`，同时恢复模型、优化器和训练进度：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  experiment=init_diffusion_lane_conditioned action=fit \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

从零训练模型与地图编码器，不加载骨干 checkpoint：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  experiment=init_diffusion_lane_conditioned action=fit \
  ckpt_path=null model.model_config.finetune=false
```

用[独立评价配置](../configs/experiment/init_diffusion_lane_conditioned_eval.yaml)完整评价 50,000 场景。必须显式指定训练得到的 checkpoint；测试使用单 GPU，检查官方集合成员完整性。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

训练监测只生成一个 batch，仍对照完整 50k GT 分布，因此监测值不能当作完整评估结果。显存不足时可调整 `data.train_batch_size`、`data.val_batch_size` 和 `data.test_batch_size`。

## 评价口径

`initial_scene_only=true` 只输出初始生成快照；`sd_gen_timestep=0` 指这个输出数组的第 0 帧，不是原始 Waymo 的第 0 帧。验证样本的初始位置、朝向和速度来自各自的 `scene_timestep`；预测以 world 坐标交给评价器。

评价器使用官方完整参考 lane 图，计算七项 vehicle 指标：`nearest_dist_jsd`、`lat_dev_jsd`、`ang_dev_jsd`、`length_jsd`、`width_jsd`、`speed_jsd`、`collision_rate`。nearest/lat JSD 乘 10，其余 JSD 乘 100；collision rate 为百分比。参考成员表为 `src/waymo_data/waymo_eval_set.pkl`，GT 统计来自 `src/waymo_data/sd_real_metric_cache.sqlite`。

完整报告保存到 `src/logs/init_diffusion_lane_conditioned_eval/<时间>/sd_agent_metrics.json`。应检查 `num_samples=50000`、`num_gt_samples=50000`、`full_membership=true` 和 `map_source=reference`。

## 比较范围

本模型沿用 GT ego 状态、agent 类型和数量，条件地图为 SMART tokens，使用现有 50 m 地图查询半径。Scenario Dreamer lane-conditioned 模型使用 AE lane latent；两者的条件信息和地图表示不同。相同的 50k 成员与指标口径便于比较结果，但这不是 Scenario Dreamer 论文结果的复现。

现有训练缓存只保存初始 agent 状态和地图 tokens，没有原始参考时刻或精确 SD lane 图元数据；其时刻与地图来源尚未逐样本追溯。本入口直接使用用户指定的缓存。验证目录则保留官方文件名、参考时刻、坐标变换和完整 lane 图，可用于严格的评估成员检查。

## 本次验证

在 `sim` 环境中通过 8 项 InitDiffusion 测试和 80 项 Scenario Dreamer 回归测试。真实数据上的短训练使用默认 batch size 256、8 workers，完成两步优化并保存 checkpoint；独立 `test` 入口加载该 checkpoint，完成 512 个场景的七项指标计算。训练 loss 为有限值，InitDiffusion 权重数值正常，冻结的地图 backbone 权重保持不变。

这验证了训练、保存、加载和评价链路，尚未完成 64 轮正式训练或 50k 完整评价。短训练的分数不能作为训练收敛结果。短训练与评价的重跑命令：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  experiment=init_diffusion_lane_conditioned \
  trainer.max_epochs=1 +trainer.max_steps=2 trainer.limit_train_batches=2 \
  model.model_config.lr_total_steps=64 \
  callbacks.model_checkpoint.dirpath=/tmp/init_diffusion_smoke/checkpoints \
  hydra.run.dir=/tmp/init_diffusion_smoke

/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/tmp/init_diffusion_smoke/checkpoints/last.ckpt \
  trainer.limit_test_batches=2 model.model_config.sd_require_full_set=false \
  hydra.run.dir=/tmp/init_diffusion_eval_smoke
```
