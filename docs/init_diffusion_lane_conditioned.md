# InitDiffusion 地图条件训练与评价

[训练配置](../configs/experiment/init_diffusion_lane_conditioned.yaml)使用已有的 `init_decoder=flow`（`InitDiffusion`），复用 SMART 的训练器、损失和采样器。训练读取 `src/waymo_data/full/training_map2_sd`，验证和测试读取 `src/waymo_data/full/scenario_dreamer_val`。

## 运行

以下命令均在 `/home/ke/code/sim` 执行，使用现有 `sim` 环境。运行入口显式设置 `paths.root_dir=/home/ke/code/sim/src`，以匹配现有数据和权重目录。

当前配置中的训练文件清单选项已注释，默认扫描训练目录。若要使用预存清单，先运行下面的命令，再给训练命令加上 `data.scenario_dreamer_train_sample_list=/home/ke/code/sim/src/waymo_data/full/training_map2_sd_files.pkl`。启用清单后，增删训练数据需重新运行命令刷新；清单只记录文件名，不读取场景内容，未刷新时新增文件不会自动进入训练。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.smart.scenario_dreamer.file_list \
  --raw-dir /home/ke/code/sim/src/waymo_data/full/training_map2_sd
```

正式训练默认从配置指定的 SMART 骨干 checkpoint 初始化，并冻结地图编码器。该骨干 checkpoint 不包含 InitDiffusion 参数，初始状态模型从新建参数开始训练。当前默认 `action=finetune`，训练 batch size 40、验证/测试 batch size 256、64 epochs；每个 epoch 用一个验证 batch 监测指标，checkpoint 按 `collision_rate` 保存。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned
```

恢复本实验的训练状态时使用 `action=fit`，同时恢复模型、优化器和训练进度：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned action=fit \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

从零训练模型与地图编码器，不加载骨干 checkpoint：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned action=fit \
  ckpt_path=null model.model_config.finetune=false
```

用[独立评价配置](../configs/experiment/init_diffusion_lane_conditioned_eval.yaml)完整评价 50,000 场景。必须显式指定训练得到的 checkpoint；测试使用单 GPU，检查官方集合成员完整性。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

训练监测只生成一个 batch，仍对照完整 50k GT 分布，因此监测值不能当作完整评估结果。显存不足时可调整 `data.train_batch_size`、`data.val_batch_size` 和 `data.test_batch_size`。

## 独立地图编码器（sep_map）

默认 `model.model_config.decoder.sep_map=false`，InitDiffusion 复用 SMART 的地图编码器。启用下面的选项后，创建结构相同但参数独立的 `init_map_encoder`；监督训练同时更新这个地图编码器与 InitDiffusion，保留共享地图和运动策略编码器的冻结状态。该选项用于 `init_decoder=flow` 的监督路径。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.sep_map=true
```

加载没有独立地图权重的骨干或共享地图 checkpoint 时，从已加载的共享地图权重初始化独立地图；加载已有 `sep_map` checkpoint 时保留其独立地图权重。把共享地图实验转换为独立地图实验应使用 `action=finetune`，重新建立包含独立地图参数的优化器；`action=fit` 用于恢复同一 `sep_map` 配置的训练。

验证、完整测试及断点续训都需要保留 `model.model_config.decoder.sep_map=true`。独立地图输出已转到 ego 局部坐标，随后只经过 InitDiffusion 的 lane projection。EMA 仍按现有范围平均 `G1`（包含 lane projection），独立地图编码器使用在线权重并随 checkpoint 保存。

## Scenario Dreamer 优化器选项

不指定 `optimizer` 时等同于 `optimizer=default`，沿用原优化器和调度器。需要使用 Scenario Dreamer 的优化器配方时，新训练增加 `optimizer=scenario_dreamer`：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned optimizer=scenario_dreamer
```

该选项使用 AdamW，基础学习率 `1e-4`、betas `(0.9, 0.999)`、epsilon `1e-7`；Linear/Conv 等权重的 weight decay 为 `1e-5`，bias、归一化层、Embedding 和其他参数不做 weight decay。梯度裁剪阈值为 `10`。学习率按 optimizer step 线性预热 500 步，此后恒定；梯度累积的每个 microbatch 不单独计步。

需要调整基础学习率或 warmup 长度时，分别覆盖 `model.model_config.lr` 和 `model.model_config.lr_warmup_steps`。该 profile 用于 InitDiffusion、Scenario Dreamer AE/LDM 等自动优化的监督训练；手动优化的 GAIL/GAN 路径不支持，会明确报错。

与[官方调度器](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/675423469766bf2fd8a6b569ef1869a6f1e76993/utils/train_helpers.py#L35-L40)一致，初始学习率为 `0`，scheduler 在每次 optimizer step 后推进；第 500 次更新完成后到达 `1e-4`，第 501 次更新开始使用完整学习率。此选项只切换优化器、学习率调度和梯度裁剪；batch size、epoch 数、数据、训练目标及 EMA 设置仍由原实验配置决定，不等于复现 Scenario Dreamer 的完整训练协议。

恢复用该选项训练的 checkpoint 时，保留相同的 `optimizer` 选项：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned optimizer=scenario_dreamer \
  action=fit ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

`action=fit ckpt_path=...` 恢复 checkpoint 的优化器和 scheduler 状态，应与原训练使用同一 profile；旧运行继续使用默认 profile，不能仅通过修改恢复命令把已有状态转换为新配方。`action=finetune` 只初始化模型权重，重新建立所选优化器和调度器，warmup 从 0 开始；它不恢复原训练进度。独立 `action=test` 评价不执行优化器更新，无需为使用此配方训练的权重额外指定该选项。

## InitDiffusion EMA

当前 lane-conditioned 训练和评价配置默认启用 `model.model_config.decoder.init_diffusion.use_ema=true`，`ema_decay=0.9999`；通用 SMART 配置仍默认关闭。EMA 只跟踪 `G1` generator 的参数，包括其中的 `lane_embed`，不平均 SMART 地图编码器或其他模块。

训练使用在线参数，每次优化器更新后更新 EMA；验证和测试在 `use_ema=true` 时临时使用平均参数，完成后恢复在线参数。与 Scenario Dreamer 一样，采用 `torch_ema` 默认的 `num_updates` 衰减预热：初期实际衰减受更新次数限制，再逐步接近配置值。

显式指定 EMA 开关和衰减：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.use_ema=true \
  model.model_config.decoder.init_diffusion.ema_decay=0.9999
```

checkpoint 同时保存在线参数、EMA shadow 参数、更新次数和衰减值。上面的 `action=fit ckpt_path=...` 命令会保留这些状态继续训练。严格加载旧的、没有 EMA 状态的模型 checkpoint 时，以已经加载的 `G1` 权重初始化 EMA，更新次数从 0 开始；无法补回此前训练的平均历史。

对同一个 checkpoint 使用在线参数评价：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt \
  model.model_config.decoder.init_diffusion.use_ema=false
```

`use_ema=false` 关闭验证/测试时的平均参数切换，不删除 checkpoint 已有的 EMA。加载带 EMA 的 checkpoint 后，其状态仍保留，并在继续训练时跟踪更新；可以再设为 `true` 使用平均参数。新建模型若始终关闭 EMA，则不积累此前的平均历史。

## 评价口径

`initial_scene_only=true` 只输出初始生成快照；`sd_gen_timestep=0` 指这个输出数组的第 0 帧，不是原始 Waymo 的第 0 帧。验证样本的初始位置、朝向和速度来自各自的 `scene_timestep`；预测以 world 坐标交给评价器。

评价器使用官方完整参考 lane 图，计算七项 vehicle 指标：`nearest_dist_jsd`、`lat_dev_jsd`、`ang_dev_jsd`、`length_jsd`、`width_jsd`、`speed_jsd`、`collision_rate`。nearest/lat JSD 乘 10，其余 JSD 乘 100；collision rate 为百分比。参考成员表为 `src/waymo_data/waymo_eval_set.pkl`，GT 统计来自 `src/waymo_data/sd_real_metric_cache.sqlite`。

完整报告保存到 `src/logs/init_diffusion_lane_conditioned_eval/<时间>/sd_agent_metrics.json`。应检查 `num_samples=50000`、`num_gt_samples=50000`、`full_membership=true` 和 `map_source=reference`。

## 比较范围

本模型沿用 GT ego 状态、agent 类型和数量，条件地图为 SMART tokens，使用现有 50 m 地图查询半径。Scenario Dreamer lane-conditioned 模型使用 AE lane latent；两者的条件信息和地图表示不同。相同的 50k 成员与指标口径便于比较结果，但这不是 Scenario Dreamer 论文结果的复现。

现有训练缓存只保存初始 agent 状态和地图 tokens，没有原始参考时刻或精确 SD lane 图元数据；其时刻与地图来源尚未逐样本追溯。本入口直接使用用户指定的缓存。验证目录则保留官方文件名、参考时刻、坐标变换和完整 lane 图，可用于严格的评估成员检查。

## 既有链路验证

以下记录来自 EMA 功能加入前，不代表本次 EMA 功能的验证结论。在 `sim` 环境中通过 8 项 InitDiffusion 测试和 80 项 Scenario Dreamer 回归测试。真实数据上的短训练使用当时的 batch size 256、8 workers，完成两步优化并保存 checkpoint；独立 `test` 入口加载该 checkpoint，完成 512 个场景的七项指标计算。训练 loss 为有限值，InitDiffusion 权重数值正常，冻结的地图 backbone 权重保持不变。

这验证了训练、保存、加载和评价链路，尚未完成 64 轮正式训练或 50k 完整评价。短训练的分数不能作为训练收敛结果。短训练与评价的重跑命令：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  data.train_batch_size=256 data.num_workers=8 \
  model.model_config.decoder.init_diffusion.use_ema=false \
  trainer.max_epochs=1 +trainer.max_steps=2 trainer.limit_train_batches=2 \
  model.model_config.lr_total_steps=64 \
  callbacks.model_checkpoint.dirpath=/tmp/init_diffusion_smoke/checkpoints \
  hydra.run.dir=/tmp/init_diffusion_smoke

/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/tmp/init_diffusion_smoke/checkpoints/last.ckpt \
  data.test_batch_size=256 model.model_config.decoder.init_diffusion.use_ema=false \
  trainer.limit_test_batches=2 model.model_config.sd_require_full_set=false \
  hydra.run.dir=/tmp/init_diffusion_eval_smoke
```

## EMA 验证

在 `sim` 环境通过 23 项 InitDiffusion 测试（含 15 项 EMA/优化器专项）及 80 项 Scenario Dreamer 回归测试。使用指定真实数据完成 CPU 短训练：4 个 batch、梯度累积 2，产生 2 次 optimizer step，checkpoint 的 EMA 更新次数也为 2。独立测试加载同一 checkpoint 后使用 EMA 且更新次数保持 2；恢复训练完成下一次 optimizer step 后，模型步数与 EMA 更新次数均为 3。在线参数与 EMA shadow 参数不同，平均权重并非仅保存的在线权重副本。

已有运行中的训练进程不会自动获得新代码；启动新训练或恢复 checkpoint 后，EMA 配置才生效。以上检查验证功能与状态恢复，不代表 EMA 一定改善最终评价分数。
