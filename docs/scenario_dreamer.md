# Scenario Dreamer init_decoder

Scenario Dreamer 是 `SMARTDecoder` 的一种初始状态 decoder。使用现有 `sim` conda 环境、`src.run`、`MultiDataModule`、`SMART_GAIL` 的监督训练分支和项目已有的 `ScenarioDreamerEvaluator`。运行时的模型和指标代码都在本仓库中，无需外部 checkout，也不使用上游 Trainer。

## 运行

在 sim 项目根目录执行：

```bash
conda activate sim

# 默认只加载 AE，随机初始化 LDM 训练
python -m src.run experiment=scenario_dreamer action=fit

# 从官方 LDM 权重继续训练
python -m src.run experiment=scenario_dreamer action=fit \
  model.model_config.decoder.scenario_dreamer.ldm_checkpoint=src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt

# 加载官方 LDM，完整验证 50,000 个场景
python -m src.run experiment=scenario_dreamer action=validate \
  model.model_config.decoder.scenario_dreamer.ldm_checkpoint=src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt \
  trainer.limit_val_batches=1.0 model.model_config.sd_require_full_set=true
```

[实验配置](../configs/experiment/scenario_dreamer.yaml)设置数据目录、初始状态任务和评价器。实际模型选择是：

```yaml
model:
  model_config:
    token_processor:
      pred_init: true
      learn_init: true
      scenario_dreamer_init: true
    decoder:
      init_decoder: scenario_dreamer  # 默认 flow
      scenario_dreamer:
        training_stage: ldm
        ae_config: null
        ae_checkpoint: src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_waymo/last.ckpt
        ldm_checkpoint: null  # 无需 LDM 权重文件，从头训练
        ldm_config: null  # 可选的 model/dataset/train 配置覆盖，仅用于从头训练
        generation_mode: initial_scene  # Joint lane + agent generation
        map_source: exact
        map_id: 0
        use_ema: true
```

`scenario_dreamer_init` 让 token processor 使用官方保存的初始场景状态，关闭原 Flow 的 refinement，并按初始状态评价。`decoder.init_decoder` 的默认值仍为 `flow`。Scenario Dreamer 当前支持监督训练，不支持现有 Flow/GAIL 的策略损失。

先做小规模验证：

```bash
python -m src.run experiment=scenario_dreamer action=validate \
  model.model_config.decoder.scenario_dreamer.ldm_checkpoint=src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt \
  trainer.limit_val_batches=1 data.val_batch_size=2 \
  model.model_config.sd_require_full_set=false

python -m src.run experiment=scenario_dreamer action=test \
  model.model_config.decoder.scenario_dreamer.ldm_checkpoint=src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt \
  trainer.limit_test_batches=1 data.test_batch_size=2 \
  model.model_config.sd_require_full_set=false

# 单步反向传播和优化器检查；不写入大型训练 checkpoint
python -m src.run experiment=scenario_dreamer action=fit \
  data.train_batch_size=1 data.num_workers=0 \
  trainer.limit_train_batches=1 trainer.limit_val_batches=0 \
  +trainer.max_steps=1 callbacks.model_checkpoint=null \
  +trainer.enable_checkpointing=false
```

小样本验证必须显式关闭 `sd_require_full_set`，否则评价器会拒绝将部分样本当作完整评估。默认仍对照完整 50k 参考分布。正式评估使用单 GPU；当前缓存评价器不聚合多卡结果。batch size 采用各实验 YAML 当前设置；显存不足时通过 `data.val_batch_size` / `data.test_batch_size` 调整。

## 权重和训练

`ldm_checkpoint: null`（默认）表示从头训练 LDM，不打开任何 LDM checkpoint，也不从中读取配置或内嵌 AE。模型结构、100 步 diffusion schedule、归一化参数、guidance 和 EMA decay 使用仓库内的 [Waymo Large 配置](../src/smart/scenario_dreamer/waymo_ldm_large.yaml)，数值取自官方发布的 checkpoint。AE 仍严格加载 `ae_checkpoint` 并冻结；EMA 从新建 LDM 参数初始化，`checkpoint_step=0`。保持 `ckpt_path=null` 即可开始全新训练。

需要修改从头训练的结构或采样设置时，可用 `ldm_config` 的 `model`、`dataset`、`train` 子项覆盖默认配置；agent/lane latent 维度必须与 AE 一致。指定 `ldm_checkpoint` 时使用 checkpoint 自带配置，不能同时传入 `ldm_config` 覆盖。

给 `ldm_checkpoint` 设置官方文件路径即可加载预训练模型。AE、LDM 和 LDM 内嵌 AE 都使用严格参数加载，恢复官方 EMA 和 checkpoint 步数；推理默认使用 EMA。路径拼错会报错，不会退回随机初始化。两份官方 Waymo checkpoint 已放在上述路径：

| 权重 | SHA-256 |
| --- | --- |
| Waymo AE | `3c3033a107de727ca1c2399a8e0df107e5eb1a84bce3d7e18cc2e01698ccf6ac` |
| Waymo LDM Large | `06a1a65e9949f55c3398aeadacde388b03a6705f2661bc273cf43e7319de4cd5` |

这些哈希已核对；指定 LDM 路径时不会每次重新读取整个 13 GB 文件计算哈希。源码固定为 [官方提交 6754234](https://github.com/princeton-computational-imaging/scenario-dreamer/tree/675423469766bf2fd8a6b569ef1869a6f1e76993)，文件来源和改动记录在 [SOURCES.json](../src/smart/scenario_dreamer/SOURCES.json)。

LDM 阶段冻结 AE，采样 AE 后验得到 latent，用官方 LDM 的联合 agent/lane 噪声预测损失训练 diffusion；记录 `train/scenario_dreamer/loss`、`agent_loss`、`lane_loss`。优化器和学习率计划继续使用 SMART。每次优化器更新后更新 EMA，EMA 也随普通 SMART checkpoint 保存和恢复。

`ckpt_path` 用于恢复本项目保存的 **SMART checkpoint**，与上面两个官方 checkpoint 路径不同：

```bash
python -m src.run experiment=scenario_dreamer action=fit \
  ckpt_path=/absolute/path/to/smart-checkpoint.ckpt
```

首次构造 decoder 会加载 AE；仅在 `ldm_checkpoint` 非空时加载官方 LDM。随后 Trainer 恢复 SMART 中的模型参数、EMA 和优化器。从头训练的 SMART checkpoint 可以保持 `ldm_checkpoint=null` 恢复，不需要官方 LDM 文件；构造时的 `ldm_config` 必须与保存时一致。

## AE 训练

[AE 实验配置](../configs/experiment/scenario_dreamer_ae.yaml)复用同一个 `init_decoder=scenario_dreamer`、数据集、SMART 优化器和 Trainer。`training_stage=autoencoder` 训练 AE，`training_stage=ldm` 训练 LDM；两个阶段分别训练。AE 实验使用官方 AE 的基础学习率 `1e-4`，优化器和学习率调度继续使用 SMART。

```bash
# 从零训练 AE：不加载 AE/LDM 权重，不创建 LDM 或 EMA
python -m src.run experiment=scenario_dreamer_ae action=fit

# 加载官方 AE 继续训练
python -m src.run experiment=scenario_dreamer_ae action=fit \
  model.model_config.decoder.scenario_dreamer.ae_checkpoint=src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_waymo/last.ckpt

# 恢复本项目保存的 AE 训练状态，包括优化器
python -m src.run experiment=scenario_dreamer_ae action=fit \
  ckpt_path=/absolute/path/to/smart-ae.ckpt

# 评价 AE 重建损失
python -m src.run experiment=scenario_dreamer_ae action=test \
  ckpt_path=/absolute/path/to/smart-ae.ckpt

# 将训练好的 AE 直接用于下一阶段的 LDM 训练
python -m src.run experiment=scenario_dreamer action=fit \
  model.model_config.decoder.scenario_dreamer.ae_checkpoint=/absolute/path/to/smart-ae.ckpt \
  model.model_config.decoder.scenario_dreamer.ldm_checkpoint=null
```

AE 的 `ae_checkpoint=null` 使用仓库内[官方 Waymo AE 结构](../src/smart/scenario_dreamer/waymo_autoencoder.yaml)随机初始化。`ae_config` 可覆盖从头训练的结构和损失权重；指定 AE checkpoint 时使用其中的结构。AE 阶段必须保持 `ldm_checkpoint=null` 和 `ldm_config=null`。LDM 阶段需要已训练的 AE。

训练使用官方联合 agent/lane 重建、agent 类型、lane 连接、KL 和分区车道数量预测损失。图适配补充 `num_lanes_after_origin` 标签；分区车道数由官方递归排序后的 partition mask 计算。没有分区样本时，条件数量损失和准确率的报告值为 0，避免空均值 NaN；总损失公式不变。

训练记录 `train/scenario_dreamer/autoencoder/*`，验证和测试分别记录 `val/scenario_dreamer/autoencoder/*`、`test/scenario_dreamer/autoencoder/*`。AE checkpoint 按 `val/scenario_dreamer/autoencoder/loss` 最小值保存。AE 验证/测试评价重建损失；后续 LDM 阶段进行 lane/agent 联合生成及 agent metrics 评价。

保存的 SMART checkpoint 包含 AE 参数和结构配置，`ae_checkpoint` 可以直接读取它，无需手动导出。`ckpt_path` 恢复同一阶段的完整训练状态；切换 AE → LDM 时使用 `ae_checkpoint`，并保持 `ckpt_path=null`。如果更改过 AE 结构，恢复训练时需保持相同的 `ae_config`。LDM 的 latent 标准化默认沿用官方统计量，自训 AE 的统计量可通过 `ldm_config.dataset` 中的 `agent_latents_mean/std`、`lane_latents_mean/std` 更新。

## Latent 预缓存

[缓存实验](../configs/experiment/scenario_dreamer_cache.yaml)使用 `src.run action=cache_latents`，复用现有数据集、token processor、图适配和 AE。输出是原始 pickle 旁路的独立缓存目录，原始数据保持可直接用于 AE 训练及生成评估。

```bash
# 默认：官方 AE，缓存全部 train 数据，支持中断后重跑
python -m src.run experiment=scenario_dreamer_cache

# 用自己训练的 AE 缓存；每份 AE 使用自己的输出目录
python -m src.run experiment=scenario_dreamer_cache \
  model.model_config.decoder.scenario_dreamer.ae_checkpoint=/absolute/path/to/smart-ae.ckpt \
  latent_cache.output_dir=/absolute/path/to/my-ae-latents/train

# LDM 训练读取官方 AE 的缓存
python -m src.run experiment=scenario_dreamer action=fit \
  data.scenario_dreamer_train_latent_cache=src/waymo_data/scenario_dreamer_latents_waymo/train

# 若使用自训 AE，其权重与缓存必须配套
python -m src.run experiment=scenario_dreamer action=fit \
  model.model_config.decoder.scenario_dreamer.ae_checkpoint=/absolute/path/to/smart-ae.ckpt \
  data.scenario_dreamer_train_latent_cache=/absolute/path/to/my-ae-latents/train
```

每个场景保存 float32 的 `agent_mu`、`agent_log_var`、`lane_mu`、`lane_log_var`，保持原始 pickle 的 ego-first agent 顺序与原始 lane 顺序。读取时重新对齐 SMART 的 ego-last 和官方递归排序。LDM 训练跳过 AE encoder，仍逐次采样 `mu + exp(0.5 * log_var) * noise` 并使用当前 LDM 配置进行标准化；缓存保留未标准化的后验，而非一次固定采样。分区 mask、条件节点及生成评价流程保持一致。AE decoder 仍用于生成结果，训练配置仍需要 `ae_checkpoint`。

默认 `latent_cache.split=train`，`max_scenes=null` 表示处理完整划分，`batch_size=16`、`device=cuda`；worker 数量使用 `data.num_workers`。`split=val/test` 按现有评价清单选取场景。训练、验证、测试可分别配置：

```yaml
data:
  scenario_dreamer_train_latent_cache: /path/to/latents/train
  scenario_dreamer_val_latent_cache: null
  scenario_dreamer_test_latent_cache: null
```

联合生成的验证/测试不需要 GT latent，通常保持后两个选项为 `null` 即可；给定车道模式可缓存对应的验证/测试数据。AE 阶段应将所有 latent cache 选项设为 `null`。

缓存包含实际 AE 权重及编码配置的 SHA-256 标识、源 pickle 内容的 SHA-256、格式版本和维度信息。训练拒绝不匹配、缺失、损坏或源数据已变化的记录，不会静默切回在线编码。加载 SMART checkpoint 后会重新验证实际恢复的 AE；加载官方 LDM 时，其内嵌 AE 也必须与缓存匹配。更换 AE 或编码配置时使用新的缓存目录。

记录按文件名哈希分到 256 个子目录，逐条原子写入。默认重跑会校验并复用已有记录、补全缺失记录；`latent_cache.overwrite=true` 可以重新编码同一份 AE 的记录。`manifest.json` 保存来源和格式，`summary.json` 保存本次处理数量、是否为部分数据、复用数量及 posterior population mean/std，可用于设置 LDM 的 latent 标准化参数。这些统计不会自动覆盖训练配置。

小规模检查可指定 `latent_cache.max_scenes=8` 和单独的 `output_dir`。这只缓存前 8 个样本；使用该部分缓存跑训练冒烟时应设置 `data.shuffle=false`、限制 batch 数。完整训练需要完整缓存。

## 数据与输入适配

训练和测试直接读取 `/home/ke/code/sim/src/waymo_data/scenario_dreamer_ae_preprocess_waymo` 中的官方 `.pkl`，不需要重建 SMART `.pt` 或地图 token 缓存。

| 用途 | 路径 | 样本选择 |
| --- | --- | --- |
| 训练 | `${paths.cache_root}/train` | 全部 973,984 个 pickle，包含普通图和分区图 |
| 验证与测试 | `${paths.cache_root}/test` | `waymo_eval_set.pkl` 中的 50,000 个文件，保持清单顺序 |

`paths.cache_root` 默认解析为项目内 `src/waymo_data/scenario_dreamer_ae_preprocess_waymo`。目录中还有 `val` 划分；当前为延续官方 50k 指标口径，训练期间的验证和最终测试都读取上述 test 子集。直接把验证目录换为 `val` 会与现有清单及 SQLite 参考统计不匹配，需要一起更换清单和参考统计。

```yaml
data:
  scenario_dreamer_preprocessed: true
  scenario_dreamer_eval_set: ${model.model_config.sd_eval_set}
  train_raw_dir: ${paths.cache_root}/train
  val_raw_dir: ${paths.cache_root}/test
  test_raw_dir: ${paths.cache_root}/test
```

`MultiDataset` 保留 pickle 中的 `agent_states`、`agent_types`、`road_points` 和显式 `edge_index_lane_to_lane / road_connection_types`；`TokenProcessor` 将初始状态接入现有 decoder。没有额外裁剪地图、重采样车道或估计车道拓扑。官方 ego-first 排列转换成 SMART ego-last，进入模型时再按上游递归排序。

数据已在官方 ego-+Y 局部坐标中，不能再按世界坐标做一次平移旋转。默认 `sd_prediction_frame=sd_local`、`sd_gen_timestep=0`，推理返回一个真实生成的初始快照，不构造 GT 历史轨迹。现有输出接口中的 z 为 0 占位值，初始场景指标只使用二维位置。

训练支持 `lg_type=0/1`：分区图的 AE 注意力只连接同侧节点，LDM 保留分区前节点作为无噪声条件。验证/测试使用官方清单中的非分区场景。容量限制仍为最多 30 个 agent、1–100 条车道，与加载的 checkpoint 配置一致。

默认 `map_source=exact` 用于读取精确的训练输入。联合推理不会编码 GT 地图或 agent latent，仅沿用输入场景的节点数量、场景类别和输出坐标参考。此前的 SMART 重建数据和 token 地图适配仍可显式选择：重建数据需保留 SD 地图，旧 token 缓存可设置 `map_source=auto/tokens`。这些不是当前实验的数据来源。

官方 AE pickle 没有 LDM latent 缓存中的 Nocturne compatibility 标签，因此默认固定 `map_id=0`；可设置为 1。标签选择会影响采样。

## 指标口径和输出

默认 `generation_mode=initial_scene`：官方 LDM 从噪声联合生成 lane 与 agent latent，再由 AE 解码 agent 状态/类型、车道点和六类车道连接。

生成的车道图经预测的 predecessor 边进行官方 compaction，再重采样用于 on-road 筛选、横向偏差和朝向偏差。最近车辆距离、尺寸、速度和碰撞率使用生成车辆。GT SQLite 缓存只用于真实参考分布，生成侧不会读取其中的车道几何、拓扑或 `metric_lanes`。联合模式若没有收到生成地图会报错。

复用现有评价器输出：

`nearest_dist_jsd`、`lat_dev_jsd`、`ang_dev_jsd`、`length_jsd`、`width_jsd`、`speed_jsd`、`collision_rate`。

JSD 保留已有官方实现的缩放；collision rate 为百分数。记录名为 `val_closed/sd/*`，test 也沿用此命名以兼容现有监控配置。完整配置检查 50,000 个参考文件的唯一覆盖和原始生成时刻，不重复计算 GT 特征。

每次验证/测试在 Hydra 输出目录生成 `sd_agent_metrics.json`，包含指标、实际样本数、参考样本数、完整清单覆盖标志、数据与指标来源摘要，以及 decoder 的 EMA、步数、地图和条件设置。配置和运行日志同样保存在该目录。

保持原有五元组返回值，新增地图通过 `generated_map` 传递。它包含 SD 局部物理坐标的 `road_points`、预测的 `road_connection_types`、批量边索引和所属场景；评价器按场景恢复局部边索引。报告中 `map_source=generated`、`decoder.mode=initial_scene`，可选导出也保存实际参与指标计算的生成车道。

若需要原来的给定地图基线，可显式切换；评价器会同步使用参考地图：

```bash
python -m src.run experiment=scenario_dreamer action=test \
  model.model_config.decoder.scenario_dreamer.ldm_checkpoint=src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt \
  model.model_config.decoder.scenario_dreamer.generation_mode=lane_conditioned
```

可把联合生成结果保存为官方格式 pickle：

```bash
python -m src.run experiment=scenario_dreamer action=test \
  model.model_config.decoder.scenario_dreamer.ldm_checkpoint=src/waymo_data/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt \
  '+model.model_config.sd_export_dir=${paths.output_dir}/generated_scenes'
```

当前仍沿用输入场景的 lane/agent 数量和固定 `map_id`，没有按官方初始概率矩阵重新采样数量；这与完整官方生成协议有差异。本次只输出所需的 7 项 agent metrics，不额外计算 lane metrics。完整 50k 分数需要实际执行完整命令后获得。

## 验证记录

在本机 `sim` 环境（Python 3.11、PyTorch 2.7.0+cu128、Lightning 2.4.0、RTX 4090）完成官方权重的严格加载。改用官方 pickle 后，`experiment=sd` 已完成 2 个真实训练 batch 的优化及随后的验证，`experiment=scenario_dreamer action=test` 已完成真实 test 数据的冒烟检查。

2026-10-01 联合模式验证：`sim` 环境中通过 8 个真实测试场景的 lane/agent 联合生成与指标计算，以及训练一步后的 8 场景验证。测试报告位于 `logs/scenario_dreamer/2026-10-01_10-40-23/sd_agent_metrics.json`；同目录 `generated_scenes` 保存了参与评价的生成样本。8 张生成地图均与 GT 不同；从这些导出样本独立重算的全部 7 项指标与报告一致（绝对误差小于 1e-12）。20 项 Scenario Dreamer 相关测试通过。

无 LDM checkpoint 训练验证：25 项 Scenario Dreamer 测试通过，新增覆盖仅加载 AE、全新 EMA、随机 LDM 反向传播及参数更新、普通 SMART 模块状态保存/恢复，以及显式 checkpoint 的完整权重加载。另在 `sim` 环境使用默认 Waymo Large 结构、真实预处理 train 数据和现有 `src.run action=fit` 完成 1 个 GPU 优化步骤（batch size 1、关闭验证及 checkpoint 写入）。此检查没有进行完整训练。

AE 阶段验证：32 项 Scenario Dreamer 测试通过，覆盖 AE 的无 checkpoint 初始化、普通/分区图损失、encoder/decoder/分区数量头的梯度、SMART train/validation/test 路由、AE 保存恢复和 AE → LDM 权重衔接。真实预处理数据上完成了从零训练及 2 场景验证、checkpoint 保存、step 1 → 2 断点续训，以及加载该 AE 后的一步 LDM 训练。保存的 AE checkpoint 位于 `logs/scenario_dreamer_ae_smoke/2026-10-01_15-53-48/lightning_logs/version_0/checkpoints/scenario_dreamer-ae-0-1.ckpt`，仅用于冒烟验证，并非收敛后的模型。本次全仓测试为 62/67 通过，剩余仍为下述 5 项既有失败。

Latent 缓存验证：39 项 Scenario Dreamer 测试通过，新增覆盖不同 batch size 下的 agent/lane 排序、分区 mask、缓存/在线后验与损失一致性、逐次随机采样、LDM 反向传播、断点补全、源文件变动/缺失/损坏检测、AE 恢复后的标识更新和各数据划分的独立配置。真实 GPU 检查缓存了 train 的前 8 个场景（`logs/scenario_dreamer_latent_smoke/cache/train`），并在禁止调用 AE encoder 的情况下通过现有入口完成一次 batch size 2 的 LDM 优化，输出目录为 `logs/scenario_dreamer_cached_training_smoke/2026-10-01_16-15-47`。全仓测试为 69/74 通过，仍为原有 5 项失败；未运行全量 973,984 场景缓存。

运行新增回归测试：

```bash
python -m unittest discover -s tests -p 'test_scenario_dreamer*.py' -v
```

联合模式新增测试覆盖 GT latent 隔离、生成地图替换、拓扑合并、跨场景边检查、导出及与官方数值实现的一致性。原有测试继续覆盖真实小型 AE/LDM 的反向传播、冻结 AE、EMA 保存恢复、批量图、车道关系方向/优先级、官方 pickle 字段与清单顺序、分区训练掩码、单帧推理、输出顺序/速度坐标和固定地图 latent 不依赖 GT agent 几何/类型。另抽查 32 个真实场景的 21,678 条车道关系，全部与官方缓存一致，车道坐标的 float32 转换误差小于 1e-6 m。完整 50k 评估尚未执行。

全仓回归中的 5 项既有失败分别位于 `test_denoiser_heading_magnitude.py`（3 项）和 `test_initial_velocity_frame.py`（2 项）；用 Git HEAD 原实现复测得到相同失败。
