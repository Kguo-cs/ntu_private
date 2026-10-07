# VectorWorld 初始化训练和评价

新增 `model.model_config.decoder.init_decoder=vectorworld`。沿用 SMART 的
监督训练、DataModule、初始场景 rollout 和 Scenario Dreamer 的七项 agent metrics。
代码和导入后的权重都在 sim 内；运行时不需要外部 VectorWorld checkout。

## 已接通的模型与接口

- 官方 motion VAE：静态 agent 状态 7 维 + 历史 motion 12 维，agent/lane latent 分别为 18/24 维。
- EGR-DiT 的 Flow、MeanFlow identity/JVP、DDPM 训练与采样，保留原生参数结构。
- `training_stage=autoencoder` 训练 VAE；`training_stage=ldm` 冻结 VAE，训练生成器。
- `training_mode=joint` 保留官方分区条件目标；`lane_conditioned` 固定全部 lane latent，优化 agent 目标，保留 before-partition agent 条件。后者是本地新增的监督训练模式。
- `generation_mode=initial_scene` 同时生成 lane 和 agent，在生成 lane 上评价 agent metrics。
- `generation_mode=lane_conditioned` 使用参考 lane latent，生成所有 agent（包括 ego），在参考 lane 上评价。
  评价只接受 `lg_type=0` 完整 lane；训练接受 `lg_type=0/1`。
- 生成器 EMA 在实际 optimizer step 后更新，随 SMART checkpoint 保存和恢复。
- 返回现有五元组：position、heading、token index、shape、世界坐标 velocity。
  解码的 motion 另存为 `generated_motion`，七项静态初始化指标不使用它。

## 权重

本地可用的四个权重已经导入：

```text
src/waymo_data/vectorworld/checkpoints/
  autoencoder.ckpt
  flow.ckpt
  meanflow.ckpt
  diffusion.ckpt
```

导入仅保留模型、嵌入的 VAE、EMA 和公开配置，删除 optimizer 状态与存储服务配置。
如需重新导入到另一目录：

```bash
conda activate sim
cd /home/ke/code/sim
python src/import_vectorworld_weights.py --output /path/to/new/checkpoints
```

生成器权重的 `ae_checkpoint=null` 合法，会使用其内嵌 VAE。独立发布 VAE 与 Flow
权重内嵌 VAE 已逐 tensor 比对一致。原生权重通过 `ae_checkpoint` /
`ldm_checkpoint` 加载；`ckpt_path` 用于恢复整个 SMART 训练 checkpoint。

## 先准备真实 motion 数据

`training_map2_sd` 精简缓存以及旧 SD AE `.pkl` 只有初始状态，缺少历史监督，
不能直接训练官方 motion VAE，也不能直接编码其 agent latent 来训练生成器。

新工具保持旧 SD 的静态状态、lane、文件名与样本顺序，补充
`agent_motion_raw`、`agent_motion_is_static`、`agent_motion_valid_mask`。
它支持从原始 TFRecords 或保留完整轨迹/SD metadata 的 SMART `.pt` 恢复真实历史，
并严格对齐 agent 来源。原数据不修改，输出有序文件列表和来源 manifest。

当前已有完整验证轨迹，可为官方测试文件补 motion：

```bash
python src/prepare_vectorworld_data.py \
  --split test \
  --input-dir src/waymo_data/scenario_dreamer_ae_preprocess_waymo/test \
  --trajectory-dir src/waymo_data/full/scenario_dreamer_val \
  --sample-list src/waymo_data/waymo_eval_set.pkl \
  --output-dir src/waymo_data/vectorworld/vae/test
```

可先加 `--limit 1` 检查；这个子集只能用于流程检查。256/50k 评价需要对应文件齐全。
已有完整训练 `.pt` 时，可直接转换：

```bash
python src/prepare_vectorworld_data.py \
  --split train --input-dir /path/to/full/training-cache \
  --output-dir src/waymo_data/vectorworld/vae/train
```

旧 SD 训练 pickle 也可从原始数据回填：

```bash
python src/prepare_vectorworld_data.py \
  --split train \
  --input-dir src/waymo_data/scenario_dreamer_ae_preprocess_waymo/train \
  --waymo-root /path/to/waymo110 \
  --output-dir src/waymo_data/vectorworld/vae/train
```

`--waymo-root` 下需有 `training/`。本机当前 `src/waymo_data/waymo110/training`
不存在，`training_map2_sd` 又没有完整轨迹，因此正式训练仍需提供这个原始目录或完整训练缓存。
工具不会用伪造的静止轨迹充当监督。真实的静止或短历史轨迹按官方规则得到合法零物理 motion。

如果数据带 `nocturne_compatible` / `map_id`，模型使用逐场景标签。旧快照缺标签时，
使用配置 `map_id=0` 作为 fallback；这与官方含 Nocturne 分类标签的训练分布有差别，
报告会记录类别与来源。缺类别标签训练的模型应保持相同条件协议，或补齐标签后训练，
不能直接假定它学到了 prior 的两个类别。联合无条件评价使用的 count prior 已与 VectorWorld 原文件逐元素比对一致。

## 训练

以下命令从 sim 根目录执行；默认数据目录为 `src/waymo_data/vectorworld/vae`。

```bash
# 用发布 VAE，从头训练 Flow 生成器
python src/run.py experiment=vectorworld optimizer=vectorworld

# 用发布 VAE，训练固定 lane 的 agent 生成器
python src/run.py experiment=vectorworld_lane_conditioned optimizer=vectorworld

# 从头训练 motion VAE
python src/run.py experiment=vectorworld_ae optimizer=vectorworld

# 其他生成目标
python src/run.py experiment=vectorworld_meanflow optimizer=vectorworld
python src/run.py experiment=vectorworld_diffusion optimizer=vectorworld
```

生成器配置使用 AdamW、1e-4、500 step warmup 后恒定学习率、EMA 0.9999、
32-true、165000 steps。默认 batch 为 32，设备数为 1，沿用本地训练环境；
官方训练配置使用 4 个设备。VAE 配置为 batch 64 / 85000 steps，其优化器沿用 SMART 的
Scenario Dreamer recipe，并不完整复制官方 VAE 的 linear schedule / 2e-5 decay。

使用新训练 VAE 时，要重新计算 latent statistics，不能继续用发布 VAE 的统计量：

```bash
python src/vectorworld_latent_stats.py \
  --ae-checkpoint /path/to/smart-vae.ckpt \
  --data-dir src/waymo_data/vectorworld/vae/train \
  --sample-list src/waymo_data/vectorworld/vae/train_files.pkl \
  --output src/waymo_data/vectorworld/my_flow.yaml \
  --max-scenes 409600 --batch-size 32 --device cuda

python src/run.py experiment=vectorworld optimizer=vectorworld \
  model.model_config.decoder.vectorworld.ae_checkpoint=/path/to/smart-vae.ckpt \
  model.model_config.decoder.vectorworld.ldm_config=/absolute/path/to/my_flow.yaml
```

统计工具逐维计算 posterior 的 `E[mu]` 和 `E[mu² + exp(logvar)]`，覆盖训练时的
posterior 采样分布；这是解析估计，与官方有限随机采样的统计估计不要求逐位相同。
输出完整生成器 YAML 和统计来源 JSON。要配置 MeanFlow/DDPM，
用对应 `waymo_meanflow.yaml` / `waymo_diffusion.yaml` 作为 `--base-config`。

## 评价

```bash
# Flow 权重，联合生成；默认前 256 个官方文件
python src/run.py experiment=vectorworld_eval optimizer=vectorworld

# 完整参考 lane 条件；仅编码 lane，直接使用已有 SD test 数据，无需 agent history
python src/run.py experiment=vectorworld_lane_conditioned_eval optimizer=vectorworld

# 切换到 MeanFlow 权重；默认按权重配置使用 1 步
python src/run.py experiment=vectorworld_eval optimizer=vectorworld \
  model.model_config.decoder.vectorworld.ldm_checkpoint=/home/ke/code/sim/src/waymo_data/vectorworld/checkpoints/meanflow.ckpt

# 评价自己训练的 SMART 生成器 checkpoint（读取其中的 VAE/生成器/EMA/配置）
python src/run.py experiment=vectorworld_lane_conditioned_eval optimizer=vectorworld \
  model.model_config.decoder.vectorworld.ldm_checkpoint=/path/to/smart-generator.ckpt

# 完整 50k 测试
python src/run.py experiment=vectorworld_eval optimizer=vectorworld \
  trainer.limit_test_batches=1.0 model.model_config.sd_require_full_set=true
```

Flow 默认使用其权重配置的 24 步 Heun；MeanFlow 的步数可用
`model.model_config.decoder.vectorworld.sampling_steps=3` 覆盖。
DDPM 步数由训练噪声调度决定。EMA 默认开启。

联合模式默认 `scene_count_source=official_prior`，从匹配的 Waymo prior 采样数量和地图类别；
用 `scene_count_source=input` 则使用参考场景的数量，属于另一种评价协议。
Lane-conditioned 模式使用输入数量。前 256 个文件沿用同一个 `waymo_eval_set.pkl` 顺序；
前提是回填后保留全部原文件名。

结果写入运行目录的 `sd_agent_metrics.json`，包括生成/参考地图来源、
decoder/采样/EMA 配置、motion 及 map 条件来源、样本数和参考分布信息。
七项指标的计算和缩放沿用当前 sim evaluator；单场景结果不能用于比较论文数值。
这里只评价初始化场景，不包含 VectorWorld 后续 DeltaSim 控制仿真指标。

## 验证记录

- 四个真实权重 strict load 成功；真实 VAE 编解码、MeanFlow 单步生成有限且 lane 条件保持。
- 真实完整轨迹转换：7 agents / 67 lanes，motion 为 [7,12]，全部有效。
- 同一真实场景的联合生成与 lane-conditioned SMART 评价均输出七项指标。
- 小模型在上述真实数据上完成 SMART 监督训练一步和 EMA 更新。
- 62 项 VectorWorld CPU 测试与 93 项 Scenario Dreamer 回归通过，覆盖三种生成器 loss/backward、motion VAE、坐标/排序/拓扑、
  完整 lane 检查、native/SMART checkpoint、EMA、数据转换与 latent statistics。

核心来源及差异见 [SOURCES.json](../src/smart/vectorworld/core/SOURCES.json)。
