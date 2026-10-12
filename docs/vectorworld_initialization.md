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

如果数据带 `nocturne_compatible` / `map_id`，模型优先使用逐场景显式标签。旧快照缺标签时，
当前配置先查询本地 `map_category_index`；没有索引才使用配置 `map_id=0` 作为 fallback。
索引现在保留原始数据 split，并使用 train、val、heldout test 三份 Nocturne 名单。
报告会记录类别与来源。缺类别标签训练的模型应保持相同条件协议，或明确标签协议后训练，
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

Lane-conditioned 默认在原始完整参考 lane 上评价 agent。原始 VectorWorld 的
`LDM.forward` 会将 lane latent 解码后的重建 lane 写回场景，因此其原生指标使用
VAE 重建地图。要比较原生口径，设置
`model.model_config.decoder.vectorworld.lane_eval_map_source=reconstructed`。

SD 原始 cache 缺少 `nocturne_compatible` 时，默认使用 sim 内的 split-aware 索引：

- `training.*` 只查 `nocturne_train_filenames.pkl`。
- `validation.*` 查 `nocturne_val_filenames.pkl` 与 `nocturne_test_filenames.pkl` 的并集；
  后者是从原始 validation 移入测试集的 heldout 场景。
- `testing.*` 设为 0，避免与 validation 的 shard/record 编号碰撞。

官方 50k 清单的新标签为 `map_id=0: 43935`、`map_id=1: 6065`。显式场景标签仍优先。
当前工作区的本地索引已更新；迁移或旧安装可执行
`python src/import_vectorworld_weights.py --metadata-only`，导入时需要三份官方名单，
评价时只读取 sim 内的 JSON。没有改写训练样本或 checkpoint。

发布源码的旧 splitless train+val 规则会得到 44165/5835；它仅保留为显式历史对照，
用 `--map-category-policy native_vae_train_plus_val_whitelist` 导入到单独目录。
加载旧索引会提示 warning，不能将它视为已修正的默认协议。结果报告新增
`map_category_policy`；本地默认值是 `split_aware_nocturne_whitelist`。


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
DDPM 步数由训练噪声调度决定。EMA 默认开启。Flow Heun 现在在 corrector 之前恢复
全部条件节点，clipping 后也恢复条件，保证每次 field 计算读到固定 lane/agent；
联合生成无条件节点时，采样结果保持不变。报告字段
`heun_condition_policy=fixed_before_corrector` 标识该修正。

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


## Lane-conditioned 差异诊断与修正（2026-10-07）

修正前的 50k 结果使用原始参考 lane：speed_jsd=0.03729、lat_dev_jsd=0.16115、
ang_dev_jsd=0.12348、collision_rate=7.0364%。它不是 joint 生成结果；也不能仅凭
`training_mode=lane_conditioned` 将公开 joint 权重视为本地 lane-conditioned 训练权重。
发布 Flow 训练对 lg_type=0 的全部 lane 加噪；完整干净 lane 是评价时施加的条件。

修正前，原始与本地 Flow Heun 的 predictor 更新 lane 后，corrector 使用暂时偏移的
lane，再于整步末尾恢复条件。现已在本地采样器中修正，并通过逐次 field 输入检查。
以下诊断工具用相同初始噪声对照旧行为、两个修正分别启用，以及两个修正同时启用：

```bash
python src/check_vectorworld_lane_conditioning.py
```

工具取官方清单中普通 testing、原生索引命中的 testing、heldout validation 各 8 个场景。
此前诊断对同一组初始 agent 噪声、Flow24/Heun/CFG4/EMA190200，实测如下；
表内的两个修正是分别启用的，不能视为修正后的完整评价结果：

| 对照 | Agent 位置平均绝对偏移 | 子集碰撞率 |
|---|---:|---:|
| 修正前采样（medium） | 基准 | 6.2112% |
| 每次 field 调用固定 lane | 0.8201 m | 9.9379% |
| 保留 split 的 heldout 标签 | 5.1302 m | 9.3333% |
| highest 矩阵精度 | 0.0207 m | 6.2112% |

偏移是逐 agent 配对的二维位置距离，包含所有 agent 类型。这 24 个场景经过分组抽样，
不代表 50k 指标；strict clamp 或语义标签在此子集上也没有保证碰撞率改善。
原生 corrector 输入的 lane latent 最大偏移为 0.7012，strict 版本为 0。
同一批生成 agent 切换原始/重建 lane，只影响与 lane 有关的评价，不能解释碰撞率差异。
旧报告保存在 `logs/vectorworld_lane_conditioning_audit.json`。当前诊断脚本默认输出
`logs/vectorworld_lane_conditioning_audit.json`，可用 `--output` 指定新的报告名。
历史对照需要 sim 内的 `nocturne_compatible_keys_native_legacy.json`；当前工作区已保留。
迁移时可用 shared map-category CLI 的 `--policy native_vae_train_plus_val_whitelist`
导入该文件。`tests/test_vectorworld_flow_conditioning.py` 覆盖每次 field 输入、部分条件、
clip 范围外条件值和 Euler，并验证 joint 采样与修正前逐值一致。
完整 50k 的修正后结果仍需重新评价；采样修正和语义标签修正不保证全部指标改善。
