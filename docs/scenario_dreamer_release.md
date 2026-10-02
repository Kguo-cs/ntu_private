# Scenario Dreamer 公开权重评估

[scenario_dreamer_release.yaml](../configs/experiment/scenario_dreamer_release.yaml) 是独立测试配置，继承现有训练配置，加载本仓库中的官方 Waymo Large LDM、内嵌 AE 和 EMA。它使用官方初始概率矩阵联合采样 agent 数量、lane 数量和 map ID，然后联合生成 lane 与 agent，并在生成车道图上评价 agent metrics。

在 `/home/ke/code/sim` 执行完整 50,000 场景评估：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run experiment=scenario_dreamer_release
```

数据仍读取 `src/waymo_data/scenario_dreamer_ae_preprocess_waymo/test`，场景清单使用 `src/waymo_data/waymo_eval_set.pkl`，GT 分布使用 `src/waymo_data/sd_real_metric_cache.sqlite`。矩阵乘法设置 `medium` 与[官方 `eval.py`](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/675423469766bf2fd8a6b569ef1869a6f1e76993/eval.py#L9) 一致，其他实验仍默认 `highest`。生成数量由概率矩阵采样，不再沿用每条输入的 GT 节点数量或固定 map ID。当前配置为单 GPU、float32 参数、`float32_matmul_precision=medium`、batch size 128、seed 0、每个条目生成一次；latent cache 全部关闭。

先检查少量场景时，必须同时关闭完整集合检查：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run experiment=scenario_dreamer_release \
  trainer.limit_test_batches=1 data.test_batch_size=8 \
  model.model_config.sd_require_full_set=false
```

这里的整数 `limit_test_batches=1` 表示一个 batch；正式配置中的浮点数 `1.0` 表示全部 batch。小样本仍对照完整 GT 分布，不能作为论文或公开权重结果的复现分数。

运行后检查 `logs/scenario_dreamer_release/<时间>/sd_agent_metrics.json`。正式结果应包含 `num_samples=50000`、`num_gt_samples=50000`、`full_membership=true`、`map_source=generated`，并在 decoder 元数据中记录官方权重、EMA 和数量采样设置。终端会打印这七项指标；JSON 保留未经过终端 float32 转换的完整精度。

应先对照[作者发布的预训练权重结果](https://github.com/princeton-computational-imaging/scenario-dreamer#pre-trained-checkpoints)。公开 Waymo Large 权重训练约 250k 步，论文使用约 165k 步，两个表的数值不同。

| 指标 | 公开 Waymo Large 权重目标 |
| --- | ---: |
| `nearest_dist_jsd` | 0.05 |
| `lat_dev_jsd` | 0.03 |
| `ang_dev_jsd` | 0.08 |
| `length_jsd` | 0.43 |
| `width_jsd` | 0.29 |
| `speed_jsd` | 0.38 |
| `collision_rate` | 4.01% |

表内 JSD 与本项目输出均使用官方缩放：nearest/lat 乘 10，其余 JSD 乘 100；collision rate 为百分数。随机采样、batch 划分和运行环境可能带来数值波动，不能要求逐位一致。

已完成一次完整 50k 评价，实测结果接近公开权重报告；具体数值及仍存在的运行设置差异见下文。冒烟结果仅验证权重加载、数量采样、生成地图和指标链路。

## 已完成的验证

- 45 项 Scenario Dreamer 回归测试通过，包括可变 agent 数量、先验索引、随机流、原 AE/LDM 训练和 latent cache。
- 本地 AE 与 Large LDM 的 SHA-256 均与官方发布值一致。
- 真实公开权重完成 64 场景导出测试和 128 场景正式数值设置测试；这两次都是小样本链路验证。
- 独立读取 64 个导出场景，逐个核对采样数量并重算七项指标，与报告在 `1e-12` 容差内一致。

## 完整 50k 实测结果

本次有效结果来自 `2026-10-01_18-53-44`，当日 21:30 完成。
[完整报告](../logs/scenario_dreamer_release/2026-10-01_18-53-44/sd_agent_metrics.json) 和
[实际运行配置](../logs/scenario_dreamer_release/2026-10-01_18-53-44/.hydra/config.yaml) 是结果依据，
不再以先前 `18-00-20` 目录的启动记录判定完成状态。

- 生成与参考均为 50,000 场景，`full_membership=true`，在生成的车道图上计算指标。
- 官方 checkpoint step 249162，EMA 开启；100 diffusion steps、lane temperature 0.75、guidance 4。
- `map_id` 分别采样 30,984 / 19,016 次，即 61.968% / 38.032%。
- 生成 364,495 辆 vehicle，其中 15,128 辆碰撞，碰撞率为 4.1504%。
- seed 0，test batch size 128，共 391 batches，测试耗时 2:36:58；float32 参数和 `medium` 矩阵计算设置。

| 指标 | 公开权重目标 | 本次实测 |
| --- | ---: | ---: |
| `nearest_dist_jsd` | 0.05 | 0.049291 |
| `lat_dev_jsd` | 0.03 | 0.031587 |
| `ang_dev_jsd` | 0.08 | 0.085929 |
| `length_jsd` | 0.43 | 0.433862 |
| `width_jsd` | 0.29 | 0.306389 |
| `speed_jsd` | 0.38 | 0.357774 |
| `collision_rate` | 4.01% | 4.150400% |

复现本次 batch 设置可显式运行：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run experiment=scenario_dreamer_release \
  data.test_batch_size=128 seed=0
```

该结果可记录为“本仓库对公开权重的 50k 复现实测”，不能声称逐位复现作者数值。
官方默认生成 batch size 为 32，本次为 128；batch 划分会改变 diffusion 随机噪声的抽取与分配。
数量采样使用独立 CPU generator，概率与 API 对齐，但随机流与官方模型初始化后的全局 RNG 不同。
这些差异可能影响最终分数；只有一次完整运行，尚未量化剩余差异的随机波动范围。

## 完整车道条件下的 agent 评价

使用独立配置 [scenario_dreamer_lane_conditioned.yaml](../configs/experiment/scenario_dreamer_lane_conditioned.yaml)：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run experiment=scenario_dreamer_lane_conditioned
```

该配置继续使用本地公开 AE/LDM 权重及 EMA，只接受有明确 `lg_type=0` 元数据的完整、非分区 lane 图。
完整图先经 AE 编码，LDM 在固定 lane latent 的条件下生成 agents；lane 和 agent 数量沿用输入场景，
`scene_count_source=input`。评价使用相应的原始参考车道，报告应为 `mode=lane_conditioned`、
`map_source=reference`，不会把 AE 重建车道作为新的生成地图参与评价。

数据清单仍为官方 50k 完整场景；默认检查完整集合。遇到 partitioned 场景、混合了 partitioned 的 batch
或缺失图类型元数据时会明确报错，不静默跳过，不把 token 地图当作可验证的完整 lane 图。
限制仅针对 lane-conditioned 推理，原有包含 partitioned 场景的 AE/LDM 训练继续可用。

先测试 8 个场景：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run experiment=scenario_dreamer_lane_conditioned \
  trainer.limit_test_batches=1 data.test_batch_size=8 \
  model.model_config.sd_require_full_set=false
```

在已有 `scenario_dreamer` / `scenario_dreamer_latent` 训练配置中，也可以设置
`model.model_config.decoder.scenario_dreamer.generation_mode=lane_conditioned`，让验证阶段使用完整车道条件；
训练阶段仍保持原 LDM loss。若从 `scenario_dreamer_release` 手动覆盖模式，需同时设置
`model.model_config.decoder.scenario_dreamer.scene_count_source=input`。

这是基于真实完整车道的条件生成任务，不能直接以本页联合 lane/agent 生成的公开目标表判断是否复现成功。

本模式已通过 54 项 Scenario Dreamer 回归测试及 8 场景公开权重 GPU 冒烟。
[冒烟报告](../logs/scenario_dreamer_lane_conditioned_smoke/2026-10-02_10-53-02/sd_agent_metrics.json)
确认 `map_source=reference`；8 个导出场景的 lane 几何与拓扑均与参考图逐项一致。
本次未运行 lane-conditioned 的完整 50k 评价。
