# Scenario Dreamer 公开权重评估

[scenario_dreamer_release.yaml](../configs/experiment/scenario_dreamer_release.yaml) 是独立测试配置，继承现有训练配置，加载本仓库中的官方 Waymo Large LDM、内嵌 AE 和 EMA。它使用官方初始概率矩阵联合采样 agent 数量、lane 数量和 map ID，然后联合生成 lane 与 agent，并在生成车道图上评价 agent metrics。

在 `/home/ke/code/sim` 执行完整 50,000 场景评估：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run experiment=scenario_dreamer_release
```

数据仍读取 `src/waymo_data/scenario_dreamer_ae_preprocess_waymo/test`，场景清单使用 `src/waymo_data/waymo_eval_set.pkl`，GT 分布使用 `src/waymo_data/sd_real_metric_cache.sqlite`。矩阵乘法设置 `medium` 与[官方 `eval.py`](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/675423469766bf2fd8a6b569ef1869a6f1e76993/eval.py#L9) 一致，其他实验仍默认 `highest`。生成数量由概率矩阵采样，不再沿用每条输入的 GT 节点数量或固定 map ID。当前配置为单 GPU、float32 参数、`float32_matmul_precision=medium`、batch size 32、seed 0、每个条目生成一次；latent cache 全部关闭。

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

完整 50k 复现仍需实际运行验证；新增此配置不代表上述目标已经复现成功。完整运行前的冒烟结果仅验证权重加载、数量采样、生成地图和指标链路。

## 已完成的验证

- 45 项 Scenario Dreamer 回归测试通过，包括可变 agent 数量、先验索引、随机流、原 AE/LDM 训练和 latent cache。
- 本地 AE 与 Large LDM 的 SHA-256 均与官方发布值一致。
- 真实公开权重完成 64 场景导出测试和 128 场景正式数值设置测试；这两次都是小样本链路验证。
- 独立读取 64 个导出场景，逐个核对采样数量并重算七项指标，与报告在 `1e-12` 容差内一致。

2026-10-01 18:00 启动的完整 50k 任务记录位于
`logs/scenario_dreamer_release/2026-10-01_18-00-20/`：`launch.json` 保存 PID 和准确命令，
`console.log` 保存进度，成功完成后写入 `sd_agent_metrics.json`。启动时估计总耗时约 3–4 小时；
以该目录最终报告为准，此文档不将尚未完成的任务记作复现成功。
