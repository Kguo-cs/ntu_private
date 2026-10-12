# 从头训练 loss 搜索：自动汇总

以下只包含已经完成并保存了七项生成指标的实际评价。固定综合分数为七项 `metric / scale` 的等权平均，越低越好；这些 scale 只是数值尺度，不来自模型权重，也不是论文的官方综合指标。

| 指标 | 固定 scale |
|---|---:|
| nearest_dist_jsd | 0.1 |
| lat_dev_jsd | 0.1 |
| ang_dev_jsd | 0.1 |
| length_jsd | 0.2 |
| width_jsd | 0.15 |
| speed_jsd | 0.2 |
| collision_rate | 5 |

生成 seed 在训练 seed 内平均，再跨训练 seed 等权平均。误差条仅为描述性训练 seed 标准差；重复评价不会增加独立训练样本量。不同学习率、评价子集、权重模式、初始化范围分别汇总，不混合排名。

lat/angle 只统计 on-road 车辆，因此同时检查车辆数量和 on-road 覆盖。coverage flags 仅提示问题，不修改官方指标或隐藏排除结果。短期训练的趋势不能证明最终完整训练已经收敛或获得全局最优。

实际评价记录：**5**；分组：**5**。配对 heatmap anchor 为 `anchor`。

## 比较组 8bcd9b085236

LR=0.0001，weights=online，samples=1024，initialization=map_encoder_and_G1_from_scratch。

| Budget | Variant | Train seeds | Score ± SD | Nearest | Lateral | Angle | Length | Width | Speed | Collision% | Vehicles/GT | On-road |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | legacy | 1 | 63.169 ± 0.000 | 0.5876 | 0.7797 | 25.6921 | 13.5908 | 10.7498 | 5.8231 | 14.2554 | 0.987 | 53.9% |
| 256 | anchor | 1 | 77.759 ± 0.000 | 0.5509 | 0.8811 | 22.3369 | 27.0189 | 14.0387 | 14.7501 | 20.9608 | 1.185 | 52.5% |
| 512 | anchor | 1 | 32.924 ± 0.000 | 0.6016 | 0.7774 | 16.3666 | 1.1065 | 2.0747 | 6.1335 | 14.9067 | 0.960 | 55.6% |
| 512 | legacy | 1 | 50.146 ± 0.000 | 0.4575 | 0.8168 | 18.0920 | 10.9016 | 11.6953 | 4.5841 | 9.7965 | 0.900 | 64.9% |
| 1024 | anchor | 1 | 32.237 ± 0.000 | 0.5461 | 0.8357 | 15.3039 | 1.9459 | 4.8763 | 2.7105 | 15.0491 | 1.053 | 56.0% |

## 图表

![learning_curves_8bcd9b085236.png](learning_curves_8bcd9b085236.png)

![metric_heatmap_8bcd9b085236_step_1024.png](metric_heatmap_8bcd9b085236_step_1024.png)

![metric_heatmap_8bcd9b085236_step_256.png](metric_heatmap_8bcd9b085236_step_256.png)

![metric_heatmap_8bcd9b085236_step_512.png](metric_heatmap_8bcd9b085236_step_512.png)

![gradient_training_traces.png](gradient_training_traces.png)

逐次记录见 [observations.csv](observations.csv)，分组统计见 [aggregate.csv](aggregate.csv)，完整协议和原始报告位置记录在 JSON 中。学习趋势仅使用同一比较组最后两个预算的实际点，未拟合所谓最终性能预测。
