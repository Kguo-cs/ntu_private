# 从头训练 loss 搜索：自动汇总

以下只包含已经完成并保存了七项生成指标的实际评价。固定综合分数为七项 `metric / scale` 的等权平均，越低越好；这些 scale 只是数值尺度，不来自模型权重，也不是论文的官方综合指标。

| 指标 | 固定 scale |
|---|---:|
| nearest_dist_jsd | 1 |
| lat_dev_jsd | 1 |
| ang_dev_jsd | 10 |
| length_jsd | 20 |
| width_jsd | 15 |
| speed_jsd | 20 |
| collision_rate | 5 |

指标使用官方 helper 输出单位：nearest/lateral JSD ×10，angle/length/width/speed JSD ×100，collision 为百分数；上述 scale 已转换为相同单位。

生成 seed 在训练 seed 内平均，再跨训练 seed 等权平均。误差条仅为描述性训练 seed 样本标准差（N≥2 时 ddof=1，N=1 时显示 0），不是置信区间；重复评价不会增加独立训练样本量。不同学习率、评价子集、权重模式、初始化范围分别汇总，不混合排名。

lat/angle 只统计 on-road 车辆，因此同时检查车辆数量和 on-road 覆盖。coverage flags 仅提示问题，不修改官方指标或隐藏排除结果。短期训练的趋势不能证明最终完整训练已经收敛或获得全局最优。

实际评价记录：**48**；分组：**8**。配对 heatmap anchor 为 `warm_winner`。

图表仅显示指定方案：`geo_mse32_col20_heading_l1_shape2, warm_winner`；下表、全部原始记录、聚合统计、排序和后期趋势仍包含所有实际方案。

## 比较组 320db4fa99dc

LR=0.0005，weights=online，samples=4096，initialization=map_encoder_and_G1_from_scratch。

| Budget | Variant | Train seeds | Score ± SD | Nearest | Lateral | Angle | Length | Width | Speed | Collision% | Vehicles/GT | On-road |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | geo_mse32_col20_heading_l1_shape2 | 3 | 0.306 ± 0.021 | 0.1665 | 0.9251 | 1.7157 | 2.0274 | 3.7149 | 1.6195 | 2.2403 | 1.016 | 78.1% |
| 1536 | warm_winner | 3 | 0.417 ± 0.110 | 0.1685 | 0.9144 | 4.6569 | 5.3176 | 6.4413 | 1.2492 | 3.0592 | 1.056 | 76.8% |
| 2048 | geo_mse32_col20_heading_l1_shape2 | 3 | 0.275 ± 0.037 | 0.1545 | 0.9246 | 0.9021 | 1.7464 | 2.4210 | 1.0133 | 2.2888 | 1.015 | 79.8% |
| 2048 | warm_winner | 3 | 0.472 ± 0.183 | 0.1532 | 0.8904 | 3.2903 | 7.5169 | 13.2823 | 1.1954 | 3.0373 | 1.056 | 78.3% |
## 比较组 4795bc325dd2

LR=0.0005，weights=ema，samples=4096，initialization=map_encoder_and_G1_from_scratch。

| Budget | Variant | Train seeds | Score ± SD | Nearest | Lateral | Angle | Length | Width | Speed | Collision% | Vehicles/GT | On-road |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | geo_mse32_col20_heading_l1_shape2 | 3 | 0.239 ± 0.008 | 0.1654 | 0.9408 | 0.7950 | 1.3650 | 1.2773 | 0.2442 | 1.6018 | 1.041 | 78.5% |
| 1536 | warm_winner | 3 | 0.257 ± 0.004 | 0.1451 | 0.9130 | 2.6107 | 1.1058 | 0.6807 | 0.1687 | 1.8449 | 1.032 | 77.1% |
| 2048 | geo_mse32_col20_heading_l1_shape2 | 3 | 0.223 ± 0.005 | 0.1614 | 0.9299 | 0.5736 | 0.9418 | 0.7948 | 0.1809 | 1.5046 | 1.024 | 79.5% |
| 2048 | warm_winner | 3 | 0.231 ± 0.001 | 0.1258 | 0.8996 | 1.7782 | 0.7983 | 0.5064 | 0.1349 | 1.6505 | 1.027 | 79.0% |

## 图表

![learning_curves_320db4fa99dc.png](learning_curves_320db4fa99dc.png)

![learning_curves_4795bc325dd2.png](learning_curves_4795bc325dd2.png)

![coverage_curves_320db4fa99dc.png](coverage_curves_320db4fa99dc.png)

![coverage_curves_4795bc325dd2.png](coverage_curves_4795bc325dd2.png)

![metric_heatmap_320db4fa99dc_step_1536.png](metric_heatmap_320db4fa99dc_step_1536.png)

![metric_heatmap_320db4fa99dc_step_2048.png](metric_heatmap_320db4fa99dc_step_2048.png)

![metric_heatmap_4795bc325dd2_step_1536.png](metric_heatmap_4795bc325dd2_step_1536.png)

![metric_heatmap_4795bc325dd2_step_2048.png](metric_heatmap_4795bc325dd2_step_2048.png)

![gradient_training_traces.png](gradient_training_traces.png)

逐次记录见 [observations.csv](observations.csv)，分组统计见 [aggregate.csv](aggregate.csv)，完整协议和原始报告位置记录在 JSON 中。学习趋势仅使用同一比较组最后两个预算的实际点，未拟合所谓最终性能预测。
