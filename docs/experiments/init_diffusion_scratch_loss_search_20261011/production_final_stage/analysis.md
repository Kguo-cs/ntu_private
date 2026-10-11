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

实际评价记录：**18**；分组：**18**。配对 heatmap anchor 为 `warm_winner`。

图表仅显示指定方案：`geo_mse32_col20_heading_l1_shape2, geometry_strong_heading_l1_width_p2, warm_winner`；下表、全部原始记录、聚合统计、排序和后期趋势仍包含所有实际方案。

## 比较组 81e7d4ad67b7

LR=0.0005，weights=ema，samples=1024，initialization=map_encoder_and_G1_from_scratch。

| Budget | Variant | Train seeds | Score ± SD | Nearest | Lateral | Angle | Length | Width | Speed | Collision% | Vehicles/GT | On-road |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 768 | geo_mse32_col20_heading_l1_shape2 | 1 | 0.283 ± 0.000 | 0.2195 | 1.0256 | 1.4408 | 1.8170 | 1.0340 | 0.5363 | 2.0127 | 0.988 | 77.8% |
| 768 | geometry_strong_heading_l1_width_p2 | 1 | 0.321 ± 0.000 | 0.2412 | 1.0439 | 1.7284 | 1.9930 | 1.1338 | 0.4200 | 2.9793 | 1.010 | 73.8% |
| 768 | warm_winner | 1 | 0.396 ± 0.000 | 0.2951 | 0.9662 | 7.2335 | 1.6563 | 0.7896 | 0.6384 | 3.0832 | 0.997 | 72.0% |
| 1536 | geometry_strong_heading_l1_width_p2 | 1 | 0.263 ± 0.000 | 0.1567 | 1.0274 | 0.9522 | 1.5142 | 0.6095 | 0.2179 | 2.1759 | 1.022 | 77.8% |
| 1536 | geo_mse32_col20_heading_l1_shape2 | 1 | 0.263 ± 0.000 | 0.2086 | 0.9984 | 1.0248 | 1.6710 | 1.0713 | 0.3731 | 1.7996 | 1.047 | 78.7% |
| 1536 | warm_winner | 1 | 0.269 ± 0.000 | 0.1679 | 0.9944 | 2.8073 | 1.3982 | 0.5848 | 0.3092 | 1.5861 | 1.023 | 78.0% |
| 2048 | geo_mse32_col20_heading_l1_shape2 | 1 | 0.238 ± 0.000 | 0.1972 | 1.0298 | 0.6185 | 1.3063 | 0.7871 | 0.3600 | 1.2008 | 1.024 | 80.1% |
| 2048 | geometry_strong_heading_l1_width_p2 | 1 | 0.242 ± 0.000 | 0.1353 | 1.0070 | 0.5949 | 1.2847 | 0.4511 | 0.3400 | 1.8991 | 1.027 | 79.0% |
| 2048 | warm_winner | 1 | 0.242 ± 0.000 | 0.1524 | 0.9334 | 2.0104 | 1.1383 | 0.5166 | 0.3167 | 1.5065 | 1.008 | 79.7% |
## 比较组 c929105420f6

LR=0.0005，weights=online，samples=1024，initialization=map_encoder_and_G1_from_scratch。

| Budget | Variant | Train seeds | Score ± SD | Nearest | Lateral | Angle | Length | Width | Speed | Collision% | Vehicles/GT | On-road |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 768 | geometry_strong_heading_l1_width_p2 | 1 | 0.408 ± 0.000 | 0.2862 | 0.9767 | 1.9385 | 2.8171 | 10.4843 | 0.5711 | 2.6563 | 0.990 | 74.0% |
| 768 | geo_mse32_col20_heading_l1_shape2 | 1 | 0.498 ± 0.000 | 0.2202 | 0.9840 | 1.2312 | 11.4563 | 17.7980 | 0.7651 | 1.8092 | 0.904 | 79.6% |
| 768 | warm_winner | 1 | 0.681 ± 0.000 | 0.3546 | 0.9073 | 11.2885 | 13.6217 | 15.8228 | 5.7532 | 1.7771 | 0.839 | 72.8% |
| 1536 | warm_winner | 1 | 0.314 ± 0.000 | 0.1479 | 0.9759 | 3.0523 | 3.7526 | 0.4336 | 0.2436 | 2.7161 | 1.055 | 78.0% |
| 1536 | geometry_strong_heading_l1_width_p2 | 1 | 0.324 ± 0.000 | 0.1598 | 1.0216 | 1.4703 | 3.5459 | 3.3481 | 1.3404 | 2.3706 | 1.021 | 79.8% |
| 1536 | geo_mse32_col20_heading_l1_shape2 | 1 | 0.337 ± 0.000 | 0.1657 | 0.9921 | 1.3696 | 4.9011 | 5.3425 | 0.1584 | 2.2914 | 0.954 | 80.3% |
| 2048 | geometry_strong_heading_l1_width_p2 | 1 | 0.290 ± 0.000 | 0.1461 | 0.9788 | 2.0135 | 1.5139 | 0.4694 | 0.4698 | 2.8534 | 1.004 | 79.3% |
| 2048 | geo_mse32_col20_heading_l1_shape2 | 1 | 0.332 ± 0.000 | 0.1659 | 1.0275 | 0.9366 | 3.7563 | 4.5365 | 1.0765 | 2.4617 | 0.973 | 80.6% |
| 2048 | warm_winner | 1 | 0.460 ± 0.000 | 0.1314 | 0.9541 | 1.8545 | 8.3271 | 13.3892 | 0.8424 | 2.9832 | 1.026 | 79.6% |
逐次记录见 [observations.csv](observations.csv)，分组统计见 [aggregate.csv](aggregate.csv)，完整协议和原始报告位置记录在 JSON 中。学习趋势仅使用同一比较组最后两个预算的实际点，未拟合所谓最终性能预测。
