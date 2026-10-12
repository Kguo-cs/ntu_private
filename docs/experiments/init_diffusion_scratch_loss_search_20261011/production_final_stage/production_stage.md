# Batch512 原训练批量确认：实际结果

本节只汇总三个已经完成的从头训练方案：每个方案 768、1536、2048 updates，各评价 online/EMA，总计 **18 次实际评价**。训练 seed=817，生成 seed=2817；使用固定 search1024 场景，因此这不是后续独立 heldout4096 的三训练 seed 部署结论，也不能证明 128 epochs 已达到最优。

共享设置：map encoder+G1 随机初始化；固定 TRAIN8192 normalizer；固定 ego 位置、朝向、尺寸和类型；速度生成；crop50m；AdamW β=(0.9,0.999)、weight_decay=0.01；32-true、medium matmul；仅 G1 EMA，共享 map encoder 仍为在线权重；20-step sampling。批量 **512**、初始 LR **5e-4**。

实际场景曝光数：768=393,216；1536=786,406；2048=1,048,524。完整训练集 493,542 个缓存，每 epoch 964 batches、末批486；2048 updates 包含两次末批，实际曝光不是简单的 2048×512。训练顺序为 seed 后固定 manifest 并逐 epoch 重复，shuffle=False；native 训练默认 shuffle=True，正式生产训练仍需长期验证这个差别。

| Short label | Recipe variant |
|---|---|
| Old recipe | `warm_winner` |
| Geometry + head L1 | `geo_mse32_col20_heading_l1_shape2` |
| Head L1 + width p2 | `geometry_strong_heading_l1_width_p2` |

## 2048 updates：原始 JSD 与覆盖

下表六个 JSD 使用自然对数，理论上界为 ln2：nearest/lateral 的 helper 输出除以10，angle/length/width/speed 除以100。collision 为百分数；S 仍按 helper 原单位尺度 [1,1,10,20,15,20,5] 计算，越低越好。仅一个训练 seed，不能从两种权重模式或多 checkpoint 产生样本 SD/置信区间。

### EMA

| Recipe | nearest | lat | angle | length | width | speed | collision% | S | On-road% | V/GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old recipe | 0.01524 | 0.09334 | 0.02010 | 0.01138 | 0.00517 | 0.00317 | 1.50649 | 0.242189 | 79.727 | 1.0076 |
| Geometry + head L1 | 0.01972 | 0.10298 | 0.00619 | 0.01306 | 0.00787 | 0.00360 | 1.20082 | 0.237820 | 80.123 | 1.0243 |
| Head L1 + width p2 | 0.01353 | 0.10070 | 0.00595 | 0.01285 | 0.00451 | 0.00340 | 1.89906 | 0.241849 | 79.047 | 1.0267 |

### Online（诊断）

| Recipe | nearest | lat | angle | length | width | speed | collision% | S | On-road% | V/GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old recipe | 0.01314 | 0.09541 | 0.01854 | 0.08327 | 0.13389 | 0.00842 | 2.98317 | 0.459818 | 79.615 | 1.0264 |
| Geometry + head L1 | 0.01659 | 0.10275 | 0.00937 | 0.03756 | 0.04536 | 0.01076 | 2.46166 | 0.331930 | 80.643 | 0.9728 |
| Head L1 + width p2 | 0.01461 | 0.09788 | 0.02014 | 0.01514 | 0.00469 | 0.00470 | 2.85342 | 0.289632 | 79.309 | 1.0043 |

## Recipe 优势与代价（2048 相对 Old recipe）

各指标 Δ% 小于0表示降低；on-road 按绝对百分点变化。所有结果完整列出，不能只看综合分数或丢弃尺寸/速度的退步。

### EMA

| Recipe vs Old recipe | nearest | lat | angle | length | width | speed | collision% | S Δ% | On-road Δpp | V/GT Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Geometry + head L1 | 29.42 | 10.32 | -69.23 | 14.76 | 52.37 | 13.65 | -20.29 | -1.80 | +0.395 | +0.0167 |
| Head L1 + width p2 | -11.17 | 7.88 | -70.41 | 12.86 | -12.66 | 7.33 | 26.06 | -0.14 | -0.681 | +0.0191 |

### Online

| Recipe vs Old recipe | nearest | lat | angle | length | width | speed | collision% | S Δ% | On-road Δpp | V/GT Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Geometry + head L1 | 26.26 | 7.69 | -49.49 | -54.89 | -66.12 | 27.79 | -17.48 | -27.81 | +1.028 | -0.0537 |
| Head L1 + width p2 | 11.21 | 2.58 | 8.58 | -81.82 | -96.49 | -44.23 | -4.35 | -37.01 | -0.306 | -0.0221 |

### EMA 综合分数的分项贡献

SΔ 的分项贡献为 `(candidate_metric-control_metric)/scale/7`；负值降低综合分数。S 的等尺度权重是本次预先定义的选择规则，不表示每项物理风险/实际任务的重要性相同。

| Recipe vs Old recipe | nearest | lat | angle | length | width | speed | collision% | Sum SΔ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Geometry + head L1 | 0.006403 | 0.013759 | -0.019883 | 0.001200 | 0.002577 | 0.000309 | -0.008734 | -0.004370 |
| Head L1 + width p2 | -0.002431 | 0.010507 | -0.020221 | 0.001046 | -0.000623 | 0.000166 | 0.011216 | -0.000340 |

## 1536→2048 实际趋势

| Recipe | Weights | S1536 | S2048 | S Δ% | On-road Δpp |
|---|---|---:|---:|---:|---:|
| Old recipe | ema | 0.269221 | 0.242189 | -10.04 | +1.715 |
| Old recipe | online | 0.314420 | 0.459818 | +46.24 | +1.642 |
| Geometry + head L1 | ema | 0.263287 | 0.237820 | -9.67 | +1.455 |
| Geometry + head L1 | online | 0.337452 | 0.331930 | -1.64 | +0.347 |
| Head L1 + width p2 | ema | 0.263113 | 0.241849 | -8.08 | +1.279 |
| Head L1 + width p2 | online | 0.324291 | 0.289632 | -10.69 | -0.521 |

## 相同 393,216 场景曝光：B64×6144 与 B512×768

这是等样本曝光的桥接，不是单因素 batch ablation。B64 使用 LR1e-4、6144 optimizer updates；B512 使用 LR5e-4、768 updates。batch、LR、optimizer 更新次数及其轨迹同时变化，AdamW 的逐 update 衰减/梯度裁剪、更新噪声和 G1 EMA 更新历程也不同。两组都未走完第一个完整 epoch，使用相同前393,216个训练缓存、训练/生成 seed、search1024 和相同 recipe。

已安装 torch_ema 默认 use_num_updates=True，实际 decay=min(0.9999,(1+n)/(10+n))。在此桥接端点，B512×768 和 B64×6144 的 EMA 平均参数年龄约 **39,372.8** 和 **39,328** 个已见场景；不能把 EMA 的有效样本平滑窗口简单说成相差8倍。仅 G1 有 EMA，map encoder 无 EMA；相近平均年龄也不意味着两条优化路径完全一致。证据见 ../fresh_batch512_multi_epoch_audit.json。

### EMA

| Recipe | B64 ×6144 S | B512 ×768 S | S Δ% | B64 on-road% | B512 on-road% |
|---|---:|---:|---:|---:|---:|
| Old recipe | 0.283256 | 0.395524 | +39.63 | 77.961 | 71.963 |
| Geometry + head L1 | 0.243395 | 0.282622 | +16.12 | 78.523 | 77.781 |
| Head L1 + width p2 | 0.256173 | 0.321445 | +25.48 | 78.074 | 73.821 |

### Online

| Recipe | B64 ×6144 S | B512 ×768 S | S Δ% | B64 on-road% | B512 on-road% |
|---|---:|---:|---:|---:|---:|
| Old recipe | 0.314144 | 0.681395 | +116.91 | 78.013 | 72.798 |
| Geometry + head L1 | 0.304786 | 0.498109 | +63.43 | 75.861 | 79.650 |
| Head L1 + width p2 | 0.290504 | 0.408052 | +40.46 | 79.067 | 73.953 |

## 图表与原始文件

Compact 图只使用三个方案的实际 checkpoint，native JSD 单位写在坐标轴；每个权重 cohort 单独绘制，因为 Ntrain=1 不绘制 SD bands。主报告可使用以下 PNG，全部数值仍保留在原始 CSV/JSON。

![ema actual production curves](../production_compact_plots/summary_81e7d4ad67b7_native.png)

![ema actual size curves](../production_compact_plots/size_81e7d4ad67b7_native.png)

![online actual production curves](../production_compact_plots/summary_c929105420f6_native.png)

![online actual size curves](../production_compact_plots/size_c929105420f6_native.png)

三个训练阶段总时间约 42.29 min（2537.447s），按 (run_path,variant) 去重计一次；不能把每个 checkpoint/online/EMA 行上的重复 train_seconds 加总。这里不包括之前的搜索和之后 heldout/full50k。

来源：[observations.json](observations.json)、[aggregate.csv](aggregate.csv)、[analysis_protocol.json](analysis_protocol.json)。部署仍按预先冻结的 heldout4096 三训练 seed、两生成 seed、EMA2048 均值与覆盖门槛选择；当前 search1024 的结果不能替代该确认。
