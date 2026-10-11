## 从头训练：batch512、三训练 seed 的 heldout4096 确认

冻结 EMA2048 终点选择 **New recipe（`geo_mse32_col20_heading_l1_shape2`）**。New recipe 相对 Old recipe 的配对 S 变化为 **-3.40% ± 2.69%**，S 在 3/3 个训练 seed 上下降。该结论来自实际从头训练和生成结果；未完成128 epochs，不能据此证明最终收敛最优。

Old recipe=`warm_winner`，New recipe=`geo_mse32_col20_heading_l1_shape2`。两者 map encoder 与 G1 都从随机参数训练，batch512、LR5e-4、默认 AdamW 和128真实epoch余弦调度、32-true、gradient clip0.5保持一致。2048 optimizer updates 对应实际1,048,524个训练样本、约2.124个full TRAIN epoch；这里使用G1 EMA和online共享map。

训练 seeds 为817/1817/2817，生成 seeds 为2817/3817。先在每个训练 seed 内平均两次生成，再在3个独立训练seed之间等权平均；表中SD是训练seed样本SD（ddof=1），不是置信区间，6次生成不等于6个独立训练实验。

下面六项JSD都是未缩放的自然对数JSD，理论上界ln2≈0.693147；helper中的nearest/lateral除10，其余JSD除100。collision保留百分比。S维持冻结的helper单位scale[1,1,10,20,15,20,5]，七项metric/scale后取均值，属于本次搜索的辅助排序分数。

### 冻结 EMA2048 主结果及配对一致性

| 指标 | Old recipe：均值 ± TRAIN SD | New recipe：均值 ± TRAIN SD | New−Old：配对变化 | 改善 TRAIN seeds |
|---|---:|---:|---:|---:|
| Nearest JSD | 0.01258 ± 0.00045 | 0.01614 ± 0.00060 | +28.46 ± 9.24% | 0/3 |
| Lateral JSD | 0.08996 ± 0.00160 | 0.09299 ± 0.00172 | +3.38 ± 1.88% | 0/3 |
| Angle JSD | 0.01778 ± 0.00175 | 0.00574 ± 0.00139 | -68.03 ± 4.44% | 3/3 |
| Length JSD | 0.00798 ± 0.00122 | 0.00942 ± 0.00046 | +20.34 ± 23.26% | 1/3 |
| Width JSD | 0.00506 ± 0.00092 | 0.00795 ± 0.00092 | +58.86 ± 18.89% | 0/3 |
| Speed JSD | 0.00135 ± 0.00046 | 0.00181 ± 0.00077 | +34.32 ± 27.49% | 0/3 |
| Collision (%) | 1.6505 ± 0.1136 | 1.5046 ± 0.0624 | -8.63 ± 5.66% | 3/3 |
| 综合 S | 0.2305 ± 0.0014 | 0.2227 ± 0.0051 | -3.40 ± 2.69% | 3/3 |
| On-road (%) | 79.04 ± 0.89 | 79.54 ± 0.57 | +0.50 ± 0.60 pp | 2/3 |
| Vehicle / GT | 1.0266 ± 0.0093 | 1.0242 ± 0.0148 | -0.22 ± 1.57% | 下降 2/3，需接近1 |

配对百分比先对每个训练seed的两次生成均值计算New/Old−1，再汇总3个seed的均值与样本SD；这不同于直接计算两臂总体均值的比例。On-road用百分点，V/GT应接近1；其下降不自动代表改善。

Lateral/angle只统计on-road车辆，因此应连同on-road覆盖一起看，较低条件JSD不能补偿大量离路车辆。指标表完整保留尺寸、速度、碰撞和覆盖的取舍，没有将综合分数最低解释成所有指标最佳。

### 各训练 seed 的实际生成波动

| TRAIN seed | Old recipe：S 均值 [GEN 最小, 最大] | New recipe：S 均值 [GEN 最小, 最大] | 配对 ΔS |
|---:|---:|---:|---:|
| 817 | 0.2290 [0.2260, 0.2319] | 0.2267 [0.2229, 0.2306] | -0.0022 |
| 1817 | 0.2311 [0.2297, 0.2326] | 0.2243 [0.2233, 0.2253] | -0.0068 |
| 2817 | 0.2315 [0.2302, 0.2329] | 0.2169 [0.2143, 0.2196] | -0.0146 |

方括号是同一checkpoint的两个生成seed范围，仅描述sampler波动；不同训练seed才是上表SD的样本单位。

### 同一批训练 seed 的实际1536→2048变化

两预算对每个配方均使用相同3个TRAIN seeds及相同2个GEN seeds。以下Δ=2048−1536，先在训练seed内配对，再汇总；它是512次真实optimizer更新的净变化。逐更新斜率Δ/512和逐seed原值保存在`heldout_stage_details.json`，不将短期斜率线性外推到128 epochs。

| 指标 | Old recipe：实际Δ均值 ± TRAIN SD | New recipe：实际Δ均值 ± TRAIN SD | Old 改善seeds | New 改善seeds |
|---|---:|---:|---:|---:|
| Nearest JSD | -0.00193 ± 0.00075 | -0.00041 ± 0.00058 | 3/3 | 2/3 |
| Lateral JSD | -0.00134 ± 0.00116 | -0.00109 ± 0.00179 | 3/3 | 3/3 |
| Angle JSD | -0.00832 ± 0.00122 | -0.00221 ± 0.00013 | 3/3 | 3/3 |
| Length JSD | -0.00307 ± 0.00028 | -0.00423 ± 0.00356 | 3/3 | 3/3 |
| Width JSD | -0.00174 ± 0.00067 | -0.00482 ± 0.00290 | 3/3 | 3/3 |
| Speed JSD | -0.00034 ± 0.00060 | -0.00063 ± 0.00060 | 2/3 | 3/3 |
| Collision (%) | -0.1943 ± 0.1125 | -0.0973 ± 0.0718 | 3/3 | 3/3 |
| 综合 S | -0.0262 ± 0.0042 | -0.0161 ± 0.0062 | 3/3 | 3/3 |
| On-road (%) | +1.97 ± 0.37 | +1.09 ± 0.57 | 3/3 | 3/3 |
| Vehicle / GT | -0.0053 ± 0.0088 | -0.0168 ± 0.0210 | 趋近1另看原值 | 趋近1另看原值 |

JSD/碰撞/S的负Δ是改善，on-road的正Δ是覆盖增加；碰撞和on-road的Δ单位都是百分点。尺寸、速度和几何的短期趋势可能不同；相同seed的后期配对比混合不同训练seed的早晚总体均值更可核对。

### Online2048：仅作诊断

Online在相同训练、生成和场景矩阵上评价，只检查权重平均影响与排名稳定性。已冻结的部署终点为EMA2048，不因online表现切换权重范围、预算或选择规则。

| 指标 | Old recipe：均值 ± TRAIN SD | New recipe：均值 ± TRAIN SD |
|---|---:|---:|
| Nearest JSD | 0.01532 ± 0.00231 | 0.01545 ± 0.00112 |
| Lateral JSD | 0.08904 ± 0.00101 | 0.09246 ± 0.00067 |
| Angle JSD | 0.03290 ± 0.01673 | 0.00902 ± 0.00378 |
| Length JSD | 0.07517 ± 0.06755 | 0.01746 ± 0.01240 |
| Width JSD | 0.13282 ± 0.12809 | 0.02421 ± 0.02012 |
| Speed JSD | 0.01195 ± 0.00956 | 0.01013 ± 0.00373 |
| Collision (%) | 3.0373 ± 0.3889 | 2.2888 ± 0.4152 |
| 综合 S | 0.4716 ± 0.1831 | 0.2752 ± 0.0368 |
| On-road (%) | 78.31 ± 1.22 | 79.84 ± 1.55 |
| Vehicle / GT | 1.0560 ± 0.0184 | 1.0147 ± 0.0487 |

### 完整门槛与复核记录

部署选择按三训练seed的EMA2048均值，要求on-road不低于Old recipe−2个百分点，且生成/GT车辆比例在[0.9,1.1]，在合格的两个finalists中取最低S；该规则在heldout生成前冻结。

| Recipe | 门槛合格 | Mean on-road (%) | 门槛最低on-road (%) | Mean V/GT | Mean S |
|---|---|---:|---:|---:|---:|
| Old recipe | True | 79.04 | 77.04 | 1.0266 | 0.2305 |
| New recipe | True | 79.54 | 77.04 | 1.0242 | 0.2227 |

heldout场景未用于本次TRAIN或此前loss搜索；这里将其用于最终配方选择后，它就是选择验证集，不能再称为完全未参与选择的test。本节只报告实际4096场景确认，不填充未发生的全量评价或完整训练结果。

[实际paired统计](paired_summary.json)、[逐训练seed表](per_training_seed.csv)、[实际阶段趋势及provenance](heldout_stage_details.json)保存原始单位、seed身份、checkpoint和数据清单hash。

### 同seed后期曲线

图中误差线/带表示3个独立训练seed的样本SD，Old/New标签与上表一致。

![实际1536与2048：online，三训练seed，JSD单位为nats](compact_plots/summary_320db4fa99dc_raw.png)

![实际1536与2048：online，三训练seed，JSD单位为nats](compact_plots/size_320db4fa99dc_raw.png)

![实际1536与2048：ema，三训练seed，JSD单位为nats](compact_plots/summary_4795bc325dd2_raw.png)

![实际1536与2048：ema，三训练seed，JSD单位为nats](compact_plots/size_4795bc325dd2_raw.png)

