# InitDiffusion 分量 loss 短训练搜索

已在 `sim` 环境实际完成 **94 个训练方案**：四轮 78 个短训方案、12 个 512 步确认训练、4 个第三种子 128 步复核；另评价了 16 个保存的中间 checkpoint。最终候选采用 **3 个训练 seed × 2 个生成 seed**，而非只选择单次最好的结果。

**本次最佳测试组合是角度 MSE 的 128 步微调。** 七指标等权、相对配对 warm checkpoint 的综合分数为 **0.8097**，下降 **19.0%**。speed / length / width JSD 分别平均下降 **38.7% / 26.8% / 46.6%**；lateral / angle 改善，nearest distance 与 collision 略有代价。这里的“最佳”限定为本次搜索和已有 checkpoint 附近的短期训练，不代表从头训练或全局最优。

冻结配方后的完整50k在线评价score为 **0.7788**，speed/length/width改善 **38.4%/36.3%/63.1%**；nearest变差11.2%，collision增加0.0899个百分点。

推荐入口：[训练配置](../configs/experiment/init_diffusion_lane_conditioned_loss_search_best.yaml)、[完整评价配置](../configs/experiment/init_diffusion_lane_conditioned_loss_search_best_eval.yaml)。保留原默认 loss；使用新 experiment 才启用该方案。最终配方和 checkpoint 选择记录在 [selected_recipe.json](experiments/init_diffusion_loss_search_20261011/selected_recipe.json)。

## 最终推荐与重复实验

推荐学习率 **2e-5**，128 optimizer steps，batch64；噪声、x0 预测目标、圆周 flow 路径、匹配和 sampler 都沿用当前设置，仅改变分量损失及其权重。

| 分量 | Loss | Time power | Weight | 残差 scale |
|---|---|---:|---:|---|
| position | L1 | 1 | 4 | 保存的数据 std |
| heading | **最短圆周角差的 MSE** | 2 | 1 | none，弧度 |
| length | MSE | 1 | 1 | 保存的数据 std |
| width | MSE | 2 | 1 | 保存的数据 std |
| velocity（当前为 scalar speed） | MSE | 1 | 2 | 保存的数据 std |
| type（当前为 separate 分支） | CE | 0 | 1 | none |
| collision | L1 | 1 | 5 | none，额外穿透量 m |

配置使用 `heading_space: angle`、`normalize_time_weights: true`、`std_floor: 0.1`；未启用额外 speed-magnitude loss。网络仍输出 cos/sin，heading loss 使用 `wrap(atan2(pred_sin,pred_cos)-theta_gt)`，没有改成另一个生成目标。

以下排序只纳入都完成三个训练种子、同样 128 步预算的最终候选。每个训练 seed 使用两个相同生成 seed，先在训练 seed 内平均，再跨训练 seed 平均。

| 方案 | Seed817 | Seed1817 | Seed2817 | 平均 score |
|---|---:|---:|---:|---:|
| **角度 MSE** | 0.7575 | 0.8751 | 0.7964 | **0.8097** |
| 向量 L1 的 A | 0.7901 | 0.8682 | 0.9891 | 0.8825 |
| 角度 L1 | 0.7451 | 0.8524 | 1.1057 | 0.9011 |
| 原 legacy loss | 1.4447 | 1.0999 | 1.0618 | 1.2022 |

前两个 seed 曾使角度 L1 看起来最好；第三个 seed 改变了结论。因此没有沿用早期赢家。角度 MSE 六次生成的 score 范围 **0.7530–0.8752**，描述性标准差 **0.0540**；生成重复嵌套于训练 seed，不能把六次结果当作六个独立训练样本来计算显著性。

| 指标 | 配对 warm 均值 | 角度 MSE 均值 | 平均配对变化 |
|---|---:|---:|---:|
| nearest_dist_jsd | 0.058015 | 0.058807 | +1.48% |
| lat_dev_jsd | 0.346108 | 0.312688 | −9.66% |
| ang_dev_jsd | 0.246149 | 0.202853 | −17.54% |
| length_jsd | 0.813212 | 0.595385 | −26.75% |
| width_jsd | 1.346023 | 0.718016 | −46.62% |
| speed_jsd | 0.431260 | 0.263890 | −38.70% |
| collision_rate | 1.974580% | 2.064530% | +4.57% |

collision 增加 **0.08995 个百分点**，不是增加 4.57 个百分点。综合 score 是每次配对比值的平均，因此不应直接用表中聚合均值相除来替代。表中 warm 均值为该赢家的实际配对 baseline，不是把不同运行的 warm 混合平均。

每 4096 场景生成车辆数平均 **30369**，warm **29745**，GT **29865**，相对 warm 多约 **2.1%**。type 生成会改变进入 vehicle metrics 的样本总体；物理分量的 JSD 不只反映单个输出头的质量。生成on-road比例约90.88%，warm91.22%，GT99.38%；lat/angle只评价这些on-road车辆。

完整比较见 [确认汇总表](experiments/init_diffusion_loss_search_20261011/confirmation_summary.csv)、[逐次配对记录](experiments/init_diffusion_loss_search_20261011/paired_observations.csv)、[确认分析](../logs/component_loss_quick/confirmation_final/aggregate.md)。同样生成子集与 full-50k GT 的次要参考比较也选择角度 MSE，score **0.8030**；这仍不是 50k 个生成场景的评价。

## 实验条件与评价范围

所有搜索均在本仓库的 `sim` 环境运行，复用 `MultiDataModule`、`SMART_GAIL.training_step`、原 AdamW 和缓存 Scenario Dreamer evaluator；没有使用外部仓库。实际运行脚本为 [quick_component_loss_sweep.py](../tools/quick_component_loss_sweep.py)。

| 条件 | 设置 |
|---|---|
| 共同初始权重 | `src/logs/sdinit512_tspeed_theta_posscale_egoVFtypeemd50/2026-10-09_19-24-16/bc/scy2l1f4/checkpoints/last.ckpt`，epoch 35 / step 34704 |
| 训练数据 | `src/waymo_data/full/training_map2_sd`，从已有文件名 manifest 按 seed 抽取固定子集，各组顺序一致 |
| 评价数据 | `src/waymo_data/full/scenario_dreamer_val/*.pt`，搜索取前 2048 个缓存样本；确认取 offset2048 的 4096 个；完整评价取 50000 个。规范化完整顺序与 `waymo_eval_set.pkl` 一致 |
| 每组训练 | 搜索128步；确认128/256/512步；batch64；新 AdamW，不恢复历史 optimizer state |
| 评价 | batch128，20步sampler；主比较在线权重，生成seed2817/3817；另单独评价保留历史状态的EMA |
| 训练 seed | 搜索817；确认817/1817/2817；每组重置RNG，重新加载同一初始参数与统计量 |
| 地图输入 | ego 坐标下 square crop，半边长 50 m，匹配初始 checkpoint 的训练条件 |
| ego 条件 | GT position、heading、length/width、type 固定；ego speed 生成 |
| 其他 agent | 数量来自输入，非 ego type 由模型生成；scalar speed、圆周 heading flow、separate type 分支 |
| 数值设置 | `32-true`，matmul `medium`，RTX 4090，梯度裁剪 0.5，DataLoader workers 0 |
| 时间权重 | `normalize_time_weights: true`，`t_eps: 0.05`；残差的 data scale 来自同一 checkpoint |
| 参考分布 | `matched`：生成和GT使用相同缓存样本；同时保留full-50k GT参考比较 |

首轮学习率为 `1e-4`，其余三轮及确认训练为 `2e-5`。128 步位于同一训练 epoch，实际学习率约为设定值的 0.999857；没有把 `max_epochs=1` 导致的 scheduler 衰减带入比较。每组初始 generator hash、训练和评价 manifest hash 见对应 `protocol.json`。主排序采用在线权重；EMA 保持历史 shadow 单独审计。

评价的是 **initial-scene vehicle metrics**。agent 在官方完整参考 lanes 上评价；网络接收裁剪后的地图 token，评价地图没有按该 crop 缩小。lateral/angle 只统计 evaluator 判定为 on-road 的车辆，length/width/speed 使用生成车辆分布，nearest distance 需要场景中至少两辆车。不同方案生成的 type 不同，因此进入指标的车辆数也不同，不能把每个直方图变化全部归因于对应物理分量。

| 输出指标 | 特征单位 | 原始 JSD 的输出倍率 |
|---|---|---:|
| nearest_dist_jsd | m | 10 |
| lat_dev_jsd | m | 10 |
| ang_dev_jsd | 度 | 100 |
| length_jsd、width_jsd | m | 100 |
| speed_jsd | m/s | 100 |
| collision_rate | 涉及至少一次 five-circle overlap 的车辆比例 | 100，即百分数 |

这里 JSD 为 `scipy.spatial.distance.jensenshannon(p,q) ** 2`，使用固定直方图范围和分箱。所有指标均为越低越好。collision 按车辆计数汇总，不能用每场景比例的简单平均代替。

搜索排序使用 `score = mean(metric / 同轮 warm_start metric)`，七项等权，warm start 为 1。这是辅助筛选准则，不是论文的官方综合指标。它消除了输出倍率差异，但仍取决于人为选定的指标权重；选型必须同时查看完整指标向量。

## 已完成搜索得到的证据

最初候选在高学习率短期切换目标时不稳定。Round1 的 mixed 得到 length/width JSD **6.856 / 8.380**，同轮 warm start 为 **0.996 / 1.383**，九个训练方案的综合 score 均差于 warm start。Round2 降低学习率后，mixed 的尺寸恢复到 **0.958 / 0.837**，但 collision 仍为 **5.388%**。因此首轮不能只用来判断 L1/MSE 哪个更优，优化步幅也是重要因素。

Round2 将 collision weight 从 mixed 的 0.1 增至 2 / 5，collision 降至 **1.983% / 1.246%**，代价是 speed JSD **0.756 / 0.894**。新 collision 项是按有效 agent pair 平均的额外穿透量，单位为 m，不能照搬 legacy `expm1` penalty 的权重。更大的系数能恢复几何约束，但不保证其他分布同步改善。

Round3 以 collision weight2 的 anchor 为参照，分别改变单个分量。下表的数值是同轮、相同初始权重和样本下的受控比较。

| 改动 | 直接观察 | 其他影响 |
|---|---|---|
| speed：MSE/p2/w1 → MSE/p1/w2 | speed JSD **0.808 → 0.208** | collision **1.875% → 2.196%**，需联合调整几何项 |
| speed：p2 → p3 | speed JSD **0.808 → 1.234** | 本次提高近干净端权重没有改善 speed 分布 |
| length：MSE/p2 → MSE/p1 | length JSD **1.177 → 0.786** | width 同时 **1.047 → 0.733** |
| heading：L1/p1 → L1/p2 | angle JSD **0.232 → 0.188** | collision **1.875% → 1.739%** |
| heading：向量 L1 → 向量 MSE，power 不变 | angle JSD **0.232 → 0.323** | 这是 cos/sin 向量 loss；不能据此外推角度 MSE |
| position：weight1 → 4 | lateral / angle **0.341 / 0.232 → 0.318 / 0.136** | collision **1.875% → 2.451%** |
| width：weight1 → 4 | width JSD **1.047 → 3.673** | 单纯加大尺寸权重不能保证尺寸分布改善 |

length 改 L1 或 p3、position 改 p2/p3 等方案也没有得到更均衡的结果。这些结论限定于该 checkpoint 附近的短期微调，尚不能外推为从头训练的一般规律。

Round4 联合检查 speed 配方、length power、heading power 以及 position/collision 系数。以下两组被选入后续确认。

| 分量 | A：偏向 speed | B：偏向 width |
|---|---|---|
| position | L1 / p1 / weight4 / data | 相同 |
| heading | L1 / p2 / weight2 / none | 相同 |
| length | MSE / p1 / weight1 / data | 相同 |
| width | MSE / p2 / weight1 / data | 相同 |
| velocity（scalar speed） | MSE / p1 / weight2 / data | MSE / p2 / weight4 / data |
| type | CE / p0 / weight1 / none | 相同 |
| collision | L1 / p1 / weight5 / none | L1 / p1 / weight3.5 / none |

| Round4 方案 | Nearest | Lateral | Angle | Length | Width | Speed | Collision % | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| warm start | 0.0623 | 0.3463 | 0.3175 | 0.9967 | 1.3842 | 0.4765 | 2.0326 | 1.000 |
| A | 0.0539 | 0.3207 | 0.1534 | 0.7103 | 1.1458 | 0.1732 | 1.9733 | 0.736 |
| B | 0.0550 | 0.3240 | 0.1584 | 0.7412 | 0.6282 | 0.4228 | 2.0388 | 0.772 |

A 在该轮搜索集上七项均低于 warm start，speed/angle 相对下降约 **63.6% / 51.7%**；B 的 width 相对下降 **54.6%**，但 speed 改善较少。B 的 collision 与 warm start 相差仅约 0.006 个百分点，不能据此声称显著变差。A、B 之间同时改变 speed 配方和 collision 系数，其差异不能全部归因于 speed power。

![Round4 指标相对同轮 warm start 的变化](experiments/init_diffusion_loss_search_20261011/search_round4.png)

图中颜色范围截断在 ±100%，格内数值保留实际百分比。原始比较见 [Round4 summary](../logs/component_loss_quick/search_round4/summary.csv)；完整 78 组结果见 [分析表](../logs/component_loss_quick/analysis_search/all_results.csv) 和 [逐轮分析](../logs/component_loss_quick/analysis_search/analysis.md)。

## 完整50000场景评价

最终组合冻结后，按照三seedscore的中位数选seed2817，完整运行 **50000个生成缓存场景**；共同warm baseline也完整生成50000个。两个报告均确认`full_membership=true`、GT50000、`unverified_generation_timestep_count=0`；matched GT和full-reference结果完全一致，没有以小生成子集冒充完整评价。

| 指标 | Warm在线权重 | 推荐在线权重 | 变化 |
|---|---:|---:|---:|
| nearest_dist_jsd | 0.049719 | 0.055298 | +11.22% |
| lat_dev_jsd | 0.344488 | 0.311335 | −9.62% |
| ang_dev_jsd | 0.212215 | 0.163094 | −23.15% |
| length_jsd | 0.712749 | 0.454002 | −36.30% |
| width_jsd | 1.319678 | 0.486455 | −63.14% |
| speed_jsd | 0.418802 | 0.258155 | −38.36% |
| collision_rate | 1.989984% | 2.079840% | +4.52% |
| 七指标等权score | 1 | **0.778808** | **−22.12%** |

这次完整评价保留了speed/shape/角度收益，也更清楚地显示nearest distance代价。collision增加 **0.08986个百分点**。它验证的是预先选定的一个median checkpoint和一个生成seed，不是三种子完整50k平均，更不是对所有候选都重跑了50k。

生成vehicle数量 **362616→365605**，增加0.82%；GT362309。on-road比例 **91.29%→91.06%**，GT99.39%；lat/angle仍是on-road车辆子集指标。原始数据：[完整summary.csv](experiments/init_diffusion_loss_search_20261011/full50k_results.csv)、[完整审计](../logs/component_loss_quick/full50k_online/analysis.md)。

### Speed和尺寸分布

![完整50k生成分布](experiments/init_diffusion_loss_search_20261011/vehicle_distributions.png)

使用官方固定分箱、每条曲线在全部bin上归一化；图中length/width只缩放横轴，没有重新归一化。另存 [完整范围PNG](experiments/init_diffusion_loss_search_20261011/vehicle_distributions_full_range.png)、[PDF](experiments/init_diffusion_loss_search_20261011/vehicle_distributions.pdf)、[统计明细](experiments/init_diffusion_loss_search_20261011/distribution_stats.csv)。

| 特征区间 | GT | Warm | 推荐 |
|---|---:|---:|---:|
| speed <1m/s | 41.70% | 36.85% | 39.50% |
| width [2.0,2.1)m | 34.82% | 26.26% | 31.63% |
| width [1.9,2.0)m | 10.78% | 6.85% | 10.23% |
| length [4.5,4.6)m | 13.52% | 11.61% | 13.25% |
| length [4.9,5.0)m | 4.79% | 6.12% | 4.92% |
| length >=10m | 0.518% | 0.236% | 0.192% |
| speed >=10m/s | 21.32% | 18.07% | 17.73% |

speed改善主要体现为最低速bin缺口从4.85缩至2.20个百分点；这只是 `<1m/s` proxy，不能用1m/s分箱区分精确停车概率。尺寸的主要峰更接近GT，但长车、高速尾部仍不足；length>=10m和speed>=10m/s甚至离GT更远。因此JSD改善不意味着每个分箱或所有尾部都改善。

官方裁剪后的bin中点近似统计显示，speed均值GT/warm/推荐约5.328/5.101/4.962m/s；length标准差约0.753/0.634/0.599m。更低JSD伴随部分均值/方差仍偏离GT，不能把这些bin近似当作原始物理值的精确矩。type生成也影响vehicle总体，现有聚合直方图无法完全拆开物理头和类型构成的贡献。

完整评价复现命令：

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/quick_component_loss_sweep.py \
  --steps 128 --scenes 50000 --eval-offset 0 --workers 0 \
  --batch-size 64 --eval-batch-size 128 --seed 2817 --eval-seed 2817 --lr 2e-5 \
  --recipe logs/component_loss_quick/selected_loss_recipe.json --variants '' \
  --evaluation-checkpoints logs/component_loss_quick/final_evaluation_checkpoints.json \
  --require-full-set --output logs/component_loss_quick/reproduce_full50k
```

## EMA：收益与历史参数分开看

按预先记录的规则，选择角度MSE三个训练seed中score居中的 **seed2817**，不是最好的seed817；选中 [last.ckpt](../logs/component_loss_quick/confirmation_seed2817/angle_mse_128/last.ckpt)。在同一4096缓存样本、两个生成seed上：

| 对比 | 相对同模式warm的score |
|---|---:|
| selected online / warm online | 0.796442 |
| selected EMA / warm EMA | 0.997805 |

EMA只改善约 **0.22%**，两个生成seed一好一坏；次要full-reference参考的EMA score为1.003374。原warm EMA自身已比warm online有更好的speed、尺寸分布，不能把candidate EMA的绝对水平当作新loss的效果。

| 指标 | warm online | selected online | warm EMA | selected EMA |
|---|---:|---:|---:|---:|
| lat_dev_jsd | 0.34622 | 0.31088 | 0.35615 | 0.35537 |
| ang_dev_jsd | 0.24569 | 0.22129 | 0.17465 | 0.17623 |
| length_jsd | 0.81466 | 0.54650 | 0.50471 | 0.48808 |
| width_jsd | 1.34885 | 0.51378 | 0.49896 | 0.49537 |
| speed_jsd | 0.43168 | 0.27972 | 0.18417 | 0.19037 |
| collision_rate | 1.96999% | 2.11748% | 1.52765% | 1.51057% |

saved EMA `num_updates`为34704→34832。实际衰减有warmup上限规则，128步后继承旧shadow的累计系数为 **96.74145%**，新参数轨迹只占3.25855%；不是直接按0.9999^128估计。本次保留完整历史EMA，没有reset。在线推荐的评价配置明确 `use_ema: false`，EMA仍在训练时跟踪。见 [EMA审计](../logs/component_loss_quick/ema_confirmation/analysis.md) / [原始指标](../logs/component_loss_quick/ema_confirmation/summary.csv)。

## 训练步数与 heading 的后续检查

前四轮选出的 A/B 继续训练 512 步时，speed/几何仍在改善，但 heading 和 length 漂移。因此保存的 128/256 checkpoint 被重新评价，没有只报告最有利的停止点。

| 方案 | 128步 score | 256步 score | 512步 score | 比较范围 |
|---|---:|---:|---:|---|
| 向量 L1 的 A | 0.8292 | 0.8590 | 1.0519 | 训练seed817/1817，各两个生成seed |
| B | 0.8858 | 0.8951 | 1.1087 | 同上 |
| 角度 MSE | 0.8163 | 未列入最终筛选 | 0.9710 | 同上 |
| 角度 L1 | 0.7988 | 未列入最终筛选 | 1.0800 | 同上 |

角度 MSE 在512步时 speed平均相对warm改善 **61.5%**、collision改善 **17.7%**，但 angle变差 **79.6%**、length变差 **27.5%**；不能因为总训练loss下降就继续延长短训。该配方的128步第三种子确认支持提前停止。这里不是“训练越久一定越差”的一般规律，而是切换现有checkpoint目标后的实际观察。

向量 component-L1 还有独立的几何问题。单位heading向量、GT角为0时，忽略时间/分量系数：

```text
L(theta) = (|cos(theta)-1| + |sin(theta)|) / 2
L(pi+delta) = 1 + |delta|/2 - delta²/4 + O(|delta|³)
```

所以反向180°可成为错误的严格局部最小值；整体旋转后loss形状也改变，向量L1并不具有旋转不变性。这已通过解析式和数值检查证明，见 [heading几何审计](../logs/component_loss_quick/heading_l1_geometry_audit.md)。向量MSE或wrapped-angle loss避免上述L1假极小值；向量MSE在180°仍有零导数的极大点。

但不能将本次差距全部归因于这个问题：改用角度L1后，seed1817的512步和seed2817的128步仍会失败。**已证明的是loss几何风险，未证明的是它单独导致某次JSD退化。** 最终推荐角度MSE是三seed生成指标的结论，而不是只凭这个推导选型。

## 系数与 time power 怎么理解

实际目标为每个分量的有效 agent 均值乘自身系数，再加独立的 collision pair 均值。MSE 对归一化残差平方求均值，L1 对其绝对值求均值；position/length/width/speed 的 data scale 使用保存的训练标准差，下限 0.1。前四轮 heading 为 cos/sin 残差；后续角度试验改用最短圆周角差，type 保持 CE。没有更换噪声、预测目标、匹配或 sampler。

`t=0` 是数据端，`t=1` 是噪声端。提高 inverse time power 会强调接近干净数据的时间段，并相对减弱高噪声段。归一化常数为：

| power | `E[max(t,0.05)^(-p)]` | 归一化最大权重，未乘分量系数 |
|---:|---:|---:|
| 0 | 1 | 1 |
| 1 | 3.9957 | 5.005 |
| 2 | 39 | 10.256 |
| 3 | 599.5 | 13.344 |

归一化使均匀时间下的**平均时间权重**为 1，并不使不同分量或 power 的平均梯度范数相等：误差、网络 Jacobian 和时间相互关联，L1/MSE 的导数也不同。若关闭时间归一化而希望保持完全相同目标，应将各项 weight 除以上表常数。

以 A/B 的 speed 为例，A 在接近噪声端的时间系数约为 `2/3.9957=0.501`，B 为 `4/39=0.103`，A 约强 4.88 倍；在 `t<=0.05` 的近干净端，A / B 为 **10.01 / 41.03**，B 约强 4.10 倍。这说明 A 加强了较高噪声段的 speed 学习，B 更集中于末段修正。它与观察到的速度分布差异相符，但还不是独立证明，因为两个候选的 collision 系数也不同。

78 个方案的 train trace 均记录到 100% 的 step 触发 0.5 梯度裁剪。Round4 A/B 的末 32 步平均未裁剪范数约为 **2.09 / 2.09**。因此不能仅凭加权 loss 数值判定某个分量支配了优化；梯度方向、共享参数冲突和 AdamW 的状态同样影响更新。0.5 是 gradient norm 上限，不是 AdamW 参数更新范数上限。collision 的尺寸输入已 detach，不能通过直接缩小 length/width 降低该项，但其共享网络梯度仍可能间接改变尺寸输出。

## 实测梯度解释与参数影响

分量梯度探测在共同warm checkpoint的四个batch上完成，没有optimizer更新。全部8083978个可训练G1参数纳入检查；捕获总目标与实际训练目标的梯度最大相对误差 **2.22e-7**，独立分量求导后相加最大相对误差 **3.44e-7**。完整数据见 [梯度JSON](../logs/component_loss_quick/gradient_probe.json) / [梯度分析](../logs/component_loss_quick/gradient_probe_analysis.md)。

下表是平均未裁剪梯度范数，以及该项在总梯度方向的带符号投影比例。它们不是AdamW参数更新量的百分比。

| 分量 | Anchor范数 / 投影 | 推荐角度MSE范数 / 投影 |
|---|---:|---:|
| position | 0.203 / 1.8% | 0.813 / 11.0% |
| heading | 0.810 / 18.4% | 0.240 / 1.9% |
| length | 0.165 / 1.9% | 0.473 / 6.1% |
| width | 1.998 / 68.6% | 1.998 / 51.4% |
| scalar speed | 0.094 / 0.4% | 0.938 / 19.1% |
| type | 0.645 / 7.6% | 0.645 / 6.3% |
| collision | 0.181 / 1.4% | 0.452 / 4.2% |

width的加权loss只有 **0.0191**，但梯度范数 **1.998**；A的position加权loss **0.6160**，梯度范数 **0.813**。因此不能看loss数值小就继续增大width weight。speed改MSE/p1/w2后，局部梯度约为anchor的10倍，而非仅weight翻倍；width自身范数基本不变，但其相对投影下降。

width/type梯度在四个batch的平均余弦为 **−0.113**，存在局部对立；多数其他分量组合近乎正交，并未发现普遍强烈的负余弦。不能把梯度范数之和除以总范数的比值直接当作冲突强度，也不能据四个静态batch证明它造成某项JSD差异。

这次探测使用 `32-true / matmul=highest`；实际训练与生成评价一直使用 `matmul=medium`。medium下独立backward的重构误差约1e-4，略超严格检查；仅改变探测精度后以未继续放宽的完整目标检查通过。它支持分量汇总正确，不是highest/medium的生成效果消融。

结合受控消融，本次可采取的判断是：

- **speed优先p1，而非一律提高p。** p1/w2增强较高噪声阶段的学习，生成speed分布优于anchor；p3此次更差。p2/w4在部分组合能改善width，但收益依赖length/collision共同设置。
- **length用MSE/p1、width用MSE/p2/w1。** 单因素与组合验证均支持不同时间权重；width增至w2/w4或p3会使自身JSD明显变差，不建议只加大尺寸权重。
- **heading用角度MSE/p2/w1。** 最终由三seed生成指标决定；局部heading梯度比角度L1弱64%并不妨碍其更稳定的生成表现，不应只因梯度小而加大weight。
- **position L1/p1/w4与collision L1/p1/w5一起保留。** position提高对lane几何有利，但单独调整会增加碰撞。collision5在本次组合中较均衡；这不意味着无限增大碰撞权重继续有效。
- **type维持CE/p0/w1。** 本次w0.5/w2没有稳定整体收益。车辆样本总体变化是评价解释中的一部分，应同时记录车辆数和on-road比例。
- **学习率与停止点是配方的一部分。** LR1e-4在切换目标时明显不稳定，2e-5/128步是实际验证的选择；不能只复制loss表然后训练512步并期待同样结果。

正的时间权重在理想、固定匹配、无限容量且按t条件化的独立回归问题中不会改变每个t的Bayes最优预测，但会改变有限模型的训练资源分配。MSE趋向条件均值、L1趋向条件中位数；噪声阶段的不确定性、多模态speed/shape分布、共享参数和几何约束使单点回归loss与总体JSD并不等价。

## 复现命令与原始证据

从 `/home/ke/code/sim` 执行，使用 `sim` Python。常规入口为：

```bash
python src/run.py experiment=init_diffusion_lane_conditioned_loss_search_best
python src/run.py experiment=init_diffusion_lane_conditioned_loss_search_best_eval
```

训练入口从共同warm checkpoint开始finetune，评价入口默认加载本次选出的seed2817 checkpoint；评价自己重新训练的权重时显式覆盖 `ckpt_path=/path/to/last.ckpt`。两个配置均通过原入口Hydra解析；训练model/data、optimizer、精度、manifest与赢家的保存设置核对一致，见 [原入口配置审计](../logs/component_loss_quick/native_config_audit/audit.json)。harness使用4个CPU线程且在每次生成前重置seed，原入口的线程/seed时序不同，因此不保证逐位相同结果。

重跑最终配方的128步训练和两个生成seed确认：

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/quick_component_loss_sweep.py \
  --steps 128 --scenes 4096 --eval-offset 2048 --workers 0 \
  --batch-size 64 --eval-batch-size 128 --seed 2817 \
  --eval-seed 2817 --extra-eval-seed 3817 --lr 2e-5 \
  --recipe logs/component_loss_quick/selected_loss_recipe.json \
  --variant-file logs/component_loss_quick/third_seed_variants.json \
  --variants angle_mse_128 \
  --output logs/component_loss_quick/reproduce_selected_2817
```

输出目录必须尚未使用。下面重跑A/B的早期128步受控搜索：

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/quick_component_loss_sweep.py \
  --steps 128 --scenes 2048 --batch-size 64 --eval-batch-size 128 \
  --workers 0 --seed 817 --eval-seed 2817 --lr 2e-5 \
  --recipe logs/component_loss_quick/round3_recipe.json \
  --variant-file logs/component_loss_quick/confirmation_variants.json \
  --variants candidate_a,candidate_b \
  --output logs/component_loss_quick/reproduce_short_ab
```

脚本默认 checkpoint 是上表的同一 warm start，`--checkpoint` 可显式指定。完整配方为 [A recipe](../logs/component_loss_quick/candidate_a_recipe.json) / [B recipe](../logs/component_loss_quick/candidate_b_recipe.json)；每个已跑方案同时保存 `config.yaml`、`last.ckpt`、`train_trace.json`、`train_summary.json` 和 `sd_agent_metrics.json`。

确认使用搜索集以外的4096个缓存样本、512步训练和第二个生成seed；下面可复现早期A/B确认协议，其结果已纳入上面的分析。

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/quick_component_loss_sweep.py \
  --steps 512 --scenes 4096 --eval-offset 2048 \
  --batch-size 64 --eval-batch-size 128 --workers 0 \
  --seed 817 --eval-seed 2817 --extra-eval-seed 3817 --lr 2e-5 \
  --recipe logs/component_loss_quick/round3_recipe.json \
  --variant-file logs/component_loss_quick/confirmation_variants.json \
  --variants anchor,legacy,candidate_a,candidate_b \
  --checkpoint-steps 128,256 \
  --output logs/component_loss_quick/reproduce_confirmation_817
```

分量梯度探测使用同一 checkpoint、前四个共同训练 batch 和相同 forward seeds，没有 optimizer step 或 EMA 更新。以下命令比较anchor、A和两种角度loss，并验证分量梯度之和能重构实际训练目标的梯度。

```bash
/home/ke/miniconda3/envs/sim/bin/python tools/probe_component_loss_gradients.py \
  --config logs/component_loss_quick/gradient_probe_highest_config.yaml \
  --checkpoint src/logs/sdinit512_tspeed_theta_posscale_egoVFtypeemd50/2026-10-09_19-24-16/bc/scy2l1f4/checkpoints/last.ckpt \
  --recipes anchor=logs/component_loss_quick/round3_recipe.json \
    candidate_a=logs/component_loss_quick/candidate_a_recipe.json \
    angle_l1=logs/component_loss_quick/angle_recipe.json \
    angle_mse=logs/component_loss_quick/angle_mse_recipe.json \
  --batches 4 --output logs/component_loss_quick/gradient_probe_reproduction.json
```

各轮原始记录：[Round1](../logs/component_loss_quick/search_round1_stable/summary.json)、[Round2](../logs/component_loss_quick/search_round2/summary.json)、[Round3](../logs/component_loss_quick/search_round3/summary.json)、[Round4](../logs/component_loss_quick/search_round4/summary.json)。协议、RNG seed、样本 hash 和 runtime 信息与每轮结果位于同一目录。

## 完整记录与运行检查

[94次训练总表](experiments/init_diffusion_loss_search_20261011/training_runs.csv)逐项记录实际recipe、seed、预算、数据范围、指标、梯度裁剪和原始文件路径。[紧凑结果目录](experiments/init_diffusion_loss_search_20261011/README.md)保存可随代码库保留的CSV、选择规则、审计和图；大checkpoint不重复复制。

[search_plan.json](experiments/init_diffusion_loss_search_20261011/search_plan.json)保存了每轮方案、根据结果追加的测试和最终选择规则。每组保存checkpoint、完整配置、训练trace、指标JSON/CSV、manifest和hash；[代码记录](../logs/component_loss_quick/code_provenance.json)与source_snapshot保留实际loss/evaluator源文件。搜索热力图、配对表、EMA/梯度审计和最终分布图分别保存，失败过程也保留日志。

原入口实测还修复了checkpoint callback的`monitor=None`与`save_top_k=3`不兼容问题，推荐配置改为`save_top_k=1`。原入口128场景评价与1步训练smoke只验证加载、执行、EMA更新和checkpoint保存，不纳入94次搜索训练计数或推荐指标排名，见 [实测入口记录](experiments/init_diffusion_loss_search_20261011/native_entry_validation.json)。harness后续也修正了保存配置中顶层seed/checkpoint元数据；已完成运行的实际参数以protocol为准，CPU审计确认模型、数据、trainer未变。

初始workers4在训练转评价时出现native SIGSEGV，随后所有纳入比较的完整运行使用workers0。最初pilot/中断运行不纳入94次完整训练排名。梯度检查medium下的数值一致性失败和highest下的通过日志均保留，未将失败结果当作成功试验。

最终检查：**31项component-loss单元测试通过**；5个sweep/分析工具通过Python编译；原生训练和评价配置成功解析，并通过原入口加载赢家checkpoint评价128场景、执行1个训练step且成功保存checkpoint；工作区diff检查通过。所有推荐指标来自实际GPU生成评价，没有用训练loss替代。

## 结论边界

当前结果反映已训练 checkpoint 附近、固定场景条件下的短期适应能力，不能代表从零训练的学习效果、完整收敛结果或公开论文的无条件生成质量。搜索过程中查看同一 2048 样本并据此选择参数，存在选择偏差。[样本划分审计](../logs/component_loss_quick/eval_split_audit.json)确认完整 50000 个 `.pt` 的规范化顺序与官方 eval manifest 一致。搜索的 2048 样本与 offset2048 的 4096 个确认样本没有预处理缓存文件重叠，但有 35 个底层 TFRecord/scene-index 相同、map_id 不同的场景。因此应称为互不重叠的预处理缓存样本，不能称为完全独立的原始 Waymo scenarios。

初始 checkpoint 的长期训练、历史 EMA、原 optimizer 状态和后续默认实验配置与本次在线短训不完全相同。`deterministic` 未启用，固定 seed 不保证 CUDA 计算逐位一致；同一 checkpoint 的各轮 warm-start 指标存在小幅波动。完整 50k GT 参考的次要指标也不能替代同样 50k 生成样本的完整评价。

本环境安装的 `torch_ema` 使用 `min(decay, (1+num_updates)/(10+num_updates))` 作为实际衰减系数。初始 checkpoint 的 `num_updates=34704`；继续 512 步时，旧 EMA shadow 的累计系数仍约为 **87.6%**。不能只按配置的 `0.9999^512` 估计，也不能把沿用历史 EMA 的短训评价解释为当前在线参数的评价。本次 EMA 对比保留历史状态，没有重置 EMA。

第三种子训练manifest为8192个文件；前两个seed的128步来自32768文件的512步训练前段。各seed内对比严格共享数据和顺序，跨seed同时改变数据选择、顺序和噪声，不只是改变参数初始化随机数。所有训练共 **16640 optimizer steps**；94次含重复配方/seed，并非94个唯一超参数组合。

搜索和确认均复用官方车辆指标，但GT ego position/heading/shape/type是输入条件，地图是参考地图。因此不能把本报告的数值直接与论文的无条件lane+agent生成表格相比。单个分量loss逼近样本对应值，JSD却比较生成总体的边缘分布；更低的重建误差不必带来更低JSD。

本次搜索是自适应局部搜索，没有穷举loss×power×weight的无限连续空间。独立缓存块确认降低了初轮选择偏差，但确认块也参与后续角度方案选择；完整50k又包含这些缓存样本，不是最终未见的测试集。未来从头训练或不同ego/map条件需重新验证；这不影响已经完成的短期微调比较。
