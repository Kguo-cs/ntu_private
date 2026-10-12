# InitDiffusion 从头训练的分量 loss 搜索报告

2026-10-11；sim 环境；已完成从头训练搜索与实际生成评价。

## 建议与适用范围

本次建议完整从头训练采用 **`geo_mse32_col20_heading_l1_shape2`**。在已冻结的batch512、LR5e-4、G1 EMA2048规则下，heldout4096的三训练seed平均S为 **0.222667**，旧配方为 **0.230538**；两臂均值相比下降 **3.41%**，配对S在 **3/3** 个训练seed下降。

New recipe相对Old recipe的三seed总体均值变化为：angle -67.7%，lateral +3.4%，collision -8.8%，speed +34.1%，length +18.0%，width +56.9%。这是综合取舍，不是所有指标同时改善。车辆on-road覆盖仍只有79.5%，明显低于heldout GT的99.33%；几何与覆盖还未解决。

实际完成 **37个不同loss配方、44个随机初始化模型、11个续训阶段及235条生成评价记录**。map encoder与G1都随机、都可训练，没有加载预训练权重、评价集normalizer或warm checkpoint。核心B64搜索到6144steps；生产B512确认到2048steps、约2.124个完整TRAIN epoch；两个finalists还实际各生成50,000个缓存样本作全量分布审计。

这是一份用quick training/evaluation预测完整训练配方的报告。推荐为本轮搜索与冻结规则下最优的已测试选择；128epochs尚未完成，不能保证全局最优或最终排名不变。下面保留预算翻转、单项退步、数据范围、失败恢复和全部原始记录。

## 数据、初始化与计算协议

| 项目 | 实际设置 |
|---|---|
| 训练数据 | `src/waymo_data/full/training_map2_sd`；预存文件名清单；493,542 个缓存文件 |
| 评价数据 | `src/waymo_data/full/scenario_dreamer_val`；主阶段固定前 1024 个缓存样本，seed2817 |
| 初始化 | `action=fit`，`ckpt_path=null`，`finetune=false`；map encoder 和 G1 随机且参与训练；不用的 agent policy 冻结 |
| TRAIN 统计 | 独立随机抽取训练集 8192 场景、76,027 agents，seed4717；不读取 checkpoint、评价集或 warm 统计 |
| 地图条件 | SMART map token 的 square crop，半边长 50 m；`sep_map=false`；评价使用完整参考 lane 图 |
| ego 条件 | GT ego position、heading、length、width、type 固定；ego speed 生成；不能当作自由生成 ego 的结果，也不能直接声称与当前 SD/VW 相同条件 |
| 状态与类别 | scalar speed；linear length/width；circular heading，预测 x0；non-ego type 使用 separate CE 生成 |
| 数量与 embedding | 生成 agent slot 总数由输入 `agent_batch.numel()` 给定；time embedding=scenario_dreamer、scale99；count/map类别 embedding 均为 none |
| 结构 | G1 hidden256、3 denoiser layers、8 heads、20 采样步；shared map encoder 1 layer、hidden128 |
| 参数量 | CPU 构造检查：map encoder 437,312 可训练参数；G1 8,083,978；冻结 agent policy 0 |
| 优化器 | 新 AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01；主阶段 LR=1e-4；clip norm=0.5 |
| 学习率调度 | 仓库现有按真实 epoch 计数的 cosine，horizon128 epochs；短阶段仍在 epoch0，不能误当作 step 调度 |
| 精度 | `32-true`；float32 matmul `medium`；RTX4090；主阶段 workers0 |
| EMA | 主阶段排名用 online；G1 从随机权重开始记录 EMA；shared map 不在 G1 EMA 中 |
| 继续训练 | 恢复 optimizer/scheduler/EMA/RNG，训练文件从已消费 step×batch 之后开始，避免重放早期样本 |

所有主阶段组合共享 seeded full training order、校准统计及评价清单。每次完整生成前重置生成 seed；训练噪声和采样流程保留当前实现。首次 workers4 主阶段评价发生 segfault，该失败记录不计入有效评价；主阶段所有组合改用 workers0 重新运行。

[TRAIN normalizer](experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json)保存 `[x,y,cosθ,sinθ,length,width,speed]` 的均值与 population std。新的 `init_diffusion.normalizer_path` 只在显式指定 JSON 时初始化固定统计，校验维度、有限值和正 scale；后续加载 checkpoint 时，checkpoint buffers 优先。默认 null 保留旧的 lazy calibration 行为。

## 指标单位、评价覆盖与筛选规则

本报告七项指标都越低越好。**正文结果表中的六项 JSD 均为未缩放值，使用自然对数，范围为 [0, ln2≈0.693147]；collision 为百分数。**原始 helper 文件和 CSV 保留返回值：nearest/lateral 为 JSD×10；angle/length/width/speed 为 JSD×100。正文分别除以 10 或 100；例如当前 helper/Lightning angle=15.3，在正文为 0.153。当前 `metric_core.finalize_metrics()` 明确乘这些倍率，`SMART.on_validation_epoch_end()` 将返回的 metrics 原样写入 JSON 并调用 `self.log(name,value)`，没有再除回去。所以当前原生 Lightning 终端与原始 JSON 也使用放大后的单位；不能把正文未缩放 JSD 直接与终端数字比较。历史版本的输出单位应按对应代码核对，不能仅凭相同指标名推断。[指标单位源码审计](experiments/init_diffusion_scratch_loss_search_20261011/native_metric_unit_audit.json)记录倍率与直接日志路径。JSD 是 Jensen-Shannon distance 的平方；直方图范围、bin 宽度和 clipping 沿用仓库对齐的官方 helper。

主指标使用 `reference_mode=matched`，生成样本与同一批 GT 场景直方图比较。`full_reference_agent_metrics` 只是相同生成子集与全 50k GT reference 的辅助比较，不等于实际生成评价了 50k 场景。

| 指标 | 正文单位 | 原始 helper 倍率 | 综合 score 的固定 scale，正文单位 |
|---|---|---:|---:|
| nearest_dist_jsd | JSD | ×10 | 0.10 |
| lat_dev_jsd | JSD | ×10 | 0.10 |
| ang_dev_jsd | JSD | ×100 | 0.10 |
| length_jsd | JSD | ×100 | 0.20 |
| width_jsd | JSD | ×100 | 0.15 |
| speed_jsd | JSD | ×100 | 0.20 |
| collision_rate | % | ×1 | 5 |

`S = mean(metric / scale)`，这些 scale 是本报告统一固定的数值尺度，修正 helper 倍率后用于后续筛选，不是论文官方综合指标，也不来自 pretrained 模型。在原始 helper 单位下使用等价 scale `[1,1,10,20,15,20,5]`，S 不因展示单位转换而改变。此前临时汇总未正确换算 helper 返回值倍率，已统一修正；原始七项指标没有因此改变。

筛选同时看七项指标、车辆数/GT、on-road 覆盖和后期曲线。lat/angle 只统计 on-road 车辆：少生成可评价车辆或让车辆偏离道路，可能改变该条件分布，不能当作所有车辆的误差下降。agent slot 总数给定，类别生成会改变其中的 vehicle 子集与直方图权重；碰撞率按生成车辆总数汇总，也受这个分母影响。on-road按车辆中心距resampled lane点不超过1.5 m判定；nearest排除只含一辆vehicle的场景，length/width/speed和碰撞分母覆盖全部生成vehicles。主阶段 GT 为 7,642 vehicles，其中 7,596 on-road，覆盖 99.398%。不同 LR、预算、评价清单、初始化范围、online/EMA 应分组比较；生成 seed 在训练 seed 内平均，不能把重复采样当成新增独立训练 seed。

## 原训练 loss 与新分量 loss 的区别

`component_loss=null` 的 legacy 实际重建总项是有效状态的 **raw MSE×0.02×时间权重**。日志中的 position/heading/shape/velocity 分量与 `w_heading/w_shape/w_vel` 并没有重新求和成为优化总项；`w_pos=0.1/5=0.02` 乘整个 raw-state MSE。当前默认 `use_l1=false` 令时间权重成为 `max(t,0.05)^-2`；scalar speed的7D状态在legacy适配中补一个dummy0成为8D，raw MSE仍含该坐标和固定零残差坐标的均值分母；新active-coordinate/agent分量均值与其并不等价。另加 separate type CE 与原有 collision loss。原 collision 使用穿透量的指数变换；新组件使用超过容许阈值后的正穿透量，单位为米，并选择 L1/MSE。因此这是一处函数形式差异，collision 的数字系数不能直接跨两条路径比较。

各分量 recipe 不改变现有噪声来源和 Hungarian matcher；它们改变的是实际优化项。新路径直接相加各分量的 active-agent/pair 均值，物理分量按固定 TRAIN std 归一化，heading 选择 vector 或最短圆周角差，时间 power 独立且可按解析期望归一化。改变 L1/MSE 会改变梯度幅度；改变 power 会改变不同噪声时间的相对权重；改变单项 weight 在 clip0.5 下还会改变全网更新方向与其他任务的有效份额。因此同一个数字系数并不保证与 legacy 同样的学习强度，不能只看日志名称推断真实 objective。

### 主阶段 anchor

| 分量 | kind | power | weight | scale |
|---|---|---:|---:|---|
| position | l1 | 1 | 2 | TRAIN std |
| heading，angle | mse | 1 | 1 | none |
| length | mse | 1 | 1 | TRAIN std |
| width | mse | 1 | 1 | TRAIN std |
| scalar speed | mse | 1 | 2 | TRAIN std |
| type | ce | 0 | 1 | none |
| collision | l1 | 1 | 1 | none |

使用 `normalize_time_weights=true`、`std_floor=0.1`。主阶段对比包括 legacy、先前 loss 配方的随机初始化版本、vector heading、position MSE、heading p2/p3、heading L1、width p2、speed p2、position/collision 加权和 collision 弱权重；每组对应的全部 overrides 保存于原始 design JSON。

## 学习率试跑与预算依赖

LR 对比实际使用 1e-4、3e-4、5e-4。128/256-step 阶段评价 512 样本；继续到 512/1024-step 时评价 1024 样本，因此只在同一行预算内横向比较。

| LR | 128-step S，512样本 | 256-step S，512样本 | 512-step S，1024样本 | 1024-step S，1024样本 |
|---|---:|---:|---:|---:|
| 1e-4 | 1.5795 | 1.1985 | 0.9203 | 0.8340 |
| 3e-4 | 2.0981 | 1.0066 | 0.8158 | 0.9037 |
| 5e-4 | 1.4316 | 1.6139 | 1.0941 | 0.9692 |

128-step 的 5e-4、256/512-step 的 3e-4、1024-step 的 1e-4 分别最好，排名发生翻转。选 1e-4 作为当前主阶段 LR，是基于后期数值稳定与 1024-step 同预算观察；没有证明 128 epochs 下 1e-4 一定最优。1024-step 的 3e-4 speed JSD=0.00557 很低，但 length/width JSD=0.04308/0.08469，不能只用 speed 指标决定 LR。

## 主阶段：12 组实际 1024-step 结果

下面同为训练 seed817、生成 seed2817、1024 个评价样本；表中六项 JSD 已还原为未缩放值。S 排名是阶段性描述，不作为单独的最终选择依据。

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm_winner | 0.6470 | 0.03896 | 0.09792 | 0.15917 | 0.01472 | 0.01405 | 0.03718 | 6.0773 | 1.066 | 64.5 |
| geometry_strong | 0.6676 | 0.04262 | 0.09664 | 0.12687 | 0.02626 | 0.01725 | 0.04177 | 7.7846 | 0.982 | 65.0 |
| legacy | 0.7585 | 0.03122 | 0.09548 | 0.15573 | 0.05059 | 0.04780 | 0.02199 | 9.0174 | 1.020 | 71.7 |
| speed_p2 | 0.8528 | 0.05643 | 0.08020 | 0.13994 | 0.02829 | 0.01518 | 0.02427 | 14.2000 | 0.938 | 58.5 |
| heading_l1 | 0.8774 | 0.05780 | 0.08634 | 0.10475 | 0.05695 | 0.00877 | 0.04575 | 15.4041 | 1.031 | 59.1 |
| vector_heading | 0.8861 | 0.05174 | 0.08368 | 0.16192 | 0.03606 | 0.01408 | 0.01938 | 14.2913 | 0.995 | 56.8 |
| anchor | 0.9257 | 0.05461 | 0.08357 | 0.15304 | 0.01946 | 0.04876 | 0.02711 | 15.0491 | 1.053 | 56.0 |
| heading_p2 | 0.9355 | 0.05141 | 0.08636 | 0.19290 | 0.01587 | 0.00750 | 0.05196 | 14.2624 | 1.043 | 55.5 |
| position_mse | 0.9664 | 0.05047 | 0.08605 | 0.14860 | 0.06628 | 0.04080 | 0.02346 | 15.9635 | 0.974 | 62.0 |
| heading_p3 | 1.0273 | 0.05519 | 0.08881 | 0.24648 | 0.01755 | 0.01932 | 0.03967 | 14.3576 | 1.066 | 55.7 |
| collision_weak | 1.0519 | 0.05413 | 0.08356 | 0.16028 | 0.06641 | 0.07179 | 0.04611 | 16.7099 | 1.060 | 56.6 |
| width_p2 | 1.0862 | 0.05313 | 0.08336 | 0.16288 | 0.18266 | 0.13881 | 0.03961 | 12.8634 | 1.049 | 56.0 |

较强 position/collision 配比在这个阶段明显降低碰撞并提高 on-road 覆盖，但不是所有几何指标都改善。`geometry_strong` 与 anchor 的 collision 为 7.7846%/15.0491%，on-road 为 65.0%/56.0%，而 lateral JSD 为 0.09664/0.08357，条件评价分布同时变化。

提高 heading power 不一定更好：anchor、heading_p2、heading_p3 在 1024 steps 的 angle JSD 为 0.15304/0.19290/0.24648；p3 更强强调小 t，没有解决生成从 t=1 开始时的高噪声误差。heading L1 将 angle JSD 降到 0.10475，但 collision 为 15.4041%，并未成为当前全指标最稳的候选。`width_p2` 在 256-step 的 width JSD=0.06805，到 1024-step 却为 0.13881；早期单点不能预测后期尺寸分布。

学习曲线图片仍使用原始 helper 单位，nearest/lateral 为 JSD×10，angle/length/width/speed 为 JSD×100，collision 为%；热图为相对 anchor 的百分比变化，不是 JSD 绝对值。

[全部主阶段组合的学习曲线，独立矢量图](experiments/init_diffusion_scratch_loss_search_20261011/main_stage/learning_curves_8bcd9b085236.pdf)。报告后文使用重新排版的紧凑图，保证页面中的字号可读。

![主阶段：同预算的七项指标与 anchor 对比](experiments/init_diffusion_scratch_loss_search_20261011/main_stage/metric_heatmap_8bcd9b085236_step_1024.png)

## 自适应追加：13组实际1024-step结果

这一轮实际完成13组更强位置权重、较弱heading/width权重、p0/p2/p3和scalar L1控制，均保存1024-step完整七项指标。它们仍只有一个训练seed，不能代替多seed确认。

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced32_l1_quarter | 0.7490 | 0.03357 | 0.10206 | 0.09180 | 0.03949 | 0.03707 | 0.04079 | 11.6021 | 1.131 | 73.8 |
| geo_l1_16 | 0.7722 | 0.03436 | 0.09865 | 0.13028 | 0.01533 | 0.00796 | 0.04195 | 12.1646 | 1.088 | 70.8 |
| geo_mse32 | 0.8135 | 0.03914 | 0.10175 | 0.11803 | 0.01435 | 0.01268 | 0.03979 | 13.7499 | 1.106 | 74.0 |
| balanced32_scalars_p2 | 0.8208 | 0.03249 | 0.10347 | 0.10847 | 0.04986 | 0.06402 | 0.03095 | 12.3520 | 1.117 | 74.3 |
| balanced32_p2 | 0.8339 | 0.02680 | 0.09850 | 0.11884 | 0.12624 | 0.23709 | 0.02548 | 5.2831 | 1.001 | 70.1 |
| balanced32 | 0.8371 | 0.03741 | 0.10212 | 0.11000 | 0.05426 | 0.06723 | 0.02265 | 12.6591 | 1.100 | 74.1 |
| pos_mse32 | 0.8378 | 0.04586 | 0.10102 | 0.08963 | 0.02171 | 0.01408 | 0.04256 | 15.4223 | 1.069 | 75.5 |
| balanced32_speed_l1 | 0.8740 | 0.03703 | 0.10204 | 0.12673 | 0.03659 | 0.03931 | 0.03047 | 14.3132 | 1.088 | 74.1 |
| balanced32_scalars_p3 | 0.9150 | 0.03616 | 0.10321 | 0.09993 | 0.12727 | 0.13515 | 0.05280 | 11.0519 | 1.031 | 73.8 |
| balanced32_shape_l1 | 0.9423 | 0.03944 | 0.10429 | 0.09297 | 0.12976 | 0.12687 | 0.04288 | 12.5999 | 1.130 | 72.7 |
| balanced32_vector | 0.9578 | 0.03653 | 0.10167 | 0.16326 | 0.07571 | 0.12280 | 0.03633 | 11.5561 | 1.085 | 73.9 |
| balanced32_p0 | 0.9912 | 0.05752 | 0.10355 | 0.08537 | 0.01440 | 0.02149 | 0.03850 | 20.3317 | 1.121 | 74.2 |
| geo_mse64 | 1.0240 | 0.05414 | 0.10434 | 0.11316 | 0.02107 | 0.01533 | 0.02894 | 20.4986 | 1.108 | 74.8 |

`pos_mse32` 的 on-road 覆盖 75.5%，高于 anchor 的 56.0%，同时其 lateral JSD=0.10102 不比 anchor=0.08357 低。`balanced32_p2` 的 collision=5.2831% 很低，但 length/width JSD=0.12624/0.23709，说明把位置和碰撞做好之后，尺寸分支可能仍然失衡。需要继续观察各项预算曲线与类别构成，不能仅按碰撞或者 S 的一个数继续加大位置系数。

### 追加交互组合：已完成的 8 组 1024-step 检查

在较强 position/collision 组合中进一步测试 heading L1，以及更大的 collision 系数。这些检查仍使用同一训练 seed、相同 1024 个评价样本和相同生成 seed；没有因此完成最终赢家或多 seed 确认。

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo_mse32_col20 | 0.5531 | 0.02517 | 0.09966 | 0.12048 | 0.03253 | 0.06329 | 0.04400 | 3.0715 | 1.078 | 74.2 |
| pos_mse32_col20 | 0.5662 | 0.02328 | 0.09856 | 0.09011 | 0.05096 | 0.07962 | 0.04249 | 4.2297 | 1.055 | 73.6 |
| geo_l1_16_col20 | 0.6096 | 0.03352 | 0.09746 | 0.16753 | 0.04822 | 0.01324 | 0.04774 | 3.5714 | 1.077 | 68.4 |
| geometry_strong_heading_l1 | 0.6171 | 0.03896 | 0.09878 | 0.07609 | 0.01380 | 0.00788 | 0.02686 | 9.6286 | 1.046 | 66.8 |
| balanced32_col20 | 0.6239 | 0.02553 | 0.10327 | 0.15637 | 0.04700 | 0.02787 | 0.05152 | 4.1864 | 1.107 | 73.5 |
| balanced32_heading_l1_w1 | 0.7053 | 0.03741 | 0.10104 | 0.04793 | 0.03598 | 0.02309 | 0.03856 | 12.7341 | 1.083 | 75.3 |
| pos_mse32_heading_l1 | 0.7641 | 0.04348 | 0.10437 | 0.05389 | 0.04730 | 0.04517 | 0.02876 | 13.2494 | 1.080 | 74.3 |
| geo_l1_16_heading_l1 | 0.8238 | 0.03870 | 0.09920 | 0.10779 | 0.01756 | 0.01263 | 0.03988 | 14.6917 | 1.082 | 72.9 |

`geometry_strong_heading_l1` 的 angle JSD=0.07609，length/width=0.01380/0.00788，speed=0.02686，collision=9.6286%，on-road=66.8%。它改善了短预算的角度分布，但不能据此外推完整 128-epoch 结果。`pos_mse32_col20` 将 collision 降到 4.2297%，代价仍包括尺寸分布与条件 on-road 指标的变化；后续选择需要兼顾覆盖与更长预算表现。

### 已完成的 width × heading 配对检查

下面每一对都逐字段核对保存的 recipe；表中只改变所列因素，position、length、speed、type 与 collision 保持相同。仍然只有一个训练 seed 与生成 seed，这是条件效应的实测，不能当作跨 seed 的普遍规律。

| 单一变化，左→右 | Angle JSD，左/右 | Length JSD，左/右 | Width JSD，左/右 | Speed JSD，左/右 | Collision%，左/右 | On-road%，左/右 |
|---|---:|---:|---:|---:|---:|---:|
| width w1→0.5；heading MSE/w0.25 | 0.11803/0.11000 | 0.01435/0.05426 | 0.01268/0.06723 | 0.03979/0.02265 | 13.75/12.66 | 74.0/74.1 |
| width w1→0.5；heading L1/w1 | 0.05389/0.04793 | 0.04730/0.03598 | 0.04517/0.02309 | 0.02876/0.03856 | 13.25/12.73 | 74.3/75.3 |
| heading MSE w0.25→1；width w1 | 0.11803/0.08963 | 0.01435/0.02171 | 0.01268/0.01408 | 0.03979/0.04256 | 13.75/15.42 | 74.0/75.5 |
| heading MSE→L1，均 w1；width w1 | 0.08963/0.05389 | 0.02171/0.04730 | 0.01408/0.04517 | 0.04256/0.02876 | 15.42/13.25 | 75.5/74.3 |

在 MSE-heading/w0.25 下，width 权重减半将 width JSD 从 0.01268 变为 0.06723；在 L1-heading/w1 下，同样减半却将 width JSD 从 0.04517 变为 0.02309。两个条件下变化方向相反，说明 width 权重不能脱离 heading 及共享网络的梯度配比单独推荐。相比之下，固定 width w1 时，将 heading MSE 从 w0.25 增至 w1 或将 MSE 改为 L1，都有不同的角度、尺寸、速度和碰撞取舍。

`geometry_strong` 与 `warm_winner` 同时改变 heading/width power，不能把差异只归因于其中一项。现已完成的 targeted4096 补齐 MSE-heading 的完整2×2 power对照，并测试 L1-heading/width-p2 和联合 geometry/shape 组合；实际终点与条件效应见后文。

## 时间分桶诊断：高噪声端仍然困难

对 4 个实际 1024-step checkpoint，用相同 256 场景、2152 agents 和同一份随机训练输入做 CPU probe；没有 optimizer 更新，paired-input SHA256 相同。下表是 clean x0 在随机训练 t 的重建误差，不是完整生成指标，不计入 JSD 排名。position 为每坐标 L1 均值，heading 为最短角差绝对值均值，speed 为 scalar L1。

| Checkpoint | t区间 | Position L1 (m/坐标) | Heading L1 (°) | Speed L1 (m/s) | Type accuracy |
|---|---|---:|---:|---:|---:|
| anchor | low，0–0.1 | 0.579 | 3.65 | 0.512 | 1.000 |
| anchor | mid，0.1–0.5 | 3.662 | 11.07 | 1.102 | 0.988 |
| anchor | high，0.5–1 | 9.092 | 33.56 | 2.915 | 0.899 |
| geometry_strong | high，0.5–1 | 8.471 | 35.20 | 2.765 | 0.908 |
| warm_winner，随机训练 | high，0.5–1 | 8.516 | 38.37 | 3.144 | 0.896 |
| pos_mse32 | high，0.5–1 | 7.745 | 33.36 | 2.866 | 0.896 |

低 t 的小误差与高 t 的大误差同时存在，说明只提升小 t 权重不能直接保证生成初始化更好。这支持继续测试更弱时间 power 或更合理的任务配比，但不是 p0 在完整训练预算下整体更优的证明。当前 `balanced32_p0` 的 angle JSD=0.08537 优于 `balanced32` 的 0.11000，而 collision=20.3317% 高于 12.6591%；各项仍有取舍。normalized p1/p2/p3 只是让时间权重的平均量级一致，各自高/低噪声端的相对梯度仍然不同。

## 继续训练：7 组实际 4096-step 结果

这 7 组从各自实际的 1024-step scratch checkpoint 继续训练，绝对终点为 4096 optimizer steps；不是重新微调 pretrained 模型。记录显示起点 `resume_global_step=1024`、`optimizer_continued=true`，每组训练清单从偏移 65,536 开始，继续消费后面的 196,608 个样本，总 exposure 为 262,144 样本，约 0.531 个 full-data epoch。训练 seed817、生成 seed2817、前 1024 个评价缓存与主阶段相同。2048/4096 steps 分别评价 online 与 G1 EMA，共 28 个完整七指标记录；shared map encoder 在两种评价中仍使用当前 online 权重。

### 4096 steps，online 权重

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm_winner | 0.3779 | 0.02156 | 0.10262 | 0.04336 | 0.01720 | 0.01700 | 0.00865 | 3.6354 | 0.979 | 76.9 |
| geometry_strong_heading_l1 | 0.3920 | 0.01848 | 0.10312 | 0.03870 | 0.01356 | 0.02691 | 0.00463 | 4.3534 | 1.025 | 78.3 |
| geometry_strong | 0.4057 | 0.02500 | 0.10176 | 0.05575 | 0.05001 | 0.01180 | 0.00431 | 3.3237 | 0.996 | 75.7 |
| geo_l1_16 | 0.4676 | 0.01549 | 0.09971 | 0.06008 | 0.01221 | 0.03804 | 0.00502 | 5.9015 | 1.004 | 80.3 |
| geo_mse32_col20 | 0.5285 | 0.02214 | 0.10728 | 0.11635 | 0.06349 | 0.05916 | 0.00681 | 2.4813 | 1.086 | 76.5 |
| legacy | 0.5632 | 0.01442 | 0.10152 | 0.09296 | 0.09588 | 0.10099 | 0.01447 | 3.1420 | 0.987 | 79.0 |
| balanced32_l1_quarter | 0.7251 | 0.02363 | 0.10753 | 0.10754 | 0.06126 | 0.08236 | 0.01428 | 8.8081 | 1.144 | 75.6 |

### 4096 steps，G1 EMA 权重

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry_strong_heading_l1 | 0.2956 | 0.02072 | 0.10588 | 0.00864 | 0.01182 | 0.00482 | 0.00427 | 3.0219 | 1.035 | 76.2 |
| warm_winner | 0.3122 | 0.02389 | 0.10091 | 0.03285 | 0.01099 | 0.00453 | 0.00341 | 2.5347 | 1.017 | 74.9 |
| geometry_strong | 0.3210 | 0.02247 | 0.10465 | 0.03009 | 0.01171 | 0.00582 | 0.00356 | 2.7994 | 1.024 | 75.6 |
| geo_mse32_col20 | 0.3405 | 0.01888 | 0.10243 | 0.07456 | 0.01075 | 0.00413 | 0.00524 | 1.5877 | 1.005 | 78.9 |
| geo_l1_16 | 0.3966 | 0.01704 | 0.10527 | 0.06015 | 0.01318 | 0.00574 | 0.00388 | 4.1420 | 1.017 | 80.5 |
| legacy | 0.4261 | 0.01665 | 0.10114 | 0.08193 | 0.02056 | 0.01948 | 0.00316 | 3.6829 | 1.023 | 78.3 |
| balanced32_l1_quarter | 0.4767 | 0.01989 | 0.10216 | 0.07319 | 0.01723 | 0.01228 | 0.00806 | 5.8824 | 1.079 | 77.9 |

在这7个先前候选中，4096-step online 的最低 S 是 `warm_winner` 的 0.3779；EMA 的最低 S 是 `geometry_strong_heading_l1` 的 0.2956。部署/评价权重不同会改变排序，不能把二者混排后宣布一个与权重无关的最优组合。在此终点，EMA 降低了全部 7 组的 width/speed JSD，并降低多数 length JSD；它只平均 G1，不是整网 EMA。

### 2048→4096 steps 的预算变化

| Variant | Online S，2048→4096 | EMA S，2048→4096 | Online on-road%，2048→4096 | EMA on-road%，2048→4096 |
|---|---:|---:|---:|---:|
| balanced32_l1_quarter | 0.6152→0.7251 | 0.5078→0.4767 | 74.5→75.6 | 75.5→77.9 |
| geo_l1_16 | 0.4773→0.4676 | 0.4613→0.3966 | 75.2→80.3 | 76.1→80.5 |
| geo_mse32_col20 | 0.4982→0.5285 | 0.4008→0.3405 | 74.0→76.5 | 75.8→78.9 |
| geometry_strong | 0.4049→0.4057 | 0.3823→0.3210 | 71.2→75.7 | 71.8→75.6 |
| geometry_strong_heading_l1 | 0.3867→0.3920 | 0.3261→0.2956 | 71.9→78.3 | 72.3→76.2 |
| legacy | 0.5679→0.5632 | 0.5328→0.4261 | 75.3→79.0 | 75.4→78.3 |
| warm_winner | 0.5906→0.3779 | 0.3823→0.3122 | 71.3→76.9 | 72.0→74.9 |

1024-step 排序没有完全预测后续表现：`balanced32_l1_quarter` 在早期追加中 S=0.7490，但 4096-step online 仍为 0.7251；`geometry_strong_heading_l1` 的 EMA angle JSD 由 2048-step 的 0.01985 降到 0.00864，同时 EMA collision 从 3.4370% 降到 3.0219%。不过它的 4096-step online collision=4.3534%，高于 `geometry_strong` 的 3.3237%。固定其他 loss 分量，仅把 geometry_strong 的 heading MSE 改为同系数 L1，4096-step EMA angle=0.03009→0.00864，S=0.3210→0.2956，但 collision=2.7994%→3.0219%；这是角度收益与碰撞取舍，不是全指标支配。

这7个候选中 best-EMA 的 on-road=76.2%，仍显著低于 matched GT=99.4%；`geo_l1_16` 的 EMA on-road 较高，为 80.5%，但 collision=4.1420%、angle JSD=0.06015。因此后续最终筛选继续把车辆数、覆盖、budget 曲线与单项指标放在一起看，不能只按 S 或较低的条件 angle 选择。

## Targeted4096：4 组实际配对检查

4 个新 recipe 都从相同随机初始化、固定 TRAIN normalizer 和 seed817 从头训练至 4096 updates，在 1024/2048/4096 steps 分别生成 online/G1 EMA 指标，共 24 个评价记录。评价清单与生成 seed2817 仍与前阶段相同；与 resumed 控制的比较使用相同原始随机参数、full train order、统计和评价 hashes，续训清单因偏移而不同，不能据此声称 CUDA bitwise 等价。

### 4096 steps，4组 online

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry_strong_heading_l1_width_p2 | 0.3345 | 0.02315 | 0.10362 | 0.01092 | 0.02535 | 0.02520 | 0.00750 | 3.1632 | 0.985 | 77.5 |
| geometry_strong_width_p2 | 0.3583 | 0.02373 | 0.09835 | 0.04821 | 0.01032 | 0.01340 | 0.00891 | 3.0970 | 0.980 | 75.7 |
| geo_mse32_col20_heading_l1_shape2 | 0.3694 | 0.01729 | 0.10374 | 0.03895 | 0.02870 | 0.02695 | 0.00770 | 3.1231 | 1.052 | 77.2 |
| geometry_strong_heading_p2 | 0.3832 | 0.02603 | 0.10246 | 0.05120 | 0.01241 | 0.03330 | 0.00479 | 2.8875 | 1.011 | 75.7 |

### 4096 steps，4组 G1 EMA

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo_mse32_col20_heading_l1_shape2 | 0.2680 | 0.02227 | 0.10353 | 0.01121 | 0.01044 | 0.00501 | 0.00738 | 1.9161 | 1.011 | 78.6 |
| geometry_strong_heading_l1_width_p2 | 0.2873 | 0.01881 | 0.10530 | 0.00831 | 0.01216 | 0.00581 | 0.00487 | 2.8167 | 1.027 | 76.6 |
| geometry_strong_width_p2 | 0.3114 | 0.02206 | 0.10104 | 0.02879 | 0.01151 | 0.00449 | 0.00397 | 2.7672 | 1.026 | 75.2 |
| geometry_strong_heading_p2 | 0.3293 | 0.02223 | 0.10383 | 0.03902 | 0.01183 | 0.00583 | 0.00369 | 2.6906 | 1.026 | 75.5 |

### 相对各自父配方的变化，4096-step G1 EMA

负 ΔJSD/ΔS/Δcollision 代表改善；on-road 用百分点。前三项是配对单因素改变；最后一项同时改变 heading kind/weight 与 length/width weights，不能单独归因于 shape 权重。

| Variant / weights | ΔS% | ΔNearest% | ΔLateral% | ΔAngle% | ΔLength% | ΔWidth% | ΔSpeed% | ΔCollision% | ΔOn-road pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry_strong_width_p2 | -3.00 | -1.79 | -3.45 | -4.31 | -1.69 | -22.83 | +11.53 | -1.15 | -0.36 |
| geometry_strong_heading_p2 | +2.59 | -1.04 | -0.79 | +29.68 | +1.00 | +0.07 | +3.56 | -3.89 | -0.02 |
| geometry_strong_heading_l1_width_p2 | -2.80 | -9.24 | -0.55 | -3.81 | +2.83 | +20.48 | +14.18 | -6.79 | +0.41 |
| geo_mse32_col20_heading_l1_shape2 | -21.31 | +17.96 | +1.08 | -84.97 | -2.81 | +21.19 | +40.91 | +20.68 | -0.34 |

### Heading power × width power 的完整 MSE 对照

各项系数与其他 loss 分量相同，下面只改变 MSE-heading/width 的 time power。S 的 difference-in-differences 为 −0.007485；只有一个配对训练/生成 seed，这描述相互作用，不是显著性检验。

| Recipe，4096 EMA | Heading kind / p | Width p | S | Angle JSD | Width JSD | Speed JSD | Collision% | On-road% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_strong | mse / 1 | 1 | 0.3210 | 0.03009 | 0.00582 | 0.00356 | 2.7994 | 75.6 |
| geometry_strong_width_p2 | mse / 1 | 2 | 0.3114 | 0.02879 | 0.00449 | 0.00397 | 2.7672 | 75.2 |
| geometry_strong_heading_p2 | mse / 2 | 1 | 0.3293 | 0.03902 | 0.00583 | 0.00369 | 2.6906 | 75.5 |
| warm_winner | mse / 2 | 2 | 0.3122 | 0.03285 | 0.00453 | 0.00341 | 2.5347 | 74.9 |
| geometry_strong_heading_l1 | l1 / 1 | 1 | 0.2956 | 0.00864 | 0.00482 | 0.00427 | 3.0219 | 76.2 |
| geometry_strong_heading_l1_width_p2 | l1 / 1 | 2 | 0.2873 | 0.00831 | 0.00581 | 0.00487 | 2.8167 | 76.6 |

MSE-heading 下，width p2 在两种 heading power 条件中都让 S 下降约 3.0%/5.2%；heading p2 在 width p1 下让 S 变差，在 width p2 下几乎持平，而且这里两项配对 angle JSD 都没有改善。相同 heading weight1 时，angle L1 相比 angle MSE 将 angle JSD 降低约 71%，两种 width power 下均如此。这比多项同时改变的 recipe 更直接支持 heading-kind 的条件效果。

L1-heading 下，width p1→p2 的 S 改善 2.80%、collision 改善 6.79%，但 width JSD 变差 20.48%、length 变差 2.83%、speed 变差 14.18%。这个组合因整体取舍入围，不能说它的 width fidelity 更好。强 geometry/shape 联合组合相对父配方 S 改善约 21.3%，主要角度下降约 85%；width、speed、collision 分别变差约 21.2%/40.9%/20.7%。它也不全指标支配另一候选。

### 机制解释及边界

解析归一化的时间权重让 uniform t 下 E[w]=1，不等于让 E[w·loss(t)] 或各分量梯度相同。clamp(t,0.05)^−p 中，p2 把约 76.9% 权重分给 t<0.1、仅约 2.56% 分给 t>0.5；p1 对应约 42.4%/17.35%。p2 更强调接近 clean data 的局部去噪，不能保证从高噪声源开始的生成更好。实际 L1-heading 配对末尾256 updates 中，width weighted-loss 中位数在 p1→p2 后由约 0.1000 降到 0.0247，尽管 numerical weight1 不变；低噪声残差较小，会改变有效任务份额。

当前 angle loss 的残差单位为弧度，评价 angle histogram 另使用度。远离0与±π分支点、未计时间和分量权重时，angle L1对角差残差的导数幅度恒定，angle MSE的幅度随角差增大；整个网络梯度还包含atan2等链式导数，±π处仍有不可微分支；它可能减轻不确定大角差对共享梯度的支配。两条路径都使用圆周角差，因此这里不是旧 sin/cos MSE 在180°附近弱梯度的问题。位置、尺寸、heading 共用 denoiser，并通过 collision geometry 耦合。当前 matcher 在 denoising 前按 GT/source normalized state 重排源噪声；改变 component loss 不会直接改变 matcher，但不同分量共享同一配对训练样本。检查的末尾更新都触发 global clip0.5；preclip 梯度中位数 L1-heading 约3.32，联合 geometry/shape 约7.72、父配方约6.17。改变 loss 配比会改变裁剪后的共享方向，但这些日志不是 per-component 梯度分解，不能据此完成因果归因。

源码还限定了一项解释：`compute_penetration()` 对预测 length/width 做了 `.detach()`，因此碰撞项不直接把梯度传回预测尺寸；缩小尺寸仍会改变其前向数值和评价碰撞率，但此项不会直接驱动尺寸输出缩小。更大的 collision weight 可以通过共享网络和裁剪改变其他任务的更新，但本轮没有测量每项的梯度分解，不能把尺寸退步直接归因于“碰撞梯度把车缩小”。

训练collision组件与评价collision_rate的范围也不同：训练使用场景内所有类型、具有有效GT尺寸的agent pairs，允许GT本身已有的pair penetration，包含零excess pair并取均值；评价只在生成type=0的vehicles中统计绝对碰撞比例。两者不是同一个函数，pedestrian/cyclist交互和GT-overlap allowance可以影响训练梯度而不直接出现在车辆碰撞率里。

尺寸和速度的逐样本重建误差与直方图 JSD 也不是同一目标。TRAIN 全类别 length/width std=1.538/0.494 m，而 search1024 的 GT vehicle clipped-bin std≈0.795/0.186 m；这些是不同数据/类别范围，不能当作泄漏或精确同分布比较，但提示全类别尺度不一定能强调车辆的窄峰与尾部。官方尺寸 bin 宽0.1 m，较小偏移就可能改变峰值桶。当前 speed 在训练时无非负约束，只在物理输出端 clamp_min(0)；第一桶覆盖0–1 m/s，不能用它推断精确停车车辆数。条件均值/中位数偏好、尾部学习和共享梯度配比是分布退步的候选解释，而不是已经隔离验证的根因。

[配对与机制记录](experiments/init_diffusion_scratch_loss_search_20261011/targeted4096_factorial_and_mechanistic_notes.md)另保留 clipped-bin 中点近似分布均值/std、原始 hashes 与全部对比。近似 moments 不能替代完整 histogram JSD；生成 type 改变 vehicle 子集，speed 0–1 m/s 第一 bin 也不是精确停车比例。

## 6144-step：4 个实际延续结果

按当时的自适应规则保留 `legacy` 与 `warm_winner` 两个控制，并从4096-step EMA结果选择满足 on-road≥72%、V/GT∈[0.8,1.2] 的两个最低S新配方；该规则在本次延续前记录，不表示整轮自适应搜索完全预注册。[选择记录](experiments/init_diffusion_scratch_loss_search_20261011/extension6144_selection.json)保存全部11组排序。

`legacy`、`warm_winner` 和 2 个 targeted finalists 从真实 4096-step checkpoint 继续到 6144 updates；optimizer/EMA 状态保留，训练清单从偏移262,144继续，不重放此前样本。5120/6144 各评价 online/EMA，共16条实测记录。batch64 总 exposure393,216 缓存样本，约0.797 epoch；未覆盖完整128epochs。

### 6144 steps，online

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry_strong_heading_l1_width_p2 | 0.2905 | 0.01681 | 0.10173 | 0.01363 | 0.01477 | 0.00267 | 0.00340 | 3.0157 | 1.033 | 79.1 |
| geo_mse32_col20_heading_l1_shape2 | 0.3048 | 0.01918 | 0.09710 | 0.01747 | 0.02291 | 0.00406 | 0.01248 | 2.9602 | 1.030 | 75.9 |
| warm_winner | 0.3141 | 0.01952 | 0.09881 | 0.01883 | 0.02292 | 0.00450 | 0.00354 | 3.3255 | 1.003 | 78.0 |
| legacy | 0.5372 | 0.01588 | 0.09963 | 0.11830 | 0.03696 | 0.04400 | 0.00803 | 4.5212 | 1.007 | 79.0 |

### 6144 steps，G1 EMA

| Variant | S | Nearest JSD | Lateral JSD | Angle JSD | Length JSD | Width JSD | Speed JSD | Collision% | V/GT | On-road% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo_mse32_col20_heading_l1_shape2 | 0.2434 | 0.02058 | 0.09859 | 0.00687 | 0.01094 | 0.00495 | 0.00461 | 1.6630 | 1.015 | 78.5 |
| geometry_strong_heading_l1_width_p2 | 0.2562 | 0.02161 | 0.09918 | 0.00560 | 0.01074 | 0.00426 | 0.00652 | 2.0732 | 0.997 | 78.1 |
| warm_winner | 0.2833 | 0.02124 | 0.10258 | 0.02544 | 0.01045 | 0.00432 | 0.00366 | 1.9546 | 0.998 | 78.0 |
| legacy | 0.3776 | 0.01412 | 0.09880 | 0.07417 | 0.01553 | 0.01461 | 0.00245 | 2.9256 | 1.011 | 79.6 |

### 同配方 4096→6144 的变化

下面按相同 recipe 和权重范围配对，不混合 online/EMA。各项百分比基于未经取整的原始 helper，JSD 单位转换不改变比值；on-road 是百分点。

| Variant / weights | ΔS% | ΔNearest% | ΔLateral% | ΔAngle% | ΔLength% | ΔWidth% | ΔSpeed% | ΔCollision% | ΔOn-road pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| geo_mse32_col20_heading_l1_shape2 / online | -17.50 | +10.91 | -6.41 | -55.15 | -20.17 | -84.92 | +61.97 | -5.21 | -1.33 |
| geometry_strong_heading_l1_width_p2 / online | -13.16 | -27.38 | -1.82 | +24.78 | -41.73 | -89.41 | -54.68 | -4.66 | +1.57 |
| legacy / online | -4.61 | +10.11 | -1.85 | +27.26 | -61.46 | -56.44 | -44.47 | +43.90 | -0.02 |
| warm_winner / online | -16.86 | -9.49 | -3.71 | -56.59 | +33.30 | -73.49 | -59.03 | -8.52 | +1.13 |
| geo_mse32_col20_heading_l1_shape2 / ema | -9.17 | -7.58 | -4.77 | -38.71 | +4.74 | -1.05 | -37.46 | -13.21 | -0.05 |
| geometry_strong_heading_l1_width_p2 / ema | -10.85 | +14.90 | -5.82 | -32.58 | -11.66 | -26.72 | +33.87 | -26.40 | +1.45 |
| legacy / ema | -11.37 | -15.20 | -2.31 | -9.48 | -24.46 | -25.03 | -22.40 | -20.56 | +1.29 |
| warm_winner / ema | -9.28 | -11.09 | +1.65 | -22.56 | -4.89 | -4.63 | +7.21 | -22.89 | +3.06 |

6144-step EMA 最低 S 是 `geo_mse32_col20_heading_l1_shape2` 的 0.2434，旧 loss recipe 为0.2833，约下降14.1%。与旧 recipe 比，新组合 angle JSD=0.00687/0.02544、collision=1.6630%/1.9546%，nearest/lateral 也下降；但 length=0.01094/0.01045、width=0.00495/0.00432、speed=0.00461/0.00366 略差，分别约 +4.7%/+14.6%/+26.2%。on-road=78.52%/77.96%，差约+0.56个百分点；两者仍远低于GT99.4%。所以这是搜索范围内、当前seed/预算/EMA条件下的综合领先，不是全指标最优。

online 的最低 S 则为 `geometry_strong_heading_l1_width_p2` 的0.2905，联合 geometry/shape 为0.3048；后者 online speed JSD=0.01248，不比旧 recipe 的0.00354好。这个阶段说明 online 与 EMA 排名会不同，因此后续生产确认固定 G1 EMA2048 为选择终点；batch512 与独立训练seed的结果另行汇总。

当前 [long_final_stage 汇总](experiments/init_diffusion_scratch_loss_search_20261011/long_final_stage/analysis.md)只纳入7个有效目录：167个真实评价记录、37个distinct recipes；11组已有4096-step终点，4组已有6144-step终点。LR pilots、失败workers目录及不同batch512的production结果均未混入此阶段。重复budget/权重记录不等于新增独立训练seed。

以下紧凑图绘制4个候选/控制，按 online/G1 EMA 分 cohort；native angle/speed/length/width单位都是JSD×100。当前只有一个训练seed，无SD阴影。

![4个finalist/control：online真实预算、覆盖、角度与速度；native单位](experiments/init_diffusion_scratch_loss_search_20261011/long_compact_plots/summary_8bcd9b085236_native.png)

![4个finalist/control：G1 EMA真实预算、覆盖、角度与速度；native单位](experiments/init_diffusion_scratch_loss_search_20261011/long_compact_plots/summary_fd721a07c315_native.png)

![4个finalist/control：G1 EMA length/width JSD；native单位](experiments/init_diffusion_scratch_loss_search_20261011/long_compact_plots/size_fd721a07c315_native.png)

| 紧凑图标签 | 完整variant |
|---|---|
| Legacy | `legacy` |
| Old recipe | `warm_winner` |
| Geometry + head L1 | `geo_mse32_col20_heading_l1_shape2` |
| Head L1 + width p2 | `geometry_strong_heading_l1_width_p2` |

多seed汇总只用训练seed的样本标准差，生成seed在训练seed内平均；阴影不是置信区间。完整2×4分析矢量图仍保留为独立附件，正文使用10×7in、基础字号11pt的紧凑图。

## Heldout 4096 场景清单与重叠核对

[heldout 清单](experiments/init_diffusion_scratch_loss_search_20261011/heldout4096_files.json)从排序缓存 index10000 以后选择，排除全部搜索用 raw Waymo keys，并对每个 raw key 只保留一个缓存。已在 CPU 上核对文件名及存在性：4096 文件、4096 个唯一 `TFRecord+scene-index` keys；与前 1024 个搜索缓存（1023 个唯一 raw keys）没有文件名或 raw-key 重叠。JSON文件 SHA256 为 `312c2bbbe12b645eb429519db39b5ca5a6155aa58eb7f99a707072d07da04a00`；按顺序将basename逐行连接后，评价协议使用的文件列表 SHA256 为 `e8f8048f505301fde1522ad4826c2997372635daf63aaa91957d68b5fe77e7e1`。两种hash的对象不同；native pickle包含完全相同的有序文件列表。首尾缓存 index 为 10000 和 14157。

这比直接取紧接搜索集之后的缓存更严格：同一 raw scene 的不同 scene_timestep 缓存不能当作新的底层场景。当前结论是文件名规则下的无 raw-scene 重叠；没有因此证明不同场景在道路地理位置或时间上互相独立。[重叠审计](experiments/init_diffusion_scratch_loss_search_20261011/heldout_overlap_audit.json)保留选择规则与计数。此处记录数据清单审计；实际heldout生成和多seed比较另行汇总，清单准备本身不算生成评价。

### 文件名后缀与 TRAIN→评价泄漏审计

评价文件名末尾是 `scene_timestep`，不是模型的 `map_id` 类别：对 64 个独立抽取的官方原始 pickle，保存的 `scene_timestep` 都与最后后缀一致。这些原始 pickle 没有 `map_id` 字段；Scenario Dreamer 另行使用 0/1 地图类别 embedding；本次 InitDiffusion 的 `map_embedding_type=none`，没有启用这项类别 embedding。当前文件名规则为 `order_TFRecord_record-index_lg-type_scene-timestep.pt`，不能把最后数字直接解释成地图类别。[原始文件证据](experiments/init_diffusion_scratch_loss_search_20261011/filename_suffix_evidence.json)保存全部 64 个检查结果。

[训练/评价泄漏审计](experiments/init_diffusion_scratch_loss_search_20261011/training_eval_leakage_audit.json)解析所有 493,542 个训练缓存的 scenario_id 文件名前缀，对应 486,994 个唯一 scenario IDs；并实际读取全部 1024 个搜索与 4096 个 heldout 缓存的 scenario_id 和源 metadata。TRAIN↔search、TRAIN↔heldout、search↔heldout 的 scenario-ID 交集均为 0，TRAIN 两项在十六进制规范化后也为 0。全部 5120 个评价缓存 metadata 与文件名的 TFRecord、record index、lg_type、scene_timestep 一致。

紧凑 TRAIN 缓存没有保留 TFRecord/index，不能声称已经逐一按 TRAIN 原始 record key 做交集；这里的 TRAIN 无泄漏证据是被预处理源码保留的 scenario_id 命名，加上实际评价 metadata。[审计说明](experiments/init_diffusion_scratch_loss_search_20261011/training_eval_leakage_audit.md)记录这一边界。搜索与 heldout 的原始 record key 和 scenario_id 两条检查都无重叠，但没有因此声称道路地理位置或采集时间独立。

## 原 batch512 / LR5e-4：三个实际从头训练确认

冻结新增 recipe 后，旧配方与两个新候选都以随机 map encoder+G1、新 AdamW、固定 TRAIN normalizer、batch512 / LR5e-4 从头训练2048 updates；训练seed817、生成seed2817，在768/1536/2048各评价online/G1 EMA，共18条真实指标。评价仍是search1024，这一步用于选择进入heldout确认的新候选，不替代独立验证。

完整TRAIN有493,542个缓存，batch512每epoch964个batch、末批486。实际2048步跨过两个完整epoch并进入第三个，checkpoint epoch和scheduler last_epoch均为2，Adam/EMA update均为2048，实际exposure=1,048,524（两次末批各少26个样本）。[实际checkpoint审计](experiments/init_diffusion_scratch_loss_search_20261011/actual_multiepoch_checkpoint_audit.json)确认所有64个map参数张量和252个G1参数张量都改变且有非零Adam状态；同seed三个配方的初始参数hash和训练顺序相同，没有加载pretrained权重。允许多epoch的quick路径仅用于fresh、zero offset；旧的单epoch续训路径仍有单独预算保护。

### 2048-step G1 EMA，实际七指标

六项JSD仍为未缩放自然对数值；collision为%。仅一个训练seed，以下不添加SD。

| Recipe | Nearest | Lateral | Angle | Length | Width | Speed | Collision% | S | On-road% | V/GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old recipe | 0.01524 | 0.09334 | 0.02010 | 0.01138 | 0.00517 | 0.00317 | 1.50649 | 0.242189 | 79.727 | 1.0076 |
| Geometry + head L1 | 0.01972 | 0.10298 | 0.00619 | 0.01306 | 0.00787 | 0.00360 | 1.20082 | 0.237820 | 80.123 | 1.0243 |
| Head L1 + width p2 | 0.01353 | 0.10070 | 0.00595 | 0.01285 | 0.00451 | 0.00340 | 1.89906 | 0.241849 | 79.047 | 1.0267 |

原批量下三个EMA分数很接近。按预先冻结的EMA2048低S及覆盖/数量门槛，`geo_mse32_col20_heading_l1_shape2`进入后续三训练seed heldout确认：比旧配方S低1.80%，angle低69.23%、collision低20.29%；但nearest、lateral、length、width、speed分别高29.42%、10.32%、14.76%、52.37%、13.65%。因此不是全指标改善。另一个新候选S仅比旧配方低0.14%，角度/nearest/width较好但collision高26.06%，只在这个训练seed参与production确认；本轮未给它补三seed，不能声称其长期一定较差。[冻结规则及选择原始数据](experiments/init_diffusion_scratch_loss_search_20261011/production_candidate_confirmation_selection.json)完整保留这次剪枝。

online2048三个S为0.459818/0.331930/0.289632，排序与EMA不同；旧配方online length/width JSD=0.08327/0.13389，远高于其EMA=0.01138/0.00517。权重模式是实质条件，后续仍按冻结的EMA终点选择，online只作为稳定性诊断。

### 同样例曝光与后期趋势

EMA1536→2048，三组S分别下降10.04%、9.67%、8.08%，on-road分别提高1.715、1.455、1.279个百分点。所有候选还在学习，不能把2048-step的优势线性外推到128epochs。

B64×6144与B512×768都实际看到393,216个训练缓存。后三者相对前者的EMA S分别高39.63%、16.12%、25.48%。这同时改变batch、LR（1e-4→5e-4）、optimizer更新次数和更新轨迹，是生产设置迁移检查，不是单因素batch ablation；短预算B64的排序不能直接复制到B512。[完整production记录](experiments/init_diffusion_scratch_loss_search_20261011/production_final_stage/production_stage.md)保留18条评价、全部单项变化、online诊断及等曝光桥接。

![三个实际production配方：EMA预算、覆盖、角度与速度；search1024，Ntrain=1](experiments/init_diffusion_scratch_loss_search_20261011/production_compact_plots/summary_81e7d4ad67b7_native.png)

![三个实际production配方：EMA length/width；search1024，Ntrain=1](experiments/init_diffusion_scratch_loss_search_20261011/production_compact_plots/size_81e7d4ad67b7_native.png)

### 当前 EMA 的自适应平均跨度

仓库构造 `torch_ema.ExponentialMovingAverage` 时沿用 installed default `use_num_updates=true`。实际第 n 次更新的 decay 是 `min(0.9999,(1+n)/(10+n))`，约到第 89,990 次才达到 0.9999 cap；本轮早期预算处于自适应阶段，不能把 nominal decay 当成全程固定值。

| 比较点 | 实际 seen examples | 终点 effective decay | 平均参数年龄，updates | 平均参数年龄，seen examples |
|---|---:|---:|---:|---:|
| batch512，768 updates | 393216 | 0.988432 | 76.900 | 39372.8 |
| batch64，6144 updates | 393216 | 0.998538 | 614.500 | 39328.0 |
| batch512，2048 updates，实际 | 1048524 | 0.995627 | 204.900 | 104893.7 |

这是按真实 adaptive EMA 系数递推的参数年龄描述，不是通过单独EMA消融得到的收益。production训练已经完成，权重模式的实际结果在前文列出。在相同 exposure 的桥接点，平均年龄约 39,373/39,328 样本，二者非常接近；原先只按 batch 比例说平滑跨度变成 8 倍，需要用这一结果修正。G1-only EMA 的范围仍不含 shared map。配置预算128个完整epoch对应123,392个optimizer updates；EMA的0.9999上限要到约epoch93.35才首次达到，当前2048步仍在自适应阶段。因此即使batch512的早期排名稳定，也没有验证完整训练后期的EMA排名。完整推导和多 epoch 源码条件见 [fresh batch512 审计](experiments/init_diffusion_scratch_loss_search_20261011/fresh_batch512_multi_epoch_audit.json)。


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

[实际paired统计](experiments/init_diffusion_scratch_loss_search_20261011/paired_finalists/paired_summary.json)、[逐训练seed表](experiments/init_diffusion_scratch_loss_search_20261011/paired_finalists/per_training_seed.csv)、[实际阶段趋势及provenance](experiments/init_diffusion_scratch_loss_search_20261011/paired_finalists/heldout_stage_details.json)保存原始单位、seed身份、checkpoint和数据清单hash。

### 同seed后期曲线

图中误差线/带表示3个独立训练seed的样本SD，Old/New标签与上表一致。



![实际1536与2048：ema，三训练seed，JSD单位为nats](experiments/init_diffusion_scratch_loss_search_20261011/paired_finalists/compact_plots/summary_4795bc325dd2_raw.png)

![实际1536与2048：ema，三训练seed，JSD单位为nats](experiments/init_diffusion_scratch_loss_search_20261011/paired_finalists/compact_plots/size_4795bc325dd2_raw.png)


## 实际生成完整50k：固定选择后的分布审计

以下是TRAIN seed817、2048-step checkpoint、G1 EMA、GEN seed2817，各实际生成50,000个样本。evaluator的require_full_set和官方membership检查均通过，matched GT也有50,000个样本；这与只拿1024生成结果对比50k GT reference不同。两组使用参考完整lane地图，生成agents；不是生成新地图。

这50,000个缓存对应43,601个唯一TFRecord+record-index keys，含6,399个额外timestep/variant；它包含此前search和heldout，不是未参与选择的独立test。此阶段只审计覆盖和分布，**没有重新选择配方**。只有一个训练seed、一个生成seed，不添加独立训练SD。

| 指标 | Old recipe | New recipe | New/Old−1 |
|---|---:|---:|---:|
| Nearest JSD | 0.012701 | 0.017303 | +36.23% |
| Lateral JSD | 0.089830 | 0.094598 | +5.31% |
| Angle JSD | 0.015819 | 0.004074 | -74.24% |
| Length JSD | 0.007539 | 0.009481 | +25.77% |
| Width JSD | 0.004167 | 0.007164 | +71.90% |
| Speed JSD | 0.001684 | 0.002245 | +33.27% |
| Collision (%) | 1.634614 | 1.512695 | -7.46% |
| 综合S | 0.226332 | 0.224097 | -0.99% |
| On-road (%) | 80.030 | 79.913 | -0.116个百分点 |
| 生成vehicles | 368,772 | 373,704 | +1.34% |

matched GT共有362,309vehicles，360,099on-road（99.390%）。完整feature counts、源文件hash、模型设置和运行精度均在各组`sd_agent_metrics.json`保存；本报告JSD除回原始自然对数单位。

![实际50,000缓存样本：相同GT的speed、length、width直方图及车辆覆盖](experiments/init_diffusion_scratch_loss_search_20261011/full50000_distributions/vehicle_distributions_and_coverage.png)

### 分布差异：相似均值不等于相似直方图

下表只使用保存的官方clipped histogram和bin中点估算均值/std，不是原始物理样本moments。size图只是缩放坐标范围，没有对显示窗口重新归一化。

| 特征 | Matched GT均值 / SD | Old recipe均值 / SD | New recipe均值 / SD |
|---|---:|---:|---:|
| Speed (m/s) | 5.328 / 5.963 | 4.893 / 5.499 | 4.645 / 5.416 |
| Length (m) | 4.885 / 0.753 | 4.831 / 0.651 | 4.835 / 0.637 |
| Width (m) | 2.163 / 0.189 | 2.144 / 0.174 | 2.147 / 0.166 |

官方speed第一桶为[0,1)m/s：GT/Old/New占比分别为41.70%/42.73%/45.42%，它不能给出精确stationary比例。重建MSE/L1、shared梯度取舍和全类别TRAIN尺度是分布差异的候选解释；未做分量梯度或尾部监督消融，不能给出单一根因。当前linear尺寸没有输出端正值约束，指标又会clip边界；本轮没有保存负尺寸计数，不用第一桶推断负尺寸数量。

## 推荐的完整从头训练配方

冻结选择为`geo_mse32_col20_heading_l1_shape2`；全50k和online诊断不改变该决定。当前网络、条件、表示和x0 objective保持一致，只把下面resolved分量loss及已确认优化器设置写入新实验配置。

| 分量 | kind | time power | weight | scale |
|---|---|---:|---:|---|
| Position | mse | 1 | 32 | data |
| Heading：圆周最短角差 | l1 | 1 | 1 | none |
| Length | mse | 1 | 2 | data |
| Width | mse | 1 | 2 | data |
| Scalar speed | mse | 1 | 2 | data |
| Non-ego type | ce | 0 | 1 | none |
| Collision | l1 | 1 | 20 | none |
| 额外speed magnitude | mse | 0 | 0 | data |

`normalize_time_weights=true`，时间权重为max(t,0.05)^−power再按uniform t的解析期望归一化；`std_floor=0.1`，data尺度来自本次TRAIN-only normalizer。type沿用separate CE，额外speed magnitude关闭；不改变噪声来源、Hungarian算法或圆周x0采样目标。

完整TRAIN493,542缓存，batch512，单GPU，32-true，matmul medium，AdamW LR5e-4/betas(0.9,0.999)/eps1e-8/weight_decay0.01，clip0.5，accumulate1。沿用原生epoch cosine、128epochs、warmup0、最低LR比例0.05；G1 adaptive EMA cap0.9999，shared map使用online权重。normalizer初始化读取固定TRAIN JSON，fresh fit的ckpt_path=null。

### 怎样解释对最终训练效果的预测

本轮按冻结终点选择`geo_mse32_col20_heading_l1_shape2`作为完整训练起点。三seed的1536→2048曲线同时提醒：Old的S下降0.0262，New下降0.0161，两者差距正在缩小；Old的角度改善更快，New的尺寸/速度改善更快。不能用当前终点排序推断所有分量最终收敛顺序。该推荐只覆盖本次条件与S权重；若目标是单独最小speed/shape JSD，综合赢家并不是目前对应指标的赢家。

128epoch相当于123,392次optimizer更新，quick确认只有2048次；adaptive EMA到约epoch93.35才达到cap，且原生full fit每epoch重新shuffle，quick多epoch重复自己的固定seeded顺序。早期loss收益、覆盖继续学习、online/EMA排序差异、采样误差和晚期平滑都会限制外推。完整训练仍需看每4epochs的同4096场景曲线，不能用quick终点作为停止已收敛的证据。

## 原生训练、评价与实际验证

已生成[完整从头训练配置](../configs/experiment/init_diffusion_lane_conditioned_scratch_best.yaml)和[全50k评价配置](../configs/experiment/init_diffusion_lane_conditioned_scratch_best_eval.yaml)。仓库默认experiment未改；运行下列命令启动128epoch随机初始化训练：

```bash
python src/run.py experiment=init_diffusion_lane_conditioned_scratch_best
```

训练每4epochs评价全部heldout4096；每4epochs保留历史checkpoint，last.ckpt每epoch更新，包含optimizer和EMA。评价必须显式传入已训练checkpoint：

```bash
python src/run.py experiment=init_diffusion_lane_conditioned_scratch_best_eval \
  ckpt_path=/absolute/path/to/your/trained/last.ckpt
```

评价使用全50k缓存、matched GT、membership检查和checkpoint的normalizer/EMA。配置未提供checkpoint时会明确拒绝解析，避免无意评价随机模型。`src/run.py`只在模型构造前统一seed，而quick每次生成前额外reset sampler seed；相同数字seed不保证两种入口位级相同采样。完整checkpoint路径和quick复核命令可从各阶段protocol与summary还原。WandB实例化当前在run.py被注释，本次没有启用外部日志。

[native CPU审计](experiments/init_diffusion_scratch_loss_search_20261011/actual_native_best_cpu_audit.json)已实际验证Hydra配置、493,542训练/4096验证/50,000测试文件范围、trainability、optimizer参数覆盖、callback保留策略、严格checkpoint加载、loss及normalizer一致，并验证缺失checkpoint拒绝和CLI覆盖。[native GPU smoke](experiments/init_diffusion_scratch_loss_search_20261011/actual_native_best_gpu_smoke.json)通过实际`src/run.py`从头跑1次optimizer更新，map/G1参数均改变，Adam/EMA状态有限、EMA update=1，selected loss和TRAIN统计一致。还实际运行了native test的一批128个matched场景，恢复所选checkpoint的2048-step G1 EMA，七项生成指标均有限。两项smoke不计入搜索模型或主比较生成指标。

固定normalizer路径为可选配置，默认null保持旧lazy行为；显式JSON仅初始化buffer，checkpoint buffer加载优先。已运行38个相关normalizer/component CPU tests、14个paired-summary tests、6个inventory tests；这些是工具与路径验证，不代表所有仓库测试或128epoch训练已通过。

## 搜索成本、失败记录与可复核文件

主比较完成94,208次optimizer updates、12,451,476次实际缓存样本曝光。按(stage,variant)去重的已知fit wall time为202.18min，唯一report的已知评价wall time为35.42min；不是GPU busy time或完整端到端成本，未加入构造、预处理、CPU图表/PDF及未知失败时间。LR/实现pilots和native smoke另列，不混入235条主比较记录。

seed1817第一次fresh训练发生native segfault；内核记录有零地址故障，但没有Python traceback，根因未确认。保留的CSV有400条已完成update日志；其失败模型/评价不计入成功结果。重启仍从随机权重开始、未resume，初始化hash和完整训练顺序与失败尝试相同，第二次两配方均健康完成；本轮没有因此改包版本或声称找到了环境根因。workers4早期失败和另一个失败pilot也在inventory单列。[失败与恢复证据](experiments/init_diffusion_scratch_loss_search_20261011/seed1817_native_failure_audit.json)可复核。

| 记录 | 文件 |
|---|---|
| 搜索阶段、实际计数、成本与来源hash | [final_search_facts.json](experiments/init_diffusion_scratch_loss_search_20261011/final_search_facts.json) |
| 全部实际指标 / CSV / 分组趋势 | [final_all_stage/analysis.md](experiments/init_diffusion_scratch_loss_search_20261011/final_all_stage/analysis.md) |
| 逐次评价observations | [final_all_stage/observations.csv](experiments/init_diffusion_scratch_loss_search_20261011/final_all_stage/observations.csv) |
| 冻结最终选择协议 | [final_deployment_selection_protocol.json](experiments/init_diffusion_scratch_loss_search_20261011/final_deployment_selection_protocol.json) |
| 配对三seed统计与选择 | [paired_finalists/paired_summary.json](experiments/init_diffusion_scratch_loss_search_20261011/paired_finalists/paired_summary.json) |
| 同seed后期曲线、online诊断 | [paired_finalists/heldout_stage_zh.md](experiments/init_diffusion_scratch_loss_search_20261011/paired_finalists/heldout_stage_zh.md) |
| 独立种子随机初始化/optimizer审计 | [actual_independent_seed_checkpoint_audit.json](experiments/init_diffusion_scratch_loss_search_20261011/actual_independent_seed_checkpoint_audit.json) |
| 生产checkpoint跨epoch审计 | [actual_multiepoch_checkpoint_audit.json](experiments/init_diffusion_scratch_loss_search_20261011/actual_multiepoch_checkpoint_audit.json) |
| TRAIN-only normalizer | [train_normalizer.json](experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json) |
| heldout数据清单及重叠审计 | [heldout_overlap_audit.json](experiments/init_diffusion_scratch_loss_search_20261011/heldout_overlap_audit.json) |
| TRAIN/评价scenario-ID泄漏检查 | [training_eval_leakage_audit.json](experiments/init_diffusion_scratch_loss_search_20261011/training_eval_leakage_audit.json) |
| 全50k直方图近似统计 | [full50000_distributions/distribution_statistics.json](experiments/init_diffusion_scratch_loss_search_20261011/full50000_distributions/distribution_statistics.json) |
| 实际原生配置生成来源 | [native_best_materialization.json](experiments/init_diffusion_scratch_loss_search_20261011/native_best_materialization.json) |
