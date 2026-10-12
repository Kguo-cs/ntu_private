# InitDiffusion 固定地图夜间搜索分析

生成时间（上海/北京时间）：2026-10-12 09:40:14 UTC+8；UTC：2026-10-12T01:40:14.662093+00:00。

本轮授权截止：2026-10-12 10:00 UTC+8（02:00 UTC）；GPU 队列 cutoff：09:35 UTC+8（01:35 UTC）。恢复没有改变截止时间。目录名20261011表示开始时的UTC日期。

实际最终训练recipe选择：**separate_speed**；完整 2 arms ×2 TRAIN ×2 GEN 的 8 个 endpoint 已核验；R=0.8975，R_attr=0.8269。训练recipe比较均固定20步推断 / t_eps=.05；后续40步与hybrid是另行预注册的推断选择。

选中 recipe：B256，effective LR=0.0005；loss 相对 control：length.time_power=2; velocity.weight=4.0; width.time_power=2；模型/表示相对 control：ego_context_heading_encoding="sincos"; speed_head_kind="separate"。完整分量配置在后文保留。

每个模型实际训练 5,242,620 TRAIN cache exposures（10.6224 等价全数据 epoch）；map encoder 来自固定 AIRL32 source，G1 从头训练。此预算结论不等于 128 epoch 的最终收敛效果。

这是完整loss recipe、模型架构与batch设置组合的相等曝光对比，不能把指标变化单独归因于独立speed head。各分量、batch和模型改动在后文完整列出。

| final arm | pooled R | R_attr | speed变化% | collision变化% | pooled / all TRAIN gates |
| --- | --- | --- | --- | --- | --- |
| control | 1.0000 | 1.0000 | 0.00 | 0.00 | PASS / PASS |
| separate_speed | 0.8975 | 0.8269 | 2.41 | -9.03 | PASS / PASS |

| final metric | control raw | challenger raw | 变化% |
| --- | --- | --- | --- |
| Near | 0.009635 | 0.009104 | -5.51 |
| Lat | 0.084949 | 0.083148 | -2.12 |
| Ang | 0.002805 | 0.002718 | -3.13 |
| Length | 0.006134 | 0.004596 | -25.07 |
| Width | 0.002612 | 0.001848 | -29.27 |
| Speed | 0.000997 | 0.001021 | 2.41 |
| Col% | 0.432765 | 0.393698 | -9.03 |

此表为 matched4096 的完整两 TRAIN 配对均值，先平均每 TRAIN 的两 GEN；六项 raw JSD 无量纲、Col% 为 vehicle-weighted 百分比，均较小为好。helper 原生值为 Near/Lat ×10、Ang/Length/Width/Speed ×100、Col% 不缩放。负变化代表改善，整体 R 不能替代逐项取舍。

| challenger TRAIN | 角色 | R | speed变化% | collision变化% | onroad% | gates |
| --- | --- | --- | --- | --- | --- | --- |
| 4817 | adaptive | 0.9453 | 9.66 | 22.23 | 86.27 | PASS |
| 5817 | independent confirmation | 0.8612 | -5.25 | -32.78 | 86.09 | PASS |

已完成full50k scope audit：仅TRAIN4817×两GEN（NTRAIN=1），R=0.9174；L/W变化-30.71% / -32.51%，speed-0.30%，collision+20.09%。full速度变化幅度很小，基本相当，不能从单次数值变化声称稳定改善；full碰撞退步与上表两TRAIN pooled变化须分别保留。full没有新增独立TRAIN，也未重新选模型。

单独的native-to-native推断cohort已完成16cells并冻结n40_eps005：同已选权重、四个fresh N20基线，R=0.9472；L/W变化-15.11% / -40.12%，speed-5.16%，collision+10.39%。这是推断设置的取舍，不改变上面的训练recipe选择，也不能与main历史N20混合算ratio。40步的model calls为2倍；完整指标、时间和小eps选项在独立附录列出。

另行预注册8cell推断cohort已冻结 **hybrid_pos0001**：相对本次四个fresh N40 anchors，R=0.9364，Near/Lat/Ang变化-13.90% / -1.61% / -4.66%，collision-24.43%；L/W/speed变化-0.108% / +0.163% / +0.064%幅度接近数值重放波动，应视为基本相当。仅最后一步position保留直接x0投影，其他特征仍用原40步路径。两TRAIN均复用现有权重，不新增独立训练；这些ratio不能与前16cell或main搜索ratio相乘。full状态与可用配置在单独附录按实际证据列出。

当前建议采用已选训练recipe **separate_speed** 与单独验证的 **hybrid40推断**；训练配置：[init_diffusion_lane_conditioned_overnight_best.yaml](<../configs/experiment/init_diffusion_lane_conditioned_overnight_best.yaml>)；hybrid评价配置：[init_diffusion_lane_conditioned_overnight_best_hybrid_sampler_eval.yaml](<../configs/experiment/init_diffusion_lane_conditioned_overnight_best_hybrid_sampler_eval.yaml>)。两GEN full50k实际完成与两GEN CPU compose/source审计均已通过。原main20训练recipe比较、新native16和hybrid8各自保留自己的基线，不合成跨cohort改善比例。下一次128epoch仍从固定预训练map、fresh G1开始，是后续任务；本次没有完成128epoch。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC=1
python tools/run_init_diffusion_safe.py \
  experiment=init_diffusion_lane_conditioned_overnight_best
python tools/run_init_diffusion_safe.py \
  experiment=init_diffusion_lane_conditioned_overnight_best_hybrid_sampler_eval
```

评价命令恢复上述实际selected checkpoint的完整模型/EMA/buffers；其预算结果和未来128epoch收敛效果须分别理解。

评价配置已默认指向本轮选中TRAIN4817的10.6224epoch预算checkpoint；完整保存路径和SHA留在来源证据中，不是未来128epoch的权重。

完整权重/EMA/buffers的CPU恢复已核验，但原N20 TEST重放没有通过1e-6数值门槛；下列结果是已保存的真实评价，native命令不承诺重放逐指标相同。失败与新native推断cohort在附录明确分开。

本轮只计 27 个已核验完成 stage、76 个实际预算/TRAIN/GEN/EMA observation。CPU 预检、未完成 checkpoint、失败尝试与计划候选不计入指标。

这次预算内结果不证明 128 epoch 的最优训练效果；128 epoch 的 native fit 是后续建议，当前没有完成该训练。

## 实际完成与缺失状态

队列文件状态：`COMPLETE`；活动项：`COMPLETE`。文件声明与实际完成核验分别列出。

实际 runtime recovery 已启动，ledger 为 [protocol.json](<experiments/init_diffusion_overnight_search_20261011/runtime_recovery/protocol.json>)。继承已完整完成的 long_seed4817_control, long_seed4817_separate_speed；余下模型按相同 recipe/seed/数据次序/预算从 exposure=0 训练，未恢复失败 checkpoint。原 18 个 frozen source 文件保持一致；新增 recovery entrypoint、loader wrapper 和 helper 有独立 SHA，原18-file freeze 并不覆盖这些新文件。执行策略已从 manual youngGC128 改为 cached public pickle_module、automatic/manual cyclic GC 均禁用、young interval0；继承 control 的运行策略不同，不能宣称相同 runtime cohort 或把差异归因于 model loss/math 变化。

实际 fresh retry `long_seed4817_separate_speed` 的 16 项初始化元数据与 archived attempt 一致，包括 G1/map hash、seed/B/LR/clip、normalizer 和 TRAIN/eval order；offset=0、optimizer 不延续。此初始化一致性不保证随后 GPU 轨迹逐 bit 相同；初始化记录本身不作为评价证据；已列成绩来自完整stage核验。

此前 control 尝试在最后已记录 update=256 后被主动终止（returncode=143），用于统一最后 deadline forecast 的已加载 controller 版本，并把全部 18 arms fresh restart。归档是有意中止的运行诊断，不计为候选模型失败、完成训练量或成绩；来源见机器审计。

| screen arm | stage 状态 | 纳入结果 | 原因 |
| --- | --- | --- | --- |
| control | success | 是 | — |
| combo_reference | success | 是 | — |
| speed_p0 | success | 是 | — |
| speed_l1_p0 | success | 是 | — |
| train_features | success | 是 | — |
| separate_speed | success | 是 | — |
| detached_speed | success | 是 | — |
| softplus_speed | success | 是 | — |
| residual_speed | success | 是 | — |
| high_noise | success | 是 | — |
| match_speed4 | success | 是 | — |
| transport_speed | success | 是 | — |
| transport_attributes | success | 是 | — |
| depth5 | success | 是 | — |
| width384 | success | 是 | — |
| normalized_speed_p0 | success | 是 | — |
| lane_center_context | success | 是 | — |
| lane_center_normalized | success | 是 | — |

| stage | 状态 | 纳入结果 | 原因 |
| --- | --- | --- | --- |
| full50000_seed4817_control | success | 是 | — |
| full50000_seed4817_separate_speed | success | 是 | — |
| long_seed4817_control | success | 是 | — |
| long_seed4817_separate_speed | success | 是 | — |
| long_seed5817_control | success | 是 | — |
| long_seed5817_separate_speed | success | 是 | — |
| medium_control | success | 是 | — |
| medium_lane_center_context | success | 是 | — |
| medium_separate_speed | success | 是 | — |

最终预算 endpoint readiness：8/8 个已核验 cell。单个已完成 cell 不能替代两 TRAIN 配对和 final selection 核验。

| final arm | TRAIN | GEN2817 | GEN3817 | stage 状态 |
| --- | --- | --- | --- | --- |
| control | 4817 | 已核验 | 已核验 | success |
| separate_speed | 4817 | 已核验 | 已核验 | success |
| control | 5817 | 已核验 | 已核验 | success |
| separate_speed | 5817 | 已核验 | 已核验 | success |

## 数据、训练来源和评价条件

TRAIN 使用 `src/waymo_data/full/training_map2_sd` 的 493,542 场景，全 iterator 按 seeded full order 连续遍历、每 epoch 重复该完整次序，保留最后的不足 batch。各 promotion 都 fresh G1/optimizer/scheduler/EMA，从 exposure=0 重新训练。

仅加载 AIRL32 checkpoint 的预训练 map encoder 并永久冻结；不恢复它的 G1、optimizer 或 EMA。训练-only 8192 场景 normalizer 固定。screen 为确认 cohort 前 1024 场景；medium/long 为同一 matched4096；full50000 为整个 scenario_dreamer_val 的 scope audit，不再择优。

本报告重新核对 native filename corpus 与已有 overlap audit：full50000 的单位是 **50,000 个 cache entries**，对应 43,601 个 unique `(source_tfrecord, record_index)` raw keys；matched4096 是 4,096 entries / 4,096 unique raw keys。medium/long 反复使用同一 4096 selection，full50k 包含之前 aggregate audit 已见条目；这些 scope 不能称为 unseen independent test，GEN repeats、不同 cache timestep 也不能当独立原场景。

source map checkpoint：[AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt](<../src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt>)；实际文件 SHA256 `610c1f94a0f573fd078e251af37c38c73255b969e52154e66546588996ef99fb`。

TRAIN normalizer：[train_normalizer.json](<experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json>)；实际文件 SHA256 `adcc5e5c232b018919b269ec06b7e8b065a49fd9f0219f3d0f432f7caf4cea23`。

实际报告中的条件：`['reference map', 'GT ego reference frame', 'GT ego position', 'GT ego heading', 'GT ego shape', 'GT ego type', 'input total agent count', 'ego/non-ego slot']`。

对应固定 ego 的 L1=0/type accuracy=1 是施加的条件，不能当作 ego 生成精度。非 ego GT type 是训练标签；推理使用生成 type，所有 vehicle metrics 都由生成 type0 的 population 决定。

精度保持 32-true、float32 matmul medium、EMA 评价、20 步采样；具体每个实际 group 的 cohort/source fingerprint 留在机器审计中。不存在通过调整 evaluator 或筛掉 offroad 生成 vehicle 来改善主指标。

## 七个指标、单位和选择规则

| helper 原名 | 表中短名 | 原生值 / raw 倍数 | population / 含义 |
| --- | --- | --- | --- |
| nearest_dist_jsd | Near | 原生/raw=10 | 全部 vehicle 的最近 center 距离（singleton 无贡献） |
| lat_dev_jsd | Lat | 原生/raw=10 | onroad vehicle 的 unsigned centerline 距离 |
| ang_dev_jsd | Ang | 原生/raw=100 | onroad vehicle 的相对 lane heading（degree histogram） |
| length_jsd | Length | 原生/raw=100 | 全部生成 vehicle 的物理属性 histogram |
| width_jsd | Width | 原生/raw=100 | 全部生成 vehicle 的物理属性 histogram |
| speed_jsd | Speed | 原生/raw=100 | 全部生成 vehicle 的物理属性 histogram |
| collision_rate | Col% | 保持百分比，不除 100 | 碰撞 vehicle 数 / 全部 vehicle 数 ×100 |

JSD 是 SciPy Jensen–Shannon distance 的平方（自然对数、无量纲），不是米或 m/s。官方 onroad 判定为到 resampled compact centerline points 距离 ≤1.5 m；lat/angle 仅比较 onroad 子集，offroad vehicle 仍进入 speed/shape/collision。更低 lat JSD 不能替代道路覆盖检查。

matched4096 与 full50000 使用相同 GT 匹配算法和固定 histogram bins，但 GT population 的范围不同，不能混合排名。局部生成报告的 full_reference_agent_metrics 另以全50k GT 对比生成子集，只作辅助诊断，不替代 matched 主指标；全量 matched scope 才与 full-real scope 一致。

先在每个 TRAIN 模型内平均配对 GEN2817/3817，再对 TRAIN 模型等权平均；screen 计划仅 GEN2817。七个 candidate/control arm 均值比值的算术平均为 R，Length/Width/Speed 三项比值平均为 R_attr。R 是 dimensionless ratio-of-arm-means，不是各场景 JSD 的平均、平均逐种子 R 或物理误差。raw/native 固定缩放在比值中抵消；collision 保持按 vehicle 总数加权的百分比。

预定 gates：Length、Width、Speed 每项 ≤control×1.15；generated onroad fraction 不比 control 低超过 0.02；生成 vehicle 数/匹配 GT 数在 [0.9,1.1]。最终 pooled 和每个 TRAIN 都需通过 gates，且 challenger pooled R<1 才替换 control。这里允许单指标最多 15% 退化，须逐项报告。

TRAIN4817 用于 adaptive screen/medium/long 选择；TRAIN5817 是冻结 recipe 后独立确认。两 GEN 不增加独立训练 N，full50000 也不增加 TRAIN N。N=1 不报告 SD；N=2 的 sample SD 仅描述两个训练模型，不能证明稳健最优，也不能把 4817 当未参与选择的 holdout。

## 实际指标与每阶段比较

### screen / ema / exposure 196,608

实际完整 TRAIN 等价 epoch：0.3984；评价 1024 场景；cohort `a1962170efdc`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1 | 0.025782 | 0.092723 | 0.041753 | 0.017268 | 0.012923 | 0.013390 | 2.644398 |
| combo_reference | 1 | 0.024448 | 0.084907 | 0.025021 | 0.018285 | 0.016828 | 0.004587 | 2.124092 |
| depth5 | 1 | 0.024282 | 0.093108 | 0.023186 | 0.011684 | 0.006315 | 0.003765 | 2.231405 |
| detached_speed | 1 | 0.022378 | 0.088620 | 0.022847 | 0.009627 | 0.006982 | 0.022766 | 1.879232 |
| high_noise | 1 | 0.031783 | 0.090170 | 0.009835 | 0.029926 | 0.021416 | 0.007746 | 1.504694 |
| lane_center_context | 1 | 0.023949 | 0.087324 | 0.016392 | 0.010001 | 0.006859 | 0.004475 | 2.402651 |
| lane_center_normalized | 1 | 0.024457 | 0.090189 | 0.014523 | 0.010758 | 0.006353 | 0.004010 | 2.389885 |
| match_speed4 | 1 | 0.020938 | 0.088340 | 0.024881 | 0.009656 | 0.003605 | 0.004365 | 2.620269 |
| normalized_speed_p0 | 1 | 0.024714 | 0.089217 | 0.017937 | 0.010372 | 0.004134 | 0.012198 | 2.099370 |
| residual_speed | 1 | 0.022329 | 0.089557 | 0.024023 | 0.009831 | 0.007478 | 0.023360 | 2.405641 |
| separate_speed | 1 | 0.024365 | 0.089129 | 0.021338 | 0.011764 | 0.007817 | 0.004931 | 1.741534 |
| softplus_speed | 1 | 0.024403 | 0.090184 | 0.021091 | 0.011584 | 0.006377 | 0.004222 | 1.857283 |
| speed_l1_p0 | 1 | 0.024597 | 0.088807 | 0.022217 | 0.010379 | 0.005901 | 0.012778 | 2.665010 |
| speed_p0 | 1 | 0.024120 | 0.086285 | 0.022233 | 0.011273 | 0.009307 | 0.011353 | 2.333333 |
| train_features | 1 | 0.024445 | 0.088358 | 0.015525 | 0.010565 | 0.004715 | 0.003850 | 1.996370 |
| transport_attributes | 1 | 0.044892 | 0.073280 | 0.051058 | 0.013188 | 0.011659 | 0.008848 | 2.401406 |
| transport_speed | 1 | 0.023998 | 0.089382 | 0.024874 | 0.010353 | 0.006643 | 0.003823 | 2.497907 |
| width384 | 1 | 0.028295 | 0.092289 | 0.030583 | 0.013582 | 0.008960 | 0.004187 | 2.434663 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/384upd | 4817 | 1.0000 | 1.0000 | 74.52 | 1.0193 | PASS |
| combo_reference | B256/768upd | 4817 | 0.8529 | 0.9012 | 75.99 | 1.0152 | FAIL |
| depth5 | B256/768upd | 4817 | 0.6845 | 0.4822 | 76.49 | 1.0299 | PASS |
| detached_speed | B256/768upd | 4817 | 0.8399 | 0.9327 | 75.85 | 1.0267 | FAIL |
| high_noise | B256/768upd | 4817 | 0.9969 | 1.3229 | 78.74 | 1.0277 | FAIL |
| lane_center_context | B256/768upd | 4817 | 0.6594 | 0.4814 | 77.09 | 1.0274 | PASS |
| lane_center_normalized | B256/768upd | 4817 | 0.6553 | 0.4714 | 77.24 | 1.0210 | PASS |
| match_speed4 | B256/768upd | 4817 | 0.6451 | 0.3881 | 76.31 | 1.0233 | PASS |
| normalized_speed_p0 | B256/768upd | 4817 | 0.7108 | 0.6105 | 76.33 | 1.0136 | PASS |
| residual_speed | B256/768upd | 4817 | 0.8871 | 0.9642 | 75.83 | 1.0261 | FAIL |
| separate_speed | B256/768upd | 4817 | 0.6758 | 0.5515 | 76.17 | 1.0264 | PASS |
| softplus_speed | B256/768upd | 4817 | 0.6580 | 0.4932 | 75.81 | 1.0159 | PASS |
| speed_l1_p0 | B256/768upd | 4817 | 0.7805 | 0.6707 | 76.11 | 1.0274 | PASS |
| speed_p0 | B256/768upd | 4817 | 0.7860 | 0.7403 | 76.19 | 1.0214 | PASS |
| train_features | B256/768upd | 4817 | 0.6132 | 0.4214 | 76.66 | 1.0162 | PASS |
| transport_attributes | B256/768upd | 4817 | 0.9985 | 0.7756 | 74.46 | 0.7266 | FAIL |
| transport_speed | B256/768upd | 4817 | 0.6906 | 0.4664 | 76.37 | 1.0166 | PASS |
| width384 | B256/768upd | 4817 | 0.7912 | 0.5975 | 75.31 | 1.0314 | PASS |

### screen / ema / exposure 393,216

实际完整 TRAIN 等价 epoch：0.7967；评价 1024 场景；cohort `a1962170efdc`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1 | 0.024517 | 0.095325 | 0.013403 | 0.020734 | 0.015578 | 0.005364 | 1.738526 |
| combo_reference | 1 | 0.023547 | 0.090599 | 0.010960 | 0.012221 | 0.006468 | 0.002739 | 1.504597 |
| depth5 | 1 | 0.022864 | 0.091084 | 0.008824 | 0.013846 | 0.007262 | 0.003086 | 1.479126 |
| detached_speed | 1 | 0.023276 | 0.090867 | 0.010054 | 0.009729 | 0.004686 | 0.011439 | 1.270894 |
| high_noise | 1 | 0.036571 | 0.088013 | 0.007439 | 0.024694 | 0.018017 | 0.004365 | 1.791127 |
| lane_center_context | 1 | 0.020833 | 0.088755 | 0.008819 | 0.010512 | 0.004903 | 0.002234 | 1.495431 |
| lane_center_normalized | 1 | 0.024051 | 0.089755 | 0.008235 | 0.009587 | 0.004250 | 0.002883 | 1.875260 |
| match_speed4 | 1 | 0.024211 | 0.089367 | 0.011157 | 0.013166 | 0.007752 | 0.002323 | 1.691156 |
| normalized_speed_p0 | 1 | 0.024346 | 0.088545 | 0.010306 | 0.010248 | 0.003451 | 0.005390 | 1.246848 |
| residual_speed | 1 | 0.026146 | 0.091562 | 0.012977 | 0.010445 | 0.003734 | 0.013434 | 1.548458 |
| separate_speed | 1 | 0.022819 | 0.087484 | 0.010994 | 0.010290 | 0.004734 | 0.002185 | 1.136679 |
| softplus_speed | 1 | 0.022632 | 0.090519 | 0.010902 | 0.011083 | 0.007867 | 0.002564 | 1.797909 |
| speed_l1_p0 | 1 | 0.024427 | 0.091579 | 0.012136 | 0.011373 | 0.008339 | 0.011794 | 1.807061 |
| speed_p0 | 1 | 0.021636 | 0.090706 | 0.012042 | 0.010948 | 0.004308 | 0.006174 | 1.598332 |
| train_features | 1 | 0.024240 | 0.089130 | 0.009686 | 0.009368 | 0.003930 | 0.002599 | 1.403903 |
| transport_attributes | 1 | 0.038830 | 0.078698 | 0.021800 | 0.012438 | 0.006423 | 0.005741 | 1.508325 |
| transport_speed | 1 | 0.022528 | 0.088040 | 0.011357 | 0.010122 | 0.005463 | 0.003466 | 1.475912 |
| width384 | 1 | 0.025656 | 0.092104 | 0.019263 | 0.017250 | 0.012862 | 0.002399 | 2.328482 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/768upd | 4817 | 1.0000 | 1.0000 | 77.51 | 1.0200 | PASS |
| combo_reference | B256/1536upd | 4817 | 0.7299 | 0.5051 | 78.25 | 1.0183 | PASS |
| depth5 | B256/1536upd | 4817 | 0.7295 | 0.5697 | 78.09 | 1.0262 | PASS |
| detached_speed | B256/1536upd | 4817 | 0.8980 | 0.9675 | 77.64 | 1.0270 | FAIL |
| high_noise | B256/1536upd | 4817 | 1.0231 | 1.0538 | 79.64 | 1.0296 | FAIL |
| lane_center_context | B256/1536upd | 4817 | 0.6482 | 0.4127 | 79.08 | 1.0245 | PASS |
| lane_center_normalized | B256/1536upd | 4817 | 0.6983 | 0.4242 | 78.34 | 1.0213 | PASS |
| match_speed4 | B256/1536upd | 4817 | 0.7565 | 0.5219 | 77.96 | 1.0234 | PASS |
| normalized_speed_p0 | B256/1536upd | 4817 | 0.7327 | 0.5735 | 78.05 | 1.0126 | PASS |
| residual_speed | B256/1536upd | 4817 | 1.0191 | 1.0826 | 77.16 | 1.0261 | FAIL |
| separate_speed | B256/1536upd | 4817 | 0.6471 | 0.4025 | 77.72 | 1.0234 | PASS |
| softplus_speed | B256/1536upd | 4817 | 0.7483 | 0.5058 | 78.05 | 1.0179 | PASS |
| speed_l1_p0 | B256/1536upd | 4817 | 1.0263 | 1.0942 | 78.01 | 1.0206 | FAIL |
| speed_p0 | B256/1536upd | 4817 | 0.8011 | 0.6518 | 77.54 | 1.0207 | FAIL |
| train_features | B256/1536upd | 4817 | 0.6632 | 0.3962 | 78.60 | 1.0105 | PASS |
| transport_attributes | B256/1536upd | 4817 | 0.9980 | 0.6941 | 77.67 | 0.7242 | FAIL |
| transport_speed | B256/1536upd | 4817 | 0.7177 | 0.4950 | 77.37 | 1.0189 | PASS |
| width384 | B256/1536upd | 4817 | 0.9849 | 0.7016 | 77.16 | 1.0235 | PASS |

### medium / ema / exposure 1,048,524

实际完整 TRAIN 等价 epoch：2.1245；评价 4096 场景；cohort `98709b04289b`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1 | 0.015527 | 0.092387 | 0.004679 | 0.011012 | 0.010336 | 0.001572 | 1.550363 |
| lane_center_context | 1 | 0.016730 | 0.090705 | 0.003205 | 0.005939 | 0.004531 | 0.001029 | 1.757351 |
| separate_speed | 1 | 0.016746 | 0.092391 | 0.004355 | 0.005200 | 0.003527 | 0.001178 | 1.614978 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/2048upd | 4817 | 1.0000 | 1.0000 | 80.64 | 1.0208 | PASS |
| lane_center_context | B256/4096upd | 4817 | 0.7871 | 0.5441 | 81.56 | 1.0212 | PASS |
| separate_speed | B256/4096upd | 4817 | 0.8020 | 0.5210 | 80.84 | 1.0258 | PASS |

### medium / ema / exposure 2,097,048

实际完整 TRAIN 等价 epoch：4.2490；评价 4096 场景；cohort `98709b04289b`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1 | 0.013829 | 0.091379 | 0.003241 | 0.007786 | 0.005396 | 0.001180 | 0.870541 |
| lane_center_context | 1 | 0.013825 | 0.088274 | 0.003153 | 0.004857 | 0.003207 | 0.001085 | 1.176068 |
| separate_speed | 1 | 0.013608 | 0.090469 | 0.003237 | 0.005531 | 0.002466 | 0.001222 | 1.041438 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/4096upd | 4817 | 1.0000 | 1.0000 | 83.04 | 1.0211 | PASS |
| lane_center_context | B256/8192upd | 4817 | 0.9182 | 0.7126 | 83.81 | 1.0250 | PASS |
| separate_speed | B256/8192upd | 4817 | 0.9103 | 0.7343 | 82.97 | 1.0250 | PASS |

### long / ema / exposure 1,747,378

实际完整 TRAIN 等价 epoch：3.5405；评价 4096 场景；cohort `98709b04289b`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 2 | 0.014697 ± 0.000548 | 0.091403 ± 0.000450 | 0.003627 ± 0.000195 | 0.007853 ± 0.000949 | 0.005748 ± 0.001035 | 0.001346 ± 0.000152 | 1.221172 ± 0.086607 |
| separate_speed | 2 | 0.015021 ± 0.000701 | 0.091515 ± 0.000099 | 0.003502 ± 0.000120 | 0.006313 ± 0.000119 | 0.003511 ± 0.000156 | 0.001255 ± 0.000189 | 1.272189 ± 0.131263 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/3413upd | 4817,5817 | 1.0000 | 1.0000 | 82.11 | 1.0195 | PASS |
| separate_speed | B256/6826upd | 4817,5817 | 0.9111 | 0.7824 | 82.27 | 1.0236 | PASS |

本组 metric/model/data/precision 条件相同，但存在不同的实际执行 runtime cohort。以下 GC/loader provenance 保留分开；本组比较不宣称运行策略相同，也不单独隔离其因果影响。

| runtime cohort | youngGC interval | entrypoint | 额外 wrapper SHA |
| --- | --- | --- | --- |
| 9cad1296c9b0 | 0 | runtime_cached_torch_loader_sweep.py | 4a0024efc498 |
| b31f8e05a842 | 128 | overnight_init_diffusion_sweep.py | 无额外 wrapper |

各 arm / TRAIN 对应的实际 runtime IDs（完整 SHA 保存在机器审计）：control={"4817":["b31f8e05a842"],"5817":["9cad1296c9b0"]}; separate_speed={"4817":["9cad1296c9b0"],"5817":["9cad1296c9b0"]}。

### long / ema / exposure 3,494,730

实际完整 TRAIN 等价 epoch：7.0809；评价 4096 场景；cohort `98709b04289b`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 2 | 0.011756 ± 0.000229 | 0.088881 ± 0.000485 | 0.002855 ± 0.000149 | 0.007122 ± 0.001214 | 0.003810 ± 0.000316 | 0.001140 ± 0.000109 | 0.599230 ± 0.106420 |
| separate_speed | 2 | 0.011071 ± 0.000701 | 0.087113 ± 0.001950 | 0.003128 ± 0.000212 | 0.005062 ± 0.000355 | 0.001941 ± 0.000215 | 0.001053 ± 0.000090 | 0.669596 ± 0.168046 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/6826upd | 4817,5817 | 1.0000 | 1.0000 | 84.41 | 1.0214 | PASS |
| separate_speed | B256/13652upd | 4817,5817 | 0.8970 | 0.7147 | 84.82 | 1.0245 | PASS |

本组 metric/model/data/precision 条件相同，但存在不同的实际执行 runtime cohort。以下 GC/loader provenance 保留分开；本组比较不宣称运行策略相同，也不单独隔离其因果影响。

| runtime cohort | youngGC interval | entrypoint | 额外 wrapper SHA |
| --- | --- | --- | --- |
| 9cad1296c9b0 | 0 | runtime_cached_torch_loader_sweep.py | 4a0024efc498 |
| b31f8e05a842 | 128 | overnight_init_diffusion_sweep.py | 无额外 wrapper |

各 arm / TRAIN 对应的实际 runtime IDs（完整 SHA 保存在机器审计）：control={"4817":["b31f8e05a842"],"5817":["9cad1296c9b0"]}; separate_speed={"4817":["9cad1296c9b0"],"5817":["9cad1296c9b0"]}。

### long / ema / exposure 5,242,620

实际完整 TRAIN 等价 epoch：10.6224；评价 4096 场景；cohort `98709b04289b`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 2 | 0.009635 ± 0.000326 | 0.084949 ± 0.001515 | 0.002805 ± 0.000106 | 0.006134 ± 0.000213 | 0.002612 ± 0.000056 | 0.000997 ± 0.000039 | 0.432765 ± 0.083484 |
| separate_speed | 2 | 0.009104 ± 0.000570 | 0.083148 ± 0.000646 | 0.002718 ± 0.000230 | 0.004596 ± 0.000058 | 0.001848 ± 0.000027 | 0.001021 ± 0.000145 | 0.393698 ± 0.089250 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/10240upd | 4817,5817 | 1.0000 | 1.0000 | 85.83 | 1.0231 | PASS |
| separate_speed | B256/20480upd | 4817,5817 | 0.8975 | 0.8269 | 86.18 | 1.0242 | PASS |

本组 metric/model/data/precision 条件相同，但存在不同的实际执行 runtime cohort。以下 GC/loader provenance 保留分开；本组比较不宣称运行策略相同，也不单独隔离其因果影响。

| runtime cohort | youngGC interval | entrypoint | 额外 wrapper SHA |
| --- | --- | --- | --- |
| 9cad1296c9b0 | 0 | runtime_cached_torch_loader_sweep.py | 4a0024efc498 |
| b31f8e05a842 | 128 | overnight_init_diffusion_sweep.py | 无额外 wrapper |

各 arm / TRAIN 对应的实际 runtime IDs（完整 SHA 保存在机器审计）：control={"4817":["b31f8e05a842"],"5817":["9cad1296c9b0"]}; separate_speed={"4817":["9cad1296c9b0"],"5817":["9cad1296c9b0"]}。

### full50000 / ema / exposure 5,242,620

实际完整 TRAIN 等价 epoch：10.6224；评价 50000 场景；cohort `88a8b6eef2a5`。不同 cohort 分开列出，不交叉排名。

| arm | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1 | 0.008998 | 0.086288 | 0.002144 | 0.005046 | 0.002419 | 0.000970 | 0.375759 |
| separate_speed | 1 | 0.008941 | 0.085132 | 0.001878 | 0.003497 | 0.001633 | 0.000967 | 0.451252 |

上表六项 JSD 为 raw dimensionless JSD；Col% 保留百分比。± 为各 TRAIN 的 GEN 均值之间的 sample SD；N=1 没有 SD。

| arm | batch/updates | TRAIN seeds | R | R_attr | onroad% | veh/GT | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | B512/10240upd | 4817 | 1.0000 | 1.0000 | 85.45 | 1.0247 | PASS |
| separate_speed | B256/20480upd | 4817 | 0.9174 | 0.7883 | 85.97 | 1.0267 | PASS |

### 冻结最终选择的逐指标结果

| metric | control raw / Col% | selected raw / Col% | control native | selected native | 变化% |
| --- | --- | --- | --- | --- | --- |
| nearest_dist_jsd | 0.009635 | 0.009104 | 0.096351 | 0.091037 | -5.51 |
| lat_dev_jsd | 0.084949 | 0.083148 | 0.849486 | 0.831480 | -2.12 |
| ang_dev_jsd | 0.002805 | 0.002718 | 0.280540 | 0.271751 | -3.13 |
| length_jsd | 0.006134 | 0.004596 | 0.613365 | 0.459591 | -25.07 |
| width_jsd | 0.002612 | 0.001848 | 0.261200 | 0.184751 | -29.27 |
| speed_jsd | 0.000997 | 0.001021 | 0.099725 | 0.102130 | 2.41 |
| collision_rate | 0.432765 | 0.393698 | 0.432765 | 0.393698 | -9.03 |

| TRAIN | 角色 | selected R | R_attr | speed变化% | L变化% | W变化% | gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4817 | adaptive selection | 0.9453 | 0.8417 | 9.66 | -27.53 | -29.62 | PASS |
| 5817 | independent confirmation | 0.8612 | 0.8111 | -5.25 | -22.49 | -28.91 | PASS |

## 旧搜索的种子敏感性（独立历史证据）

旧 [final_selection](<experiments/init_diffusion_frozen_map_search_20261011/final_selection.json>) 的 2 arms ×3 TRAIN ×2 GEN 共 12 cells 已按六个 actual stage 的 execution manifest SHA、terminal DONE、实际 train/eval filename manifest SHA 和指标报告核验。各 TRAIN 内平均两 GEN，再平均三个 TRAIN；独立训练 N=3，817 是 adaptive discovery，1817/2817 是冻结后确认。

旧 pooled 规则选择 **combo_batch256_ego_sincos**：combo R=0.89118、speed ratio=1.08612，pooled gates=PASS。本节不重新加载 checkpoint tensors，不把这些旧结果混入本轮 TRAIN4817/5817。

| metric | 旧 control raw / Col% | 旧 combo raw / Col% | 旧 control native | 旧 combo native | 变化% |
| --- | --- | --- | --- | --- | --- |
| nearest_dist_jsd | 0.013359 | 0.013522 | 0.133591 | 0.135217 | 1.22 |
| lat_dev_jsd | 0.090733 | 0.090602 | 0.907327 | 0.906018 | -0.14 |
| ang_dev_jsd | 0.003140 | 0.003289 | 0.313997 | 0.328875 | 4.74 |
| length_jsd | 0.007852 | 0.005547 | 0.785196 | 0.554730 | -29.35 |
| width_jsd | 0.004471 | 0.002856 | 0.447122 | 0.285625 | -36.12 |
| speed_jsd | 0.001041 | 0.001130 | 0.104075 | 0.113039 | 8.61 |
| collision_rate | 1.251629 | 0.937164 | 1.251629 | 0.937164 | -25.12 |

| 旧 TRAIN | 角色 | R | speed ratio | speed变化% | 按本轮逐TRAIN gates |
| --- | --- | --- | --- | --- | --- |
| 817 | adaptive discovery | 0.92750 | 1.10968 | 10.97 | PASS |
| 1817 | post-freeze confirmation | 0.96407 | 1.36395 | 36.39 | FAIL |
| 2817 | post-freeze confirmation | 0.82197 | 0.88947 | -11.05 | PASS |

若把本轮更严格的 pooled+每 TRAIN gates 用于诊断这份旧矩阵，结论为 FAIL。TRAIN1817 的 speed ratio 超过 1.15，而 TRAIN2817 的 speed 改善；这说明均值会遮住失败种子，支持本轮增加逐 TRAIN 门槛。它不改写旧冻结选择规则，也不证明某个单独参数导致种子差异。

## Loss 和模型配置的对照

control 的计划 loss；同一个分量仍可独立选择 kind、时间 power、系数和 TRAIN scale：

| 分量 | kind | time power | weight | scale |
| --- | --- | --- | --- | --- |
| position | mse | 1 | 32 | data |
| heading | l1 | 1 | 1.0 | none |
| length | mse | 1 | 2.0 | data |
| width | mse | 1 | 2.0 | data |
| velocity | mse | 1 | 2 | data |
| type | ce | 0 | 1 | none |
| collision | l1 | 1 | 20.0 | none |
| speed_magnitude | mse | 0 | 0.0 | data |

计划全局设置：normalize_time_weights=True，std_floor=0.1，heading_space=angle。实际权重为 max(t,0.05)^(-p)/C_p；t≈0 接近 clean data、t≈1 接近 source noise，生成从 1 到 0。因此较大的 p 更强调低噪声/接近 clean 的误差。C_p 是固定 Uniform(t) 解析均值，p1=3.995732、p2=39、p3=599.5；不是逐 batch 或改后的抽样密度归一化。

| 计划 arm | B | loss 相对 control | 模型 / transport 相对 control |
| --- | --- | --- | --- |
| control | 512 | 与 control 相同 | 与 control 相同 |
| combo_reference | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos" |
| speed_p0 | 256 | length.time_power=2; velocity.time_power=0; width.time_power=2 | ego_context_heading_encoding="sincos" |
| speed_l1_p0 | 256 | length.time_power=2; velocity.kind=l1; velocity.time_power=0; width.time_power=2 | ego_context_heading_encoding="sincos" |
| train_features | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; state_feature_normalization="train" |
| separate_speed | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; speed_head_kind="separate" |
| detached_speed | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; speed_head_detach_context=true; speed_head_kind="separate" |
| softplus_speed | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; speed_x0_parameterization="softplus" |
| residual_speed | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; speed_x0_parameterization="time_residual" |
| high_noise | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; time_sample_power=0.5 |
| match_speed4 | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; matching_weights={"velocity":4.0} |
| transport_speed | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | distribution_loss={"speed":2.0,"time_min":0.5}; ego_context_heading_encoding="sincos" |
| transport_attributes | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | distribution_loss={"speed":2.0,"length":2.0,"width":2.0,"time_min":0.5}; ego_context_heading_encoding="sincos" |
| depth5 | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | denoiser_num_layers=5; ego_context_heading_encoding="sincos" |
| width384 | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | denoiser_hidden_dim=384; ego_context_heading_encoding="sincos" |
| normalized_speed_p0 | 256 | length.time_power=2; velocity.time_power=0; width.time_power=2 | ego_context_heading_encoding="sincos"; state_feature_normalization="train" |
| lane_center_context | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; map_condition_types="lanes_and_edges" |
| lane_center_normalized | 256 | length.time_power=2; velocity.weight=4.0; width.time_power=2 | ego_context_heading_encoding="sincos"; map_condition_types="lanes_and_edges"; state_feature_normalization="train" |

上表来自实际保存的 search_protocol，但尚未完成的 arm 不能据此宣称这些选项已产生性能变化；容量/梯度/CPU 一步预检也不是收敛验证。

具体地，speed_p0 相对 combo_reference 同时把 velocity time power 从 1 改为 0、weight 从 4.0 改为 2.0；不能把它的结果归因于纯 power 改动。normalized_speed_p0 在这组联合 loss 改动上再加入输入归一化，需作为另一组联合干预解释。

以下为 verified stage 真正保存的配置（不是仅解析计划）：

| arm / config | loss hash | optimizer / LR / clip | G1 hidden/layers | 输入 / head / x0 | map / t sampler |
| --- | --- | --- | --- | --- | --- |
| control / [15f931d7](<../logs/init_diffusion_overnight_20261011/full50000_seed4817_control/control/config.yaml>) | 5f341f6a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| separate_speed / [cead7fdc](<../logs/init_diffusion_overnight_20261011/full50000_seed4817_separate_speed/separate_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/separate/identity | road_edges/1.0 |
| control / [5ef7fdb2](<../logs/init_diffusion_overnight_20261011/long_seed4817_control/control/config.yaml>) | 5f341f6a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| separate_speed / [fb8b7763](<../logs/init_diffusion_overnight_20261011/long_seed4817_separate_speed/separate_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/separate/identity | road_edges/1.0 |
| control / [597e2b8d](<../logs/init_diffusion_overnight_20261011/long_seed5817_control/control/config.yaml>) | 5f341f6a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| separate_speed / [03ab4741](<../logs/init_diffusion_overnight_20261011/long_seed5817_separate_speed/separate_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/separate/identity | road_edges/1.0 |
| control / [a7b9118f](<../logs/init_diffusion_overnight_20261011/medium_control/control/config.yaml>) | 5f341f6a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| lane_center_context / [29bfc402](<../logs/init_diffusion_overnight_20261011/medium_lane_center_context/lane_center_context/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/identity | lanes_and_edges/1.0 |
| separate_speed / [e19fdd6a](<../logs/init_diffusion_overnight_20261011/medium_separate_speed/separate_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/separate/identity | road_edges/1.0 |
| combo_reference / [07202d2e](<../logs/init_diffusion_overnight_20261011/screen_combo_reference/combo_reference/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| control / [952bc771](<../logs/init_diffusion_overnight_20261011/screen_control/control/config.yaml>) | 5f341f6a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| depth5 / [eae47ea9](<../logs/init_diffusion_overnight_20261011/screen_depth5/depth5/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/5 | none/shared/identity | road_edges/1.0 |
| detached_speed / [6d037e0b](<../logs/init_diffusion_overnight_20261011/screen_detached_speed/detached_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/separate/identity | road_edges/1.0 |
| high_noise / [9298eac4](<../logs/init_diffusion_overnight_20261011/screen_high_noise/high_noise/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/0.5 |
| lane_center_context / [76eae730](<../logs/init_diffusion_overnight_20261011/screen_lane_center_context/lane_center_context/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/identity | lanes_and_edges/1.0 |
| lane_center_normalized / [8a1ed320](<../logs/init_diffusion_overnight_20261011/screen_lane_center_normalized/lane_center_normalized/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | train/shared/identity | lanes_and_edges/1.0 |
| match_speed4 / [ac71c24c](<../logs/init_diffusion_overnight_20261011/screen_match_speed4/match_speed4/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| normalized_speed_p0 / [9cd1573c](<../logs/init_diffusion_overnight_20261011/screen_normalized_speed_p0/normalized_speed_p0/config.yaml>) | 23f75c06 | default/0.0005/0.5 | 256/3 | train/shared/identity | road_edges/1.0 |
| residual_speed / [02591672](<../logs/init_diffusion_overnight_20261011/screen_residual_speed/residual_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/time_residual | road_edges/1.0 |
| separate_speed / [2373eda5](<../logs/init_diffusion_overnight_20261011/screen_separate_speed/separate_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/separate/identity | road_edges/1.0 |
| softplus_speed / [acaa590e](<../logs/init_diffusion_overnight_20261011/screen_softplus_speed/softplus_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/softplus | road_edges/1.0 |
| speed_l1_p0 / [c9cf2fce](<../logs/init_diffusion_overnight_20261011/screen_speed_l1_p0/speed_l1_p0/config.yaml>) | cc624b76 | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| speed_p0 / [f840a8d8](<../logs/init_diffusion_overnight_20261011/screen_speed_p0/speed_p0/config.yaml>) | 23f75c06 | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| train_features / [b1416ebf](<../logs/init_diffusion_overnight_20261011/screen_train_features/train_features/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | train/shared/identity | road_edges/1.0 |
| transport_attributes / [50685e74](<../logs/init_diffusion_overnight_20261011/screen_transport_attributes/transport_attributes/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| transport_speed / [17fa459a](<../logs/init_diffusion_overnight_20261011/screen_transport_speed/transport_speed/config.yaml>) | f041ca3a | default/0.0005/0.5 | 256/3 | none/shared/identity | road_edges/1.0 |
| width384 / [f6cafdf5](<../logs/init_diffusion_overnight_20261011/screen_width384/width384/config.yaml>) | f041ca3a | default/0.0005/0.5 | 384/3 | none/shared/identity | road_edges/1.0 |

以下裁剪记录由实际 train_summary 和完整 pre-update train_trace 重算交叉检查。旧实验仅提供机制线索，不计入本轮 overnight 指标或 TRAIN N：

| 阶段 / 范围 | updates | clip | clip_fraction | 首32步 grad norm | 末32步 grad norm | 实际来源 |
| --- | --- | --- | --- | --- | --- | --- |
| long_seed4817_control / control / 本轮 verified stage | 10240 | 0.50 | 1.0000 | 169.727 | 4.337 | [f221f23e](<../logs/init_diffusion_overnight_20261011/long_seed4817_control/control/train_summary.json>) |
| long_seed4817_separate_speed / separate_speed / 本轮 verified stage | 20480 | 0.50 | 1.0000 | 170.096 | 2.346 | [88c16d46](<../logs/init_diffusion_overnight_20261011/long_seed4817_separate_speed/separate_speed/train_summary.json>) |
| long_seed5817_control / control / 本轮 verified stage | 10240 | 0.50 | 1.0000 | 160.866 | 2.120 | [a5d65953](<../logs/init_diffusion_overnight_20261011/long_seed5817_control/control/train_summary.json>) |
| long_seed5817_separate_speed / separate_speed / 本轮 verified stage | 20480 | 0.50 | 1.0000 | 121.174 | 2.033 | [faede007](<../logs/init_diffusion_overnight_20261011/long_seed5817_separate_speed/separate_speed/train_summary.json>) |
| medium_control / control / 本轮 verified stage | 4096 | 0.50 | 1.0000 | 169.630 | 3.474 | [f4bbb09c](<../logs/init_diffusion_overnight_20261011/medium_control/control/train_summary.json>) |
| medium_lane_center_context / lane_center_context / 本轮 verified stage | 8192 | 0.50 | 1.0000 | 178.907 | 4.229 | [d03bee8e](<../logs/init_diffusion_overnight_20261011/medium_lane_center_context/lane_center_context/train_summary.json>) |
| medium_separate_speed / separate_speed / 本轮 verified stage | 8192 | 0.50 | 1.0000 | 177.635 | 4.081 | [0c296e8a](<../logs/init_diffusion_overnight_20261011/medium_separate_speed/separate_speed/train_summary.json>) |
| screen_combo_reference / combo_reference / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 186.288 | 5.516 | [c2973b26](<../logs/init_diffusion_overnight_20261011/screen_combo_reference/combo_reference/train_summary.json>) |
| screen_control / control / 本轮 verified stage | 768 | 0.50 | 1.0000 | 166.861 | 7.472 | [f1662c3a](<../logs/init_diffusion_overnight_20261011/screen_control/control/train_summary.json>) |
| screen_depth5 / depth5 / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 168.466 | 16.872 | [eca7f9b3](<../logs/init_diffusion_overnight_20261011/screen_depth5/depth5/train_summary.json>) |
| screen_detached_speed / detached_speed / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 161.916 | 8.747 | [429739c8](<../logs/init_diffusion_overnight_20261011/screen_detached_speed/detached_speed/train_summary.json>) |
| screen_high_noise / high_noise / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 24.086 | 3.472 | [78afe925](<../logs/init_diffusion_overnight_20261011/screen_high_noise/high_noise/train_summary.json>) |
| screen_lane_center_context / lane_center_context / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 179.244 | 6.582 | [cbb84817](<../logs/init_diffusion_overnight_20261011/screen_lane_center_context/lane_center_context/train_summary.json>) |
| screen_lane_center_normalized / lane_center_normalized / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 184.710 | 6.742 | [2cd47b65](<../logs/init_diffusion_overnight_20261011/screen_lane_center_normalized/lane_center_normalized/train_summary.json>) |
| screen_match_speed4 / match_speed4 / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 188.216 | 7.922 | [e3e1820c](<../logs/init_diffusion_overnight_20261011/screen_match_speed4/match_speed4/train_summary.json>) |
| screen_normalized_speed_p0 / normalized_speed_p0 / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 172.233 | 6.510 | [f212bf8f](<../logs/init_diffusion_overnight_20261011/screen_normalized_speed_p0/normalized_speed_p0/train_summary.json>) |
| screen_residual_speed / residual_speed / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 184.625 | 5.592 | [1c909fb4](<../logs/init_diffusion_overnight_20261011/screen_residual_speed/residual_speed/train_summary.json>) |
| screen_separate_speed / separate_speed / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 171.980 | 6.022 | [239a595e](<../logs/init_diffusion_overnight_20261011/screen_separate_speed/separate_speed/train_summary.json>) |
| screen_softplus_speed / softplus_speed / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 187.541 | 6.541 | [4a16e926](<../logs/init_diffusion_overnight_20261011/screen_softplus_speed/softplus_speed/train_summary.json>) |
| screen_speed_l1_p0 / speed_l1_p0 / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 180.609 | 7.700 | [3b6442ed](<../logs/init_diffusion_overnight_20261011/screen_speed_l1_p0/speed_l1_p0/train_summary.json>) |
| screen_speed_p0 / speed_p0 / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 190.245 | 7.829 | [ba520422](<../logs/init_diffusion_overnight_20261011/screen_speed_p0/speed_p0/train_summary.json>) |
| screen_train_features / train_features / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 175.633 | 7.462 | [81064a37](<../logs/init_diffusion_overnight_20261011/screen_train_features/train_features/train_summary.json>) |
| screen_transport_attributes / transport_attributes / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 311.390 | 33.625 | [3c2f0efa](<../logs/init_diffusion_overnight_20261011/screen_transport_attributes/transport_attributes/train_summary.json>) |
| screen_transport_speed / transport_speed / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 183.934 | 8.809 | [9c7407c4](<../logs/init_diffusion_overnight_20261011/screen_transport_speed/transport_speed/train_summary.json>) |
| screen_width384 / width384 / 本轮 verified stage | 1536 | 0.50 | 1.0000 | 227.444 | 15.809 | [fa820147](<../logs/init_diffusion_overnight_20261011/screen_width384/width384/train_summary.json>) |
| TRAIN817 / new_control / 旧实验历史证据 | 4096 | 0.50 | 1.0000 | 147.507 | 3.596 | [0daa15b1](<../logs/component_loss_frozen_map/long_seed817_new_control/new_control/train_summary.json>) |
| TRAIN817 / combo_batch256_ego_sincos / 旧实验历史证据 | 8192 | 0.50 | 1.0000 | 150.365 | 4.029 | [acc2edbb](<../logs/component_loss_frozen_map/long_seed817_combo_batch256_ego_sincos/combo_batch256_ego_sincos/train_summary.json>) |
| TRAIN1817 / new_control / 旧实验历史证据 | 4096 | 0.50 | 1.0000 | 210.373 | 4.702 | [6ce97042](<../logs/component_loss_frozen_map/confirm_seed1817_new_control/new_control/train_summary.json>) |
| TRAIN1817 / combo_batch256_ego_sincos / 旧实验历史证据 | 8192 | 0.50 | 1.0000 | 144.232 | 2.917 | [672b786b](<../logs/component_loss_frozen_map/confirm_seed1817_combo_batch256_ego_sincos/combo_batch256_ego_sincos/train_summary.json>) |

强全局裁剪会把总梯度缩放到阈值附近，因此增大一个分量的 weight 不等于等比例增大参数更新，也会改变各分量的相对梯度贡献。上述触发率说明这可能影响权重干预的效果；它不证明裁剪造成任何已观测的 JSD 改善或恶化。

所有 speed scalar arm 使用 `[pos,cos/sin,L,W,speed]`，沿生成 heading 重建 velocity；speed norm 已直接监督，没有再添加 vector magnitude loss。time_residual 维持 x0 目标但 head 梯度带 t 因子；高-noise t=U^0.5 改变抽样密度且未 importance-correct，不能描述成仅降低方差的等权实验。

默认 road_edges 保留旧权重与数值路径；lanes_and_edges opt-in 只增加 type0/1/2 中心线 query，attention source/crop/radius/k 不变。训练缓存中心线是稀疏片段，naive lane-distance 辅助项未加入；不能用 road edge 的 position 当 centerline。

已有 CPU 权重积分复核：保持 Uniform(t) 解析归一化常数时，t=U^0.5 下 p1 的平均有效权重约 0.488，p2 约 0.179（Uniform 下均为 1）。high_noise 同时干预时间密度及分量平均权重；这些是机制计算，不是新实验的性能观察。

## 真实图形、运行成本与不足

![screen 已完成终点：六项 JSD 与 collision 的 candidate/control 比值及 R；星号表示 gates FAIL。](experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/screen_endpoint_c1.png)

screen 已完成终点：六项 JSD 与 collision 的 candidate/control 比值及 R；星号表示 gates FAIL。

可缩放原图：[screen_endpoint_c1.svg](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/screen_endpoint_c1.svg>)。

![medium 已完成终点：六项 JSD 与 collision 的 candidate/control 比值及 R；星号表示 gates FAIL。](experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/medium_endpoint_c1.png)

medium 已完成终点：六项 JSD 与 collision 的 candidate/control 比值及 R；星号表示 gates FAIL。

可缩放原图：[medium_endpoint_c1.svg](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/medium_endpoint_c1.svg>)。

![long实际epoch曲线（1/2）；第二张下右图为TRAIN N，N=1不画SD。每一budget的实际TRAIN membership和相同source在metadata保留；未配齐TRAIN只作原始均值描述。](experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/long_actual_epochs_c1_p1.png)

long实际epoch曲线（1/2）；第二张下右图为TRAIN N，N=1不画SD。每一budget的实际TRAIN membership和相同source在metadata保留；未配齐TRAIN只作原始均值描述。

可缩放原图：[long_actual_epochs_c1_p1.svg](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/long_actual_epochs_c1_p1.svg>)。

![long实际epoch曲线（2/2）；第二张下右图为TRAIN N，N=1不画SD。每一budget的实际TRAIN membership和相同source在metadata保留；未配齐TRAIN只作原始均值描述。](experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/long_actual_epochs_c1_p2.png)

long实际epoch曲线（2/2）；第二张下右图为TRAIN N，N=1不画SD。每一budget的实际TRAIN membership和相同source在metadata保留；未配齐TRAIN只作原始均值描述。

可缩放原图：[long_actual_epochs_c1_p2.svg](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/long_actual_epochs_c1_p2.svg>)。

![main N20训练recipe比较的full50k speed/length/width histogram与onroad/vehicle count；TRAIN4817×两GEN，只作scope audit。不是native40或hybrid40的full直方图。](experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/full50000_attributes_coverage.png)

main N20训练recipe比较的full50k speed/length/width histogram与onroad/vehicle count；TRAIN4817×两GEN，只作scope audit。不是native40或hybrid40的full直方图。

可缩放原图：[full50000_attributes_coverage.svg](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/full50000_attributes_coverage.svg>)。

![sampler16：相对fresh N20 anchors。每个面板使用自己的新跑基线，不表示跨阶段复合收益；两套TRAIN权重复用、各两GEN，不新增训练N。负值表示该指标改善。](experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/sampler_steps_actual.png)

sampler16：相对fresh N20 anchors。每个面板使用自己的新跑基线，不表示跨阶段复合收益；两套TRAIN权重复用、各两GEN，不新增训练N。负值表示该指标改善。

![hybrid8：相对另行重新执行的fresh N40 anchors。每个面板使用自己的新跑基线，不表示跨阶段复合收益；两套TRAIN权重复用、各两GEN，不新增训练N。负值表示该指标改善。](experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/sampler_hybrid_actual.png)

hybrid8：相对另行重新执行的fresh N40 anchors。每个面板使用自己的新跑基线，不表示跨阶段复合收益；两套TRAIN权重复用、各两GEN，不新增训练N。负值表示该指标改善。

可缩放原图：[sampler_tradeoffs_actual.svg](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/sampler_tradeoffs_actual.svg>)；实际grid/decision/source/file SHA：[sampler_tradeoffs_actual.metadata.json](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/sampler_tradeoffs_actual.metadata.json>)。

嵌入图的 actual 数据、独立 TRAIN membership、source SHA 与文件 SHA：[metadata.json](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/metadata.json>)。

| stage | fit秒（一次） | eval秒 | B / updates | 秒/update |
| --- | --- | --- | --- | --- |
| full50000_seed4817_control | 0.0 | 453.7 | B512/10240 | — |
| full50000_seed4817_separate_speed | 0.0 | 452.1 | B256/20480 | — |
| long_seed4817_control | 3269.4 | 108.0 | B512/10240 | 0.3193 |
| long_seed4817_separate_speed | 3517.0 | 110.0 | B256/20480 | 0.1717 |
| long_seed5817_control | 3161.8 | 105.6 | B512/10240 | 0.3088 |
| long_seed5817_separate_speed | 3461.8 | 107.1 | B256/20480 | 0.1690 |
| medium_control | 1355.4 | 72.3 | B512/4096 | 0.3309 |
| medium_lane_center_context | 1563.2 | 73.4 | B256/8192 | 0.1908 |
| medium_separate_speed | 1472.7 | 70.9 | B256/8192 | 0.1798 |
| screen_combo_reference | 288.9 | 8.7 | B256/1536 | 0.1881 |
| screen_control | 274.5 | 8.8 | B512/768 | 0.3574 |
| screen_depth5 | 296.6 | 9.7 | B256/1536 | 0.1931 |
| screen_detached_speed | 289.1 | 8.7 | B256/1536 | 0.1882 |
| screen_high_noise | 288.8 | 8.8 | B256/1536 | 0.1880 |
| screen_lane_center_context | 290.7 | 9.0 | B256/1536 | 0.1892 |
| screen_lane_center_normalized | 293.8 | 9.0 | B256/1536 | 0.1913 |
| screen_match_speed4 | 263.7 | 8.7 | B256/1536 | 0.1717 |
| screen_normalized_speed_p0 | 267.4 | 8.6 | B256/1536 | 0.1741 |
| screen_residual_speed | 290.1 | 8.8 | B256/1536 | 0.1889 |
| screen_separate_speed | 289.1 | 8.7 | B256/1536 | 0.1882 |
| screen_softplus_speed | 286.4 | 8.7 | B256/1536 | 0.1865 |
| screen_speed_l1_p0 | 290.7 | 8.8 | B256/1536 | 0.1893 |
| screen_speed_p0 | 289.4 | 8.7 | B256/1536 | 0.1884 |
| screen_train_features | 286.9 | 8.7 | B256/1536 | 0.1868 |
| screen_transport_attributes | 272.0 | 8.0 | B256/1536 | 0.1771 |
| screen_transport_speed | 267.3 | 8.7 | B256/1536 | 0.1740 |
| screen_width384 | 307.9 | 10.3 | B256/1536 | 0.2004 |

已完成主搜索 fit 合计 22934.4 秒，所有实际 eval cell 合计 1712.4 秒。fit 时间对每个 stage 只计一次。失败尝试成本不冒充这些成功 fit 的训练量。

最终预算实际冻结为每 TRAIN 5242620 TRAIN cache exposures（10.6224 等价全数据 epoch），B512=10240 updates，B256=20480 updates；这是选择的预算，完成情况以上面的 verified stage 为准。

## 已有证据给出的机制与边界

以下是本轮之前 CPU 数据排查与源码设计依据，不能与夜间 TRAIN4817/5817 的新结果混合计 N：

旧 TRAIN817/1817 配对中尺寸和 collision 可改善而 speed gate 失效；短期优势可能反转。标量 speed 已直接监督，低速 [0,1) 过多而高速尾部不足；1 m/s bins 无法识别 exact stationary 或最终 clamp 前负值。

4096 native cache 的真实初始化 speed/L/W bins 与 matched GT 的 29,698 vehicle counts 完全一致，未发现这些属性的评价时刻或单位错位；8192 TRAIN cache scan 只校验校准样本，历史原始 track/timestep 来源仍受缓存缺字段限制。

更大 loss weight、独立 head、输入归一化、time sampling 和 matching 都是待因果验证的干预。MSE 的条件均值在精确学习/积分下可对应正确概率流，不能只因 MSE 就断言分布必然坍缩。生成 type 会改变 vehicle histogram population，需一起看 counts 与 onroad。

比较只在同 metric/helper、物理场景 manifest、source、conditioning、实际 exposure 的组内进行。B256 与 B512 的 optimizer/EMA updates 不同，same exposure 不是 pure batch 因果隔离。GT 地图条件实验也不能直接与 SD/VW 的联合地图生成论文数值宣称优劣。

## Native 配置与后续执行

实际 materializer 已通过：[init_diffusion_lane_conditioned_overnight_best.yaml](<../configs/experiment/init_diffusion_lane_conditioned_overnight_best.yaml>)、[init_diffusion_lane_conditioned_overnight_best_eval.yaml](<../configs/experiment/init_diffusion_lane_conditioned_overnight_best_eval.yaml>)；四个来源 checkpoint 的 metadata/EMA/normalizer/frozen-map 核验与 Hydra composition 记录有效。

后续fresh训练与当前建议推断设置的精确命令见报告开头；128epoch尚未执行/完成。历史main20训练recipe比较的native重放配置仍保留，默认同一本次TRAIN4817权重：

```bash
python tools/run_init_diffusion_safe.py \
  experiment=init_diffusion_lane_conditioned_overnight_best_eval
```

上述20步重放仍受已记录的1e-6数值复现失败限制；未将其重标为新sampler基线。

Native full fit shuffle=True，quick fixed-order search shuffle=False；full fit 的 epoch-based128epoch cosine schedule 与此处预算结果不能视为最终效果保证。native eval 恢复 map/G1/normalizer/完整 EMA；fit 只加载 map、fresh G1，不 resume evidence checkpoint。

## 原严格 sampler 复现审计（失败保留）

本附录复用主搜索已选 checkpoint，仅改变推理 steps / t_eps；实际 training updates=0，不改变主模型选择。GEN 先在每 TRAIN 内平均，grid最多复用两套训练权重，full以实际列出的TRAIN为准，不新增独立训练 N。TRAIN4817参与自适应模型选择，TRAIN5817是已冻结模型的独立确认；sampler同时使用这两套权重择优，因此并未获得新的独立 sampler 确认。改变 t_eps 会改变末端积分路径，不能都解释成纯 ODE 步长精度改善。

原预注册grid的首个N20实际未通过1e-6原生单位复现门槛，未接受任何successful sampler cell，也没有任何changed-step结果；原driver/selector/protocol保持不变，永久排除选优。主搜索8+4实际结果和选中训练recipe不受此失败重新选择。证据：[actual_native_default20_parity_failure_review.json](<experiments/init_diffusion_overnight_search_20261011/actual_native_default20_parity_failure_review.json>)。

相同checkpoint/seed的两个native结果与原冻结runner自重放也都不能达到1e-6；CPU记录的map/G1/EMA/normalizer完全一致。该现象没有证明native restoration有误，CUDA scatter非确定性是待验证机制，source noise逐元素一致尚未测，重复同TRAIN/GEN不会增加独立N。来源：[sampler_numerical_repeatability_independent_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/sampler_numerical_repeatability_independent_cpu_review.json>)。

推理选择准则已在实际 UTC 2026-10-11T22:37:45.721171+00:00、本cohort所有 sampler cell 启动之前冻结，源SHA见 [sampler_selection_protocol.json](<experiments/init_diffusion_overnight_search_20261011/sampler_selection_protocol.json>)。完整20cells且四个default20全部通过≤1e-6原生parity后，计算GEN→TRAIN均值与R7；L/W/speed≤baseline×1.15、onroad≥baseline−.02、vehicle/GT在.9..1.1须同时通过pooled及每TRAIN。只选eligible最小R<1，否则保留20步；精确R tie依次更少steps、eps更接近训练.05、option名。full50k禁止重选。

尚未发布已冻结 sampler 设置：尚无实际 frozen_sampler.json。

### sampler grid / FAILED_DEFAULT20_PARITY

首个default20实际复现失败，旧1e-6门槛保持不变；该grid永久排除选优；不报告 sampler 改善或最佳设置。

### sampler full / not_started_or_no_completed_cell

尚无实际 sampler summary；不报告 sampler 改善或最佳设置。

## 新 native-to-native sampler 实测附录

本附录复用主搜索已选 checkpoint，仅改变推理 steps / t_eps；实际 training updates=0，不改变主模型选择。GEN 先在每 TRAIN 内平均，grid最多复用两套训练权重，full以实际列出的TRAIN为准，不新增独立训练 N。TRAIN4817参与自适应模型选择，TRAIN5817是已冻结模型的独立确认；sampler同时使用这两套权重择优，因此并未获得新的独立 sampler 确认。改变 t_eps 会改变末端积分路径，不能都解释成纯 ODE 步长精度改善。

这是一项新建并另行预注册的native-to-native推断实验：四个重新执行的N20是唯一比较基线，不复用旧失败结果、不宣称与主endpoint通过1e-6。四options×两TRAIN×两GEN=16cells；仅eligible且pooled R≤.98才选更低R，否则保留fresh N20。2%是对观察到数值波动设置的描述性margin，不是置信区间或统计显著性保证。

推理选择准则已在实际 UTC 2026-10-12T00:56:36.784553+00:00、本cohort所有 sampler cell 启动之前冻结，源SHA见 [native_sampler_selection_protocol.json](<experiments/init_diffusion_overnight_search_20261011/native_sampler_selection_protocol.json>)。完整16cells、四个fresh N20 anchors均实际完成后，计算GEN→TRAIN均值与R7；L/W/speed≤baseline×1.15、onroad≥baseline−.02、vehicle/GT在.9..1.1须同时通过pooled及每TRAIN。只选eligible最小R≤.98，否则保留fresh20步；精确R tie依次更少steps、eps更接近训练.05、option名。full50k禁止重选。

实际已冻结推理设置 **n40_eps005**（40 steps，base eps=0.05），由完整实际grid按上述预注册规则选择。相对fresh N20的模型求值次数为×2.00；这是计算次数比，不是未经实测的wall-time倍率。冻结artifact：[frozen_native_sampler.json](<experiments/init_diffusion_overnight_search_20261011/frozen_native_sampler.json>)。主模型配方选择保持不变。

### sampler grid / verified_complete

实际完成 16/16 cells；只有 execution rc0、native marker、checkpoint/source/scene/metric artifact SHA 与 hook metadata 全部通过才纳入。来源：[summary.json](<../logs/init_diffusion_native_sampler_cohort_20261012/grid/summary.json>)。

fresh N20 anchors 已核验 4/4 cells；这是新的配对基线，没有old-main parity成功声明。

| option | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n10_eps005 | 2 | 0.010747 ± 0.000585 | 0.082523 ± 0.000908 | 0.002755 ± 0.000247 | 0.005621 ± 0.000246 | 0.003047 ± 0.000052 | 0.001330 ± 0.000131 | 0.717050 ± 0.174586 |
| n20_eps005 | 2 | 0.009085 ± 0.000561 | 0.083158 ± 0.000643 | 0.002715 ± 0.000223 | 0.004585 ± 0.000057 | 0.001853 ± 0.000016 | 0.001021 ± 0.000144 | 0.393693 ± 0.089257 |
| n40_eps0001 | 2 | 0.008418 ± 0.000639 | 0.083585 ± 0.000092 | 0.002678 ± 0.000231 | 0.004406 ± 0.000029 | 0.001755 ± 0.000285 | 0.000992 ± 0.000138 | 0.333345 ± 0.063843 |
| n40_eps005 | 2 | 0.009772 ± 0.000744 | 0.084983 ± 0.001054 | 0.002805 ± 0.000160 | 0.003892 ± 0.000161 | 0.001110 ± 0.000018 | 0.000969 ± 0.000178 | 0.434581 ± 0.074259 |

六项为 raw JSD，Col% 为百分比；± 是两套复用 TRAIN 权重的 GEN 均值间 sample SD，并非各cell都对应独立训练。以下 ratio/R 均对同checkpoint、同TRAIN/GEN、本cohort重新执行的anchor。

| option | pooled R | Speed ratio | Length ratio | Width ratio | Col ratio | pooled / each TRAIN gates |
| --- | --- | --- | --- | --- | --- | --- |
| n10_eps005 | 1.3119 | 1.3020 | 1.2258 | 1.6439 | 1.8213 | False / False |
| n20_eps005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | True / True |
| n40_eps0001 | 0.9492 | 0.9716 | 0.9609 | 0.9468 | 0.8467 | True / True |
| n40_eps005 | 0.9472 | 0.9484 | 0.8489 | 0.5988 | 1.1039 | True / True |

| option | TRAIN | R | Speed ratio | Col ratio | gates |
| --- | --- | --- | --- | --- | --- |
| n10_eps005 | 4817 | 1.3165 | 1.2669 | 1.8399 | FAIL |
| n10_eps005 | 5817 | 1.3074 | 1.3450 | 1.7956 | FAIL |
| n20_eps005 | 4817 | 1.0000 | 1.0000 | 1.0000 | PASS |
| n20_eps005 | 5817 | 1.0000 | 1.0000 | 1.0000 | PASS |
| n40_eps0001 | 4817 | 0.9631 | 0.9705 | 0.8286 | PASS |
| n40_eps0001 | 5817 | 0.9360 | 0.9730 | 0.8718 | PASS |
| n40_eps005 | 4817 | 0.9484 | 0.9750 | 1.0663 | PASS |
| n40_eps005 | 5817 | 0.9474 | 0.9158 | 1.1558 | PASS |

这是完整配对 grid 的描述性比较，本报告不创建 sampler selection。若另行冻结一个推理设置并做 full50k，必须引用实际完整 grid SHA；不能用未完成 grid 或事后 full50k 重选。

| option | native run 秒/cell 均值 | 范围秒 | 实测时间/本次anchor | model calls/本次anchor |
| --- | --- | --- | --- | --- |
| n10_eps005 | 15.881 | 15.818–15.947 | 0.834 | 0.50 |
| n20_eps005 | 19.034 | 18.999–19.070 | 1.000 | 1.00 |
| n40_eps0001 | 25.450 | 25.403–25.540 | 1.337 | 2.00 |
| n40_eps005 | 25.540 | 25.396–25.654 | 1.342 | 2.00 |

时间来自各实际cell的eval_seconds，覆盖native run的模型/restore/data setup和TEST，不包含子进程启动前的Python imports。40步模型求值次数为2倍，实测约1.34倍时间；不把计算次数与wall-time等同。

实现含义：eps是低t时Euclidean速度分母的floor，不是solver终止时间；各option均匀时间网格1→0。令最后一次预测为x0_hat，v=(x_t−x0_hat)/max(t,eps)，x_next=x_t+(t_next−t)v。N20/eps=.05最后t=.05、t_next=0，最终x=x0_hat；N40/eps=.05最后t=.025，最终x=.5x_t+.5x0_hat；N40/eps=.0001最后分母为真实.025，最终x=x0_hat。该公式适用于pos/L/W/speed。圆周heading使用真实t与机器epsilon，separate type最终argmax最后的clean logits；不能把上述半混合公式套到heading或最终type。训练t由U(0,1)抽样，t<.05并非未训练范围。来源：[scale_flow.py（681–697、1565、1696、1803）](<../src/smart/diffusion/scale_flow.py>)。

因此40/.05的结果同时涉及更多模型预测和最后一次Euclidean低t damping；额外残余变化可能影响宽度/速度分布，是符合源码的机制假设，尚未以可控实验隔离因果。不能把收益全部解释成数值截断误差下降，或声称模型训练目标改变。

两40步选项的实际R：eps=.05为0.947242，eps=.0001为0.949189，差+0.001947。eps=.0001的碰撞相对fresh20为-15.33%，而eps=.05有更强的尺寸收益。差距小于2%择优margin的量级，该margin针对相对N20而非两40选项之间；结合数值重放波动，不能称eps=.05是稳健最优。预注册选择保持不变。

### sampler full / verified_complete

实际完成 2/2 cells；只有 execution rc0、native marker、checkpoint/source/scene/metric artifact SHA 与 hook metadata 全部通过才纳入。来源：[summary.json](<../logs/init_diffusion_native_sampler_cohort_20261012/full/summary.json>)。

本native sampler driver未导出feature histogram bins；这里只能报告实际七项metrics/counts/coverage，不能据此分析新40步生成结果的stationary或尺寸尾部分布。

| TRAIN | GEN | frozen option | Near | Lat | Ang | Length | Width | Speed | Col% | onroad% | veh/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4817 | 2817 | n40_eps005 | 0.009595 | 0.086679 | 0.001975 | 0.002787 | 0.000901 | 0.000947 | 0.472475 | 85.22 | 371025/362309 |
| 4817 | 3817 | n40_eps005 | 0.009998 | 0.086678 | 0.002015 | 0.002741 | 0.000931 | 0.000935 | 0.416622 | 85.37 | 371800/362309 |

| TRAIN | GEN repeats | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4817 | 2 | 0.009796 | 0.086679 | 0.001995 | 0.002764 | 0.000916 | 0.000941 | 0.444548 |

上表是各TRAIN内两GEN的raw JSD均值；同一checkpoint的生成重复不增加独立训练N，不提供TRAIN间SD。full50k仅展示已冻结sampler的实际原生TEST metrics/counts；50,000 cache entries不是50,000个未见独立原始场景。不同GT population的4096 grid不混算ratio，也不依据full结果重新选sampler。

这次full没有重新执行本cohort的baseline full anchor，不能报告native full配对R或把历史main20当新的fresh full对照。

实际40步standalone评价配置已通过两GEN的native builder/hook/数据/EMA纯CPU compose核验：[init_diffusion_lane_conditioned_overnight_best_sampler_eval.yaml](<../configs/experiment/init_diffusion_lane_conditioned_overnight_best_sampler_eval.yaml>)；metadata：[selected_sampler_native_materialization.json](<experiments/init_diffusion_overnight_search_20261011/selected_sampler_native_materialization.json>)。以下命令采用已落地YAML的默认4817checkpoint/EMA20480，不改变训练20步设置：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC=1
python tools/run_init_diffusion_safe.py \
  experiment=init_diffusion_lane_conditioned_overnight_best_sampler_eval
```

第二GEN可追加sampler_eval_seed=3817。未来更换checkpoint时应同步实际saved EMA update计数；没有128epoch已训练权重或bitwise replay保证。

## 追加 position-only hybrid 推断附录（单独8cell cohort）

本附录复用主搜索已选 checkpoint，仅改变推理 steps / t_eps；实际 training updates=0，不改变主模型选择。GEN 先在每 TRAIN 内平均，grid最多复用两套训练权重，full以实际列出的TRAIN为准，不新增独立训练 N。TRAIN4817参与自适应模型选择，TRAIN5817是已冻结模型的独立确认；sampler同时使用这两套权重择优，因此并未获得新的独立 sampler 确认。改变 t_eps 会改变末端积分路径，不能都解释成纯 ODE 步长精度改善。

本追加实验另行预注册两options×两TRAIN×两GEN=8cells；四个重新执行的N40/eps=.05为唯一配对anchor，hybrid仅在0<t<.05将pos维0:2的velocity分母floor改为1e-4，heading/L/W/speed/type路径保留。完整8cells且R≤.98与pooled AND每TRAIN五gates全通过才选择hybrid，否则保留fresh N40。前native16选择和主训练recipe保持不变。两TRAIN权重都参与这次选择，不新增独立holdout；full只对该次冻结option作scope audit，不能重选。实际40步最后t约.025大于1e-4，因此hybrid position使用真实t；1e-4 floor在此次最后更新并未生效，这里没有单独优化该floor数值。

推理选择准则已在实际 UTC 2026-10-12T01:13:40.138311+00:00、本cohort所有 sampler cell 启动之前冻结，源SHA见 [hybrid_sampler_selection_protocol.json](<experiments/init_diffusion_overnight_search_20261011/hybrid_sampler_selection_protocol.json>)。完整8cells、四个fresh N40 anchors均实际完成后，计算GEN→TRAIN均值与R7；L/W/speed≤baseline×1.15、onroad≥baseline−.02、vehicle/GT在.9..1.1须同时通过pooled及每TRAIN。只选eligible最小R≤.98，否则保留fresh40步；精确R tie依次更少steps、eps更接近训练.05、option名。full50k禁止重选。

实际已冻结推理设置 **hybrid_pos0001**（40 steps，base eps=0.05），由完整实际grid按上述预注册规则选择。相对fresh N40的模型求值次数为×1.00；这是计算次数比，不是未经实测的wall-time倍率。冻结artifact：[frozen_hybrid_sampler.json](<experiments/init_diffusion_overnight_search_20261011/frozen_hybrid_sampler.json>)。主模型配方选择保持不变。

### sampler grid / verified_complete

实际完成 8/8 cells；只有 execution rc0、native marker、checkpoint/source/scene/metric artifact SHA 与 hook metadata 全部通过才纳入。来源：[summary.json](<../logs/init_diffusion_hybrid_sampler_cohort_20261012/grid/summary.json>)。

fresh N40 anchors 已核验 4/4 cells；这是新的配对基线，没有old-main parity成功声明。

| option | TRAIN N | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_pos0001 | 2 | 0.008409 ± 0.000661 | 0.083603 ± 0.000099 | 0.002677 ± 0.000222 | 0.003898 ± 0.000152 | 0.001110 ± 0.000016 | 0.000968 ± 0.000180 | 0.328401 ± 0.061516 |
| n40_eps005 | 2 | 0.009767 ± 0.000761 | 0.084973 ± 0.001049 | 0.002808 ± 0.000145 | 0.003902 ± 0.000176 | 0.001108 ± 0.000022 | 0.000967 ± 0.000179 | 0.434581 ± 0.074259 |

六项为 raw JSD，Col% 为百分比；± 是两套复用 TRAIN 权重的 GEN 均值间 sample SD，并非各cell都对应独立训练。以下 ratio/R 均对同checkpoint、同TRAIN/GEN、本cohort重新执行的anchor。

| option | pooled R | Speed ratio | Length ratio | Width ratio | Col ratio | pooled / each TRAIN gates |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_pos0001 | 0.9364 | 1.0006 | 0.9989 | 1.0016 | 0.7557 | True / True |
| n40_eps005 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | True / True |

| option | TRAIN | R | Speed ratio | Col ratio | gates |
| --- | --- | --- | --- | --- | --- |
| hybrid_pos0001 | 4817 | 0.9336 | 1.0012 | 0.7635 | PASS |
| hybrid_pos0001 | 5817 | 0.9388 | 0.9999 | 0.7457 | PASS |
| n40_eps005 | 4817 | 1.0000 | 1.0000 | 1.0000 | PASS |
| n40_eps005 | 5817 | 1.0000 | 1.0000 | 1.0000 | PASS |

这是完整配对 grid 的描述性比较，本报告不创建 sampler selection。若另行冻结一个推理设置并做 full50k，必须引用实际完整 grid SHA；不能用未完成 grid 或事后 full50k 重选。

| option | native run 秒/cell 均值 | 范围秒 | 实测时间/本次anchor | model calls/本次anchor |
| --- | --- | --- | --- | --- |
| hybrid_pos0001 | 25.525 | 25.431–25.616 | 1.000 | 1.00 |
| n40_eps005 | 25.528 | 25.434–25.630 | 1.000 | 1.00 |

### sampler full / verified_complete

实际完成 2/2 cells；只有 execution rc0、native marker、checkpoint/source/scene/metric artifact SHA 与 hook metadata 全部通过才纳入。来源：[summary.json](<../logs/init_diffusion_hybrid_sampler_cohort_20261012/full/summary.json>)。

本native sampler driver未导出feature histogram bins；这里只能报告实际七项metrics/counts/coverage，不能据此分析新40步生成结果的stationary或尺寸尾部分布。

| TRAIN | GEN | frozen option | Near | Lat | Ang | Length | Width | Speed | Col% | onroad% | veh/GT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4817 | 2817 | hybrid_pos0001 | 0.007744 | 0.085453 | 0.001817 | 0.002790 | 0.000900 | 0.000946 | 0.372207 | 85.54 | 371030/362309 |
| 4817 | 3817 | hybrid_pos0001 | 0.008334 | 0.085695 | 0.001837 | 0.002742 | 0.000932 | 0.000935 | 0.329745 | 85.69 | 371802/362309 |

| TRAIN | GEN repeats | Near | Lat | Ang | Length | Width | Speed | Col% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4817 | 2 | 0.008039 | 0.085574 | 0.001827 | 0.002766 | 0.000916 | 0.000941 | 0.350976 |

上表是各TRAIN内两GEN的raw JSD均值；同一checkpoint的生成重复不增加独立训练N，不提供TRAIN间SD。full50k仅展示已冻结sampler的实际原生TEST metrics/counts；50,000 cache entries不是50,000个未见独立原始场景。不同GT population的4096 grid不混算ratio，也不依据full结果重新选sampler。

这次full没有重新执行本cohort的baseline full anchor，不能报告native full配对R或把历史main20当新的fresh full对照。

实际hybrid40步standalone评价配置已通过两GEN的native builder/hook/数据/EMA纯CPU compose核验：[init_diffusion_lane_conditioned_overnight_best_hybrid_sampler_eval.yaml](<../configs/experiment/init_diffusion_lane_conditioned_overnight_best_hybrid_sampler_eval.yaml>)；metadata：[selected_hybrid_sampler_native_materialization.json](<experiments/init_diffusion_overnight_search_20261011/selected_hybrid_sampler_native_materialization.json>)。以下命令采用已落地YAML的默认4817checkpoint/EMA20480，不改变训练20步设置：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC=1
python tools/run_init_diffusion_safe.py \
  experiment=init_diffusion_lane_conditioned_overnight_best_hybrid_sampler_eval
```

第二GEN可追加sampler_eval_seed=3817。未来更换checkpoint时应同步实际saved EMA update计数；没有128epoch已训练权重或bitwise replay保证。

重新生成实际分析和本报告：

```bash
python tools/overnight_search_analysis.py --figures
python tools/build_overnight_sampler_figures.py
python tools/build_overnight_search_report.py
```

该 builder 不启动 GPU、不生成 PDF、不写 selection 或 completion marker。PDF 和视觉核验由单独流程完成。

## 附录A：恢复与内存诊断（不参与性能排名）

控制台异常和 incomplete stage 保留为诊断，不能把已有部分 checkpoint 当完成 endpoint；恢复/重跑需单独记录，不将失败尝试加到模型已消费样本数。局部 GC/隔离进程是运行环境 workaround，正常独立读取失败附近文件也不能证明 SIGSEGV 的根因。

已记录 CPU probe 的真实 JSON 与 lifecycle review 已交叉核验；没有在生成报告时重跑 probe：

| CPU policy | 实际 scene loads | 未回收 local unpickler classes | 确认 class/closure cycles |
| --- | --- | --- | --- |
| disabled | 400 | 400 | 400 |
| young_after_batch | 400 | 0 | 0 |

400 次真实 scene load 中，禁用自动 GC 时残留 400 个局部 unpickler class；在 batch 后只收集 generation0 的 probe 中为 0。该 Python 对象生命周期证据不证明原训练 SIGSEGV 的根因；CPU probe 不包含完整 Lightning/GPU 训练，也不能外推为百万次 load 的 RSS 预测。

另一个实际已执行的 384-update GPU runtime pilot 已通过 source artifact SHA、terminal DONE、三次 EMA eval、实际 GC event 与 memory log 核验；它不是主搜索 candidate，也不增加 TRAIN N。记录到 3 次 generation0 collection（step128/256/384），不代表长任务已经安全。

| GPU pilot step | allocated MiB | reserved MiB / GiB | peak RSS KiB / GiB |
| --- | --- | --- | --- |
| 1 | 201.41 | 6,252.0 / 6.11 | 2,448,228 / 2.33 |
| 256 | 200.61 | 13,854.0 / 13.53 | 2,843,524 / 2.71 |

这两个实际采样点的 allocated 约 201 MiB，reserved 上升到 13,854 MiB（13.53 GiB），peak RSS 约 2.71 GiB。reserved 包括 PyTorch allocator cache，不能直接当作 live tensor 量或泄漏；两点和短 pilot 不能证明长训练内存有界，更不能证明 GC 修复了 native crash 的根因。阶段间保留的旧 framework/model cycles 仍是单独风险，后续需继续记录。

后续真实主搜索失败：`long_seed4817_separate_speed`，returncode=-11，最后已记录 update=5760；faulthandler 是否处于 Garbage-collecting：True。该失败尝试没有完成训练或最终评价，不计成绩、exposure 或 TRAIN N；实际失败说明短期 GC pilot 不能保证长训安全，也没有证明 native crash 的根因。完整原始 execution/log 的 exact SHA 审计副本保留于 [long_seed4817_separate_speed_2922fc1e1b55.execution.json](<experiments/init_diffusion_overnight_search_20261011/report_runtime_failures/long_seed4817_separate_speed_2922fc1e1b55.execution.json>) 和 [long_seed4817_separate_speed_2922fc1e1b55.log](<experiments/init_diffusion_overnight_search_20261011/report_runtime_failures/long_seed4817_separate_speed_2922fc1e1b55.log>)。恢复策略和 fresh retry 必须另行记录，不把原未完成 checkpoint 当作最终 endpoint。

这次失败独立子进程实际耗时 1005.2 秒；它是额外运行成本，没有加进成功 fit 的模型训练曝光量。

新 loader 的已有实际 CPU 记录已按 source SHA 和输出字段复核：100 个不同 native cache entries 中，Python 类型/key顺序/metadata 及 1100 个 tensor 的 dtype/shape/stride/storage-offset/bytes（含 signed zero/NaN bits）一致；wrapper class 创建1次、复用99次。来源：[cached_torch_loader_100_real_scene_equivalence.json](<experiments/init_diffusion_overnight_search_20261011/cached_torch_loader_100_real_scene_equivalence.json>)。报告自身没有重跑或导入 Torch。

另一个 no-GC CPU probe 反复读取2个实际 native caches 共 20,000 次，末端 local class=1、保留 tensor=0，RSS 471,372→472,652 KiB。来源：[gc_off_20k_cached_actual_scene_load_probe.json](<experiments/init_diffusion_overnight_search_20261011/gc_off_20k_cached_actual_scene_load_probe.json>)。这不包含 PyG/Lightning/model/TensorFlow forward，既不是20,000个独立场景，也不证明完整 GPU 长训内存有界或解决 SIGSEGV 根因。

上述 CPU/GPU memory diagnostics 全部排除在性能表、R、gates、训练种子统计及模型选择之外。报告 builder 只读已有记录，自身不运行 GPU。

## 附录B：EMA与长训练运行边界

实际已完成 control checkpoint 的 global step / EMA n / AdamW step 均为 10,240；EMA nominal cap=.9999，当前解析 decay=.9991219512，尚未 cap。实际保存 LR=0.0004928824，是 initial LR 的 98.5765%。128epoch 的低 LR 阶段尚未观测。证据：[ema_optimizer_128epoch_transfer_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/ema_optimizer_128epoch_transfer_cpu_review.json>)。

| EMA kernel 状态 | B | n | 平均迭代年龄（场景曝光） | 最近95%系数质量（场景曝光） |
| --- | --- | --- | --- | --- |
| 实际 control n，解析 kernel | 512 | 10240 | 524315 | 1485234 |
| 解析 long B256 kernel | 256 | 20480 | 524290 | 1484978 |
| 未来128epoch解析 kernel | 512 | 123392 | 5101071 | 15336640 |
| 未来128epoch解析 kernel | 256 | 246784 | 2559596 | 7668320 |

当前 long 等曝光下 adaptive EMA 的场景单位平均迭代年龄几乎相同，不能仅凭 B256 的更新数翻倍断言窗口更长。cap 首次在 n=89,990 生效，B512≈93.35场景epoch、B256≈46.68；cap后同 per-update decay 对应B256约半个场景窗口。表中 kernel 是相关参数迭代的解析系数历史，既不是原始数据权重/独立N，也不是128epoch指标预测；B256 long 项未由这份 CPU review 读取候选 checkpoint。

cached loader 消除了已测的 Unpickler 类增长源，但生产训练 RSS 并未因此得到完整有界证明。以下是日志中确实存在的 peak RSS 样本；当前训练次数/内存诊断不增加性能 evidence N。

| 实际 step | peak RSS GiB |
| --- | --- |
| 1 | 2.350 |
| 14336 | 4.261 |
| 19456 | 4.771 |

128epoch运行可行性尚未验证，不做线性内存外推。已通过 CPU 合同测试的 native launcher 默认监控自有 child RSS 18 GiB / host MemAvailable 1.5 GiB，每2秒采样；超限停止并记录实际原因，但不保证checkpoint已保存或精确续训。RSS不汇总worker子树，短峰可能发生在采样之间。证据：[native_guard_long_memory_boundary_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/native_guard_long_memory_boundary_cpu_review.json>)。

## 证据索引

- [search_protocol.json](<experiments/init_diffusion_overnight_search_20261011/search_protocol.json>)
- [selection_protocol_review.json](<experiments/init_diffusion_overnight_search_20261011/selection_protocol_review.json>)
- [source_freeze.json](<experiments/init_diffusion_overnight_search_20261011/source_freeze.json>)
- [current_protocol_units_scope_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/current_protocol_units_scope_cpu_review.json>)
- [current_protocol_units_scope_cpu_review.md](<experiments/init_diffusion_overnight_search_20261011/current_protocol_units_scope_cpu_review.md>)
- [time_weighting_direction_cpu_audit.json](<experiments/init_diffusion_overnight_search_20261011/time_weighting_direction_cpu_audit.json>)
- [current_long_budget_independent_review.json](<experiments/init_diffusion_overnight_search_20261011/current_long_budget_independent_review.json>)
- [screen_control_pre_deadline_revision_interruption.json](<experiments/init_diffusion_overnight_search_20261011/screen_control_pre_deadline_revision_interruption.json>)
- [data_metric_review.md](<experiments/init_diffusion_overnight_search_20261011/data_metric_review.md>)
- [data_metric_review.json](<experiments/init_diffusion_overnight_search_20261011/data_metric_review.json>)
- [design_model.md](<experiments/init_diffusion_overnight_search_20261011/design_model.md>)
- [lane_distance_design.md](<experiments/init_diffusion_overnight_search_20261011/lane_distance_design.md>)
- [map_condition_types_cpu_audit.json](<experiments/init_diffusion_overnight_search_20261011/map_condition_types_cpu_audit.json>)
- [default_compatibility_cpu_audit.json](<experiments/init_diffusion_overnight_search_20261011/default_compatibility_cpu_audit.json>)
- [controller_cpu_audit.json](<experiments/init_diffusion_overnight_search_20261011/controller_cpu_audit.json>)
- [runtime_failure_handoff_cpu_audit.json](<experiments/init_diffusion_overnight_search_20261011/runtime_failure_handoff_cpu_audit.json>)
- [gc_object_lifecycle_cpu_review.md](<experiments/init_diffusion_overnight_search_20261011/gc_object_lifecycle_cpu_review.md>)
- [gc_object_lifecycle_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/gc_object_lifecycle_cpu_review.json>)
- [gc_disabled_two_scene_cpu_probe.json](<experiments/init_diffusion_overnight_search_20261011/gc_disabled_two_scene_cpu_probe.json>)
- [gc_young_two_scene_cpu_probe.json](<experiments/init_diffusion_overnight_search_20261011/gc_young_two_scene_cpu_probe.json>)
- [runtime_gc_gpu_pilot_review.json](<experiments/init_diffusion_overnight_search_20261011/runtime_gc_gpu_pilot_review.json>)
- [runtime_gc_pilot_execution.json](<experiments/init_diffusion_overnight_search_20261011/runtime_gc_pilot_execution.json>)
- [report_cpu_transient_exit139.json](<experiments/init_diffusion_overnight_search_20261011/report_cpu_transient_exit139.json>)
- [report_watcher_session.json](<experiments/init_diffusion_overnight_search_20261011/report_watcher_session.json>)
- [report_watcher.json](<experiments/init_diffusion_overnight_search_20261011/report_watcher.json>)
- [medium_protocol.json](<experiments/init_diffusion_overnight_search_20261011/medium_protocol.json>)
- [frozen_long_protocol.json](<experiments/init_diffusion_overnight_search_20261011/frozen_long_protocol.json>)
- [final_selection.json](<experiments/init_diffusion_overnight_search_20261011/final_selection.json>)
- [full50000_audit.json](<experiments/init_diffusion_overnight_search_20261011/full50000_audit.json>)
- [queue_complete.json](<experiments/init_diffusion_overnight_search_20261011/queue_complete.json>)
- [native_best_materialization.json](<experiments/init_diffusion_overnight_search_20261011/native_best_materialization.json>)
- [summary.json](<experiments/init_diffusion_overnight_search_20261011/analysis/summary.json>)
- [actual_observations.csv](<experiments/init_diffusion_overnight_search_20261011/analysis/actual_observations.csv>)
- [metadata.json](<experiments/init_diffusion_overnight_search_20261011/analysis/figures/metadata.json>)
- [metadata.json](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/metadata.json>)
- [report_runtime_failures.json](<experiments/init_diffusion_overnight_search_20261011/report_runtime_failures.json>)
- [sampler_report_appendix_plan.md](<experiments/init_diffusion_overnight_search_20261011/sampler_report_appendix_plan.md>)
- [protocol.json](<experiments/init_diffusion_overnight_search_20261011/runtime_recovery/protocol.json>)
- [events.json](<experiments/init_diffusion_overnight_search_20261011/runtime_recovery/events.json>)
- [archive_ledger.json](<experiments/init_diffusion_overnight_search_20261011/runtime_recovery/attempt1/archive_ledger.json>)
- [first_paired_TRAIN4817_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/runtime_recovery/first_paired_TRAIN4817_cpu_review.json>)
- [verification_bridge.json](<experiments/init_diffusion_overnight_search_20261011/runtime_recovery/verification_bridge.json>)
- [cached_torch_loader_100_real_scene_equivalence.json](<experiments/init_diffusion_overnight_search_20261011/cached_torch_loader_100_real_scene_equivalence.json>)
- [gc_off_20k_cached_actual_scene_load_probe.json](<experiments/init_diffusion_overnight_search_20261011/gc_off_20k_cached_actual_scene_load_probe.json>)
- [runtime_young_gc_failure_cached_loader_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/runtime_young_gc_failure_cached_loader_cpu_review.json>)
- [analysis_runtime_cohort_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/analysis_runtime_cohort_cpu_review.json>)
- [ema_optimizer_128epoch_transfer_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/ema_optimizer_128epoch_transfer_cpu_review.json>)
- [ema_optimizer_128epoch_transfer_cpu_review.md](<experiments/init_diffusion_overnight_search_20261011/ema_optimizer_128epoch_transfer_cpu_review.md>)
- [long_control_10240_optimizer_ema_metadata_cpu.json](<experiments/init_diffusion_overnight_search_20261011/long_control_10240_optimizer_ema_metadata_cpu.json>)
- [native_guard_long_memory_boundary_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/native_guard_long_memory_boundary_cpu_review.json>)
- [native_guard_long_memory_boundary_cpu_review.md](<experiments/init_diffusion_overnight_search_20261011/native_guard_long_memory_boundary_cpu_review.md>)
- [sampler_runtime_parity_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/sampler_runtime_parity_cpu_review.json>)
- [sampler_inference_driver_review.json](<experiments/init_diffusion_overnight_search_20261011/sampler_inference_driver_review.json>)
- [sampler_pre_run_parity_diagnostic_review.json](<experiments/init_diffusion_overnight_search_20261011/sampler_pre_run_parity_diagnostic_review.json>)
- [sampler_pre_run_parity_diagnostic_review.md](<experiments/init_diffusion_overnight_search_20261011/sampler_pre_run_parity_diagnostic_review.md>)
- [sampler_selection_protocol.json](<experiments/init_diffusion_overnight_search_20261011/sampler_selection_protocol.json>)
- [native_sampler_selection_protocol.json](<experiments/init_diffusion_overnight_search_20261011/native_sampler_selection_protocol.json>)
- [frozen_native_sampler.json](<experiments/init_diffusion_overnight_search_20261011/frozen_native_sampler.json>)
- [actual_native_default20_parity_failure_review.json](<experiments/init_diffusion_overnight_search_20261011/actual_native_default20_parity_failure_review.json>)
- [sampler_numerical_repeatability_independent_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/sampler_numerical_repeatability_independent_cpu_review.json>)
- [final_actual_tradeoff_independent_cpu.json](<experiments/init_diffusion_overnight_search_20261011/final_actual_tradeoff_independent_cpu.json>)
- [actual_final_population_scope_review.json](<experiments/init_diffusion_overnight_search_20261011/actual_final_population_scope_review.json>)
- [actual_final_histogram_coverage_review.json](<experiments/init_diffusion_overnight_search_20261011/actual_final_histogram_coverage_review.json>)
- [hybrid_sampler_selection_protocol.json](<experiments/init_diffusion_overnight_search_20261011/hybrid_sampler_selection_protocol.json>)
- [frozen_hybrid_sampler.json](<experiments/init_diffusion_overnight_search_20261011/frozen_hybrid_sampler.json>)
- [native_sampler_protocol_independent_cpu_review.json](<experiments/init_diffusion_overnight_search_20261011/native_sampler_protocol_independent_cpu_review.json>)
- [selected_sampler_native_materialization.json](<experiments/init_diffusion_overnight_search_20261011/selected_sampler_native_materialization.json>)
- [actual_selected_native_full_scope_review.json](<experiments/init_diffusion_overnight_search_20261011/actual_selected_native_full_scope_review.json>)
- [selected_hybrid_sampler_native_materialization.json](<experiments/init_diffusion_overnight_search_20261011/selected_hybrid_sampler_native_materialization.json>)
- [actual_selected_hybrid_native_full_scope_review.json](<experiments/init_diffusion_overnight_search_20261011/actual_selected_hybrid_native_full_scope_review.json>)
- [sampler_tradeoffs_actual.metadata.json](<experiments/init_diffusion_overnight_search_20261011/analysis/report_figures/sampler_tradeoffs_actual.metadata.json>)
