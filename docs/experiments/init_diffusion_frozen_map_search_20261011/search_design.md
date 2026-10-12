# InitDiffusion 固定预训练 map encoder 搜索设计

本文件是搜索前的只读源码与历史实测分析，不包含本轮 GPU 训练结果。用户授权从 `src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt` 提取 map encoder，固定地图网络，并允许调其他参数。此次应把 G1、optimizer、scheduler、EMA 从头初始化，不加载旧 initializer 或 policy 权重。

## 为什么值得重新搜索

上一轮随机 map + G1 的 37 个 loss 配方中，`geo_mse32_col20_heading_l1_shape2` 在 B512、LR5e-4、EMA2048 的三 TRAIN seed 上比 `warm_winner` 的综合 S 低 3.41%。收益主要来自 angle 与 collision，nearest/lateral/width/speed 均在 3/3 seeds 退步；shape 与 speed 不构成全指标改善。1536→2048 更新中，旧配方综合 S 改善更快，尚未证明完整 128 epochs 的排名。

固定已经训练的 map 特征，既减少早期条件漂移，也取消 map 参数分担共享 clip0.5 的影响。旧 loss 的排序不能直接迁移：从本轮两个控制配方重新开始，才可判断哪些结果源于更好的地图条件，哪些源于 loss。固定 map 不保证条件 on-road 接近 GT，地图语义与初始化任务仍可能不匹配。

## 首要审计与公平规则

1. map keys 必须严格按当前 `encoder.map_encoder` 的名称、shape 和 dtype 校验，不能用全模型 `strict=False` 静默漏载。记录 checkpoint 文件 SHA256、提取 keys/参数 hash、训练前后 map hash；参数 `requires_grad=False`、无 grad、Adam state 不含 map。map 模块保持 eval，包括 dropout/BatchNorm 等可变行为；先确认该 encoder 的实际层结构。
2. 固定的是权重与运行行为；地图 token 输入、裁剪范围仍是模型条件的一部分。主搜索沿用 square50 与当前 map token 文件。核验预训练 checkpoint 对应 encoder 配置，若需结构变化，先做兼容检查；不能同时改 map 结构又声称用了同一固定 encoder。
3. G1 fresh、未使用旧 G1 EMA，agent policy 冻结；每个 TRAIN seed 的同结构候选保持相同初始 G1 hash、完整训练 manifest 与 batch 顺序。结构改动会改变 RNG 消耗，须记录公共参数初始化是否相同，不能声称结构变体的全网初始化相同。
4. 直接复用 TRAIN-only 的 8192-scene scalar-speed/linear-size normalizer。log-size 或 vector-velocity 必须在**同一 TRAIN calibration manifest** 上重新拟合所需表征，保存维度/单位/hash；不能把 linear 长宽的 mean/std 用于 log 长宽。
5. B512 主搜索、32-true、matmul medium、clip0.5、AdamW 默认、128 real-epoch cosine 作为共同条件。全 493542 场景 seeded manifest 连续迭代，跨 epoch 重复完整集合；不可反复训练第一小批样本。续训保留 optimizer/scheduler/EMA/RNG，恢复真正未消费样本，不按 `step*B` 忽略 epoch 末尾 partial batch。
6. 评价与原先比较保持 matched GT、生成 agent type、固定 ego position/heading/shape/type、只生成 ego speed、reference map、20 sampler steps、实际 on-road 与 vehicle-count 计数。不要把 SD/VW 的不同输入条件混入本轮排名。

## 24 个探索 fresh 模型，再加 4 个独立 seed 确认

默认 `B512, LR5e-4, uniform per-scene t, adaptive G1 EMA cap0.9999`。每个 fresh 模型保存 128 和 512 updates，128 只检查失稳和明显坏样本；**不因 early JSD 差淘汰**。512 作为首轮筛选，至少保留 4 个多指标候选继续到 1024/2048。固定 map 后预计比随机地图阶段更快，但任何长期判断均来自延长曲线，不来自一个 128-step 点。

### 第一阶段：14 个 loss 控制或单因素

除 Old control 外均从 New control 出发，只改变表中项。未列出项保持 New：position MSE/p1/w32；heading wrapped-angle L1/p1/w1；length/width MSE/p1/w2；speed MSE/p1/w2；type CE/p0/w1；collision L1/p1/w20；所有分量的 inverse-time weight 按 Uniform(t) 解析期望归一化。

| 名称 | 相对 New 的改动 | 用途 |
|---|---|---|
| old_control | `warm_winner` 完整旧 recipe：position L1/p1/w4，heading angle MSE/p2/w1，L MSE/p1/w1，W MSE/p2/w1，speed MSE/p1/w2，collision L1/p1/w5 | 固定 map 下旧配方基线 |
| new_control | 无 | 之前 scratch 选中配方的迁移基线 |
| col10 | collision w10 | 缓解几何项对 speed/shape 的竞争 |
| col5 | collision w5 | 检验碰撞收益的代价曲线 |
| shape4 | length、width w4 | 增加细窄尺寸直方图监督 |
| speed4 | scalar speed w4 | 检验低速峰不足是否仅权重问题 |
| width4 | 只 width w4 | width 较大退步的单分量定位 |
| heading_mse1 | heading angle MSE/p1/w1 | 固定 map 后是否仍需要 L1 |
| heading_p0 | heading L1/p0/w1 | 提升高噪声全局 heading 约束 |
| heading_p2 | heading L1/p2/w1 | 精细低噪声修正，与上一轮结果对照 |
| shape_p2 | length、width p2/w2 | shape 精细低噪声学习 |
| speed_p0 | speed MSE/p0/w2 | 降低 x0 的小 t 强权重，强化噪声起点的速度条件预测 |
| pos16 | position MSE/p1/w16 | 检验位置权重与 clip 竞争 |
| pos_l1_16 | position L1/p1/w16 | 检验位置鲁棒性；须披露 loss 幅度改变 |

优先延长 Old、New、最佳 coverage/Pareto 候选及最佳 shape/speed 候选。若同名候选重合，用下一 Pareto 候选补足 4 组。不能把某项 JSD 很差但 angle 很好的模型全部按 S 提前晋级，也不能以 vehicle 数减少而碰撞下降视为成功。

### 第二阶段：8 个参数或表示单因素

第一阶段确定一个暂时可用的 loss anchor 后，**从 fresh G1** 训练下列 8 组。不是从其 512-step checkpoint 微调。表示/结构变体仅更改一项；其余条件固定。

| 名称 | 唯一变化 | 注意事项 |
|---|---|---|
| lr2e4 | LR2e-4 | 固定 map 可使高 LR 不再适合；和 anchor5e-4 对照 |
| lr1e3 | LR1e-3 | 允许更快的 fresh G1 学习，但看梯度与后期波动 |
| batch256 | B256、LR仍5e-4 | 在**同训练 exposure**比较：主组512-step对应本组1024-step，2048对应4096；报告 optimizer 更新数差异 |
| shape_uniform | shape source U(-sqrt3,sqrt3)，归一化坐标 | bounded prior 是否帮助窄尺寸分布；Gaussian anchor 仍保留 |
| speed_uniform | speed source 同标准化 uniform | 改变负源比例与尾部；生成非负 clamp 的边界质量须实际计数 |
| size_log | 全路径 log(length/width) | 新 TRAIN-only log normalizer；物理输出与评价不改变 |
| edge_mlp | denoiser edge embedding=mlp | 改 G1 相对几何表示，不改固定 map encoder FourierEmbedding |
| count_embedding | count_embedding_type=scenario_dreamer | 加当前输入已知 agent slot 与 map token count；count_lane_source保持map_tokens |

第二阶段先用512-equivalent exposure评价512-search场景，再把最多两个明确改进、无覆盖/数量退步的单因素延长。若各单因素无稳定改进，不强行组合。

### 第三阶段：最多 2 个有依据的组合，最终 2 臂 N=3

最多将上述阶段中**已经通过较长预算**的 2 个兼容改动合并， fresh 起点检查组合收益，共计最多24个探索 fresh 模型。不要同时改变 loss、LR、表示、embedding 后失去归因。

冻结最终候选与控制两个配方，TRAIN seeds817/1817/2817，GEN seeds2817/3817。seed817保留/延长到2048与4096；两个额外TRAIN seeds各自两个配方 fresh，因此总 fresh 上限28。主要终点EMA4096；EMA2048用于趋势，online仅辅助，不根据结果切换权重模式或终点。

## 排名、门槛与长期预测

上一轮辅助分数 S=mean(native metric/[1,1,10,20,15,20,5]) 中，lateral 的 native值约0.9、占比偏大，而 speed约0.1→除20只有0.005，尺寸也较小；它容许以较大的 speed/width 相对退步换取 angle/collision 收益。继续保留 S 作为跨轮描述，但**不把它当唯一推荐标准**。

建议本轮在模型结果出来之前固定以下选择方式：

- 排除 V/GT不在[0.9,1.1]，以及 on-road比固定-map Old control低2个百分点以上的候选；同时报告与GT约99.4%的绝对覆盖差。已有门槛只防止退步，不能证明覆盖足够。
- 首先列七指标+coverage的 Pareto 前沿。三训练 seed 先平均 GEN，再 TRAIN等权；SD单位是TRAIN seed，不能把6次生成当N6。
- 用最终确认 Old control 的量级计算固定相对 composite `R = mean(metric/control_metric)`，七项等权。control_metric设一个预先指定小正floor防止接近零不稳定；同时给S和逐项比值。若要强调尺寸速度，可另报告 R_attr=mean(L/oldL,W/oldW,speed/oldSpeed)，不随结果调整权重。
- 推荐默认“综合最优”满足R<1、coverage/count门槛，并且 L/W/speed 各相对控制退步不超过15%；找不到则诚实保留两套推荐：综合/安全与尺寸速度。15%是预先设定实用容忍度，不是统计显著性。
- 不对512→4096短期曲线线性外推128 epochs。至少检查最后三个预算点、独立seed一致性、单项是否仍剧烈改善、ranking是否反转。即便4096稳定，也称“已测试中长期前景最好”，不称证明完整训练全局最优。

heldout4096若沿用上一轮集合，应叫 validation confirmation，不是未触碰test；若另选新集合，排除旧search1024/heldout4096的raw scenario key，并记录唯一rawkeys/hash/与TRAIN scenarioID交叉。过去50k全集合聚合审计已经覆盖整个缓存，所以也不能宣称从未见过该分布。50000完整评价放在最后两臂，不用其重选模型。

## 暂时不做或需新实现的参数

- **G1 num_freq_bands/容量的伪扫描**：`InitDenoiser.__init__` 将 num_freq_bands=hidden_dim//2；`InitDiffusion._make_args()` 写死 G1 hidden256、3layers、8heads。decoder.num_freq_bands不是当前G1频带控制。固定 map 后更不能改 map频带维度再加载同一权重。要调G1容量必须先显式接通参数并测试。
- **非均匀 time sampler**：`Flow._sample_time`当前只有per-sceneUniform。Beta/logit-normal若改变采样分布，会使 `ComponentLoss.time_weight` 内Uniform解析期望不再等于实际均值；若想保留同一目标，需importance权重 q(t)^-1，端点密度/clip与方差须说明。更易审计的是直接现有p0/p1/p2，所以先不加sampler结构。
- **Hungarian修改**：当前按GT-type、scene分组，非ego源噪声按完整归一化状态平方距离配对，heading两列是cos/sin经各自std归一化的弦距，不是角度测地距离。统一heading标准差与圆周cost可能更合适，但改变source↔target耦合后训练目标样本本身改变，需要单独实验。不要把`type_match_weight`当当前separate模式的有效超参：它仅joint非feature路径有效。
- **angular_velocity objective、joint type、stationary branch**：用户允许其他参数，但本轮主目标是可归因的x0 initializer改进。它们改变训练目标/维度/标签处理，留作下一独立结构阶段，不和固定map损失筛选同时引入。
- **sigma_h**：本轮heading_noise=circular，sigma_h不生效，不能将改变其数值记录为实际候选。
- **ego_context sincos**：源码支持，但当前固定ego姿态条件与相对theta输入的分支仍可能有±π跳变。它是合理独立候选；本轮8个参数位若不做count embedding，可替换为sincos，并保持freshG1、输出条件与loss相同。

## 应记录的额外诊断

除七metric、coverage、counts、耗时、hash外，保存 speed精确0比例、[0,1)bin比例、负scalar输出clamp数量、物理shape<=0与输出极端值比例。hist midpoint均值不能替代原始moments。每分量weighted/raw loss、t分桶重建误差、preclip总grad、按position/heading/shape/speed单项grad份额可帮助解释；只记录总grad不足以证明哪个分量抢占梯度。

当前 collision 训练对预测shape `.detach()`，不会直接通过尺寸分支反向鼓励缩车；但forward碰撞值、采样车辆类型和共享网络仍影响报告。尺寸速度变差可能来自x0条件平均、高噪声源混合、不同type共用TRAINstd以及clip后的共享更新方向，需要上述单因素证据，不能仅凭一个loss权重推断因果。

## 关键本地证据

- `docs/init_diffusion_scratch_loss_search_20261011.md`：实际37recipe、三seed2048与6144趋势、50k结果。
- `docs/experiments/init_diffusion_scratch_loss_search_20261011/final_deployment_selection_protocol.json`：旧冻结score/gates。
- `tools/scratch_component_loss_sweep.py`：fullmanifest、延长训练、online/EMA评价、训练样本与RNG来源。
- `src/smart/diffusion/scale_flow.py`：source、per-scene t、source-target matcher及component loss。
- `src/smart/diffusion/denoiser.py`：logsize全路径、edge embedding、hardcoded频带。
- `src/smart/diffusion/initial_diffusion.py`：hardcodedG1容量、EMA、representation/normalizercheckpoint检查。

