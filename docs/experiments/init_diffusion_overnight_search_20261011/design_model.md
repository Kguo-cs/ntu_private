# 固定预训练地图的 InitDiffusion 夜间模型实验设计

日期：2026-10-11。截止时间：2026-10-12 10:00 北京时间（02:00 UTC）。本文件记录 CPU 源码复核、已完成实验的证据和待测假设；新选项尚未经过 GPU 训练，不代表它们改善了指标。GPU 实验由 root 的串行队列执行。

## 已有结果给出的约束

评价保留原 lane-conditioned 条件：参考地图、GT ego 坐标系、固定 ego 的位置/heading/尺寸/type、输入 agent 总数和 ego 槽位；ego 的 speed 与其他车辆属性仍生成。固定 AIRL map encoder，仅 G1、optimizer、scheduler 和 EMA 从头开始。保持 TRAIN normalizer、匹配后的评价 reference 和官方 helper 不变。

下表是实际长训练的 endpoint EMA 结果，每个训练种子先平均 GEN 2817/3817 的 4096 场景。所有数值使用 helper 原生单位：lat JSD ×10，length/width/speed JSD ×100，collision 为百分比。

| TRAIN seed / arm | lat | length | width | speed | collision |
| --- | ---: | ---: | ---: | ---: | ---: |
| 817 / control，B512 ×4096 updates | 0.913912 | 0.618442 | 0.317088 | 0.090875 | 1.251773 |
| 817 / combo，B256 ×8192 updates | 0.908401 | 0.480569 | 0.221688 | 0.100842 | 0.930521 |
| 1817 / control，B512 ×4096 updates | 0.907444 | 0.797407 | 0.363201 | 0.087231 | 1.334101 |
| 1817 / combo，B256 ×8192 updates | 0.912466 | 0.589930 | 0.321925 | 0.118978 | 1.038081 |

这些 endpoint 的实际 scene exposure 都是 2,097,048。combo 同时改了 batch、ego sin/cos、尺寸 time power 2 和 speed weight 4，所以不能把结果归因于一个参数。它的 speed 相对 control 分别恶化 10.97% 和 36.39%；N=1 的短期速度优势不足以预测长期优势。1817 的 speed gate 已失败，其他维度改善不能掩盖这项退化。2817 的完整最终确认在本文件写入时尚未齐备，不提前命名最终 winner。

证据文件：[817 长训练](../init_diffusion_frozen_map_search_20261011/long_three_arm_forecast_review.json)、[1817 独立配对](../init_diffusion_frozen_map_search_20261011/confirmation_seed1817_pair_review.json)、[训练后分布](../init_diffusion_frozen_map_search_20261011/long_baseline_distribution_review.json)、[短期阶段](../init_diffusion_frozen_map_search_20261011/phase2_selection.json)。

TRAIN817 control 的 histogram 还显示：speed [0,1) m/s 的 GT 比例 41.8378%，生成 43.4944%，已经过量；[15,20) 从 GT 7.0240% 降到生成 6.0266%。width [2.0,2.2) 从 GT 57.0880% 升到生成 62.2540%，长/宽尾部不足。1 m/s 的 histogram 不能证明 exact-zero 或 clamp 前负值的数量。因此暂不加 stationary 分支，也不把所有问题解释为负 speed 被 clamp。

## 当前实现的具体机制

当前标量速度状态是 `[x,y,cos(theta),sin(theta),length,width,speed]`，不是 vx/vy。speed 的 x0 loss 已直接监督速度 magnitude；vector 版本的 magnitude auxiliary 对它重复，现有代码会拒绝该组合。

1. `InitDenoiser._original_state_embedding` 原来把物理 length/width/speed 直接送进 `proj_in_m_delta`。noise 和 loss 已使用 TRAIN mean/std，但模型输入投影没有同步标准化。width 与 speed 的尺度不同，输出共享 trunk/head，存在条件数和梯度耦合的可能；这是源码机制，是否影响指标仍需实验。
2. 原 speed x0 head 可输出负值，训练不 clamp；最后 `get_output` 才 `clamp_min(0)` 并沿生成 heading 重建 velocity。最终 clamp 会把负输出压成零，但 coarse histogram 尚不能量化这种贡献。
3. Hungarian 在每个 scene 的 GT type 组内匹配 source 整行，ego 单独配对；现有标准化代价共同包含位置、cos/sin、尺寸和速度。重调速度匹配权重改变 coupling，可能同时改变几何和速度目标，不能与 speed head 同时首次引入。
4. t=0 为 data、t=1 为 noise，每场景共享随机 t。反推 Euclidean velocity 为 `(latent-x0)/clamp(t,0.05)`；高 noise 的 x0 回归容易趋于条件均值，使速度/尺寸主峰更集中。增大 loss 权重并不能自动恢复尾部。
5. 当前 p=1 时间权重会把约 25.03% 的归一化 loss 质量放在 t<0.05，而 t>0.5 仅约 17.35%。p=2 对应约 51.28% 和 2.56%；p=3 为 66.72% 和 0.25%。这里是 uniform t、`max(t,0.05)^(-p)` 和现有解析归一化下的质量，不是实际每个分量的梯度比例。尺寸 p2 的成功不意味着速度也应采用更大的 p。

## 五个单因素模型候选

用最终真实选择的参考配置做 anchor；selection 尚未完成前可用明确标注的 control/combo 诊断 anchor，不称为 best。每个 arm 只覆盖下列一个选项，其他 loss、LR、batch、TRAIN normalizer、source noise、匹配、条件、采样步数都保持 anchor 一致。可先运行前两项，第三/四项用于验证具体速度机制，第五项在成本允许时进行。

| 优先级 / arm | 精确配置覆盖 | 要验证的机制 | 主要风险与诊断 |
| --- | --- | --- | --- |
| 1 / feature_train | `state_feature_normalization: train` | L/W/speed 输入使用保存的 TRAIN mean/std，缓解量纲差异；几何位置、角度、map 边仍保持物理单位 | 可能削弱 pretrained shared input 对物理尺度的表达；看 speed/shape 同时变化与 geometry gates |
| 2 / separate_speed | `speed_head_kind: separate`，`speed_head_detach_context: false` | 给速度独立 MLP 输出容量，其他 state/type 输出不变；共享 trunk 仍能收到 speed 梯度 | 额外参数不等于解除 trunk 的梯度竞争；观察 speed、type population、lat/collision |
| 3 / residual_speed | `speed_x0_parameterization: time_residual` | 用 `x0_s=s_t+t*r_s` 学从当前 noisy speed 到 clean speed 的修正；低 t 有自然恒等路径 | 目标仍是 x0，但 head 的梯度多了 t 因子，等效时间权重改变；不可宣称只是无影响的重参数化 |
| 4 / positive_speed | `speed_x0_parameterization: softplus` | x0 在训练、Flow 转换和采样全过程非负，检验只在最后 clamp 的错配 | softplus 接近零时梯度弱，可能更难学零/低速，或丢失 moving tail；已有低速过量使该候选不能预设为最优 |
| 5 / capacity384 | `denoiser_hidden_dim: 384`，层数保持 3 | 检验共享表示/尾部条件容量不足，避免把更大 loss weight 当唯一手段 | attention/MLP 成本增加；需要同 exposure 比较并记录 wall time，可能欠训或破坏短期排序 |

`speed_head_detach_context: true` 是 separate_speed 若牺牲几何时的后续诊断，而不是与 separate 首次实验合并：它阻断 speed loss 回到共享特征的梯度，但位置/尺寸等 loss 仍更新 trunk，也可能导致 speed 欠拟合。不要首轮同时叠加 normalization、separate、softplus、residual。

旧 InitDiffusion 外层 `hidden_dim=128` 不能用来控制 G1：旧 `_make_args` 给 denoiser hardcode hidden256/layers3，map encoder 的 hidden128 是另一模块。新增 `denoiser_hidden_dim` 和 `denoiser_num_layers` 已由 root 透传，只有这些显式选项才控制 G1。384 需保持 head 数可整除；固定 map128 通过 `lane_embed` 映射到 denoiser hidden。

## 与 root 负责的 transport 实验如何组合

先与上述模型候选分开比较，避免无法归因。值得优先筛选的是 speed 的 `time_power: 0`（weight 保持原值），或 `time_sample_power: 0.5` 的 high-noise 覆盖；后者当前实现是 `t=U^power`，没有 importance correction，因此改变了有效训练目标/归一化期望，不是仅减少 Monte Carlo 方差。p0 与采样同时改变会失去单因素解释。

其次可以小权重 `distribution_loss`，仅在 high-noise 子集、按 GT type 的 minibatch 中比较 clean prediction/target 排序值，直接针对边缘分布收缩。它新增辅助目标，不等价于逐 agent x0 loss，也没有保证与位置/heading 的联合分布正确；需要 shape/collision gates。推荐先 speed-only，再必要时加 size，不与更高 speed weight 同时引入。

匹配权重属于第三类独立 transport intervention：例如降低速度对全行 Hungarian 的牵制，再观察 speed 与几何共同变化。不要一次同时换 source、改匹配和改 head。已测 speed uniform 在短期没有给出可靠改善；目前没有依据把它作为默认。

### 暂缓增加采样步数

默认 20 步的最后 t=0.05，等于 t_eps。40/80 步末端 t<0.05，Euclidean velocity 仍除以 clamp(t,0.05)，圆周 heading 却使用真实 t；因此更密网格不是纯 ODE 精度提升。恒定理想 x0=0、source=1 的 40 步 Euclidean update 仍留下约 0.0125 的残余。先保留 `sampling_steps: 20`；若改 solver 或 terminal x0，需要 opt-in 独立实验和 CPU endpoint 测试，不把步数变化解读为等价 sampler。

## 公平筛选和保留标准

所有模型 arm 均 fresh G1/optimizer/EMA，只加载同一固定预训练 map 和 TRAIN normalizer，不从旧 G1 微调。使用完整 TRAIN iterator 连续遍历；记录实际 consumed scenes，不能只反复使用前几 batch。batch 相同时按 updates 对齐；B256/B512 时按实际 exposure 对齐并明确 optimizer/EMA updates 不同。

推荐先在固定 discovery TRAIN seed 做 1024/2048 update screen（若 B512，则与 B256 对齐 exposure），每个 endpoint 使用同一 1024 场景、两个配对 GEN seeds；保留改善随 budget 持续的 arm，再用完整 4096 确认 cohort。shortlist 至少延伸到与旧 endpoint 相同的 2,097,048 scenes；优先在独立 TRAIN seed 再确认，而非把重复 GEN 当更多训练模型。受截止时间约束，实际无法完成的确认明确列为 pending，不称 128-epoch 最优。

选择时先平均同一训练模型的两个 GEN，再计算 candidate/control 逐 metric 比值。保留七指标 R 和属性 R_attr，但同时查看单指标 gates：优先要求 speed 不恶化，并保留原 geometry/collision/onroad gates；若总 R 改善但 speed 或安全/几何 gate 失败，不推荐作为综合最佳。报告 speed 每个 1 m/s bin、moving tail、shape tail 和生成 vehicle/type 数量；更低 vehicle 数可能使 collision 下降，并不证明布局改善。

建议额外诊断 clamp 前负 speed 比例、exact-zero/near-zero 比例和 GT/generated vehicle population，当前已有粗 histogram 不够回答该机制。诊断不要改变 official metric/helper、reference lane 范围或主选择 cohort。

## 新发现：地图输出缺少 lane query

进一步复核确认：固定 SMART map encoder 的 attention source 包括 centerline，但其 returned/query 默认只有 type4/5 road edges。因此 InitDiffusion 接收到的显式 map position/orientation 是边界而非 lane centerline。现已加入默认关闭的 `map_condition_types: lanes_and_edges`，只扩展返回 query 为 `{0,1,2,4,5}`，source/crop/radius/k/66 权重不变。root 将增加 laneedges-only 与 laneedges+TRAIN feature normalization 两臂；这是比增加模型容量更直接的结构假设。

实际 TRAIN32 CPU 检查 default 旧/新 map 输出 bitwise 一致，source 4211 不变，query 1689→3203，66 frozen tensors 与 fresh G1 初始化 hash 一致。中心线辅助 loss 暂缓：当前缓存的稀疏 5 m 片段有纵向缺口，无原 polyline ID/full sd_map，naive finite-segment 距离不能充当 lateral distance。见 [专项设计](lane_distance_design.md) 和 [真实 batch 审计](map_condition_types_cpu_audit.json)。

## 已实现选项与 CPU 兼容性证据

denoiser 的四个新选项默认分别是 `none / identity / shared / false`。默认无新增参数和 RNG draw，identity 返回原 prediction 对象。独立 speed head 在 legacy 权重初始化之后创建，不扰动同 seed 的 shared 权重。TRAIN normalization 只改网络输入特征，不改 physical source、loss target、Hungarian 或地图坐标。speed 参数化在 x0 进入 Flow 之前应用，完整路径一致。

CPU 检查：新增 speed head/feature normalization/endpoint/gradient/eval_mask 测试 8 项通过；已有 scalar-speed suite 14 项通过。与添加选项前源码 snapshot 的默认 state tensors 和投影输出完全相同。按真实旧 control config、seed817 构造的 G1 初始 hash 仍是 `66ae5d81e82e26af2336043337fff6191b92c0ae010369c314f3fa315c224090`；真实旧 4096 checkpoint strict 恢复全部 741 tensor states，EMA counter=4096。未初始化 CUDA。详见 [CPU 兼容性审计](denoiser_options_cpu_audit.json)。

固定 source map checkpoint SHA256：`610c1f94a0f573fd078e251af37c38c73255b969e52154e66546588996ef99fb`；map 66 tensor SHA256：`09b7f55a4c78816794dd5ffcdce36760af5aa5afe038a1bf8cd34f168c51f720`。线性尺寸/标量 speed TRAIN normalizer SHA256：`adcc5e5c232b018919b269ec06b7e8b065a49fd9f0219f3d0f432f7caf4cea23`。新模型选项需要 fresh 训练和对应 metadata；CPU 兼容性不证明其学习效果更好。
