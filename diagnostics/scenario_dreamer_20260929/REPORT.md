# Scenario Dreamer 数据与训练审计

审计日期：2026-09-29。范围：当前 `init_bc` → `training_map2_sd` / `scenario_dreamer_val` → 初始场景 Flow → SD 指标，以及本地 `last.ckpt` 和已有训练日志。仅新增本目录诊断材料，没有修改训练代码、配置、缓存或 checkpoint，没有启动重训。

结论：已有证据优先指向训练源分布错误、学习率调度单位错误、checkpoint 加载和评估使用方式；抽样没有发现 SD agent 数据的整体坐标、类型或尺寸错位。具体因素对最终质量的贡献仍需受控重训验证，不能由静态审计直接分摊。

## 1. 首先分清正在比较哪个模型

- 当前 `src/waymo_data/last.ckpt`：`epoch=87`、`global_step=84832`，含 **255 个 `encoder.init_decoder.*` tensor**，是训练过的初始场景生成器。
- `2026-09-27_18-41-43` 的完整 50k 结果碰撞率 **30.7746%**。但该运行的 `.hydra/config.yaml:3` 加载的是 `AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt`。实际检查该文件，**没有任何 `encoder.init_decoder.*` 权重**。该运行 TensorBoard 仅记录 step 0–4，epoch 始终为 0。
- `2026-09-28_12-04-50` 同样加载 AIRL checkpoint，仅记录 step 0–6。其 `limit_val_batches: 1` 是整数，即只验证一个 batch；前一运行的 `1.0` 才是全量验证。
- `src/run.py:127` 使用 `load_state_dict(..., strict=False)`，不检查返回的 missing keys。旧 AIRL 权重缺少的新生成器参数会保留随机初始化。因此上述很差的历史结果能证明这些短运行效果差，**不能代表已训练 88 个 epoch 的 last.ckpt**。
- 本次未找到当前 last.ckpt 对应的已完成全量质量结果；`2026-09-29_13-54-09` 的日志能证明加载了它，但没有对应的完整指标。应先给这个精确 checkpoint 做独立、无更新的验证。

历史全量结果文件：`src/logs/xflow512_l1_v1_matchraw_nocol_20/2026-09-27_18-41-43/sd_agent_metrics.json`。其 `full_membership=true`、`reference_mode=matched`、未验证生成帧数为 0。362,309 辆生成车中 111,499 辆碰撞。角度/长度/宽度/速度的缩放后 JSD 分别为 29.52 / 59.18 / 20.95 / 38.79。由 onroad 特征计数推算，生成车约 27.29% 距中心线超过 1.5m，GT 约 0.61%；这是指标的中心线阈值定义，不能当作道路边界越界率。

修复方向：验证前断言生成器权重没有 missing keys，记录 checkpoint 文件指纹、global_step 和数据清单；迁移旧行为模型与恢复初始生成器训练必须明确区分。

## 2. 优先修复：固定 Ego 参与 Hungarian，训练与推理源分布不同

代码位置：

- `src/smart/diffusion/scale_flow.py:165`：采样 Gaussian noise。
- `scale_flow.py:173`：把 Ego 的 noise 改成 Ego 真值。
- `scale_flow.py:175`：对全部 agent 匹配。
- `src/smart/diffusion/diffusion_utils.py:311`、`:365`：只按场景、类型分组，没有排除固定 Ego。
- `scale_flow.py:233`：匹配后再次固定 Ego。
- `scale_flow.py:898`：推理直接采 Gaussian 再固定 Ego，没有上述匹配。

Hungarian 优化全体匹配代价，并不保证 Ego 留在自身索引。若某个普通车辆拿到了 Ego 的源点，随后又把 Ego 强制放回真值，源集合中就出现两个完全相同的 Ego 状态，同时少了一个 Gaussian 源点。这不再只是集合内的排列变化。

**已验证**：使用 last.ckpt 的 normal_mean/normal_scale，seed=817，512 个训练场景，每场景 8 次，共 4,096 次。**2,676 次（65.332%）发生 Ego 源点转移并在覆盖后重复**。脚本逐次断言了重复状态确实相等。

这证明当前训练和推理源分布存在差异；是否以及多大程度导致碰撞、位置或朝向变差，需要修复后重训对比。

修复方向：固定 Ego 不参与 assignment，只对非 Ego 按场景和类型匹配；验证每组非 Ego 噪声在匹配前后仍是同一集合的排列。仅改变推理不能消除旧权重已经学到的训练偏差。

证据：`noise_matching_stats.json`；可复现脚本：`noise_matching_audit.py`。

## 3. 优先修复：学习率在 128 次更新内完成了 128 epoch 的衰减

- `configs/model/smart.yaml:5`：`lr_total_steps: ${trainer.max_epochs}`，得到 128。
- `src/smart/model/smart_gail.py:751`：按 optimizer step 计算 cosine，超过 total 后 clamp。
- `smart_gail.py:785`：scheduler 明确设置 `interval: step`。

按当前逻辑，初始 `5e-4` 在第 128 次更新降至 `2.5e-5`。493,542 个训练文件、单卡 batch=512，约 964 batch/epoch，意味着第一轮约 13.3% 时已经衰减到底。

**checkpoint 实证**：last.ckpt 保存的 `lr_total_steps=128`、`lr=0.0005`、scheduler `last_epoch=84832`、`_last_lr=[2.5e-5]`，optimizer 的实际 lr 同样为 `2.5e-5`。这些与当前错误调度一致。checkpoint 没有保存逐步学习率曲线或历史 Python 函数，不能声称逐步实测了全部训练过程；若该训练一直用此逻辑，则约 99.85% 的更新发生在最低 LR。

修复方向：保留 step 调度时用实际 optimizer 总更新次数，计入 DDP、梯度累积和训练 batch 数；或明确改成 epoch 调度并统一参数单位。先检查学习率曲线，再训练长任务。这个问题会改变优化速度与收敛路径，但最低 LR 并非零，不能单凭它断言模型完全没学到。

## 4. 数据处理存在确定缺陷，但抽样未发现整体特征损坏

当前链路：Waymo proto → SD 选帧/筛选 → SMART 世界坐标 agent/map → `data2initmap.py` 缓存 → `TokenProcessor` → Ego 局部八维状态。

### 4.1 入口参数实际上没有传入

`src/data_preprocess_scenario_dreamer.py:216-217` 只调用 `wm2argo(file_path, split, target, None)`；传递 `rng / scene_timestep / frame_manifest / save_scene_info` 的代码在 205–214 行被注释。

结果：`--seed` 不控制实际选帧，指定 `current` 仍随机选帧，manifest 无法重放，scene info 开关失效，`seen/total` 不更新。隔离提取实际入口函数并替换 I/O 后确认调用 `kwargs={}`；有效 manifest 也会最终报缺少场景，处理成功仍会报 `Saved 0`。

输出目录复用，文件名又包含随机帧，因此重跑可能留存同一 scenario 的多个版本。**实际训练目录**有 493,542 文件、486,994 个唯一 scenario ID；4,326 个 scenario 有多个文件，最多 6 个，多出 6,548 个文件。多帧本身未必有害，但应由明确采样策略产生；仅凭现有缓存无法确认这些重复都是误重跑，也无法追溯各批处理参数。

修复方向：恢复参数传递、manifest 写入和计数；新数据写独立版本目录，保留来源、物理时刻、过滤配置和处理代码版本。核实重复来源后再决定采样权重，不要直接删除。

### 4.2 Source index 与输出行不一致

`data_preprocess_scenario_dreamer.py:161` 保存 `scene['source_index']`，而 `scenario_dreamer_filter.py:453-470` 已把输出 agent 重排成 Ego-last。对应字段应为 `scene['output_source_index']`（515 行）。三车合成验证中输出 ID 是 `[100,102,101]`，错误索引为 `[1,0,2]`，正确为 `[0,2,1]`。

当前 `data2initmap.py` 未把该索引带入训练缓存，故主要影响数据追溯和后续对齐，不能将其直接认定为当前 loss 异常原因。

### 4.3 抽样结果和排除项

固定 seed=817，各抽 1,024 个训练/验证场景：

| 检查 | 训练 | 验证 |
|---|---:|---:|
| agent 总数 | 9,895 | 9,323 |
| 每场景平均 agent 数 | 9.663 | 9.104 |
| NaN/Inf | 0 | 0 |
| 非正 length/width | 0 | 0 |
| 非法类型 | 0 | 0 |
| 超出 SD 64×64m 正方形（含数值容差） | 0 | 0 |
| 仅 Ego 的场景 | 25 | 35 |
| 选定时刻无效 agent / Ego-last 错误 | 缓存不含原始 mask/role | 0 / 0 |

验证集转换回保存的 SD 参考顺序与坐标后，位置最大绝对误差 **0.001974m**；速度误差小于 `9.4e-7`，heading 向量误差小于 `4.8e-7`，尺寸误差小于 `4.8e-7m`。因此抽样不支持“整体坐标旋转、长宽顺序或速度单位错误”。这不等于验证了全量每条数据。

last.ckpt 的八维 normal_scale 与抽样统计按现有 heuristic 算出的尺度比值为 **0.958–1.018**，不支持“这个 checkpoint 用了严重过时的归一化尺度”这一解释。代码仍只用首次 batch 初始化归一化，未来重新训练时应改为固定数据统计或同步统计。

训练随机帧均值 46.45，验证均值 10.05。官方训练也随机选择有效 Ego 帧，官方测试数据的有效帧范围与训练不同，不能仅凭此差异判为 bug。核心 SD 的裁图、图划分、选 agent、移除离中心线车辆等六个函数与本地官方实现去 docstring 后 AST 相同。

证据：`data_stats.json`；脚本：`data_audit.py`。

## 5. 评估和 checkpoint 选择会掩盖真实问题

### 5.1 选模指标写死

`src/smart/model/smart.py:491` 把 `val_closed/wosac_likelihood/metametric` 设成 **0.65**，`configs/callbacks/model_checkpoint.yaml:7` 正好监控这个值保存 top-k。

因此 `valmeta=0.6500` 无法表示 SD 生成质量，也无法选出最佳模型。它与真实算出的 collision/JSD 是两回事。应监控真实 SD 指标，并同时保留碰撞率和各属性分布，避免单项改进掩盖另一项退化。

### 5.2 部分验证与全量 GT 不匹配

当前 `smart.py:40` 导入 `old_gen_metrics`，严格评估器的初始化、update、compute 在 238、268、484 行被注释。旧路径使用全 50k 的 GT 分布，即使仅生成一部分场景，也不核验覆盖率、重复或参考帧。

把前 512 个场景的 GT 本身作为“生成结果”与 full50k 比较，也得到长度 JSD 0.54094、角度 0.28116、速度 0.16655。说明部分验证的非零差异包含样本组成偏差。单卡完整 50k 时不会触发该项偏差；它不足以解释历史长度 JSD 59.18 等大幅错误。

本地 SQLite 的 backend signature 与当前代码/依赖匹配，未发现缓存过期证据。

### 5.3 多 GPU 索引错误

`src/smart/datasets/scalable_dataset.py:61` 使用 `idx // num_gpus`，但 `len()` 未改变。若进程可见 G 张卡，遍历 N 个逻辑样本只会访问排序后的前约 N/G 个文件，其余不读。

真实 DistributedSampler 的 24 样本实验：G=4 只读前 6 个文件，每个重复 4 次。验证又仅 rank0 生成（`smart.py:258`），对 full50k 参考造成更大偏差。应删除手工除法，让 DistributedSampler 唯一负责分片，并汇总各 rank 的指标统计。

本机实测只有一张 4090，当前本机该问题不触发；远端多卡训练是否受影响需核实每个进程可见 GPU 数量，不能从本机推断。

## 6. 训练运行方式和次级建模候选

- **finetune 不等于 resume**：`run.py:127-134` 仅恢复模型权重，丢弃 optimizer moments 和 scheduler。以默认 `5e-4` 再加载处于 `2.5e-5` 的 last.ckpt，LR 会跳到 20 倍；如果配置为 `1e-4`，是 4 倍。恢复同一训练应使用完整 checkpoint；刻意迁移则显式设新 LR。前述 AIRL 短运行不能作为“last 恢复后崩坏”的证据。
- **配置在审计期间发生了用户侧修改**：最初读取 `init_bc.yaml:8` 为 `1e-40`，14:15 再读为 `1e-4`。本次未修改它。几乎零学习率只解释当时配置，不解释历史 `5e-4` 的训练。
- **验证过密**：`val_check_interval=2e-7` 会被 Lightning 换算为每训练 batch 验证一次；配 `limit_val_batches: 1.0` 就是每更新一次扫 50k。验证用途应走独立 validate，训练用途应设合理间隔。
- **无条件画图**：`token_processor.py:451` 每 batch 绘图并 `plt.show()`（367 行），`old_gen_metrics.py:591` 每验证 batch 再绘图。会阻塞交互后端或消耗计算/内存；属于当前调试状态，不能推断历史都如此。
- **Ego-only batch**：当前监督将 Ego 固定且令 t=0，实测该 batch 的监督 loss、梯度为 0。batch=1 时尤其浪费更新；大 batch 可只在有效非 Ego 上归一化。两次不同样本抽样比例约 2.4%–3.5%，不是主体数据。
- **损失与约束**：`diffusion_utils.py:435-438` 的 raw L1 权重为位置 0.02、heading 0.1、尺寸 0.04、速度 0.2；确定性 loss 分支未用传入的 normal_scale 归一化。`scale_flow.py:491` 硬编码 `use_col=False`。这些是建模选择，不是已证明 bug；可测 normalized loss、分维梯度、碰撞/道路辅助损失。
- **地图条件**：SD agent 目标在 64×64m 内，地图仍走原 SMART map，而 `init_map_range=100`。`finetune` 又冻结 map encoder（`smart.py:208`）。较大条件地图本身不错误，但应消融统一地图区域、解冻地图编码器的收益；不应直接宣称这是离路根因。

## 7. 建议的修复与实验顺序

1. 给精确的 last.ckpt 做无训练更新的单卡全 50k 基线；验证生成器权重覆盖、场景清单和参考时刻，记录碰撞、中心线阈值比例、各项 JSD 与生成尺寸范围。不要用硬编码的 0.65 判断。
2. 修复固定 Ego 的匹配，确认非 Ego 源集合不增不减；统一 scheduler 单位。保留同一缓存、同一划分、同一初始权重和更新预算，分别比较原逻辑 / 只修 matching / 只修 scheduler / 两者都修，才能判断因果贡献。
3. 恢复正确的 checkpoint monitor 和严格评估器；若需要 DDP，同步修复索引与多 rank 汇总。关闭训练中的逐 batch 绘图。
4. 修复预处理 CLI 和 manifest，在独立版本目录重建小规模数据并核对旧新结果；确认多帧重复的来源后再重建全量。
5. 最后再尝试损失权重、collision/offroad 约束和地图编码器消融。先解决可证实的管线错误，再调模型容量。

本次完成的是可复现诊断，没有执行上述重训消融，也未宣称修复后指标一定提升。

## 复现

从仓库根目录，在现有 sim 环境运行；均为 CPU 读取，不执行训练：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ke/miniconda3/envs/sim/bin/python diagnostics/scenario_dreamer_20260929/data_audit.py --output /tmp/sd_data_stats_recheck.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/ke/miniconda3/envs/sim/bin/python diagnostics/scenario_dreamer_20260929/noise_matching_audit.py --sample-list diagnostics/scenario_dreamer_20260929/noise_matching_stats.json --output /tmp/sd_noise_stats_recheck.json
```

`noise_matching_stats.json` 保存了完整抽样文件清单。数据审计对排序后路径用固定 seed 抽样；数据目录和 checkpoint 改变后结果自然可能变化。`audit_snapshot.json` 记录审计源文件指纹及核实过的运行来源。
