# InitDiffusion 地图条件训练与评价

[训练配置](../configs/experiment/init_diffusion_lane_conditioned.yaml)使用已有的 `init_decoder=flow`（`InitDiffusion`），复用 SMART 的训练器、损失和采样器。训练读取 `src/waymo_data/full/training_map2_sd`，验证和测试读取 `src/waymo_data/full/scenario_dreamer_val`。

## 运行

以下命令均在 `/home/ke/code/sim` 执行，使用现有 `sim` 环境。运行入口显式设置 `paths.root_dir=/home/ke/code/sim/src`，以匹配现有数据和权重目录。

当前配置中的训练文件清单选项已注释，默认扫描训练目录。若要使用预存清单，先运行下面的命令，再给训练命令加上 `data.scenario_dreamer_train_sample_list=/home/ke/code/sim/src/waymo_data/full/training_map2_sd_files.pkl`。启用清单后，增删训练数据需重新运行命令刷新；清单只记录文件名，不读取场景内容，未刷新时新增文件不会自动进入训练。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.smart.scenario_dreamer.file_list \
  --raw-dir /home/ke/code/sim/src/waymo_data/full/training_map2_sd
```

正式训练默认从配置指定的 SMART 骨干 checkpoint 初始化，并冻结地图编码器。该骨干 checkpoint 不包含 InitDiffusion 参数，初始状态模型从新建参数开始训练。当前默认 `action=finetune`，训练 batch size 512、验证 batch size 256、测试 batch size 256、128 epochs，checkpoint 按 `collision_rate` 保存。每 4 个 epoch 使用完整验证集评价。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned
```

恢复本实验的训练状态时使用 `action=fit`，同时恢复模型、优化器和训练进度：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned action=fit \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

从零训练模型与地图编码器，不加载骨干 checkpoint：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned action=fit \
  ckpt_path=null model.model_config.finetune=false
```

用[独立评价配置](../configs/experiment/init_diffusion_lane_conditioned_eval.yaml)完整评价 50,000 场景。必须显式指定训练得到的 checkpoint；测试使用单 GPU，检查官方集合成员完整性。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

当前 `limit_val_batches=1.0` 使用完整验证集，GT 对照为完整 50k 分布。若通过覆盖值只跑部分 batch，不能把该结果当作完整评价；应检查报告中的 `num_samples` 和 `full_membership`。显存不足时可调整 `data.train_batch_size`、`data.val_batch_size` 和 `data.test_batch_size`。

## 是否固定 Ego

当前 lane-conditioned 训练和独立评价配置使用 `fix_ego=true`，固定 GT ego；通用模型配置/API 默认也为 `true`。需要生成 ego 时，训练和评价都设置如下：

```yaml
model:
  model_config:
    decoder:
      init_diffusion:
        fix_ego: false
```

`false` 会完整切换监督训练和确定性采样：ego 参与 source 抽样，使用场景的正常 timestep，参与原有状态重建、碰撞项以及已启用的 angular-velocity/speed magnitude loss。启用 `use_ego_embedding` 时，ego 使用自己抽到的 source，Hungarian 只排列 non-ego 的 source；关闭时 ego 也进入匹配集合。`generate_type=false` 时匹配按 scene/type 分组，`true` 时只按 scene 分组。每一步和最终输出都不再恢复 GT ego，scalar speed 输出也不再用 GT ego 速度向量覆盖。ego 的位置、heading、尺寸和速度都可以改变；是否生成其类型由 `generate_type` 控制。`fix_ego` 本身不改变 loss 公式及系数，变化的是参与训练的 agent 集合。

GT ego pose/保存的场景变换仍用于建立局部坐标系和条件地图；生成后不重新居中地图。lane-conditioned 路径中的 ego pose context 是重复的静态 pose，转入 ego 坐标系后为零；其他通用路径仍可能使用已有的 ego history context。agent 总数量仍来自输入；当前实验使用 `generate_type=false`，保留输入类别，启用后才生成类别。评价报告记录实际 `fix_ego`、`generate_type`、`use_ego_embedding` 和 conditioning。

训练、恢复和评价应保持同一设置。仅切换 `fix_ego` 不改变模型参数结构，可以加载固定-ego 权重继续训练，但直接切换旧模型评价时，ego 是此前未被监督生成的节点，结果不代表已训练好的全-agent 模型。当前选项支持监督 Flow 和初始化评价；SDE/RL/refiner 路径仍要求 `fix_ego=true`。

显式保持固定 ego 行为时，训练和评价命令同时追加：

```bash
model.model_config.decoder.init_diffusion.fix_ego=true
```

## Ego / Non-Ego 身份编码

当前 lane-conditioned 训练和评价启用 `use_ego_embedding=true`；通用模型配置和 API 默认 `false`，保留旧模型的参数结构。显式开启的配置如下：

```yaml
model:
  model_config:
    decoder:
      init_diffusion:
        use_ego_embedding: true
```

denoiser 增加 `nn.Embedding(2, hidden_dim)`，以输入 `ego_mask` 指定 `0=non-ego`、`1=ego`，在每个 timestep 加到 agent features，随后进入共享 graph attention。ego slot 的身份在训练和采样中保持不变；这里提供已知身份条件，不增加 `is_ego` 分类 head，也不从生成结果中选择哪一辆车作为自车。SMART 每个场景预留的 ego 行继续作为 ego 输出。

训练时，即使 `fix_ego=false`，ego 的 source 也单独保留，Hungarian 只在同场景的 non-ego 之间匹配；不会将其他 agent 的 source 交换到 ego slot。原有匹配代价、连续状态 loss 和已启用的 type CE 均保持不变。`use_ego_embedding=false` 保留原来的匹配行为。

`use_ego_embedding` 与 `fix_ego`、`generate_type` 独立：当前 `fix_ego=true`、`generate_type=false` 仍固定 GT ego 并使用 GT agent type，开启身份编码不会自动改为全-agent 生成。需要同时生成 ego 和全部类别时，训练、恢复和评价都使用：

```bash
model.model_config.decoder.init_diffusion.use_ego_embedding=true \
model.model_config.decoder.init_diffusion.fix_ego=false \
model.model_config.decoder.init_diffusion.generate_type=true
```

身份编码支持 GT type 与生成 type 两种模式；它属于 `G1`，由相同优化器和 EMA 更新。当前选项面向监督确定性 Flow 和初始化评价。新增 embedding 必须经过训练：旧 checkpoint 可用于 `action=finetune` 初始化已有参数，再训练新身份编码；不能启用选项后直接评价缺少该 embedding 的旧权重。加载这样的旧权重时，直接推理会报错，`G1` EMA 则从已加载的在线参数重新开始。评价旧模型需设置 `use_ego_embedding=false`，并保留该模型训练时的其他配置。

## 生成 Agent Type

当前 lane-conditioned 训练和独立评价使用 `generate_type=false`、`type_loss_weight=1.0`；通用 SMART 配置和 InitDiffusion API 默认同为 `generate_type=false`，保留 GT type 条件行为。需要生成全部类别时使用如下配置：

```yaml
model:
  model_config:
    decoder:
      init_diffusion:
        fix_ego: false
        generate_type: true
        type_loss_weight: 1.0
```

类别为 `0=vehicle`、`1=pedestrian`、`2=cyclist`。生成模式增加独立的三维 type Flow state，连续 agent 状态仍保持 vector 模式的 8 维或 speed 模式的 7 维：

```text
y_gt = one_hot(GT_type, 3), epsilon_type ~ N(0, I_3)
y_t = (1 - t) * y_gt + t * epsilon_type
type_clean = softmax(type_logits)
dy/dt = (y_t - type_clean) / max(t, t_eps)
L_type = mean_active(cross_entropy(type_logits, GT_type))
L_total = L_original + type_loss_weight * L_type
```

type state 与连续状态使用相同的场景 timestep，并在确定性 Euler 采样中一起更新。denoiser 从有噪声的三维 type state 建立 embedding，类型 head 使用共享图 attention 后的 features；未固定 agent 的 GT 类别仅作为训练目标，不直接用作 type embedding。类型 source 始终为标准 Gaussian，与 position/shape/velocity 的 source 选项独立。现有连续状态、圆周 heading 目标及其权重保持原样，额外增加独立 CE；CE 对未固定且 `0<t<1` 的 agent 取均值，不使用 reconstruction 的 `t^-3` 时间加权。

生成类别模式的 Hungarian 匹配只按 scene 分组，不再使用 GT type 分组；启用 `use_ego_embedding` 时仅匹配 non-ego。原有 context 中的三类 GT 数量替换为 `[总数量, 0, 0]`，包括已缓存的 `ego_feat`；总 agent 数量仍由输入指定，各类别的数量由生成结果决定。log 尺寸的缺失标注填充值也不按 GT type 分类，避免类别经输入填充值泄漏。

`fix_ego=false` 时 ego 的类别和连续状态全部参与生成；`fix_ego=true` 时保留 GT ego 类别和状态，并将 ego 排除在类型 CE 之外。最终类别取最后一次 clean logits 的 `argmax`。输出使用预测类别更新 `agent["type"]` 及对应的 motion token libraries，再选择速度 token；SD evaluator 也按预测类别筛选生成车辆，GT 对照仍使用真实类别。当前生成类别模式支持监督确定性 Flow、`initial_scene_only=true`、`scenario_dreamer_init=true` 和 cached SD evaluator；SDE/RL/refiner 及其他评价路径暂不支持。

训练日志增加 `train/type_loss`、`train/type_weighted_loss` 和 `train/type_accuracy`。新 head 属于 `G1`，优化器和 EMA 一起更新它。checkpoint 保存 `generate_type`；训练、恢复和评价必须保持同一设置。旧的 GT-type initializer 没有训练过类型 head，需要用 `action=finetune` 继续训练新 head 和生成路径，不能直接作为类型生成模型评价；切换模式时 `G1` EMA 从加载后的在线参数重新开始。只有 SMART 骨干的默认 checkpoint 可继续用于初始化新实验。

评价旧 GT-type 模型时，训练恢复或评价命令追加以下覆盖项，并保留旧模型的速度表示、heading、ego 和其他配置：

```bash
model.model_config.decoder.init_diffusion.generate_type=false
```

## Position、Shape、Velocity 的 Uniform Source

三个组可以独立选择 source，默认均为 `gaussian`，保留已有行为。需要全部改为 uniform 时，训练命令追加：

```bash
  model.model_config.decoder.init_diffusion.pos_source=uniform \
  model.model_config.decoder.init_diffusion.shape_source=uniform \
  model.model_config.decoder.init_diffusion.velocity_source=uniform
```

每个选项只接受 `gaussian` 或 `uniform`，也可以只开启其中一个。训练加噪与评价初始采样共用相同实现；是否固定 ego 由 `fix_ego` 控制，source 选项不修改 Hungarian 代价。圆周 heading 仍由 `heading_noise` 独立控制，插值路径和训练 objective 不变。

uniform 在模型的标准化坐标中独立采样 `u ~ U(-√3, √3)`，再使用现有、随权重保存的 normalizer 计算 `x_source = normal_mean + normal_scale * u`（scale 下限仍为 `1e-6`）。这样与 Gaussian source 保持相同的均值和方差；范围是每个坐标的 `mean ± √3 * std`，不是数据的 min/max。位置为 x/y 各自均匀采样，不是圆盘均匀采样。

`shape_source=uniform` 在 `size_representation=log` 时采样 log(length/width)，该 source 对应的物理尺寸呈有界 log-uniform 分布；生成结果仍由学习到的 Flow 决定。linear 模式下直接采样尺寸，source 仍可能为负，不额外 clamp。`velocity_source=uniform` 在 vector 模式下对 vx/vy 各自采样，速度模长不服从均匀分布；speed 模式下只作用于 scalar speed 的源值，内部有噪声的 speed 仍可能为负。

恢复训练或复现评价时，应显式使用该 checkpoint 训练时的同一组 source 设置；评价配置继承训练配置，也支持上述覆盖值。配置和 `sd_agent_metrics.json` 都记录 source 选项。权重结构不变，允许加载原 Gaussian 权重做实验，但只在评价时切换 source 会改变已学习路径的起点分布，不能当作已训练好的 uniform 模型。建议重新训练后比较完整评价指标。当前 Gaussian score 的 SDE/PPO 分支不支持 uniform source。

## 保留速度向量，增加 Speed Magnitude Loss

设置 `velocity_representation=vector` 时，输入和预测保留 `vx,vy`；当前 lane-conditioned 配置使用 `speed`，启用这个辅助项时须同时切换为 vector。
新增 `speed_loss_weight`（默认 `0.0`，关闭）允许在原有向量重建与碰撞损失之外，独立监督速度模长：

```text
s_pred = norm(v_pred), s_gt = norm(v_gt)
L_speed = mean_active(((s_pred - s_gt) / speed_scale)^2)
L_total = L_match_original + L_collision + speed_loss_weight * L_speed
```

只统计未固定且 `0<t<1` 的 agent；`fix_ego=false` 时包括 ego，无有效 agent 时辅助项为可微的零。
该项使用均匀时间权重，不再乘原来 reconstruction 的 `t^-3`。
现有 `vx,vy` MSE、朝向目标、尺寸表示和噪声/采样路径保持原来的定义。
内部 agent 坐标系与最终世界坐标系通过旋转关联，速度模长一致，因此监督的物理量与 `speed_jsd` 对齐。
`match_loss` 已包含加权辅助项；`vel_loss` 继续记录原向量重建分量，不合并 magnitude loss。

`speed_loss_scale=null` 默认复用现有 normalizer 的速度均值和 population 标准差：
`speed_scale = max(sqrt(sum(mean_v^2 + std_v^2)), 1 m/s)`，即 RMS speed，**不是 speed 标准差**。
这些统计沿用原有首次拟合与 checkpoint 保存方式，不重新拟合或增加 learnable scale。
也可配置固定正值（单位 m/s），例如 `speed_loss_scale=10.0`；分母不依赖预测值，也不随每个 batch 重新计算。
低精度输入的模长运算使用 float32。预测恰为零时 magnitude 项梯度为零，原向量 MSE 仍能将其推向非零 GT。

从权重 `1.0` 开始尝试，并监测归一化辅助项与其他损失的量级：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.velocity_representation=vector \
  model.model_config.decoder.init_diffusion.speed_loss_weight=1.0
```

训练日志新增 `train/speed_loss`、`train/speed_weighted_loss` 和 `train/speed_scale`。
没有新增网络 head、参数或 buffer，因此可以继续训练已有的 vector 模型，EMA 和 checkpoint 架构兼容。
可与 `size_representation=log`、两种 heading objective、sep_map 和条件 embedding 组合。
此项用于监督重建；refiner 的策略目标不支持。
标量 `velocity_representation=speed` 已直接监督 speed，因此开启本向量辅助项会报错。
评价时建议保留对应训练配置，报告会记录权重和归一化设置；这些选项本身不改变采样计算。

## 标量 Speed 输入和预测

要启用标量模式，训练和评价都设置
`model.model_config.decoder.init_diffusion.velocity_representation=speed`。
内部 Flow 状态从 8 维变为 7 维：
`[x,y,cos(theta),sin(theta),length,width,speed]`，其中 GT speed 是
`norm(local_vel)`，单位 m/s。denoiser 的连续输入、噪声、normalizer 和输出都只有一个 speed 分量；
不再预测 vx/vy。启用 angular-velocity objective 时，在这 7 维输出后追加一个角速度，共 8 维网络输出。

speed 使用原来的 x0 reconstruction 方式和时间权重；与 heading 的角速度目标分开。
训练直接比较预测 speed 与 GT speed，不用重建的 vx/vy 做监督，也不把侧向速度当作错误。
共享 loss 接口内部补一个恒为 0 的字段，保留原来的每维 `1/8` 系数；
现有 `train/vel_loss` 日志在 speed 模式记录未加权的 scalar speed MSE。
位置、尺寸、collision loss、地图条件和 EMA 路径继续沿用。

有噪声的 speed 和网络 raw speed 保留为实数，负预测能获得正常的纠正梯度；
只在最终物理输出将 speed 截为非负。输出先构造 agent-heading 坐标系的 `[speed,0]`，
再旋转为世界速度：`vx=speed*cos(global_heading)`、`vy=speed*sin(global_heading)`。
motion token 选择使用同一 `[speed,0]`；连续初始状态评价仍直接使用生成的物理速度。
`fix_ego=true` 时保留 GT ego 原始速度向量，不强制投影到自身 heading；`false` 时 ego 也使用预测 speed 和 heading 重建速度。

这个表示假设生成 agent 的运动方向与 heading 相同，不表达侧滑或倒车方向。
speed 模式当前支持确定性监督训练和采样；SDE/refiner 路径需使用 vector。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.velocity_representation=speed
```

独立评价配置也需覆盖为 speed 模式；评价报告记录实际 `velocity_representation`。
输入、输出投影及 normalizer 参数形状已经改变，需要训练新的 speed 模型。
旧 vx/vy checkpoint 的训练恢复和评价必须显式设置
`model.model_config.decoder.init_diffusion.velocity_representation=vector`，
同时保留该 checkpoint 对应的 heading_noise 和 heading_objective。
其他未显式启用 speed 的实验仍默认 vector。

## Log 空间 Length / Width

通用配置默认 `size_representation=linear`。要训练 log 尺寸模型，覆盖：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.size_representation=log
```

内部状态的字段 4、5 改为 `log(length)`、`log(width)`（自然对数，物理尺寸单位为米）。
输入编码、normalizer 的均值/标准差、Gaussian 噪声端点、Flow 插值、denoiser 的 x0 预测和采样积分全部使用这个表示。
因此尺寸插值在物理空间对应几何插值；负的 log 值是合法状态，不做截断。
现有 MSE 和时间权重不变，尺寸项比较 log 值，所以相同的尺寸比例误差有相同的尺寸重建误差。
`train/shape_loss` 记录 log 尺寸的分量 MSE，`shape_std` 也改为 log 尺寸标准差；与 linear 模式的绝对数值不能直接比较。

碰撞损失对预测和 GT 都先 `exp` 回物理尺寸，再计算原来的额外重叠损失。
最终初始化输出同样 `exp` 回米，现有 agent metrics 和运动策略读取物理尺寸；只有 `fix_ego=true` 时 ego 条件尺寸保持原值。
log 的有效尺寸标注必须为有限正值。训练默认 `invalid_size_policy=mask`：
将非正值、NaN、Inf 视为缺失尺寸坐标，保留该 agent 的数量、位置、heading、速度和其他有效尺寸监督。
缺失坐标的输入暂用同类型有效尺寸的几何均值（再依次退回 batch 几何均值、SMART 的标称尺寸）；
这个替代值不进入尺寸重建 loss 或 normalizer 统计。没有任何有效尺寸观测的坐标使用标准 log Gaussian prior。
涉及未知 GT 尺寸的碰撞 pair 不计算监督碰撞 loss，其余 pair 继续使用物理尺寸及原来的 GT 重叠允许量。
这不会把无效尺寸夹成极小正值当作训练目标，也不会删除真实 agent。
首次遇到无效尺寸会打印 agent、batch、type 和原值，相关 batch 记录
`train/size_invalid_fields`、`train/size_invalid_agents`。

需要检查数据时设置 `model.model_config.decoder.init_diffusion.invalid_size_policy=error`。
评价始终拒绝无效 GT 尺寸，包含重复使用的 masked 训练缓存。
已确认训练缓存 `2f0cdd3fea29917c_0_26.pt` 的一个 vehicle length 为 `-0.0979066`，
来源是 SD 预处理使用最后有效帧的尺寸；原 Waymo 轨迹的有效帧平均尺寸为正。
缓存文件保持原样。log/exp 对 float16、bfloat16 输入使用 float32，
不通过输出端 clamp 堆积边界值；如果 exp 溢出或下溢到零则报错。
本选项支持监督训练和确定性采样，可与 vector/speed、两种 heading objective、EMA、sep_map 和条件 embedding 组合；SDE/RL/refiner 暂不支持。

评价时也必须覆盖同一选项：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/absolute/path/to/log_size_init_diffusion.ckpt \
  model.model_config.decoder.init_diffusion.size_representation=log
```

需要重新训练 log 尺寸模型。旧 linear 初始化权重的尺寸 head 和 normalizer 都使用米，不能改配置后直接续训或评价；
checkpoint 记录表示并拒绝混用。仍允许用只有 SMART 骨干权重的 checkpoint 初始化新的 log 实验。
评价报告记录实际 `size_representation`。保持 `linear` 可继续加载原来的初始化权重。

## 圆周 Heading Flow Matching

当前配置使用 circular 路径和 x0 heading 目标。需要独立角速度 Flow Matching 时，训练和评价同时设置：

```yaml
heading_noise: circular
heading_objective: angular_velocity
heading_flow_loss_weight: 1.0
```

对每个可生成的 agent（`fix_ego=false` 时包括 ego），噪声方向 `theta_noise` 在 `[-π, π)` 均匀采样，计算最短角差
`delta=wrap(theta_noise-theta_gt)`，构造 `theta_t=wrap(theta_gt+t*delta)`。
默认输入为上述 7 维 speed 状态；vector 模式仍是原来的 8 维状态。
heading 始终为单位向量。位置、尺寸和 speed（或 vx/vy）保留线性插值和 x0 预测。

共享 denoiser 图特征后增加一个 scalar head，直接预测 angular velocity `omega_pred`，
单位为 rad / unit flow time，目标为 `d(theta_t)/dt = delta`。
`heading_loss=(omega_pred-delta)^2` 在未固定 agent 的有效训练时间上计算；
使用普通标量 MSE，不对误差 wrap、不套用 x0 loss 的 `1/t^3` 权重，也不使用 heading normalizer。
原 cos/sin 的 x0 reconstruction 项已从总 loss 中移除，避免两个目标同时约束 heading。
其余状态维度保留原来的每维系数、8 维平均分母和时间权重。
总 loss 为 `Euclidean_x0_loss + heading_flow_loss_weight * heading_loss + collision_loss`；
`train/heading_loss` 记录未乘该系数的 angular-velocity MSE。

采样直接使用 `theta_next=wrap(theta_t+dt*omega_pred)`，从 t=1 积分到 t=0。
collision loss 使用 `theta_x0=wrap(theta_t-t*omega_pred)` 得到的 clean heading，
并保留原来的 collision 权重。`fix_ego=true` 时 ego 固定、angular velocity 为 0，不参与 angular loss；`false` 时 ego 也学习和积分角速度。
内部生成状态在 speed 模式为 7 维，在 vector 模式为 8 维；已有物理输出和评价接口保持一致。

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned
```

评价使用同一 `heading_noise` 和 `heading_objective`；独立评价配置继承上述设置。
`heading_flow_loss_weight` 只调节训练角速度项，默认 1.0，可通过
`model.model_config.decoder.init_diffusion.heading_flow_loss_weight=0.1` 覆盖。
`sigma_h` 在 circular 模式不参与计算；评价报告记录实际 objective 和 loss weight。
EMA 包含新的 angular head，`sep_map` 的训练和 EMA 路径继续沿用。
当前 circular 模式支持确定性 Flow 采样；SDE/PPO 分支要求 Gaussian/x0 设置。

新增 angular head 的参数需要训练。旧 x0 checkpoint 不能直接作为 angular-velocity 模型恢复训练或评价；
本实验默认的 SMART 骨干 checkpoint 不含 InitDiffusion 参数，可以继续用于初始化新的训练。

保留此前的“仅改圆周路径、objective 不变”选项：训练和评价同时覆盖
`model.model_config.decoder.init_diffusion.heading_objective=x0`。
这时没有 angular head，预测所选速度表示的 clean x0、使用 reconstruction loss，
采样由 `omega=wrap(theta_t-theta_x0)/t` 得到角速度；加载此前的 circular/x0 权重还需设置 `velocity_representation=vector`。

## Heading 噪声标准差

同时设置 `heading_noise=gaussian` 和 `heading_objective=x0` 时，
当前实验默认 `model.model_config.decoder.init_diffusion.sigma_h=5.0`。
heading 的噪声端点为 `sigma_h * N(0, I₂)`：cos/sin 两维均值为 0，使用同一个标准差，
不读取 heading 的首批数据均值或标准差，也不再乘以已有的 `normal_scale`。
其他状态分量继续使用所选 source，ego 是否固定由 `fix_ego` 控制，线性 Rectified Flow 路径保留。
训练及采样共用同一噪声转换函数，已加载 checkpoint 中的旧 heading normalizer 不会覆盖此选项。

`5.0` 接近此前 head8 checkpoint 的 heading 噪声均方标准差（约 `5.0044`）。
`model_args` 不用于设置此值；通过 Hydra 的 `init_diffusion.sigma_h` 显式配置，例如：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.heading_noise=gaussian \
  model.model_config.decoder.init_diffusion.heading_objective=x0 \
  model.model_config.decoder.init_diffusion.sigma_h=2.5
```

评价同一模型时使用相同覆盖值。这个参数保存在运行配置中，评价报告记录实际 `sigma_h` 和 `heading_noise`；
恢复 checkpoint 的权重不会自动切换当前配置的 `sigma_h`。
切换噪声设置后需训练或微调，以比较对应的学习效果。若复查原有经验噪声模型，训练和评价都设置
`model.model_config.decoder.init_diffusion.heading_noise=gaussian`、
`model.model_config.decoder.init_diffusion.heading_objective=x0` 和
`model.model_config.decoder.init_diffusion.sigma_h=null`。其他未显式启用本选项的实验仍默认使用经验噪声。

## Scenario Dreamer 时间编码

`model.model_config.decoder.init_diffusion.time_embedding_type=scenario_dreamer`
复用本仓库 Scenario Dreamer 的 `TimestepEmbedder`：256 维 sin/cos 特征，
再经 `Linear(256, hidden_dim) → SiLU → Linear(hidden_dim, hidden_dim)`，无 LayerNorm。
两层时间 MLP 的权重按官方 `Normal(0, 0.02)` 初始化，bias 为零；隐藏维度仍使用 InitDiffusion 自己的设置。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.time_embedding_type=scenario_dreamer
```

Flow 的 `t=0` 是干净状态，`t=1` 是噪声。此模式直接编码 `99*t`，
对应公开 Scenario Dreamer 的 100 个 timestep（0…99），保留连续时间，不取整或反转。
`time_embedding_scale` 默认 `99.0`，可单独覆盖；只有 `fix_ego=true` 时 ego 时间为 0，否则与本场景其他 agent 的时间一致。
本选项应用到主 denoiser 和可选 refiner，不改变状态表示、噪声分布、loss、采样步数或优化器；EMA 自动包含时间 MLP。

默认 `time_embedding_type=legacy` 保留当前 `1-t` 的时间特征和原有参数结构。
切换为 `scenario_dreamer` 会改变时间 MLP 的参数结构，需要新训练；可使用默认的不含 InitDiffusion 权重的 SMART 骨干初始化。
已有 legacy InitDiffusion checkpoint 不能直接作为新模式的续训或评价权重，Scenario Dreamer 的完整公开权重也不能直接加载到 InitDiffusion。
恢复训练或使用 `experiment=init_diffusion_lane_conditioned_eval` 评价新权重时，必须保留相同的 `time_embedding_type` 和 `time_embedding_scale`；评价报告记录这两个设置。

## Scenario Dreamer agent / lane 数量编码

设置 `count_embedding_type=scenario_dreamer` 后，InitDiffusion 复用本仓库 SD 的
`LabelEmbedder(max_count + 1, hidden_dim, dropout_prob=0)`，用场景整数数量直接查表，
权重初始化为 `Normal(0, 0.02)`。agent count 包含 ego，统计完整场景，不随 `eval_mask` 改变。
agent 数量 embedding 加到 agent hidden features；lane 数量 embedding 加到条件地图节点，
再通过 map→agent attention 影响 agents。该选项保留已有 attention 架构、loss 和采样流程；ego context 是否包含 GT 类型数量仍由 `generate_type` 决定，生成类别时只提供总 agent 数量。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.count_embedding_type=scenario_dreamer
```

当前 `training_map2_sd` 缓存只有 SMART tokens，没有精确 SD lane graph。
默认 `count_lane_source=map_tokens`，在训练和评价中都统计实际送入 InitDiffusion 的条件地图 token 数，
即现有类型筛选和地图范围裁剪后的 `initial_map_feature` 行数。此数量不是 SD compact-lane 数；
即使评价样本具有 `sd_map`，也不会自动切换数量定义。

若训练和评价数据都提供了 SD lane graph，可显式选择 `count_lane_source=scenario_dreamer`，
以 `sd_map` 中当前 `lg_type` 的 graph 行数作为 lane count；缺少 metadata 时明确报错，不回退到 token 数。
此选项不改变条件地图的 SMART 表示。

SD 公开模型的 count vocab 上限为 agents 30、lanes 100。
InitDiffusion 默认采用 `count_max_num_agents=128`、`count_max_num_lanes=1024`，
为现有 SMART 场景和地图 tokens 提供更大的词表；上限是包含端点的，数量 0 有独立 embedding。
超出上限会报错，不截断数量；需要更大词表时，应在新训练前调整这些选项。

默认 `count_embedding_type=none` 不新增参数，保留旧权重结构。
启用后主 denoiser、可选 refiner 和 EMA 都包含两张 count embedding 表，地图缓存不会被反复累加 embedding。
新增表需要训练，已有不含这些参数的 InitDiffusion 权重不能直接用于新模式的 strict 续训/评价。
恢复训练和独立评价必须使用同一数量来源与词表上限；`sd_agent_metrics.json` 记录这些设置。

## Scenario Dreamer map_id / scene-type 编码

`map_embedding_type=scenario_dreamer` 复用 SD 的 scene-type `LabelEmbedder`。
Waymo 的真实 `map_id` 是 Nocturne 场景类别（0/1），不是地图文件编号；
SD 用 `scene_idx = 2 * lg_type + map_id` 编码成四类。
默认 `map_label_dropout=0.1`，因此表有五行，最后一行为训练时的 null label；
embedding 权重初始化为 `Normal(0, 0.02)`。
每个 forward 对每个场景只抽一次标签 dropout，再将同一个 embedding 加到 agent 和地图节点。

当前抽查的 `training_map2_sd` 和 `scenario_dreamer_val` 缓存均未保存真实类别标签。
默认 `map_id_source=fixed` 使用明确配置的类别，训练、评价口径一致，不根据地图几何或文件名推断类别。
下面的命令可直接用于当前缓存；其中 `map_id=0` 是固定条件，不表示样本具有已确认的真实类别 0：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.map_embedding_type=scenario_dreamer \
  model.model_config.decoder.init_diffusion.map_id_source=fixed \
  model.model_config.decoder.init_diffusion.map_id=0
```

若训练和评价数据都提供有效逐场景标签，使用 `map_id_source=metadata`。
沿用共享 reader 的来源优先级，读取 canonical `vectorworld_map_id`/valid mask、
`nocturne_compatible` 或 `map_id`。缺少任何场景标签时明确报错，reader 的无效零占位值不会作为类别 0 使用。
嵌套 `scenario_dreamer` 标签在 dataset metadata 整理前转为 batch-safe canonical 字段。
此选项不自动启用只在评价文件名上可用的 whitelist index，以避免训练和评价使用不同标签来源。

`map_lg_type=0` 默认表示 InitDiffusion 使用的完整 SMART 条件地图；
不会根据文件名中的原始 SD graph variant 自动切换。
设置 `map_lg_type=null` 时，要求从 `sd_map.lg_type` 或 agent 的逐场景 `lg_type` metadata 读取 0/1 标签，
可按官方公式处理混合 graph types；也可显式配置固定的 `map_lg_type=1`。

默认 `map_embedding_type=none` 保持旧参数结构和行为。启用后主 denoiser、可选 refiner 与 EMA 均包含
scene embedding，不修改地图缓存，不改变 loss 或现有 sampler。
新增表需要训练；strict 恢复训练和评价需要保留相同的模式、标签来源、graph-type 设置与 dropout 配置。
实际设置保存在运行配置及 `sd_agent_metrics.json` 中。

## Denoiser 关系编码：Fourier / MLP

默认 `model.model_config.decoder.init_diffusion.edge_embedding_type=fourier`，保留现有 FourierEmbedding。切换为 `mlp` 后，InitDiffusion denoiser 的 agent–agent 和 map–agent 关系使用已有 `MLPEmbedding`，直接编码 `[local_x, local_y, relative_heading]`；输出维度仍为 denoiser 的 hidden dimension。启用 refiner 时，其关系编码也使用同一选项。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.edge_embedding_type=mlp
```

这个选项只切换 denoiser 内部关系编码。地图编码器（包括 `sep_map`）、运动策略、agent 状态和时间编码继续使用原实现；MLP 关系不使用 `num_freq_bands`。训练 loss、优化器和采样方法沿用现有配置，EMA 自动跟踪所选关系编码的参数。

恢复训练和评价 MLP checkpoint 时都需要保留 `model.model_config.decoder.init_diffusion.edge_embedding_type=mlp`，例如在 `experiment=init_diffusion_lane_conditioned_eval` 命令中加入同一覆盖项。默认训练配置指定的 SMART 骨干 checkpoint 没有 InitDiffusion 权重，可以用它初始化新的 MLP 实验。已经训练的 Fourier InitDiffusion checkpoint 与 MLP 关系架构不同，不能直接作为 MLP 的续训或评价 checkpoint；本选项不自动转换关系权重和 EMA。

## 独立地图编码器（sep_map）

默认 `model.model_config.decoder.sep_map=false`，InitDiffusion 复用 SMART 的地图编码器。启用下面的选项后，创建结构相同但参数独立的 `init_map_encoder`；监督训练同时更新这个地图编码器与 InitDiffusion，保留共享地图和运动策略编码器的冻结状态。该选项用于 `init_decoder=flow` 的监督路径。

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.sep_map=true
```

加载没有独立地图权重的骨干或共享地图 checkpoint 时，从已加载的共享地图权重初始化独立地图；加载已有 `sep_map` checkpoint 时保留其独立地图权重。把共享地图实验转换为独立地图实验应使用 `action=finetune`，重新建立包含独立地图参数的优化器；`action=fit` 用于恢复同一 `sep_map` 配置的训练。

验证、完整测试及断点续训都需要保留 `model.model_config.decoder.sep_map=true`。独立地图输出已转到 ego 局部坐标，随后只经过 InitDiffusion 的 lane projection。启用 EMA 时，独立地图编码器与 `G1`（包含 lane projection）一起跟踪平均权重；评价先使用地图 EMA 编码，再使用 `G1` EMA 进行 projection 和采样。

## Scenario Dreamer 优化器选项

不指定 `optimizer` 时等同于 `optimizer=default`，沿用原优化器和调度器。需要使用 Scenario Dreamer 的优化器配方时，新训练增加 `optimizer=scenario_dreamer`：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned optimizer=scenario_dreamer
```

该选项使用 AdamW，基础学习率 `1e-4`、betas `(0.9, 0.999)`、epsilon `1e-7`；Linear/Conv 等权重的 weight decay 为 `1e-5`，bias、归一化层、Embedding 和其他参数不做 weight decay。梯度裁剪阈值为 `10`。学习率按 optimizer step 线性预热 500 步，此后恒定；梯度累积的每个 microbatch 不单独计步。

需要调整基础学习率或 warmup 长度时，分别覆盖 `model.model_config.lr` 和 `model.model_config.lr_warmup_steps`。该 profile 用于 InitDiffusion、Scenario Dreamer AE/LDM 等自动优化的监督训练；手动优化的 GAIL/GAN 路径不支持，会明确报错。

与[官方调度器](https://github.com/princeton-computational-imaging/scenario-dreamer/blob/675423469766bf2fd8a6b569ef1869a6f1e76993/utils/train_helpers.py#L35-L40)一致，初始学习率为 `0`，scheduler 在每次 optimizer step 后推进；第 500 次更新完成后到达 `1e-4`，第 501 次更新开始使用完整学习率。此选项只切换优化器、学习率调度和梯度裁剪；batch size、epoch 数、数据、训练目标及 EMA 设置仍由原实验配置决定，不等于复现 Scenario Dreamer 的完整训练协议。

恢复用该选项训练的 checkpoint 时，保留相同的 `optimizer` 选项：

```bash
/home/ke/miniconda3/envs/sim/bin/python -m src.run \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned optimizer=scenario_dreamer \
  action=fit ckpt_path=/absolute/path/to/init_diffusion.ckpt
```

`action=fit ckpt_path=...` 恢复 checkpoint 的优化器和 scheduler 状态，应与原训练使用同一 profile；旧运行继续使用默认 profile，不能仅通过修改恢复命令把已有状态转换为新配方。`action=finetune` 只初始化模型权重，重新建立所选优化器和调度器，warmup 从 0 开始；它不恢复原训练进度。独立 `action=test` 评价不执行优化器更新，无需为使用此配方训练的权重额外指定该选项。

## InitDiffusion EMA

当前 lane-conditioned 训练和评价配置默认启用 `model.model_config.decoder.init_diffusion.use_ema=true`，`ema_decay=0.9999`；通用 SMART 配置仍默认关闭。EMA 跟踪 `G1` generator 的参数，包括其中的 `lane_embed`、已启用的类型 head 和 ego/non-ego embedding。当 `sep_map=true` 时，还跟踪独立的 `init_map_encoder`，共用同一个开关、衰减值和更新时机；共享 SMART 地图编码器和运动策略不做平均。

训练使用在线参数，每次优化器更新后更新 EMA；验证和测试在 `use_ema=true` 时临时使用平均参数，完成后恢复在线参数。与 Scenario Dreamer 一样，采用 `torch_ema` 默认的 `num_updates` 衰减预热：初期实际衰减受更新次数限制，再逐步接近配置值。

显式指定 EMA 开关和衰减：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  model.model_config.decoder.init_diffusion.use_ema=true \
  model.model_config.decoder.init_diffusion.ema_decay=0.9999
```

checkpoint 同时保存在线参数、EMA shadow 参数、更新次数和衰减值；`sep_map` 的地图 EMA 也单独保存这些状态。上面的 `action=fit ckpt_path=...` 命令会保留这些状态继续训练。严格加载旧的、没有 EMA 状态的模型 checkpoint 时，以已经加载的对应模块权重初始化 EMA，更新次数从 0 开始；无法补回此前训练的平均历史。旧 `sep_map` checkpoint 如果只有 `G1` EMA，则保留 `G1` 的历史，地图 EMA 从已加载的在线地图权重开始。

对同一个 checkpoint 使用在线参数评价：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/absolute/path/to/init_diffusion.ckpt \
  model.model_config.decoder.init_diffusion.use_ema=false
```

`use_ema=false` 关闭验证/测试时的平均参数切换，不删除 checkpoint 已有的 EMA。加载带 EMA 的 checkpoint 后，其状态仍保留，并在继续训练时跟踪更新；可以再设为 `true` 使用平均参数。新建模型若始终关闭 EMA，则不积累此前的平均历史。

## 评价口径

`initial_scene_only=true` 只输出初始生成快照；`sd_gen_timestep=0` 指这个输出数组的第 0 帧，不是原始 Waymo 的第 0 帧。验证样本的初始位置、朝向和速度来自各自的 `scene_timestep`；预测以 world 坐标交给评价器。

评价器使用官方完整参考 lane 图，计算七项 vehicle 指标：`nearest_dist_jsd`、`lat_dev_jsd`、`ang_dev_jsd`、`length_jsd`、`width_jsd`、`speed_jsd`、`collision_rate`。nearest/lat JSD 乘 10，其余 JSD 乘 100；collision rate 为百分比。参考成员表为 `src/waymo_data/waymo_eval_set.pkl`，GT 统计来自 `src/waymo_data/sd_real_metric_cache.sqlite`。

完整报告保存到 `src/logs/init_diffusion_lane_conditioned_eval/<时间>/sd_agent_metrics.json`。应检查 `num_samples=50000`、`num_gt_samples=50000`、`full_membership=true` 和 `map_source=reference`。

## 比较范围

本模型沿用输入 agent 总数量，当前 `fix_ego=true`、`generate_type=false` 固定 GT ego 并使用输入类别；`use_ego_embedding=true` 提供 ego slot 的身份条件，不改变这两个设置。条件地图为 SMART tokens，使用现有 50 m 地图查询半径。设置 `fix_ego=false`、`generate_type=true` 后才能与 Scenario Dreamer/VectorWorld 对齐生成 ego 和类别的条件；GT ego 坐标框架与已有 context 仍保留。Scenario Dreamer/VectorWorld 使用 AE/VAE lane latent，模型结构和地图表示仍不同。相同的 50k 成员、指标口径以及对齐的 ego/type 条件便于比较结果，这不是 Scenario Dreamer/VectorWorld 论文结果的复现。

现有训练缓存只保存初始 agent 状态和地图 tokens，没有原始参考时刻或精确 SD lane 图元数据；其时刻与地图来源尚未逐样本追溯。本入口直接使用用户指定的缓存。验证目录则保留官方文件名、参考时刻、坐标变换和完整 lane 图，可用于严格的评估成员检查。

## 既有链路验证

以下记录来自 EMA 功能加入前，不代表本次 EMA 功能的验证结论。在 `sim` 环境中通过 8 项 InitDiffusion 测试和 80 项 Scenario Dreamer 回归测试。真实数据上的短训练使用当时的 batch size 256、8 workers，完成两步优化并保存 checkpoint；独立 `test` 入口加载该 checkpoint，完成 512 个场景的七项指标计算。训练 loss 为有限值，InitDiffusion 权重数值正常，冻结的地图 backbone 权重保持不变。

这验证了训练、保存、加载和评价链路，尚未完成 64 轮正式训练或 50k 完整评价。短训练的分数不能作为训练收敛结果。短训练与评价的重跑命令：

```bash
/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned \
  data.train_batch_size=256 data.num_workers=8 \
  model.model_config.decoder.init_diffusion.use_ema=false \
  trainer.max_epochs=1 +trainer.max_steps=2 trainer.limit_train_batches=2 \
  model.model_config.lr_total_steps=64 \
  callbacks.model_checkpoint.dirpath=/tmp/init_diffusion_smoke/checkpoints \
  hydra.run.dir=/tmp/init_diffusion_smoke

/home/ke/miniconda3/envs/sim/bin/python src/run.py \
  paths.root_dir=/home/ke/code/sim/src \
  experiment=init_diffusion_lane_conditioned_eval \
  ckpt_path=/tmp/init_diffusion_smoke/checkpoints/last.ckpt \
  data.test_batch_size=256 model.model_config.decoder.init_diffusion.use_ema=false \
  trainer.limit_test_batches=2 model.model_config.sd_require_full_set=false \
  hydra.run.dir=/tmp/init_diffusion_eval_smoke
```

## EMA 验证

在 `sim` 环境通过 23 项 InitDiffusion 测试（含 15 项 EMA/优化器专项）及 80 项 Scenario Dreamer 回归测试。使用指定真实数据完成 CPU 短训练：4 个 batch、梯度累积 2，产生 2 次 optimizer step，checkpoint 的 EMA 更新次数也为 2。独立测试加载同一 checkpoint 后使用 EMA 且更新次数保持 2；恢复训练完成下一次 optimizer step 后，模型步数与 EMA 更新次数均为 3。在线参数与 EMA shadow 参数不同，平均权重并非仅保存的在线权重副本。

已有运行中的训练进程不会自动获得新代码；启动新训练或恢复 checkpoint 后，EMA 配置才生效。以上检查验证功能与状态恢复，不代表 EMA 一定改善最终评价分数。
