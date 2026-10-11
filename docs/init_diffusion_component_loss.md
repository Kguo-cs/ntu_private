# InitDiffusion 逐分量 loss

当前默认 `component_loss: null` 保留原训练与采样行为。非空配置启用独立的 position、heading、length、width、velocity、type、collision 和可选 speed_magnitude loss。旧的 `matching_loss` 总项是完整状态的 raw MSE 乘 `w_pos`；其分项 L1/MSE 和其他权重不能控制实际优化总项。新路径直接求各分量的加权和，不修改网络、噪声、Hungarian 匹配或 sampler。

## 已验证的搜索结果

[搜索报告](init_diffusion_loss_search_20261011.md)记录了实际短训练、逐项消融和多 seed 确认。最终选择的[训练配置](../configs/experiment/init_diffusion_lane_conditioned_loss_search_best.yaml)与[评价配置](../configs/experiment/init_diffusion_lane_conditioned_loss_search_best_eval.yaml)使用下面的组合：

| 分量 | kind | time_power | weight | scale |
|---|---|---:|---:|---|
| position | l1 | 1 | 4.0 | data |
| heading（最短圆周角差，rad） | mse | 2 | 1.0 | none |
| length | mse | 1 | 1.0 | data |
| width | mse | 2 | 1.0 | data |
| velocity（当前为 scalar speed） | mse | 1 | 2.0 | data |
| type | ce | 0 | 1.0 | none |
| collision | l1 | 1 | 5.0 | none |

启用 `normalize_time_weights: true`、`heading_space: angle`；物理分量的 data scale 使用已保存的训练标准差，最小值为 `std_floor: 0.1`。heading 与 type 不按标准差缩放。各权重作用在分量均值上，不再随 state 为 7/8/10/11 维而改变。

这是当前搜索范围内的最优短训练组合，不是任意数据、训练长度下的全局最优。对同一个 epoch35 warm checkpoint，以新 AdamW、LR `2e-5`、batch64、128 optimizer steps、square 50 m 地图 token 范围进行微调；ego position/heading/shape/type 固定，ego speed 与 non-ego 属性生成，type 使用 separate CE。3 个训练 seed 与每个训练 seed 下 2 个生成 seed，在 4096 个预处理评价样本上确认。七项指标相对各自 warm baseline 的平均比值为 `0.8097`；speed/length/width 的平均配对变化分别为 `-38.7%/-26.8%/-46.6%`，nearest distance 和 collision 分别为 `+1.5%/+4.6%`。它改善综合目标，仍存在指标取舍。完整 50k 评价与实际 EMA 权重检查以搜索报告为准。

最终 checkpoint 取 winning recipe 中综合分数位于中位数的训练 seed2817。原始 [mixed 候选配置](../configs/experiment/init_diffusion_lane_conditioned_mixed_loss.yaml)保留为初始对照，包含 cos/sin L1、较小 position/collision 权重等不同设置；它不是最终推荐组合。使用已测试训练配方和选中 checkpoint 的完整评价：

```bash
python src/run.py experiment=init_diffusion_lane_conditioned_loss_search_best

python src/run.py experiment=init_diffusion_lane_conditioned_loss_search_best_eval
```

## 配置与时间权重

```yaml
model:
  model_config:
    decoder:
      init_diffusion:
        component_loss:
          normalize_time_weights: true
          std_floor: 0.1
          heading_space: angle
          components:
            position: {kind: l1, time_power: 1, weight: 4.0, scale: data}
            heading: {kind: mse, time_power: 2, weight: 1.0, scale: none}
            length: {kind: mse, time_power: 1, weight: 1.0, scale: data}
            width: {kind: mse, time_power: 2, weight: 1.0, scale: data}
            velocity: {kind: mse, time_power: 1, weight: 2.0, scale: data}
            type: {kind: ce, time_power: 0, weight: 1.0, scale: none}
            collision: {kind: l1, time_power: 1, weight: 5.0, scale: none}
```

每项可以选择 `kind: l1|mse`、`time_power: 0|1|2|3`、非负 `weight` 和 `scale: none|data|正数`。weight 为 0 关闭该项；power 为 0 表示无时间加权。缺省分量配置由引擎补齐并记录在 checkpoint 和评价报告中：物理项 MSE/p2/weight1，type 使用现有类别分支对应的 CE 或 feature MSE/p0，collision L1/p1/weight0.1，speed_magnitude 默认关闭。通常应显式写出全部参与比较的设置。

对当前每个场景均匀采样的 `t ~ U(0,1)`，定义：

```text
w_p(t) = max(t, 0.05)^(-p)
L_c = lambda_c * mean_active_agents[
    w_p(t) / E[w_p(t)] * mean_valid_coordinates(rho(error / scale))
]
L_total = sum(reconstruction components, type, optional speed_magnitude) + collision
```

固定字段、缺失尺寸和 t=0/1 不参与重建均值与梯度。每个 agent 内按有效坐标平均，再按至少有一个有效坐标的 agent 平均；不会把额外 dummy speed 坐标或其他分量的通道数算入分母。长度和宽度分别计算，且沿用 linear/log 表示。scale=data 使用固定的 checkpoint 训练统计，不对每个 batch 重新拟合。

| power | E[w] | 未归一化最大值 | 归一化最大值 |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 1 |
| 1 | 3.9957 | 20 | 5.005 |
| 2 | 39 | 400 | 10.256 |
| 3 | 599.5 | 8000 | 13.344 |

归一化是解析常数，不是 batch 均值；它保留同一个 power 对小 t 的相对强调，同时避免不同 power 的平均梯度规模相差几十倍。`normalize_time_weights: false` 使用原始权重。若关闭归一化又希望保持相同 loss，把各分量 weight 除以表中的 E[w]。

可直接覆盖某个分量：

```bash
python src/run.py experiment=init_diffusion_lane_conditioned_loss_search_best \
  model.model_config.decoder.init_diffusion.component_loss.components.length.time_power=2
```

## Heading、type 与碰撞

`heading_space: vector` 比较现有 cos/sin 的分量误差；`angle` 比较最短圆周角差，可以使用 L1 或 MSE，方向为零时使用安全 fallback。它只影响训练 loss。angular_velocity 模式比较角速度预测与原最短弧速度目标，误差不做 wrap；同样使用配置的 kind/power/weight。angle 和 angular_velocity 没有已保存的角度 std，scale 需为 none 或指定正数。

separate/joint 的 type 默认保留 CE；也允许选择 L1/MSE，此时比较 softmax 概率与 one-hot，采样类别分支保持原样。feature 模式比较三个 raw type 状态的 L1/MSE，不能用 CE。CE 的 scale 必须为 none；推荐继续使用 power0/weight1，避免小噪声 timestep 主导分类。

新 collision 项直接比较物理尺寸下的正额外穿透量，保留 GT 同一对车辆已存在的重叠允许量。L1 比较米，MSE 比较米的平方；不使用 legacy 的 expm1 穿透变换。它独立按有效 pair 平均，pair 时间权重取两个端点中较大的值；完全固定的 ego 在 t=0 仍可作为碰撞条件。缺失尺寸不进入解码与几何。collision 没有训练穿透 std，scale 需为 none 或指定物理数值。该变化是新目标的一部分，比较实验时应明确记录。

vector velocity 下的可选 speed_magnitude 也可独立配置，例如：

```yaml
speed_magnitude: {kind: mse, time_power: 1, weight: 0.2, scale: data}
```

它比较速度向量模长。data scale 使用保存的 RMS speed（最小 1 m/s）；旧的 speed_loss_weight/speed_loss_scale 作为此项缺省权重/尺度。scalar-speed 表示已经由 velocity 项监督，不能再启用 speed_magnitude。

## 日志与兼容

`train/component/<name>_loss` 为未加权的有效分量均值；`train/component/<name>_weighted_loss` 为该项实际加入优化目标的贡献。现有 match_loss 包含 type 和辅助速度项，col_loss 单独相加；不会重复加入 type 或碰撞。现有 ego L1/type accuracy、type 指标和 EMA 日志继续保留。

损失配置写入 InitDiffusion checkpoint extra state 和 sd_agent_metrics.json 的 decoder.component_loss。改变 loss 不改变权重形状，旧 checkpoint 可以用于新目标的 finetune；训练目标不会从 checkpoint 元数据自动覆盖当前 Hydra 设置。新选项仅用于监督训练，RL/SDE/refiner 路径会明确拒绝配置，避免悄悄忽略新参数。

比较时保持同一评价集合、ego 条件、type 流程、地图范围、optimizer、步数和 seed。搜索报告包含单因素消融与候选组合验证；更改训练长度、数据条件或 EMA 使用方式后，应重新验证组合。不要将不同生成场景数的 JSD 直接当作 loss 优劣依据。

## 本次验证

在 sim 环境中，31 项新增数值/集成测试通过，覆盖独立 objective/power/weight、解析时间归一化、type 分支、碰撞单次计入、速度模长、固定字段/缺失尺寸/时间端点的零梯度、真实 denoiser backward 和 checkpoint 元数据。三个 Hydra 入口均可组合。

关闭新选项时，speed/vector 模式的初始化参数、训练 loss、参数梯度和 20 步生成输出与修改前 HEAD 完全一致。扩大检查执行 129 项测试，有 6 条旧断言失败；同样的失败已在内存加载修改前 HEAD 源码后复现，来自仍要求全 ego 生成的配置断言及仍要求 p3 的 loss 数值断言。当前基线使用 p2 并固定部分 ego 字段，本次未改动这些基线设置或旧测试。随后已经执行真实训练和 JSD 评价搜索；设置、原始记录与最终选择见[搜索报告](init_diffusion_loss_search_20261011.md)。两份 tested-best 原生配置通过 CPU-only Hydra 组合检查；[配置审计](../logs/component_loss_quick/native_config_audit/audit.json)记录与 winning run 的数据、模型、optimizer、精度和 manifest 对比，以及原生运行与 harness 的日志、CPU thread 和评价 reseed 方式差异。
