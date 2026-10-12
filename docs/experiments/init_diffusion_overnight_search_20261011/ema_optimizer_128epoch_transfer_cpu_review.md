# EMA 与 128 epoch 迁移边界（CPU 核验）

实际完成的 control checkpoint：global_step、EMA n、AdamW state.step 都为 **10240**，EMA 配置上限为 **0.9999**，当前实际衰减为 **0.9991219512**，尚未达到上限。对应 **5,242,620 场景曝光 / 10.62244 epoch**。candidate 的 20480 更新目前按源代码和已冻结协议投影，未标为完成的实际 checkpoint；其衰减预计为 **0.9995607613**，同样尚未达到上限。

`torch_ema` 每次先令 n 增加 1，再使用 `min(0.9999,(n+1)/(n+10))`。上限首次生效于 **n=89990**。当前自动监督训练每个 generator optimizer update 更新一次 EMA，测试不会增加 n。

| 批量 | 每完整 epoch 的 update 数 | 上限首次生效 | 128 epoch 的 update 数 |
|---|---:|---:|---:|
| 512 | 964（末 batch 486） | 93 整 epoch +338 updates；93.35064 场景 epoch | 123392 |
| 256 | 1928（末 batch 230） | 46 整 epoch +1302 updates；46.67535 场景 epoch | 246784 |

下表使用真实 partial batch 数量，对完整 EMA 权重递推求参数迭代的平均年龄和最近 95% 系数质量范围。它们表示模型参数迭代的历史，不能解释成原始数据权重、独立样本数或性能改善的因果证据。

| 状态 | 批量 | 平均迭代年龄，场景曝光单位 | 最近 95% 参数迭代质量范围，场景曝光单位 |
|---|---:|---:|---:|
| long 10240，实际 n | 512 | 524315 | 1485234 |
| long 20480，计划 n | 256 | 524290 | 1484978 |
| 完整 128 epoch，理论 | 512 | 5101071 | 15336640 |
| 完整 128 epoch，理论 | 256 | 2559596 | 7668320 |

在当前等曝光 long 预算，adaptive EMA 的场景单位窗口几乎相同；不能仅凭 candidate 的 n 翻倍，就宣称它的 EMA 场景窗口更长。达到固定衰减上限后，B256 的窗口约为 B512 的一半。完整 128 epoch 会进入当前短期运行尚未覆盖的新 EMA 区间。

实际 optimizer 为 AdamW，betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01。control 最后实际 LR 为 **0.0004928824226**，仍是初始 0.0005 的 **98.5765%**；scheduler 的实际 last_epoch=10。源码使用 epoch cosine（warmup=0，128 epoch，最低比例0.05），没有观察到完整低 LR 阶段。B256 在相同曝光下仍多一倍 AdamW 更新，Adam moments 与 decoupled weight decay 的样本单位跨度也会变化。

这些证据支持把已完成长预算结果作为配方筛选依据，不能据此给出可信的 128 epoch 指标数值预测。所有理论值都依赖当前 493542 场景、单 GPU、accumulation=1、保留末 partial batch、EMA 从 n=0 开始且公式保持原样。详见同名 JSON 中的实际 checkpoint SHA、源代码 SHA 和计算方法。未启动 GPU，也未修改运行中的源文件。
