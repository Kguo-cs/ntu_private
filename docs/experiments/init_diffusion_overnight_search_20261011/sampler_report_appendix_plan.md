# 可选采样器附录接口（尚无实测结论）

这份记录只规定如何把后续真实采样器结果接入报告，不宣称任何选项已完成、改善或获选。主模型选择、fixed-map fresh-G1 搜索和 full50k scope audit 独立保留，不把 sampler 运行增加到 TRAIN N。

当前独立 evaluator 是 `tools/evaluate_init_diffusion_sampler.py`；主队列完整完成、actual final selection 和 native checkpoint materialization 核验后才能执行。它只恢复实际选中模型的完整 state、EMA、normalizer 和 map，不训练，也不改已冻结主链源码。

## 接收真实结果的条件

1. 读取 `logs/init_diffusion_sampler_inference_20261012/grid/plan.json` 与 `summary.json`，校验主 selection、checkpoint 和 driver 的实际 SHA。计划 grid 是五选项 × TRAIN4817/5817 × GEN2817/3817，共 20 cells。
2. 每个实际 cell 用 driver 的 `validate_completed_cell(..., require_execution=True)` 复核：配置、scene manifest、native metrics、sampler audit、cell summary 的 SHA；独立子进程 execution success/returncode0 和日志中的 `DONE_INFERENCE_ONLY`。未完整结束、诊断或失败 cell 不纳入比较。
3. 默认20步、eps=.05 的四个 actual cell 必须先与主 endpoint 对照，七个原生指标的绝对差都 ≤1e-6。driver 不通过即停止；报告不能先假定默认复测等价。
4. 完整 grid 才能比较全部选项；partial summary 只列已完成事实和缺失状态，不排名、不补零、不自动选择。root 明确写入 `frozen_sampler.json` 后才记录实际选中 sampler；它须绑定 complete grid 的真实文件 SHA，full scope 只审计该冻结选项，不再择优。

## 计划选项与解释边界

| option | steps | inference t_eps |
| --- | --- | --- |
| n10_eps005 | 10 | .05 |
| n20_eps005 | 20 | .05 |
| n40_eps005 | 40 | .05 |
| n40_eps0001 | 40 | .0001 |
| n80_eps0001 | 80 | .0001 |

主训练保持20步和原训练 t_eps；这些是独立 inference-only 干预。40步而仍用 eps=.05 会在最后时间点进入 Euclidean denominator clamp，所以不等于纯 ODE 数值精度改善；降低 inference eps 也改变实际末段采样路径。圆周 heading 路径需单独保留其真实时间分母解释，不混同 Euclidean clamp。

## 报告呈现

在主结论和完整搜索过程后追加单独附录，标题明确为“同一选中权重的 inference-only sampler”。先写真实 complete/missing 状态与执行边界，再列七项 raw JSD/Col%、原生缩放、onroad 与 vehicle/GT counts、steps/eps、elapsed time、paired GEN seeds 和 checkpoint provenance。

每个 TRAIN 内先平均两 GEN，再平均两个 TRAIN；N=2 不改变，TRAIN4817 仍用于此前 adaptive selection，不能称新独立验证。相对 actual n20_eps005 计算 R、R_attr、pooled 和逐 TRAIN gates，不拿主链不同 population 的指标作新 control。full50k 是已见 scope 的 audit，不增加独立 TRAIN N。

driver 当前不导出 feature histograms；因此不能凭 sampler 指标表画新的 speed/shape 分布。若之后实际添加原生 bins 导出，须单独核验 exact GT counts/bin edges、生成 population 和四个 actual scope cells 后再接图。主模型 full50k 的 histogram 图只属于原主 sampler，不移作新 sampler 的证据。

本接口为 CPU 静态准备；未运行 GPU、未写 sampler selection、未生成 PDF 或完成 marker。
