# 从头训练搜索：实验清单与成本审计

UTC snapshot: `2026-10-11T10:06:15.910363+00:00`。此文件只读取实际文件，可在排队任务完成后重跑。

模型数按成功的实际训练生命周期计数：同 seed/初始张量哈希但不同 loss 的独立训练仍分别计一个模型；续训阶段不重复计新模型；只评价加载的模型不计新训练。失败和运行中阶段不计成功模型。

| Scope | Fresh completed models | Continuations | Eval-only stages | Successful metric records | Updates completed | Actual scene exposures | Known fit min | Pending | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_search | 37 | 11 | 0 | 167 | 79,872 | 5,111,808 | 101.68 | 0 | 1 |
| production | 3 | 0 | 0 | 18 | 6,144 | 3,145,572 | 42.29 | 0 | 0 |
| heldout | 4 | 0 | 4 | 48 | 8,192 | 4,194,096 | 58.21 | 0 | 0 |
| full_set | 0 | 0 | 2 | 2 | 0 | 0 | 0.00 | 0 | 0 |
| lr_pilots | 3 | 3 | 0 | 12 | 3,072 | 196,608 | 3.09 | 0 | 0 |
| failed_attempts | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0 | 2 |
| implementation_pilots | 1 | 1 | 0 | 2 | 32 | 2,048 | 0.03 | 0 | 0 |

主比较已完成生成指标记录 **235**；实际评价过的 loss 名称 **37**、resolved loss 哈希 **37**。这不是未来预计数量。

每个记录是一次生成评价，其完整 GT 参考指标是同次评价的附加结果，不另加一次。online、EMA、不同生成 seed 或 checkpoint 属于多次评价，不增加独立训练 seed N。

## 阶段清单

| Category / Run | Variant | Kind | Status | Start→end | Stage updates | Examples | Metric rows | Fit seconds once |
|---|---|---|---|---:|---:|---:|---:|---:|
| core_search / search1024_w0 | anchor | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.12290571599988 |
| core_search / search1024_w0 | collision_weak | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.94133950499963 |
| core_search / search1024_w0 | geometry_strong | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.93172264500026 |
| core_search / search1024_w0 | heading_l1 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.90106153200031 |
| core_search / search1024_w0 | heading_p2 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.10368870900129 |
| core_search / search1024_w0 | heading_p3 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.88549095899907 |
| core_search / search1024_w0 | legacy | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 69.35015417500108 |
| core_search / search1024_w0 | position_mse | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.01061992300129 |
| core_search / search1024_w0 | speed_p2 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.00701749200016 |
| core_search / search1024_w0 | vector_heading | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.56284891900032 |
| core_search / search1024_w0 | warm_winner | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.96686254500128 |
| core_search / search1024_w0 | width_p2 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.87755129099969 |
| core_search / adaptive1024_w0 | balanced32 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.72852114899979 |
| core_search / adaptive1024_w0 | balanced32_l1_quarter | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.89735368900074 |
| core_search / adaptive1024_w0 | balanced32_p0 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.8320754080014 |
| core_search / adaptive1024_w0 | balanced32_p2 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 76.19804802200088 |
| core_search / adaptive1024_w0 | balanced32_scalars_p2 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 76.12932194899986 |
| core_search / adaptive1024_w0 | balanced32_scalars_p3 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.99385983899992 |
| core_search / adaptive1024_w0 | balanced32_shape_l1 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 76.098147460998 |
| core_search / adaptive1024_w0 | balanced32_speed_l1 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.68408159799947 |
| core_search / adaptive1024_w0 | balanced32_vector | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.91821708399948 |
| core_search / adaptive1024_w0 | geo_l1_16 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 76.33644929900038 |
| core_search / adaptive1024_w0 | geo_mse32 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.93246890599949 |
| core_search / adaptive1024_w0 | geo_mse64 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 76.77866863100098 |
| core_search / adaptive1024_w0 | pos_mse32 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 74.95101472200076 |
| core_search / extra1024_w0 | geo_l1_16_heading_l1 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.58202237799924 |
| core_search / extra1024_w0 | geometry_strong_heading_l1 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 76.28340643000047 |
| core_search / extra1024_w0 | pos_mse32_heading_l1 | fresh | failed | 0→None | None | None | 0 | None |
| core_search / extra1024_recovery | balanced32_col20 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.80944639500012 |
| core_search / extra1024_recovery | balanced32_heading_l1_w1 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.83056501300234 |
| core_search / extra1024_recovery | geo_l1_16_col20 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.01947983499849 |
| core_search / extra1024_recovery | geo_mse32_col20 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.16272873900016 |
| core_search / extra1024_recovery | pos_mse32_col20 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.52140420200158 |
| core_search / extra1024_recovery | pos_mse32_heading_l1 | fresh | complete | 0→1024 | 1024 | 65536 | 3 | 75.73603065299903 |
| core_search / long4096_w0 | balanced32_l1_quarter | continuation | complete | 1024→4096 | 3072 | 196608 | 4 | 251.870922033002 |
| core_search / long4096_w0 | geo_l1_16 | continuation | complete | 1024→4096 | 3072 | 196608 | 4 | 251.75899512199976 |
| core_search / long4096_w0 | geo_mse32_col20 | continuation | complete | 1024→4096 | 3072 | 196608 | 4 | 225.9304825649997 |
| core_search / long4096_w0 | geometry_strong | continuation | complete | 1024→4096 | 3072 | 196608 | 4 | 251.0966266430005 |
| core_search / long4096_w0 | geometry_strong_heading_l1 | continuation | complete | 1024→4096 | 3072 | 196608 | 4 | 258.0252082070001 |
| core_search / long4096_w0 | legacy | continuation | complete | 1024→4096 | 3072 | 196608 | 4 | 242.28500915400218 |
| core_search / long4096_w0 | warm_winner | continuation | complete | 1024→4096 | 3072 | 196608 | 4 | 250.78073231999952 |
| core_search / targeted4096_w0 | geo_mse32_col20_heading_l1_shape2 | fresh | complete | 0→4096 | 4096 | 262144 | 6 | 302.2566360310011 |
| core_search / targeted4096_w0 | geometry_strong_heading_l1_width_p2 | fresh | complete | 0→4096 | 4096 | 262144 | 6 | 313.26810924500023 |
| core_search / targeted4096_w0 | geometry_strong_heading_p2 | fresh | complete | 0→4096 | 4096 | 262144 | 6 | 301.4717974310006 |
| core_search / targeted4096_w0 | geometry_strong_width_p2 | fresh | complete | 0→4096 | 4096 | 262144 | 6 | 304.15923264999947 |
| core_search / extend6144_w0 | geo_mse32_col20_heading_l1_shape2 | continuation | complete | 4096→6144 | 2048 | 131072 | 4 | 166.11463166199974 |
| core_search / extend6144_w0 | geometry_strong_heading_l1_width_p2 | continuation | complete | 4096→6144 | 2048 | 131072 | 4 | 168.63083113999892 |
| core_search / extend6144_w0 | legacy | continuation | complete | 4096→6144 | 2048 | 131072 | 4 | 160.26551538900094 |
| core_search / extend6144_w0 | warm_winner | continuation | complete | 4096→6144 | 2048 | 131072 | 4 | 166.94861817200217 |
| production / batch512_seed817 | geo_mse32_col20_heading_l1_shape2 | fresh | complete | 0→2048 | 2048 | 1048524 | 6 | 855.2528493119971 |
| production / batch512_seed817 | warm_winner | fresh | complete | 0→2048 | 2048 | 1048524 | 6 | 840.8616408760026 |
| production / batch512_alternative_seed817 | geometry_strong_heading_l1_width_p2 | fresh | complete | 0→2048 | 2048 | 1048524 | 6 | 841.3329182170019 |
| heldout / heldout_seed817 | geo_mse32_col20_heading_l1_shape2 | evaluation_only | complete | 2048→2048 | 0 | 0 | 4 | 0.0 |
| heldout / heldout_seed817 | warm_winner | evaluation_only | complete | 2048→2048 | 0 | 0 | 4 | 0.0 |
| heldout / heldout_bridge_seed817 | geo_mse32_col20_heading_l1_shape2 | evaluation_only | complete | 1536→1536 | 0 | 0 | 4 | 0.0 |
| heldout / heldout_bridge_seed817 | warm_winner | evaluation_only | complete | 1536→1536 | 0 | 0 | 4 | 0.0 |
| heldout / batch512_seed1817 | geo_mse32_col20_heading_l1_shape2 | fresh | complete | 0→2048 | 2048 | 1048524 | 8 | 887.9450551500013 |
| heldout / batch512_seed1817 | warm_winner | fresh | complete | 0→2048 | 2048 | 1048524 | 8 | 864.2044311859972 |
| heldout / batch512_seed2817 | geo_mse32_col20_heading_l1_shape2 | fresh | complete | 0→2048 | 2048 | 1048524 | 8 | 867.558757630999 |
| heldout / batch512_seed2817 | warm_winner | fresh | complete | 0→2048 | 2048 | 1048524 | 8 | 872.6342001839985 |
| full_set / full50000_seed817 | geo_mse32_col20_heading_l1_shape2 | evaluation_only | complete | 2048→2048 | 0 | 0 | 1 | 0.0 |
| full_set / full50000_seed817 | warm_winner | evaluation_only | complete | 2048→2048 | 0 | 0 | 1 | 0.0 |
| lr_pilots / lr256 | lr_1e4 | fresh | complete | 0→256 | 256 | 16384 | 2 | 16.31243765599902 |
| lr_pilots / lr256 | lr_3e4 | fresh | complete | 0→256 | 256 | 16384 | 2 | 16.077080711000235 |
| lr_pilots / lr256 | lr_5e4 | fresh | complete | 0→256 | 256 | 16384 | 2 | 16.027201674000025 |
| lr_pilots / lr1024 | lr_1e4 | continuation | complete | 256→1024 | 768 | 49152 | 2 | 45.715327131001686 |
| lr_pilots / lr1024 | lr_3e4 | continuation | complete | 256→1024 | 768 | 49152 | 2 | 45.61188793600013 |
| lr_pilots / lr1024 | lr_5e4 | continuation | complete | 256→1024 | 768 | 49152 | 2 | 45.821709944999384 |
| failed_attempts / search1024 | anchor | fresh | failed | 0→1024 | 1024 | 65536 | 0 | None |
| failed_attempts / failed_seed1817_segfault_20261011_163945 | warm_winner | fresh | failed | 0→None | None | None | 0 | None |
| implementation_pilots / pilot16 | anchor | fresh | complete | 0→16 | 16 | 1024 | 1 | None |
| implementation_pilots / pilot_resume32 | anchor | continuation | complete | 16→32 | 16 | 1024 | 1 | 2.035855708001691 |

## 时间、曝光和失败的限制

`train_trace.json` 每行记录 train batch 开始时的 pre-update global_step；连续且唯一的 trace 行数等于此阶段 optimizer updates，末行 step+1 为最终 global endpoint。trace 没有 elapsed/timestamp/实际 batch size；曝光使用真实 manifest 长度、批量和末批大小计算，并与完成 checkpoint/summary 元数据交叉核对。

可靠训练耗时来自 harness 用 monotonic clock 包住 trainer.fit 后写出的 train_seconds，每个 (run_path,variant) 只计一次（重复评价行取最大相同阶段记录值）。STEP 日志 elapsed 起点是 Diagnostics 构造，只有近似进度证据；没有 trace/checkpoint 的崩溃最后日志只是已确认更新下界，不能宣称精确最终步数或加成可靠 fit 成本。若保留 CSV，连续 train step 行的数量是另列的诊断性已完成更新证据，不会增加成功模型、主比较更新数或生成指标数。

已知 fit 时间是 wall time，并非 GPU kernel busy time。未计模型构造、训练/评价之间开销、预处理、统计校准、CPU 图表/PDF、未知失败耗时和未来任务。评价秒数只加每个唯一实际报告一次，缺少计时的报告另记。因此这些时间是有范围限制的已知成本，不是完整端到端总时长。

native smoke 在 JSON 中单列；LR pilots、implementation pilots 和失败阶段不并入主结果指标/成功模型数。全部输入 identity、初始化、resolved loss、trace schema、checkpoint global_step/epoch/optimizer/scheduler 和 pending 证据见 search_inventory.json。

## 审计提示

- Complete report recovered before/no summary row: /home/ke/code/sim/logs/component_loss_scratch/pilot16/anchor/eval_step_16/sd_agent_metrics.json
