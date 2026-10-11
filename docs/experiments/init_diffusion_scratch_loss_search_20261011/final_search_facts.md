# 从头训练搜索：最终实际文件汇总

UTC: `2026-10-11T10:06:18.012304+00:00`。全量阶段实际 DONE 后才生成；数目由保存文件推导，没有填入预计结果。

主比较：**235 次生成评价，37 个 loss 名称，37 个 resolved loss 哈希；44 个成功从头训练模型，11 个续训阶段，6 个只评价阶段**。

| Phase | Metric records | Fresh models | Continuations | Eval-only stages | Updates | Actual scene exposures | Known fit min | Known eval min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core_search | 167 | 37 | 11 | 0 | 79,872 | 5,111,808 | 101.68 | 12.42 |
| production | 18 | 3 | 0 | 0 | 6,144 | 3,145,572 | 42.29 | 1.36 |
| heldout | 48 | 4 | 0 | 4 | 8,192 | 4,194,096 | 58.21 | 14.14 |
| full_set | 2 | 0 | 0 | 2 | 0 | 0 | 0.00 | 7.50 |

已完成主比较实际 optimizer updates=94,208，实际样本曝光=12,451,476；已知去重 fit wall time 202.18 min，已知评价 35.42 min。重复 checkpoint、生成 seed、online/EMA 行不重复增加 fit 成本。

这些是已知 wall time，不是 GPU busy time或完整端到端成本；没有加入模型构造、数据预处理、CPU 图表/PDF、未知失败时间。LR/实现 pilots、失败阶段和 native smoke在清单中单列，不并入主比较。保留失败CSV的400次更新仅为诊断证据，不进入成功更新/模型数。

全部14个合法主根目录均保留在 final_all_stage；无新增图表，全部实际记录、分组、后期趋势和表格均保存。各评价 cohort 分开汇总，不将搜索1024、heldout4096和full50000混合排名。full50000不改变此前冻结的heldout部署选择。

六项JSD遵循helper输出单位：nearest/lateral×10，其余JSD×100；原始JSD为自然对数，理论上界ln2。碰撞为百分数。综合S的scale为[1,1,10,20,15,20,5]；训练seed样本SD用ddof=1（N=1为0），生成重复不增加独立训练N。

[全部实际指标和表格](final_all_stage/analysis.md)、[observations](final_all_stage/observations.json)、[inventory](search_inventory.json)、[精确facts及hash](final_search_facts.json)。

工具验证：6个CPU inventory单元测试通过；范围为曝光/trace计数、评价矩阵、cost去重、重复报告和失败CSV诊断，不代表GPU训练测试或完整原生训练验证。
