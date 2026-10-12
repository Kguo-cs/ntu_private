实际复跑已证明：相同checkpoint、TRAIN4817/GEN2817、4096场景、B128、EMA20480、N20/.05下，原冻结runner和两次native指标都存在超过1e-6的波动。因此不能只归因native接口。生成vehicle30449/collision116始终相同，onroad数在26236–26238之间。

原runner复跑与历史结果的length直方图L1计数差34、width差14，总vehicle30449；GT直方图完全相同。length JSD差−0.00294980、width差−0.00018428。两次native width差0.00159181、angle差−0.00155231。0.1m尺寸bin及JSD会放大少量边界计数变化，但保存的bin不能证明每个agent的物理偏差大小，也不能证明noise逐元素相同。

旧网格保持失败且排除；不得修改旧1e-6门槛。新native推理cohort应先冻结独立协议、重新完整评价自己的4个N20基线，再比较所有同weights/同TRAIN/GEN/manifest选项。R≤.98是保守选择规则，不能当这些少数复跑得到的统计置信界。主训练选择和full50k不被重选。
