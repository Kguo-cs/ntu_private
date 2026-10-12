默认 N20/.05 的首个实际 native TEST 完成32批，但六项JSD未通过绝对1e-6复现门槛，网格已停止，不能解释其他采样选项。最大差为width −0.00124698、length −0.00106791；collision完全相同。

输入4096顺序、GT29698、generated30449、collision116、B128、EMA20480、GEN2817、float32/medium及全部decoder条件一致。generated onroad相差1车（26237对26236）。主评价和native都在CPU严格加载同一endpoint后搬GPU；主评价也重新构造model，并非直接使用训练内存。

优先做同native独立重复和原冻结runner eval-only重复，判断历史main与native是否稳定。若仍有差异，需实际首批CUDA RNG/噪声/EMA/map/buffer/output/type/edge证据。PyG CUDA散射聚合和seed重置时机仅为待验证方向；当前不能归因，也不放宽容差。
