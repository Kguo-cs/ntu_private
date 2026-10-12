# 训练 lane-distance 辅助 loss 的可行性复核

结论：当前 TRAIN 缓存提供可信的道路中心线**片段**，其坐标可与 InitDiffusion x0 对齐，可微距离计算成本很低；但中心线片段存在系统性纵向缺口，暂不实现把有限片段距离当 lateral distance 的辅助 loss。增加 lane query 的 conditioning 是另一个独立实验，已经 opt-in 接通，不改变评价或对生成结果自动 snap。

## 真实训练 batch

CPU 使用实际 TRAIN817 full-order manifest 的前 32 个场景，native `MultiDataset` 和 `TokenProcessor`，未加载模型 checkpoint、未初始化 CUDA。此诊断样本不用于估计完整训练集比例。详情见 [lane_distance_cpu_feasibility.json](lane_distance_cpu_feasibility.json)。

原缓存只有 `tokenized_agent`、`tokenized_map`；诊断访问 `map_save` 后的空 store 不包含 `traj_pos`。缓存没有 `sd_map`、原 Waymo polyline、lane ID 或 `ln_id`。本批有 21,579 个地图 token、223 个 non-ego GT vehicle。

| 现有字段 | 类型 / shape | 坐标或语义 |
| --- | --- | --- |
| `tokenized_map.position` | float32 `[M,2]` | token 起点，world 坐标 |
| `tokenized_map.orientation` | float32 `[M]` | token world heading |
| `tokenized_map.traj_pos_local` | float32 `[M,2,2]` | token-local 后两个点；起点需补 `[0,0]` |
| `tokenized_map.type` | uint8 `[M]` | 0 freeway、1 surface street、2 stop-sign lane，均来自道路中心线；3 bike lane；4/5 road edge；6/7/8 road line；9 crosswalk |
| `tokenized_map.batch` | int64 `[M]` | PyG 场景索引 |
| `tokenized_agent.initial_pos` | float32 `[N,2]` | 对齐初始化时刻的 world center |
| `tokenized_agent.initial_heading` | float32 `[N]` | 同一时刻 world heading |
| `tokenized_agent.type` | int64 `[N]` | GT type，0 为 vehicle |
| `tokenized_agent.ego_mask` | bool `[N]` | ego slot，固定位置，辅助项需排除 |
| `tokenized_agent.batch` | int64 `[N]` | 场景索引 |

`InitDenoiser.get_input` 的 x0 前两维是 ego-x-forward local position，与对 `initial_pos` 调用 native `transform_to_local` 逐 tensor 精确一致。centerline 应只选 type `{0,1,2}`，不能使用 bike lane、road line、crosswalk 或 road edges 来充当中心线。

几何重建：补零起点后的 token-local points 先按 `orientation` 旋转，再加 `position` 得到 world points；之后按每场景 ego position/heading 调用 native `transform_to_local`。直接把 `traj_pos_local` 当 ego-local 是错误的。

若其他数据路径有 `agent.sd_map`，其 `lanes` 是官方 compact centerline，bridge 已排除 undefined/bike lane；它处在 SD +Y-forward frame，不能直接与 InitDiffusion x0 相减。需先 `world = R(-sd_map.angle) * lanes + sd_map.center`，再转当前 ego-local；当前实际训练缓存没有此分支，不能假定可用。

## 为什么暂不加入 naive 辅助 loss

`src/smart/utils/preprocess.py::preprocess_map` 把 polyline 切成约 5 m 片段，对非 road-edge 只保留每 10 个片段中的一个；road edge 则每 2 个保留一个。token preprocessing 随后只保存 tokenized map，丢弃原 polyline 和连接 ID。不能按相邻 token 的空间距离随意拼接，在平行车道、交叉路口和曲线处会连错。

真实本批中，GT vehicle 到 `{0,1,2}` 有限中心线片段的最近欧氏距离 median 为 **3.6127 m**，70.40% 大于 2 m。采用全部缓存片段或原 square50 起点 crop，所有 vehicle 的最近距离在本批相同，因而这里主要不是 crop 不足。相比已有评价的参考地图约 99.4% GT onroad，这个 proxy 与真实 lateral distance 不等价；片段端点的 longitudinal 缺口很可能造成大量距离。该比例是 32 场景诊断值，不是完整 TRAIN 分布。

虽然 `relu(d(pred_center,lanes)-detach(d(GT_center,lanes)))` 保留每辆 GT 的距离阈值，它仍可能把预测往短片段端点的纵向方向推，损害联合位置分布；这个损失也允许向中心线过度收缩，不能保证 lat JSD 改善。无限延长每条短线也不可靠：远处交叉线延长后可能使任意中心点得到虚假的近零距离。

若以后保留 full centerline 和正确 scene transform，可再试 opt-in 非 ego GT vehicle 的超额距离 loss，保持 Hungarian 后 GT 行不变、GT distance detach，跳过无 lane/无有效 vehicle 的场景。不要用生成 type 筛训练 loss，不自动修正输出，不改 metric helper/reference/filtering。至少同时检查 lat 分布、onroad population、heading、nearest-distance 和 collision，不能只看 loss 下降。

## 可微与成本

CPU probe 使用 point-to-segment 投影：每条相邻点线段计算投影比例并 clamp 到 `[0,1]`，求距离的 hard minimum；它对预测位置几乎处处可微，nearest-segment 切换处非光滑。GT center 添加 `[0.3,-0.4]` m perturbation 后，31 个有 non-ego vehicle 的场景均有 finite/nonzero 梯度，单场景辅助 forward+backward 平均约 0.207 ms（2 CPU threads，不是 GPU wall-time 预测）。

SMART square50 每场景本批有 10–151 个 centerline tokens，每 token 两条线段。32 场景共 23,282 个 vehicle/segment pairs；按本批线性外推 B256/B512 为约 0.186/0.373 million pairs，几个 float32 中间矩阵和 O(pairs) 操作，计算本身轻量。需要按场景处理，不能构造全部 batch vehicle ×全部 batch segments 的矩阵。若改为 full dense lanes，成本按点数比例增加，需 chunk 并实测 GPU peak memory/latency。本次主要阻碍是 proxy 数据的正确性，而非 compute。

## 独立的地图条件实验已接通

原 `SMARTMapDecoder` 返回的 query 是 type4/5 road edges，lane/type3/roadline/crosswalk 仅参与固定 map attention 的 source。InitDiffusion 的 `map_feature`/`initial_map_feature` 只保留 `pt_token,position,orientation,batch`，丢掉 type 和 polyline；因此不能从这里直接提取可靠的中心线辅助目标。sep_map=false 时 raw tokenized_map 也没有自动传入 Flow.update，未来若要辅助项应显式传 typed geometry，而非猜测混合 position 的类型。

新 `map_condition_types: lanes_and_edges` 由 InitDiffusion 设置 `token_processor.init_map_output_types`。MapDecoder 仅将 returned/query mask 扩展为 `{0,1,2,4,5}`，attention sources、crop、radius、k、参数和 frozen weight 不变。默认 `road_edges` 完全保留旧行为。root 会串行测试 laneedges-only 和 laneedges+TRAIN feature normalization 两个 arm，不把组合 arm 的提升归因于单个因素。

实际预训练 map 的 32 场景 CPU 检查：square50 sources 4,211 保持不变，returned queries 从 1,689 增到 3,203；新增 1,419 surface-street、95 stop-sign lane queries（本批没有 type0）；world/ego-local position 和 heading 逐 tensor 精确符合 native transform。旧 source snapshot 与当前默认 map 输出所有字段 bitwise 相同，66 个 map tensors/冻结状态/hash 保持不变，fresh seed817 G1 hash 保持不变。5 项 CPU unit tests 通过。详见 [map_condition_types_cpu_audit.json](map_condition_types_cpu_audit.json)。

这些证据只证明类型、坐标、可微性和默认兼容性，没有证明 laneedges 的学习指标更好；最终采用与否需要 fresh/equal-exposure 的 paired TRAIN/GEN 评价。
