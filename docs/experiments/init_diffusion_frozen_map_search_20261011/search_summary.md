# 固定 map encoder 的 InitDiffusion 搜索汇总

实际有效记录：84；分层组：7。自动读取当前已完成 summary，因此运行中报告可刷新。

主评价为 EMA。GEN 先在 TRAIN seed 内平均，N 和 sample SD(ddof=1)只跨独立 TRAIN seeds；N=1 的 SD=0 是描述值，不是无不确定性。

S=mean(native metric/[1,1,10,20,15,20,5])；R=七项相对 new_control 的 ratio 平均；R_attr=length/width/speed 三项 ratio 平均，均越低越好。R 的排序使用 paired TRAIN seed 后的 arm-mean ratios，另完整保存逐 seed ratio/SD。

筛选门槛：onroad≥同组 control−0.02；V/GT∈[0.9,1.1]；L/W/speed 各项≤control×1.15。无匹配 control/覆盖 metadata/GEN seed pairing 时不排名。不同 evaluation cohort 不混排名。

单位：nearest/lat 为 JSD×10；angle/length/width/speed 为 JSD×100；collision 为百分比。JSON 同时保存 raw JSD；S/R 都是搜索辅助分数，不是论文指标。

## ema / B512 / updates 128 / exposure 65,536

cohort `f4507cfec996`，eval manifest `c552af064913`，1024 场景。

| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| speed4 `f72cd9` | 1 | 1.0426±0.0000 | 0.8318 | 0.6141 | 57.27% | 0.910 | PASS / provisional |
| old_control `7f5adb` | 1 | 1.1250±0.0000 | 0.9173 | 0.7031 | 54.84% | 0.994 | FAIL / provisional |
| speed_p0 `399c31` | 1 | 1.0639±0.0000 | 0.9272 | 0.8664 | 56.75% | 0.985 | FAIL / provisional |
| shape_p2 `bdcb4e` | 1 | 1.1210±0.0000 | 0.9567 | 0.8679 | 55.08% | 0.979 | FAIL / provisional |
| heading_p0 `5bc911` | 1 | 1.0491±0.0000 | 0.9663 | 0.9721 | 56.50% | 0.968 | PASS / provisional |
| heading_mse1 `6d8e69` | 1 | 1.1384±0.0000 | 0.9678 | 0.8902 | 56.39% | 0.975 | PASS / provisional |
| col5 `0d706b` | 1 | 1.1439±0.0000 | 0.9720 | 0.8959 | 57.59% | 0.947 | PASS / provisional |
| pos16 `252a61` | 1 | 1.1326±0.0000 | 0.9758 | 0.9157 | 56.75% | 0.954 | PASS / provisional |
| new_control `297ebd` | 1 | 1.1144±0.0000 | 1.0000 | 1.0000 | 57.37% | 0.963 | PASS / provisional |
| pos_l1_16 `8c8a2e` | 1 | 1.1077±0.0000 | 1.0152 | 1.0025 | 54.90% | 0.953 | FAIL / provisional |
| width4 `52e53c` | 1 | 1.1663±0.0000 | 1.0183 | 0.9857 | 56.38% | 0.961 | FAIL / provisional |
| heading_p2 `f0f08c` | 1 | 1.1520±0.0000 | 1.0206 | 1.0170 | 56.51% | 0.977 | PASS / provisional |
| col10 `057d5f` | 1 | 1.1696±0.0000 | 1.0252 | 1.0122 | 56.52% | 0.991 | FAIL / provisional |
| shape4 `200410` | 1 | 1.1966±0.0000 | 1.0765 | 1.1260 | 55.65% | 0.970 | FAIL / provisional |

**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：

| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| col10 | 0.03% | 1.70% | 3.49% | -16.16% | -2.15% | 21.97% | 8.75% | 1 |
| col5 | 0.27% | 1.02% | 5.75% | -15.73% | -30.13% | 14.64% | 4.57% | 1 |
| heading_mse1 | -0.17% | -0.92% | 12.80% | -19.96% | -11.04% | -1.95% | -1.29% | 1 |
| heading_p0 | 3.80% | -1.95% | -16.09% | 3.42% | -4.95% | -6.85% | -0.94% | 1 |
| heading_p2 | 0.10% | 0.27% | 7.20% | -2.27% | -4.56% | 11.93% | 1.76% | 1 |
| new_control | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1 |
| old_control | 18.38% | 2.24% | 13.25% | -37.68% | -33.99% | -17.40% | -2.73% | 1 |
| pos16 | -0.30% | 1.24% | 7.75% | -15.38% | -21.42% | 11.52% | -0.35% | 1 |
| pos_l1_16 | 19.04% | -4.85% | 5.84% | -27.34% | -2.98% | 31.07% | -10.13% | 1 |
| shape4 | 3.17% | -2.00% | 11.96% | -13.74% | -20.42% | 71.95% | 2.59% | 1 |
| shape_p2 | 0.81% | 2.85% | 9.89% | -21.98% | -31.33% | 13.67% | -4.23% | 1 |
| speed4 | 3.31% | -2.92% | 7.94% | -44.02% | -39.29% | -32.46% | -10.30% | 1 |
| speed_p0 | 3.22% | -4.53% | -7.81% | -38.99% | -35.98% | 34.90% | -1.80% | 1 |
| width4 | 4.08% | 0.85% | 11.21% | -14.50% | -22.68% | 32.89% | 0.97% | 1 |

通过 gate 的 R 排名：speed4@f72cd9510d7f → heading_p0@5bc911b51200 → heading_mse1@6d8e69bbd18e → col5@0d706be440ad → pos16@252a61b7c35b → new_control@297ebde7d0cb → heading_p2@f0f08cf03f86

通过 gate 的七指标 Pareto：col5@0d706be440ad, heading_mse1@6d8e69bbd18e, heading_p0@5bc911b51200, heading_p2@f0f08cf03f86, new_control@297ebde7d0cb, pos16@252a61b7c35b, speed4@f72cd9510d7f

## ema / B512 / updates 256 / exposure 131,072

cohort `f4507cfec996`，eval manifest `c552af064913`，1024 场景。

| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| heading_p0 `5bc911` | 1 | 0.4718±0.0000 | 0.8904 | 0.8315 | 71.22% | 1.015 | PASS / provisional |
| speed4 `f72cd9` | 1 | 0.4883±0.0000 | 0.9054 | 0.7450 | 72.63% | 1.004 | PASS / provisional |
| clip2 `fba538` | 1 | 0.5074±0.0000 | 0.9136 | 0.7417 | 70.27% | 1.021 | PASS / provisional |
| ego_sincos `5cc9b3` | 1 | 0.4659±0.0000 | 0.9140 | 0.8787 | 70.99% | 1.008 | PASS / provisional |
| optimizer_sd `9baa67` | 1 | 0.5661±0.0000 | 0.9736 | 0.7861 | 69.89% | 1.003 | PASS / provisional |
| shape_uniform `8e0abb` | 1 | 0.4665±0.0000 | 0.9763 | 0.9916 | 72.06% | 1.015 | FAIL / provisional |
| shape_p2_speed4 `010d7c` | 1 | 0.5114±0.0000 | 0.9949 | 0.8971 | 71.64% | 1.038 | PASS / provisional |
| new_control `297ebd` | 1 | 0.5213±0.0000 | 1.0000 | 1.0000 | 70.86% | 1.007 | PASS / provisional |
| speed_p0 `399c31` | 1 | 0.4660±0.0000 | 1.0234 | 1.2061 | 72.40% | 1.010 | FAIL / provisional |
| count_embedding `f73a3b` | 1 | 0.5199±0.0000 | 1.0380 | 1.0134 | 69.50% | 1.021 | FAIL / provisional |
| shape_p2 `bdcb4e` | 1 | 0.5350±0.0000 | 1.0780 | 1.1002 | 70.79% | 1.004 | FAIL / provisional |
| speed_uniform `1f9853` | 1 | 0.5346±0.0000 | 1.0805 | 1.0547 | 70.37% | 1.030 | FAIL / provisional |
| heading_mse1 `6d8e69` | 1 | 0.5158±0.0000 | 1.1302 | 1.2070 | 72.24% | 1.009 | FAIL / provisional |
| col10 `057d5f` | 1 | 0.8322±0.0000 | 1.3045 | 0.9843 | 72.33% | 1.008 | PASS / provisional |
| edge_mlp `9cb3d5` | 1 | 0.9022±0.0000 | 1.3200 | 0.6222 | 65.07% | 0.995 | FAIL / provisional |
| lr1e3 `311a23` | 1 | 0.8935±0.0000 | 1.4154 | 0.9271 | 68.09% | 1.017 | FAIL / provisional |
| heading_p2 `f0f08c` | 1 | 0.7645±0.0000 | 1.4440 | 1.2444 | 69.48% | 1.008 | FAIL / provisional |
| col5 `0d706b` | 1 | 0.9795±0.0000 | 1.5445 | 1.1420 | 71.94% | 1.013 | FAIL / provisional |
| pos16 `252a61` | 1 | 0.7791±0.0000 | 1.5454 | 1.5796 | 61.49% | 1.040 | FAIL / provisional |
| width4 `52e53c` | 1 | 0.8941±0.0000 | 1.5509 | 1.2896 | 65.38% | 1.023 | FAIL / provisional |
| lr2e4 `c5390a` | 1 | 0.6379±0.0000 | 1.6161 | 2.1474 | 68.29% | 1.018 | FAIL / provisional |
| pos_l1_16 `8c8a2e` | 1 | 0.8554±0.0000 | 1.6209 | 1.4167 | 57.75% | 1.013 | FAIL / provisional |
| shape4 `200410` | 1 | 0.9659±0.0000 | 1.7902 | 1.6100 | 60.22% | 1.037 | FAIL / provisional |
| old_control `7f5adb` | 1 | 0.9807±0.0000 | 1.8086 | 1.4762 | 55.33% | 0.998 | FAIL / provisional |
| size_log `3227ad` | 1 | 0.8996±0.0000 | 2.3310 | 3.3272 | 66.81% | 1.032 | FAIL / provisional |

**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：

| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip2 | 17.58% | -2.70% | 0.85% | -1.86% | -20.49% | -55.15% | 1.30% | 1 |
| col10 | 64.38% | 3.05% | -10.28% | 5.40% | 6.34% | -16.44% | 160.69% | 1 |
| col5 | 98.65% | 2.72% | 26.01% | 26.74% | 36.20% | -20.32% | 211.14% | 1 |
| count_embedding | 20.80% | -0.78% | 5.25% | 32.10% | 16.21% | -44.28% | -2.73% | 1 |
| edge_mlp | 96.99% | -5.98% | 95.87% | -27.96% | -35.71% | -49.66% | 150.46% | 1 |
| ego_sincos | 2.17% | -1.88% | -5.37% | -2.78% | 11.96% | -45.58% | -18.68% | 1 |
| heading_mse1 | 23.22% | -0.83% | 30.21% | 20.51% | 78.86% | -37.28% | -23.58% | 1 |
| heading_p0 | 9.03% | -0.56% | -26.64% | -7.18% | -19.59% | -23.79% | -8.02% | 1 |
| heading_p2 | 35.74% | -3.34% | 184.87% | 31.78% | 36.00% | 5.55% | 20.16% | 1 |
| lr1e3 | 80.73% | -3.84% | 86.18% | -30.37% | 75.31% | -66.80% | 149.56% | 1 |
| lr2e4 | 27.24% | -5.14% | 60.43% | 148.09% | 172.95% | 23.18% | 4.51% | 1 |
| new_control | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1 |
| old_control | 96.64% | -15.19% | 237.23% | 120.08% | 51.66% | -28.89% | 104.46% | 1 |
| optimizer_sd | 4.06% | -0.04% | 20.16% | -4.87% | -15.68% | -43.62% | 21.53% | 1 |
| pos16 | 68.70% | -8.97% | 59.56% | 96.08% | 77.30% | 0.50% | 88.64% | 1 |
| pos_l1_16 | 103.39% | -10.86% | 122.59% | 79.87% | 64.83% | -19.70% | 94.51% | 1 |
| shape4 | 79.50% | -6.84% | 179.53% | 92.03% | 75.03% | 15.94% | 117.95% | 1 |
| shape_p2 | 20.36% | -1.08% | 3.47% | 35.58% | 9.74% | -15.25% | 1.80% | 1 |
| shape_p2_speed4 | 19.88% | -0.09% | 23.04% | 4.39% | 7.67% | -42.92% | -15.53% | 1 |
| shape_uniform | 10.57% | -0.06% | 5.72% | 14.60% | 24.43% | -41.54% | -30.29% | 1 |
| size_log | 78.52% | -7.43% | 16.17% | 275.91% | 448.71% | -26.45% | 146.28% | 1 |
| speed4 | 22.34% | 1.65% | 1.17% | -11.25% | -19.53% | -45.71% | -14.87% | 1 |
| speed_p0 | 6.31% | -0.77% | -28.74% | 22.95% | 10.18% | 28.71% | -22.26% | 1 |
| speed_uniform | 19.30% | 0.15% | 29.16% | 17.00% | 46.41% | -47.00% | -8.70% | 1 |
| width4 | 78.01% | -4.74% | 94.50% | 9.93% | 67.23% | 9.72% | 130.96% | 1 |

通过 gate 的 R 排名：heading_p0@5bc911b51200 → speed4@f72cd9510d7f → clip2@fba5388dcf34 → ego_sincos@5cc9b3bc543b → optimizer_sd@9baa671e9aa6 → shape_p2_speed4@010d7ce0e5d8 → new_control@297ebde7d0cb → col10@057d5f0271ca

通过 gate 的七指标 Pareto：clip2@fba5388dcf34, ego_sincos@5cc9b3bc543b, heading_p0@5bc911b51200, new_control@297ebde7d0cb, optimizer_sd@9baa671e9aa6, shape_p2_speed4@010d7ce0e5d8, speed4@f72cd9510d7f

## ema / B256 / updates 512 / exposure 131,072

cohort `f4507cfec996`，eval manifest `c552af064913`，1024 场景。

| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| batch256 `010d7c` | 1 | 0.3119±0.0000 | — | — | 75.62% | 1.024 | no control |

**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：

| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch256 | — | — | — | — | — | — | — | 0 |

通过 gate 的 R 排名：无

通过 gate 的七指标 Pareto：无

## ema / B512 / updates 512 / exposure 262,144

cohort `f4507cfec996`，eval manifest `c552af064913`，1024 场景。

| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| shape_uniform `8e0abb` | 1 | 0.3049±0.0000 | 0.7874 | 0.5366 | 77.09% | 1.020 | PASS / provisional |
| optimizer_sd `9baa67` | 1 | 0.3037±0.0000 | 0.7964 | 0.5406 | 76.19% | 1.016 | PASS / provisional |
| ego_sincos `5cc9b3` | 1 | 0.3129±0.0000 | 0.8033 | 0.4593 | 75.90% | 1.016 | PASS / provisional |
| shape_p2 `bdcb4e` | 1 | 0.2939±0.0000 | 0.8157 | 0.5752 | 76.37% | 1.015 | PASS / provisional |
| count_embedding `f73a3b` | 1 | 0.3042±0.0000 | 0.8309 | 0.5395 | 75.51% | 1.029 | PASS / provisional |
| clip2 `fba538` | 1 | 0.3127±0.0000 | 0.8557 | 0.5243 | 75.40% | 1.022 | PASS / provisional |
| shape_p2_speed4 `010d7c` | 1 | 0.3166±0.0000 | 0.8581 | 0.5906 | 75.78% | 1.019 | PASS / provisional |
| speed4 `f72cd9` | 1 | 0.3167±0.0000 | 0.8685 | 0.6891 | 76.55% | 1.007 | PASS / provisional |
| pos_l1_16 `8c8a2e` | 1 | 0.3050±0.0000 | 0.8973 | 0.8096 | 74.50% | 1.011 | PASS / provisional |
| heading_p0 `5bc911` | 1 | 0.3070±0.0000 | 0.9209 | 0.8577 | 75.46% | 1.018 | FAIL / provisional |
| speed_uniform `1f9853` | 1 | 0.3304±0.0000 | 0.9364 | 0.7257 | 76.74% | 1.019 | PASS / provisional |
| speed_p0 `399c31` | 1 | 0.3155±0.0000 | 0.9877 | 0.9337 | 76.16% | 1.017 | FAIL / provisional |
| edge_mlp `9cb3d5` | 1 | 0.3765±0.0000 | 0.9979 | 0.5071 | 72.62% | 1.005 | FAIL / provisional |
| new_control `297ebd` | 1 | 0.3287±0.0000 | 1.0000 | 1.0000 | 75.75% | 1.014 | PASS / provisional |
| pos16 `252a61` | 1 | 0.3118±0.0000 | 1.0733 | 1.0742 | 74.13% | 1.013 | FAIL / provisional |
| heading_mse1 `6d8e69` | 1 | 0.3480±0.0000 | 1.0749 | 0.7269 | 75.23% | 1.013 | PASS / provisional |
| col10 `057d5f` | 1 | 0.3796±0.0000 | 1.0952 | 1.0609 | 76.82% | 1.013 | FAIL / provisional |
| shape4 `200410` | 1 | 0.3608±0.0000 | 1.1176 | 1.0042 | 75.60% | 1.008 | FAIL / provisional |
| lr1e3 `311a23` | 1 | 0.3761±0.0000 | 1.1357 | 0.9117 | 74.45% | 1.010 | PASS / provisional |
| width4 `52e53c` | 1 | 0.3581±0.0000 | 1.1466 | 1.1012 | 75.49% | 1.015 | FAIL / provisional |
| lr2e4 `c5390a` | 1 | 0.3448±0.0000 | 1.2097 | 1.2545 | 74.66% | 1.023 | FAIL / provisional |
| size_log `3227ad` | 1 | 0.3678±0.0000 | 1.2678 | 1.5506 | 75.95% | 1.010 | FAIL / provisional |
| heading_p2 `f0f08c` | 1 | 0.3937±0.0000 | 1.2979 | 1.0700 | 75.51% | 1.005 | FAIL / provisional |
| old_control `7f5adb` | 1 | 0.4494±0.0000 | 1.3609 | 0.8061 | 71.58% | 1.017 | FAIL / provisional |
| col5 `0d706b` | 1 | 0.6739±0.0000 | 1.5935 | 0.8314 | 77.15% | 1.010 | PASS / provisional |

**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：

| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip2 | 7.00% | 0.43% | 49.52% | -58.22% | -62.62% | -21.87% | -15.28% | 1 |
| col10 | -20.05% | 3.48% | 4.95% | 19.20% | -0.07% | -0.86% | 59.98% | 1 |
| col5 | 51.31% | 2.85% | -0.62% | -13.72% | -36.09% | -0.77% | 412.49% | 1 |
| count_embedding | 3.25% | 0.89% | 35.05% | -58.48% | -63.14% | -16.51% | -19.44% | 1 |
| edge_mlp | -17.81% | -2.36% | 109.91% | -50.76% | -72.98% | -24.14% | 56.67% | 1 |
| ego_sincos | 11.40% | 3.46% | 14.83% | -62.00% | -71.66% | -28.54% | -5.19% | 1 |
| heading_mse1 | 6.45% | 0.52% | 149.72% | -24.85% | -45.22% | -11.87% | -22.30% | 1 |
| heading_p0 | 9.78% | 0.83% | -5.67% | -19.18% | -41.27% | 17.77% | -17.64% | 1 |
| heading_p2 | 28.86% | -0.42% | 149.40% | 19.77% | -19.50% | 20.72% | 9.72% | 1 |
| lr1e3 | 24.54% | 2.35% | 86.36% | 11.31% | 10.47% | -48.28% | 8.22% | 1 |
| lr2e4 | 4.39% | 0.89% | 60.53% | -42.89% | -61.33% | 180.58% | 4.62% | 1 |
| new_control | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1 |
| old_control | 24.37% | -2.22% | 218.41% | -5.28% | -50.30% | -2.59% | 70.27% | 1 |
| optimizer_sd | 1.35% | 1.56% | -1.57% | -46.13% | -71.06% | -20.63% | -6.07% | 1 |
| pos16 | 53.80% | -1.48% | 28.26% | 12.17% | -21.25% | 31.34% | -51.55% | 1 |
| pos_l1_16 | -2.61% | 0.74% | 3.32% | -17.52% | -47.84% | 8.24% | -16.19% | 1 |
| shape4 | -0.27% | 0.60% | 57.59% | -18.91% | -25.83% | 46.01% | 23.11% | 1 |
| shape_p2 | 3.75% | 2.16% | 18.63% | -54.96% | -79.98% | 7.51% | -26.11% | 1 |
| shape_p2_speed4 | 1.33% | 1.86% | 25.31% | -47.12% | -55.11% | -20.61% | -4.97% | 1 |
| shape_uniform | 0.70% | 4.53% | -7.21% | -54.17% | -62.57% | -22.29% | -7.81% | 1 |
| size_log | 4.67% | 2.47% | 6.02% | 62.58% | 66.41% | 36.19% | 9.10% | 1 |
| speed4 | -0.50% | 1.34% | 1.47% | -27.00% | -39.57% | -26.71% | -1.07% | 1 |
| speed_p0 | 13.42% | 0.82% | 15.04% | -20.04% | -41.38% | 41.54% | -18.01% | 1 |
| speed_uniform | -4.63% | 3.00% | 37.67% | -45.27% | -37.56% | 0.54% | 1.77% | 1 |
| width4 | 18.14% | -0.22% | 37.97% | 3.14% | -24.79% | 52.00% | 16.35% | 1 |

通过 gate 的 R 排名：shape_uniform@8e0abba5490c → optimizer_sd@9baa671e9aa6 → ego_sincos@5cc9b3bc543b → shape_p2@bdcb4ed434f3 → count_embedding@f73a3b9a216a → clip2@fba5388dcf34 → shape_p2_speed4@010d7ce0e5d8 → speed4@f72cd9510d7f → pos_l1_16@8c8a2ec0ccd7 → speed_uniform@1f985387534c → new_control@297ebde7d0cb → heading_mse1@6d8e69bbd18e → lr1e3@311a236ef60c → col5@0d706be440ad

通过 gate 的七指标 Pareto：clip2@fba5388dcf34, count_embedding@f73a3b9a216a, ego_sincos@5cc9b3bc543b, heading_mse1@6d8e69bbd18e, lr1e3@311a236ef60c, new_control@297ebde7d0cb, optimizer_sd@9baa671e9aa6, pos_l1_16@8c8a2ec0ccd7, shape_p2@bdcb4ed434f3, shape_p2_speed4@010d7ce0e5d8, shape_uniform@8e0abba5490c, speed4@f72cd9510d7f, speed_uniform@1f985387534c

## ema / B256 / updates 1024 / exposure 262,144

cohort `f4507cfec996`，eval manifest `c552af064913`，1024 场景。

| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| batch256 `010d7c` | 1 | 0.2657±0.0000 | — | — | 77.13% | 1.019 | no control |
| combo_batch256_ego_sincos `5cc9b3` | 1 | 0.2765±0.0000 | — | — | 77.61% | 1.016 | no control |

**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：

| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch256 | — | — | — | — | — | — | — | 0 |
| combo_batch256_ego_sincos | — | — | — | — | — | — | — | 0 |

通过 gate 的 R 排名：无

通过 gate 的七指标 Pareto：无

## ema / B512 / updates 768 / exposure 393,216

cohort `f4507cfec996`，eval manifest `c552af064913`，1024 场景。

| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| ego_sincos `5cc9b3` | 1 | 0.2757±0.0000 | 0.8532 | 0.5353 | 76.69% | 1.015 | PASS / provisional |
| shape_p2_speed4 `010d7c` | 1 | 0.2729±0.0000 | 0.8810 | 0.6104 | 77.54% | 1.019 | PASS / provisional |
| shape_p2 `bdcb4e` | 1 | 0.2811±0.0000 | 0.8929 | 0.5927 | 78.17% | 1.013 | PASS / provisional |
| optimizer_sd `9baa67` | 1 | 0.2834±0.0000 | 0.8969 | 0.5549 | 77.58% | 1.020 | PASS / provisional |
| speed_uniform `1f9853` | 1 | 0.2794±0.0000 | 0.9065 | 0.6579 | 78.66% | 1.017 | PASS / provisional |
| clip2 `fba538` | 1 | 0.2829±0.0000 | 0.9119 | 0.5989 | 77.59% | 1.014 | PASS / provisional |
| shape_uniform `8e0abb` | 1 | 0.2818±0.0000 | 0.9263 | 0.7666 | 77.91% | 1.017 | PASS / provisional |
| speed4 `f72cd9` | 1 | 0.2840±0.0000 | 0.9517 | 0.7921 | 77.88% | 1.009 | PASS / provisional |
| new_control `297ebd` | 1 | 0.2759±0.0000 | 1.0000 | 1.0000 | 77.39% | 1.008 | PASS / provisional |
| size_log `3227ad` | 1 | 0.2962±0.0000 | 1.0167 | 0.8413 | 76.91% | 1.015 | PASS / provisional |
| edge_mlp `9cb3d5` | 1 | 0.2990±0.0000 | 1.0413 | 0.5768 | 74.94% | 1.016 | FAIL / provisional |
| count_embedding `f73a3b` | 1 | 0.2913±0.0000 | 1.0848 | 1.0836 | 78.02% | 1.010 | FAIL / provisional |
| lr1e3 `311a23` | 1 | 0.3243±0.0000 | 1.0896 | 0.7170 | 75.27% | 1.026 | FAIL / provisional |
| lr2e4 `c5390a` | 1 | 0.2978±0.0000 | 1.1124 | 0.8940 | 76.57% | 1.019 | FAIL / provisional |
| old_control `7f5adb` | 1 | 0.3456±0.0000 | 1.4122 | 0.7405 | 74.42% | 1.011 | FAIL / provisional |

**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：

| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip2 | 9.11% | 4.87% | 38.53% | -43.18% | -50.55% | -26.62% | 6.14% | 1 |
| count_embedding | -7.42% | 3.47% | 33.66% | 1.08% | 41.81% | -17.82% | 4.61% | 1 |
| edge_mlp | -21.15% | 3.01% | 147.21% | -41.67% | -58.36% | -26.93% | 26.78% | 1 |
| ego_sincos | -3.31% | 5.17% | 27.46% | -54.40% | -53.20% | -31.82% | 7.36% | 1 |
| lr1e3 | 11.12% | 2.34% | 68.84% | -22.41% | -15.05% | -47.44% | 65.33% | 1 |
| lr2e4 | 7.30% | 4.27% | 78.56% | -45.91% | -61.10% | 75.20% | 20.39% | 1 |
| new_control | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 1 |
| old_control | 3.79% | 0.91% | 325.53% | -12.36% | -53.89% | -11.60% | 36.17% | 1 |
| optimizer_sd | -9.55% | 4.78% | 46.64% | -45.04% | -61.02% | -27.48% | 19.52% | 1 |
| shape_p2 | -3.28% | 5.99% | 31.54% | -44.30% | -66.21% | -11.67% | 12.97% | 1 |
| shape_p2_speed4 | -0.52% | 2.30% | 26.84% | -45.74% | -53.26% | -17.86% | 4.92% | 1 |
| shape_uniform | -17.75% | 8.40% | 23.29% | -21.13% | -17.99% | -30.89% | 4.48% | 1 |
| size_log | 1.55% | 6.85% | 31.47% | -38.37% | -3.94% | -5.30% | 19.43% | 1 |
| speed4 | -9.56% | 5.83% | 20.85% | -21.22% | -25.75% | -15.40% | 11.41% | 1 |
| speed_uniform | -11.23% | 3.69% | 21.43% | -50.43% | -62.80% | 10.59% | 23.32% | 1 |

通过 gate 的 R 排名：ego_sincos@5cc9b3bc543b → shape_p2_speed4@010d7ce0e5d8 → shape_p2@bdcb4ed434f3 → optimizer_sd@9baa671e9aa6 → speed_uniform@1f985387534c → clip2@fba5388dcf34 → shape_uniform@8e0abba5490c → speed4@f72cd9510d7f → new_control@297ebde7d0cb → size_log@3227adb0e837

通过 gate 的七指标 Pareto：clip2@fba5388dcf34, ego_sincos@5cc9b3bc543b, new_control@297ebde7d0cb, optimizer_sd@9baa671e9aa6, shape_p2@bdcb4ed434f3, shape_p2_speed4@010d7ce0e5d8, shape_uniform@8e0abba5490c, speed4@f72cd9510d7f, speed_uniform@1f985387534c

## ema / B256 / updates 1536 / exposure 393,216

cohort `f4507cfec996`，eval manifest `c552af064913`，1024 场景。

| 配方 | TRAIN N | S mean±SD | R | R_attr | onroad | V/GT | gates / ranking |
|---|---:|---:|---:|---:|---:|---:|---|
| batch256 `010d7c` | 1 | 0.2714±0.0000 | — | — | 78.06% | 1.017 | no control |
| combo_batch256_ego_sincos `5cc9b3` | 1 | 0.2729±0.0000 | — | — | 78.27% | 1.017 | no control |

**逐分量相对 new_control 的百分比变化**（正数退步；按 paired TRAIN arm means）：

| 配方 | near | lat | angle | length | width | speed | collision | paired TRAIN N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch256 | — | — | — | — | — | — | — | 0 |
| combo_batch256_ego_sincos | — | — | — | — | — | — | — | 0 |

通过 gate 的 R 排名：无

通过 gate 的七指标 Pareto：无

## 实际成本与当前状态

完成 fresh 27，continuation 4，eval-only 0；partial 1。

已保存连续 trace 确认的实际 updates 19,712；实际 TRAIN cache exposures 9,306,112。1 stages 仍无精确更新记录，不用 planned budget 冒充。

已知去重 fit 109.93 分钟，test 6.55 分钟；不包含 setup/preprocess/未知失败耗时，不是 GPU busy 时间。

| stage | kind | status | actual updates | actual samples | metric rows |
|---|---|---|---:|---:|---:|
| combo_batch256_ego_sincos/combo_batch256_ego_sincos | fresh | complete | 1536 | 393216 | 2 |
| long_seed817_new_control/new_control | fresh | partial_or_running | None | None | 0 |
| loss_interaction/shape_p2_speed4 | fresh | complete | 768 | 393216 | 3 |
| phase1/col10 | fresh | complete | 512 | 262144 | 3 |
| phase1/col5 | fresh | complete | 512 | 262144 | 3 |
| phase1/heading_mse1 | fresh | complete | 512 | 262144 | 3 |
| phase1/heading_p0 | fresh | complete | 512 | 262144 | 3 |
| phase1/heading_p2 | fresh | complete | 512 | 262144 | 3 |
| phase1/new_control | fresh | complete | 512 | 262144 | 3 |
| phase1/old_control | fresh | complete | 512 | 262144 | 3 |
| phase1/pos16 | fresh | complete | 512 | 262144 | 3 |
| phase1/pos_l1_16 | fresh | complete | 512 | 262144 | 3 |
| phase1/shape4 | fresh | complete | 512 | 262144 | 3 |
| phase1/shape_p2 | fresh | complete | 512 | 262144 | 3 |
| phase1/speed4 | fresh | complete | 512 | 262144 | 3 |
| phase1/speed_p0 | fresh | complete | 512 | 262144 | 3 |
| phase1/width4 | fresh | complete | 512 | 262144 | 3 |
| phase1_extend/new_control | continuation | complete | 256 | 131072 | 1 |
| phase1_extend/old_control | continuation | complete | 256 | 131072 | 1 |
| phase1_extend/shape_p2 | continuation | complete | 256 | 131072 | 1 |
| phase1_extend/speed4 | continuation | complete | 256 | 131072 | 1 |
| phase2/clip2 | fresh | complete | 768 | 393216 | 3 |
| phase2/count_embedding | fresh | complete | 768 | 393216 | 3 |
| phase2/edge_mlp | fresh | complete | 768 | 393216 | 3 |
| phase2/ego_sincos | fresh | complete | 768 | 393216 | 3 |
| phase2/lr1e3 | fresh | complete | 768 | 393216 | 3 |
| phase2/lr2e4 | fresh | complete | 768 | 393216 | 3 |
| phase2/optimizer_sd | fresh | complete | 768 | 393216 | 3 |
| phase2/shape_uniform | fresh | complete | 768 | 393216 | 3 |
| phase2/speed_uniform | fresh | complete | 768 | 393216 | 3 |
| phase2_batch256/batch256 | fresh | complete | 1536 | 393216 | 3 |
| phase2_size_log/size_log | fresh | complete | 768 | 393216 | 3 |

## 排除的 worker benchmark 诊断

`worker2_benchmark` 是 DataLoader workers=2 性能 pilot，默认不计入主模型数、主 metric rows 或任何模型配方排名。实际记录和成本保留如下，不是丢弃实验。

排除 pilot 有效 metric rows 1，stage 1；已知实际 updates 128 / sample exposures 65,536；fit 0.43 分钟、test 0.01 分钟。未保存 trace 的成本仍标 unknown。

| pilot | status | actual updates | actual samples | metric rows |
|---|---|---:|---:|---:|
| worker2_benchmark/new_control | complete | 128 | 65536 | 1 |

## 独立保留的 map-policy comparison

`map_policy_comparison_*` 及其所有 nested evaluator stages 是额外的随机训练地图/预训练固定地图对比，绝不用于主搜索排名、候选选择或主模型成本计数。独立成本仍记录，结果由各 comparison 目录的 comparison_summary.json 汇总。

有效 metric rows 0，eval-only stages 0；实际新增 TRAIN updates 0 / exposures 0，已知 test 0.00 分钟。


## 解释范围

所有排名只描述已测试 budget/cohort。quick trend 无法证明 128 epoch 最终最优；新4096确认集仍位于之前审计过的50k池中，不是 pristine unseen test。

817 是自适应搜索及long阶段选择 seed；1817/2817 两个 TRAIN seeds 在冻结候选对之后才用于确认。最终可合法报告总 TRAIN N=3，但只有 **2 个独立确认 seeds**，不能写成3次独立确认。主分数仍按全部3个TRAIN等权计算，同时 JSON 保留仅1817/2817的敏感性汇总。每 TRAIN 有2个 GEN 重复时也不是 N=6。

paired summary 已保存在各 group 的 comparison/per_training_seed，independent_confirmation_only_sensitivity 是额外核验，不替代事先声明的 all-three primary。
