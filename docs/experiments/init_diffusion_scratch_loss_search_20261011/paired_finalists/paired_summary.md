# Paired scratch finalist confirmation

Primary endpoint: **ema / 2048 steps**. Selected variant: **geo_mse32_col20_heading_l1_shape2**.

Independent training seeds are the sample units. Generation seeds are averaged inside each training seed. SD uses ddof=1 (N=1: 0), is descriptive, and is not a confidence interval. Each training seed's generation range is recorded separately in JSON/CSV.

## ema: primary

Complete endpoint: True; paired training N=3; generation repetitions per training seed=2.

JSD values below use natural logarithms, with theoretical maximum ln2 (helper nearest/lateral divided by10, other JSD by100); collision is percent.

| Metric | Control mean ± sample SD | Candidate mean ± sample SD | Paired change mean ± sample SD |
|---|---:|---:|---:|
| nearest_dist_jsd | 0.01258 ± 0.00045 | 0.01614 ± 0.00060 | +28.46 ± 9.24 % |
| lat_dev_jsd | 0.08996 ± 0.00160 | 0.09299 ± 0.00172 | +3.38 ± 1.88 % |
| ang_dev_jsd | 0.01778 ± 0.00175 | 0.00574 ± 0.00139 | -68.03 ± 4.44 % |
| length_jsd | 0.00798 ± 0.00122 | 0.00942 ± 0.00046 | +20.34 ± 23.26 % |
| width_jsd | 0.00506 ± 0.00092 | 0.00795 ± 0.00092 | +58.86 ± 18.89 % |
| speed_jsd | 0.00135 ± 0.00046 | 0.00181 ± 0.00077 | +34.32 ± 27.49 % |
| collision_rate | 1.65054 ± 0.11363 | 1.50459 ± 0.06242 | -8.63 ± 5.66 % |
| score | 0.23054 ± 0.00138 | 0.22267 ± 0.00510 | -3.40 ± 2.69 % |
| generated_onroad_fraction | 79.04066 ± 0.89279 | 79.54011 ± 0.57095 | +0.50 ± 0.60 pp |
| vehicle_count_ratio | 1.02655 ± 0.00925 | 1.02421 ± 0.01480 | -0.22 ± 1.57 % |

## online: diagnostic only

Complete endpoint: True; paired training N=3; generation repetitions per training seed=2.

JSD values below use natural logarithms, with theoretical maximum ln2 (helper nearest/lateral divided by10, other JSD by100); collision is percent.

| Metric | Control mean ± sample SD | Candidate mean ± sample SD | Paired change mean ± sample SD |
|---|---:|---:|---:|
| nearest_dist_jsd | 0.01532 ± 0.00231 | 0.01545 ± 0.00112 | +2.45 ± 17.81 % |
| lat_dev_jsd | 0.08904 ± 0.00101 | 0.09246 ± 0.00067 | +3.85 ± 1.76 % |
| ang_dev_jsd | 0.03290 ± 0.01673 | 0.00902 ± 0.00378 | -67.48 ± 16.43 % |
| length_jsd | 0.07517 ± 0.06755 | 0.01746 ± 0.01240 | -0.30 ± 134.87 % |
| width_jsd | 0.13282 ± 0.12809 | 0.02421 ± 0.02012 | +179.81 ± 456.51 % |
| speed_jsd | 0.01195 ± 0.00956 | 0.01013 ± 0.00373 | +3.68 ± 35.62 % |
| collision_rate | 3.03729 ± 0.38894 | 2.28877 ± 0.41524 | -22.99 ± 21.47 % |
| score | 0.47159 ± 0.18312 | 0.27520 ± 0.03675 | -33.68 ± 31.03 % |
| generated_onroad_fraction | 78.31160 ± 1.22106 | 79.83819 ± 1.55042 | +1.53 ± 1.92 pp |
| vehicle_count_ratio | 1.05605 ± 0.01841 | 1.01467 ± 0.04871 | -3.89 ± 5.05 % |

Only the complete frozen EMA endpoint may select deployment. Online results and other budgets cannot reselect. The larger full-50k audit is separate from this decision.

The reported training cost deduplicates each (run_path, variant) stage using its maximum elapsed time; it covers only stages present in this input, not the whole search.
