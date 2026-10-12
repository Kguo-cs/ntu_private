# Compact scratch-loss report figures

PNG only, based on actual analyzer records. Each cohort is plotted separately.

| Short label | Full recipe variant |
|---|---|
| Old recipe | `warm_winner` |
| Geometry + head L1 | `geo_mse32_col20_heading_l1_shape2` |
| Head L1 + width p2 | `geometry_strong_heading_l1_width_p2` |

JSD units: `native`. Current native angle/speed/size metrics are natural-log JSD x100; raw JSD has maximum ln2.
Error bars/bands use independent training-seed SD, not confidence intervals.

![summary_81e7d4ad67b7_native.png](summary_81e7d4ad67b7_native.png)
![size_81e7d4ad67b7_native.png](size_81e7d4ad67b7_native.png)
![summary_c929105420f6_native.png](summary_c929105420f6_native.png)
![size_c929105420f6_native.png](size_c929105420f6_native.png)
