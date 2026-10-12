# Final targeted stage: CPU design recommendation

Add `geometry_strong_heading_l1_width_p2` as the fourth targeted arm, then freeze recipe expansion and confirm finalists. This is a proposal, not a tested improvement.

Only change the current `geometry_strong_heading_l1` width time power from 1 to 2. Keep position L1/data/p1/w4; circular-angle heading L1/p1/w1; length MSE/data/p1/w1; width MSE/data/p2/w1; speed MSE/data/p1/w2; collision L1/p1/w5; type CE/p0/w1. Time weights remain normalized.

The MSE-heading p1/p2 × width p1/p2 controls complete a clean 2×2 power experiment using `geometry_strong`, `warm_winner`, `geometry_strong_width_p2`, and `geometry_strong_heading_p2`. Adding the L1-heading width-p2 arm tests whether the strong heading result survives the width schedule. The stronger-map/shape/collision arm changes several factors and supplies an alternative recipe, not a single-factor explanation.

Actual 4096-step snapshot (one training seed, one generation seed):

| Variant | Weights | S ↓ | Angle JSD ×100 | Length JSD ×100 | Width JSD ×100 | Speed JSD ×100 | Collision % | On-road % | Vehicles / GT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry_strong_heading_l1 | ema | 0.2956 | 0.864 | 1.182 | 0.482 | 0.427 | 3.022 | 76.2 | 1.035 |
| warm_winner | ema | 0.3122 | 3.285 | 1.099 | 0.453 | 0.341 | 2.535 | 74.9 | 1.017 |
| geometry_strong | ema | 0.3210 | 3.009 | 1.171 | 0.582 | 0.356 | 2.799 | 75.6 | 1.024 |
| geo_l1_16 | ema | 0.3966 | 6.015 | 1.318 | 0.574 | 0.388 | 4.142 | 80.5 | 1.017 |
| legacy | ema | 0.4261 | 8.193 | 2.056 | 1.948 | 0.316 | 3.683 | 78.3 | 1.023 |
| balanced32_l1_quarter | ema | 0.4767 | 7.319 | 1.723 | 1.228 | 0.806 | 5.882 | 77.9 | 1.079 |
| warm_winner | online | 0.3779 | 4.336 | 1.720 | 1.700 | 0.865 | 3.635 | 76.9 | 0.979 |
| geometry_strong_heading_l1 | online | 0.3920 | 3.870 | 1.356 | 2.691 | 0.463 | 4.353 | 78.3 | 1.025 |
| geometry_strong | online | 0.4057 | 5.575 | 5.001 | 1.180 | 0.431 | 3.324 | 75.7 | 0.996 |
| geo_l1_16 | online | 0.4676 | 6.008 | 1.221 | 3.804 | 0.502 | 5.902 | 80.3 | 1.004 |
| legacy | online | 0.5632 | 9.296 | 9.588 | 10.099 | 1.447 | 3.142 | 79.0 | 0.987 |
| balanced32_l1_quarter | online | 0.7251 | 10.754 | 6.126 | 8.236 | 1.428 | 8.808 | 75.6 | 1.144 |

S = mean(metric / scale), scales nearest=1, lateral=1, angle=10, length=20, width=15, speed=20, collision=5 in the helper output units. This auxiliary score is not a paper metric.

The L1-heading incumbent improves EMA S by 5.3% and angle JSD by 73.7% relative to `warm_winner`, with road coverage +1.32 points. It increases collision by 19.2% (3.02% vs 2.53%). Its online S is worse than warm control, so the advantage depends on evaluating EMA; the recipes are not Pareto ordered.

Continuation and confirmation:

- Keep the current EMA incumbent and warm control for 6144 updates; choose the other two only after actual targeted results and 2048→4096 trends are available. Require road fraction ≥.65 and vehicle count within ±20% of matched GT, then inspect full histogram and coverage details.
- Rank online and EMA separately. Declare the intended deployment weights before choosing the final recipe.
- Freeze after this stage. Confirm two finalists using original batch512, three fresh training seeds, paired ordered data/evaluation and generation seeds, and a disjoint heldout scene set.
- Report updates and seen scenes. 6144×64 = 393216 scenes matches 768×512; 1024 updates at batch512 sees 524288 scenes. Equal updates alone is not equal exposure.
- EMA .9999 is per update; keeping it at batch512 increases its smoothing horizon in scenes eightfold. Hold the intended production EMA setting constant across finalists and report it.
- Three independent training seeds establish early-budget reproducibility. Generation repetitions estimate sampler noise, not training variation. ~.53/.80 epoch at 4096/6144 steps cannot prove a full128-epoch optimum.

Source: `logs/component_loss_scratch/long4096_w0/summary.json`; SHA256 `eea6cb315620a0af1fa28e234595e7d248b32a2847f3001ef2a9c5b1758865e9`. The seventh long arm and all targeted arms were still absent from this snapshot; no metrics are invented for them.
