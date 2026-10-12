# Main4096 results and fixed extension rule

The main seven recipes have completed. Retain `legacy` and `warm_winner` as controls; after all four targeted recipes finish, select the two lowest-S eligible non-control EMA recipes at4096 updates on the same1024 search scenes. Require generated on-road fraction ≥.72 and generated/GT vehicle count between .8 and1.2. Freeze these thresholds, auxiliary score, candidate pool, and primary EMA mode now; report online results separately.

This guard is reasonable for early-stage selection. It is not a final map-adherence threshold: matched GT on-road fraction is99.398%, and current main EMA coverage is74.9–80.5%. Angle/lateral JSD uses only eligible on-road vehicles, so report on-road counts and all-feature histogram mass alongside the score. All main seven currently pass the gate; the current main-only non-control choices are `geometry_strong_heading_l1` and `geometry_strong`, subject to actual targeted results.

The rule is fixed for the next adaptive stage; it was not preregistered before the completed earlier sweep. Do not loosen the gate or change metric weights after seeing target outcomes. If fewer than two non-controls qualify, report that shortage.

Actual primary EMA ranking at4096:

| Rank | Recipe | S ↓ | Angle JSD ×100 | Collision % | Road % | Vehicles/GT |
|---:|---|---:|---:|---:|---:|---:|
| 1 | geometry_strong_heading_l1 | 0.295633 | 0.864 | 3.022 | 76.22 | 1.0349 |
| 2 | warm_winner | 0.312231 | 3.285 | 2.535 | 74.90 | 1.0170 |
| 3 | geometry_strong | 0.321019 | 3.009 | 2.799 | 75.56 | 1.0237 |
| 4 | geo_mse32_col20 | 0.340520 | 7.456 | 1.588 | 78.92 | 1.0055 |
| 5 | geo_l1_16 | 0.396648 | 6.015 | 4.142 | 80.49 | 1.0173 |
| 6 | legacy | 0.426052 | 8.193 | 3.683 | 78.31 | 1.0233 |
| 7 | balanced32_l1_quarter | 0.476734 | 7.319 | 5.882 | 77.88 | 1.0789 |

S = mean(metric / scale), scales nearest=1, lateral=1, angle=10, length=20, width=15, speed=20, collision=5 in helper output units. It is an auxiliary score, not an official paper metric.

Online rank changes, restricted to the same seven recipes at all budgets:

| Recipe | 1024 rank / S | 2048 rank / S | 4096 rank / S |
|---|---:|---:|---:|
| balanced32_l1_quarter | 5 / 0.749037 | 7 / 0.615192 | 7 / 0.725051 |
| geo_l1_16 | 7 / 0.772170 | 3 / 0.477328 | 4 / 0.467559 |
| geo_mse32_col20 | 1 / 0.553141 | 4 / 0.498203 | 5 / 0.528542 |
| geometry_strong | 4 / 0.667612 | 2 / 0.404943 | 3 / 0.405731 |
| geometry_strong_heading_l1 | 2 / 0.617127 | 1 / 0.386723 | 2 / 0.392008 |
| legacy | 6 / 0.758466 | 5 / 0.567945 | 6 / 0.563192 |
| warm_winner | 3 / 0.647015 | 6 / 0.590587 | 1 / 0.377869 |

EMA was first measured at2048; do not splice the1024 online ranking into an EMA learning curve. The L1-heading recipe stays first in EMA at2048 and4096. Warm moves third→second and geometry_strong second→third, but their2048 scores differ by only0.000006 and should be treated as tied. Geo-MSE32/col20 stays fourth in EMA and supplies the lowest main collision rate, trading against angular quality.

Linear forecasts are unreliable:

- L1-heading online S falls .617127→.386723, then rises to .392008. Strong-geometry online S similarly improves early then plateaus .404943→.405731.
- Geo-MSE32/col20 begins first at1024 but ends fifth online at4096; its score worsens after2048. Warm begins third, falls to sixth, then finishes first online. Early slope and early rank predict neither late rank nor slope.
- Online scalar JSD oscillates while EMA improves. Track the same evaluation weights through time and independently confirm training-seed variation.

Qualitative final-training forecast and confirmation:

- Circular-angle L1 is promising from scratch because it supplies a useful angular correction across large errors and now gives the strongest EMA angular distribution. Its full-training advantage is not proven.
- Separate width and heading powers with the targeted2×2 controls; warm’s late recovery argues against selecting from1024 results alone. The stronger-map/shape/collision targeted arm is a multi-factor alternative, not a causal single-factor test.
- Use actual6144 results plus three fresh training seeds at original batch512 and paired disjoint heldout evaluation to choose the final recipe. Preserve the intended EMA setting and keep online diagnostics.
- Compare seen scenes as well as updates:6144×64 =393216, equivalent to768×512, and only~.797epoch of493542 scenes. Fixed EMA decay is per update, so changing batch size changes its smoothing horizon in scenes.
- Do not emit predicted128-epoch JSD numbers or claim the global optimum. The evidence supports a shortlist and qualitative trends, with seed and budget uncertainty.

Targeted candidate pool: `geometry_strong_heading_l1_width_p2`, `geometry_strong_width_p2`, `geometry_strong_heading_p2`, `geo_mse32_col20_heading_l1_shape2`. No targeted result was present in this snapshot.


Original-batch confirmation addendum:

Fresh batch512 training to2048 updates, with checkpoints768/1536/2048 (~.797/1.593/2.125epochs), should take priority over spending all confirmation compute on three sub-epoch repetitions. It reveals behavior after repeat passes while768 provides the sample-matched bridge. Confirm with two additional training seeds at the same2048 budget if feasible; shorter-seed results are separate-budget evidence.

Keep real full-manifest epoch boundaries and scheduler steps. Enable multi-epoch only for fresh runs until resume offsets/RNG/epoch state are generalized. Preserve genuine scheduler advances and undo only the artificial partial-final-stage step. A repeated fixed shuffled order is useful for pairing but should be disclosed if normal full training reshuffles each epoch. Two epochs still cannot establish128-epoch convergence.
