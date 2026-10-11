# Targeted4096 factorial and interaction evidence

All37 distinct loss recipes have actual runs. The four targeted recipes completed4096 updates. The fixed gate selects `geo_mse32_col20_heading_l1_shape2` and `geometry_strong_heading_l1_width_p2` for6144 continuation, alongside `legacy` and `warm_winner` controls. This is a shortlist, not a full-training optimum.

Metrics below use official helper output units: nearest/lateral raw JSD×10, angle/length/width/speed raw JSD×100; collision is percent. S is mean(metric/scale), with scales1,1,10,20,15,20,5. EMA is the primary cohort; online remains separate.

Clean time-power/kind family at4096 EMA:

| Recipe | Heading | Width power | S ↓ | Angle JSD×100 | Width JSD×100 | Speed JSD×100 | Collision % | On-road % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_strong | msep1,w1 | 1 | 0.321019 | 3.009 | 0.582 | 0.356 | 2.799 | 75.56 |
| geometry_strong_width_p2 | msep1,w1 | 2 | 0.311399 | 2.879 | 0.449 | 0.397 | 2.767 | 75.20 |
| geometry_strong_heading_p2 | msep2,w1 | 1 | 0.329336 | 3.902 | 0.583 | 0.369 | 2.691 | 75.54 |
| warm_winner | msep2,w1 | 2 | 0.312231 | 3.285 | 0.453 | 0.341 | 2.535 | 74.90 |
| geometry_strong_heading_l1 | l1p1,w1 | 1 | 0.295633 | 0.864 | 0.482 | 0.427 | 3.022 | 76.22 |
| geometry_strong_heading_l1_width_p2 | l1p1,w1 | 2 | 0.287347 | 0.831 | 0.581 | 0.487 | 2.817 | 76.63 |

The four MSE-heading recipes form a clean2×2 heading-power×width-power comparison. Widthp2 improves composite S at either heading power (~3.0% and5.2%); headingp2 worsens S when widthp1 and is essentially tied when widthp2. It does not improve the angular metric here. The descriptive interaction in S is−.007485, but one paired seed cannot establish statistical significance. Width-only and warm differ by only.000831 in S.

Heading angleL1 atp1 lowers angular JSD by~71% versus matching angleMSE at either width power. These comparisons keep headingweight1 and other component settings fixed. This provides more direct evidence for the heading-kind effect than the multi-factor recipe below.

L1-heading widthp2 versusp1 improves S2.80% and collision6.79%, but worsens width JSD20.48%, length2.83% and speed14.18%. It is selected for the composite tradeoff, not because width fidelity improved. Its binned width mean moves2.1477→2.1446m while GT is2.1631m; spread moves slightly towardGT.1856m (.1685→.1706m), illustrating why a small mean/spread change does not guarantee lower histogram JSD.

Strong geometry/shape interaction:

| Recipe | S ↓ | Angle JSD×100 | Length JSD×100 | Width JSD×100 | Speed JSD×100 | Collision % | On-road % | Vehicles/GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| geo_mse32_col20 | 0.340520 | 7.456 | 1.075 | 0.413 | 0.524 | 1.588 | 78.92 | 1.0055 |
| geo_mse32_col20_heading_l1_shape2 | 0.267968 | 1.121 | 1.044 | 0.501 | 0.738 | 1.916 | 78.57 | 1.0107 |
| geometry_strong_heading_l1_width_p2 | 0.287347 | 0.831 | 1.216 | 0.581 | 0.487 | 2.817 | 76.63 | 1.0267 |

The joint strong-map recipe retains posMSE/data/p1/w32 and collisionL1/p1/w20; changes headingMSE/p1/w.25→angleL1/p1/w1 and length/widthMSE/p1 weights1→2. S improves21.31% against its parent, mainly from angle−84.97%. Width, speed and collision worsen21.19%,40.91%,20.68%; length improves2.81% and road drops.344points. These are multiple simultaneous changes. Doubling shape weights is not established as the reason for success.

Compared with the other selected L1W2 recipe, its lower composite comes mainly from collision1.916% versus2.817%, with higher angular and speed JSD. It is not Pareto superior. Both preserve acceptable vehicle counts and road coverage under the fixed early-stage gate; road still falls well short ofGT99.398%.

Mechanistic interpretation, with source and empirical limits:

- `component_loss.py:85` uses normalized `clamp(t,.05)^(-p)`. Smallt is near clean data. Normalization sets E[w]=1 under uniformt; it does not equalize E[wL(t)] or component gradient norms. p2 assigns76.9% of weight tot<.1 versus42.4% underp1 and only2.56% tot>.5 versus17.35%. It emphasizes local denoising much more than high-noise generation.
- In the actual last256 updates, the L1-head pair’s median width weighted loss falls.09998→.02472 underp2 with numericalweight1 unchanged. Low-noise residuals are easier; a higher time power can reduce the effective contribution after multiplying by residuals.
- AngleL1 uses constant residual-gradient magnitude away from its cusp, whereas angleMSE’s magnitude scales with angular error. Reduced domination by uncertain large-error headings is a plausible explanation for the angular distribution improvement. Both current losses use circular-angle residuals; this is not the old sin/cos-MSE weak-gradient-at180deg issue.
- Sizes, heading and positions share the denoiser and collision geometry, and matching can couple correspondence. All inspected256-step tails exceed globalclip=.5: L1-head preclip median3.32; joint strong-map7.72 versusparent6.17. Changing component weights therefore changes shared gradient directions under clipping. Scalar loss values do not provide per-component gradient decompositions.
- Binned moments remain biased: jointrecipe lengthmean/std4.834/.587m vsGT4.892/.795m; width2.148/.173 vsGT2.163/.186m; speed4.959/5.512 vsGT5.588/6.140m/s. These are midpoint estimates of fixed bins, and generated vehicle-type composition also affects them. Speed firstbin0–1m/s is not exact stationary frequency.

Pairing/provenance: all4096 EMA records share seed817, generationseed2817, random generator/map hashes, full train-order hash, calibration/eval hashes and nominalLR1e-4. Remaining-manifest hashes differ between fresh and resumed runs by design; continuation order/state was separately audited. No bitwise CUDA equivalence is claimed.

Freeze further loss expansion. Use actual6144 continuation, fresh originalbatch512 checkpoints768/1536/2048, independent training seeds and disjoint heldout evaluation to decide the final recipe. Do not produce numeric128-epoch forecasts from this sub-epoch search.
