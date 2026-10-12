# Hybrid position sampler: independent CPU review

Six isolated CPU tests and the float32 N40 timetable passed. Only one positive timestep is below 0.05: 0.02500000037. With dt=-t, the original 0.05 denominator moves the final position halfway toward x0; the position-only 0.0001 denominator reaches x0. The global floor, non-position velocity, x0 prediction, fixed ego masks, training loss and EMA are preserved.

This is a separate inference cohort: four fresh N40 anchors and four hybrid cells reuse two TRAIN checkpoints and two GEN seeds on the same 4096 scenes. Selection requires all eight actual successful cells and pooled/per-TRAIN gates. Full50k validates the frozen choice only. The old 1e-6 N20 replay failure remains excluded. No GPU outcome is inferred from these CPU tests.

Position metrics and collision may change. Non-position outputs are equal within the same CPU call; separate CUDA runs have observed replay drift, so bitwise pairing is not claimed. The wrapper restores on successful test end; interruption leaves only an isolated failed child, never an eligible row.
