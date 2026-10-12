# Training/evaluation leakage audit

All 493,542 training cache filenames parse as `scenario_id/lg_type/scene_timestep`, representing 486,994 unique scenario IDs. We loaded `scenario_id` and source metadata from all 5,120 selected search and heldout cache files.

Training/search and training/heldout scenario-ID intersections are both zero, including hexadecimal-normalized comparison. Search/heldout scenario-ID intersection is also zero. Actual evaluation metadata agrees with every filename's TFRecord, record index, graph type and timestep.

The compact training cache drops TFRecord and record index, so direct training matching by that raw key is unavailable. `data2initmap.py` preserves the scenario-ID filename produced by `data_preprocess_scenario_dreamer.py`, which uses the raw Scenario protobuf `scenario_id`. This supports no scenario-ID leakage for the selected evaluation scenes, without claiming a direct training TFRecord-index audit.

The final evaluation filename suffix is `scene_timestep`, separate from the 0/1 `map_id` embedding category. Independent original-cache validation found that the explicit saved `scene_timestep` matches the suffix in all 64 checked official pickles; see `filename_suffix_evidence.json`.
