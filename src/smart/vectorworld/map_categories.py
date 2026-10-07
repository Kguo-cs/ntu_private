"""Shared split-aware Waymo labels, with an explicit legacy cache policy."""
from ..scenario_dreamer.map_categories import (
    KEY_PATTERN, SCENE_PATTERN, SOURCE_RAW_SPLITS, SPLIT_POLICY, NATIVE_POLICY,
    MapCategoryIndex, apply_category_index, classify_category, import_category_keys,
    load_category_keys, scene_category_key,
)
