#!/usr/bin/env bash

CODE_DIR="$PROJECT_ROOT/data_processing/waymo"

cd "$CODE_DIR"
# ignore all the tf warnings
python generate_waymo_dataset.py generate_waymo_dataset.mode=train

python -m sd.data_processing.waymo.generate_waymo_dataset generate_waymo_dataset.mode=val

python -m sd.data_processing.waymo.generate_waymo_dataset generate_waymo_dataset.mode=test

python -m sd.data_processing.waymo.add_nocturne_compatible_val_scenarios_to_test