"""Opt-in model/transport experiments using the existing scratch training chain."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import scratch_component_loss_sweep as sweep

PREFIX = "model.model_config.decoder.init_diffusion."
sweep.SEARCH_CONFIG_KEYS.update({PREFIX + key for key in (
    "matching_weights", "time_sample_power", "distribution_loss", "denoiser_hidden_dim",
    "denoiser_num_layers", "sampling_steps", "state_feature_normalization",
    "speed_x0_parameterization", "speed_head_kind", "speed_head_detach_context",
    "map_condition_types",
)})

if __name__ == "__main__":
    sweep.main()
    if os.environ.get("SIM_SWEEP_DEFER_CYCLIC_GC") == "1" or "--defer-cyclic-gc" in sys.argv:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
