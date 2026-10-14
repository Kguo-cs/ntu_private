"""Native sweep with the architecture search fields explicitly authorized."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import runtime_checkpointed_sweep as native

native.sweep.SEARCH_CONFIG_KEYS.update({
    'model.model_config.decoder.' + key for key in
    ('hidden_dim', 'num_freq_bands', 'num_heads', 'head_dim', 'num_map_layers',
     'pl2pl_radius', 'pt2pt_neighbor')})
native.sweep.SEARCH_CONFIG_KEYS.update({
    'model.model_config.decoder.init_diffusion.' + key for key in
    ('denoiser_num_freq_bands', 'denoiser_num_heads', 'denoiser_head_dim')})

if __name__ == '__main__':
    native.runtime.main()
