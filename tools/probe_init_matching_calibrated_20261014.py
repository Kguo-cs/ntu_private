"""Re-run the prior endpoint-weight probe with the actual frozen TRAIN stats."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import probe_init_matching_20261013 as prior
from tools.run_wandb_cc_scratch_20261013 import NORMALIZER, read, save, sha, now

ART = ROOT / 'output/reports/init_diffusion_overnight_20261014/matching_probe_calibrated'
prior.ART = ART
prior.OPTIONS = {'baseline': None, **{
    f'heading{s}_speed{s}': dict(position=1., heading=float(s), length=1., width=1., velocity=float(s))
    for s in (2, 4)}}
instantiate = prior.hydra.utils.instantiate


def calibrated(*args, **kwargs):
    value = instantiate(*args, **kwargs)
    if hasattr(value, 'encoder') and hasattr(value.encoder, 'init_decoder'):
        value.encoder.init_decoder._load_train_normalizer(str(NORMALIZER))
    return value


if __name__ == '__main__':
    prior.hydra.utils.instantiate = calibrated
    prior.main()
    result = read(ART / 'results.json')
    result.update(normalizer_sha256=sha(NORMALIZER), wrapper_sha256=sha(__file__),
        corrected_utc=now(), correction='Prior probe silently fitted first-batch statistics. This run explicitly loads frozen TRAIN8192 statistics before get_input/source drawing.')
    save(ART / 'results.json', result)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
