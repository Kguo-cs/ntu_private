"""Audit how the full-training recipe differs from native search configurations."""
from __future__ import annotations
import copy
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now
ART = ROOT / 'output/reports/init_diffusion_expanded_20261014'
RUNS = ROOT / 'logs/init_diffusion_expanded_20261014'


def differences(a, b, prefix=''):
    result = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() | b.keys():
            result.update(differences(a.get(key, '<absent>'), b.get(key, '<absent>'),
                                      (prefix + '.' + key).lstrip('.')))
    elif a != b:
        result[prefix] = dict(full=a, native=b)
    return result


def main():
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        base = compose(config_name='run', overrides=['experiment=init_diffusion_657_n20_road_k8_full'])
    with open_dict(base):
        base.paths.root_dir = str(ROOT / 'src')
        base.paths.output_dir = str(ART / 'full_recipe_preflight')
        base.paths.log_dir = str(ART / 'full_recipe_preflight')
    variants = read(ART / 'variants.json')
    rows = read(ART / 'observations.json')
    results = []
    allowed = {'decoder.init_diffusion.collision_guidance': (None, '<absent>'),
               'decoder.init_diffusion.sampling_exact_x0': (False, '<absent>'),
               'decoder.init_diffusion.warm_start_from_ema': (False, '<absent>'),
               'decoder.init_diffusion.component_loss.components.speed_magnitude':
                   (dict(kind='mse', time_power=0, weight=0., scale='data'), '<absent>'),
               'sd_r_reference': (str(ROOT / 'configs/metrics/iu3fa4gx_full50000.json'), None),
               'sd_require_full_set': (True, False),
               'decoder.init_diffusion.normalizer_path':
                   (str(ROOT / 'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'), None)}
    for arm in sorted({r['variant'] for r in rows if r['budget'] == 2048}):
        row = next(r for r in rows if r['variant'] == arm and r['budget'] == 2048)
        cfg_path = Path(row['checkpoint']).parent / 'config.yaml'
        native = OmegaConf.load(cfg_path)
        full = copy.deepcopy(base)
        for key, value in variants[arm]['__config'].items():
            OmegaConf.update(full, key, value, force_add=True)
        for component, overrides in variants[arm].items():
            if not component.startswith('__'):
                for option, value in overrides.items():
                    OmegaConf.update(full, 'model.model_config.decoder.init_diffusion.component_loss.components.'
                                     + component + '.' + option, value, force_add=True)
        full.model.model_config.lr = variants[arm].get('__lr', full.model.model_config.lr)
        a = OmegaConf.to_container(full.model.model_config, resolve=True)
        b = OmegaConf.to_container(native.model.model_config, resolve=True)
        diff = differences(a, b)
        for key, values in diff.items():
            assert key in allowed and (values['full'], values['native']) == allowed[key], (arm, key, values)
        assert full.action == 'fit' and full.ckpt_path is None
        assert full.model.model_config.finetune and not full.model.model_config.decoder.sep_map
        assert full.trainer.max_epochs == 128 and full.trainer.check_val_every_n_epoch == 8
        assert full.data.shuffle and full.data.num_workers == 8 and full.data.train_batch_size == 512
        assert full.trainer.limit_val_batches == 1.0 and isinstance(full.trainer.limit_val_batches, float)
        assert full.model.model_config.sd_expected_scenes == 50000
        assert full.model.model_config.sd_require_full_set and full.model.model_config.sd_reference_mode == 'matched'
        assert sha(a['decoder']['init_diffusion']['normalizer_path']) == row['calibration_sha256']
        assert a['initial_map_checkpoint'] == row['map_checkpoint']
        source = a['decoder']['init_diffusion'].get('attribute_source')
        if source:
            assert sha(ROOT / 'configs/initialization_sources/train_attribute_quantiles_20261014.pt') == source['sha256']
        results.append(dict(arm=arm, native_config=str(cfg_path), native_config_sha256=sha(cfg_path),
                            model_config_differences=diff, normalizer_sha256=row['calibration_sha256'],
                            map_checkpoint_sha256=row['map_checkpoint_sha256']))
    save(ART / 'full_recipe_preflight.json', dict(status='PASS', generated_utc=now(),
        observations_sha256=sha(ART / 'observations.json'), results=results,
        scope='Hydra composition and model-configuration parity for completed2048-update arms; '
              'not a full128epoch outcome or a runtime smoke test of a final selected configuration.',
        intended_differences=['Full recipe reshuffles every epoch; quick runner repeats a fixed seeded order.',
            'Full recipe loads the same normalizer JSON in its constructor; native runner copies its buffers explicitly.',
            'Full recipe evaluates complete50000 scenes and reports ratios against iu3fa4gx; screen uses4096 and paired controls.',
            'Full recipe uses8 data workers; quick runner uses0.'],
        default_parity='Explicit None/False sampling settings match absent defaults; the extra speed_magnitude term has zero weight.'))
    print({'status': 'PASS', 'arms_checked': len(results)})


if __name__ == '__main__':
    main()
