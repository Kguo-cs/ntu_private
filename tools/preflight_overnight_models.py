"""CPU-only real-batch smoke checks, excluded from all GPU search evidence."""
import argparse
import copy
import json
import pickle
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from tools.continue_overnight_init_diffusion import ART, design_arms, save


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--variants', default='')
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to empty for this CPU-only diagnostic')
    torch.set_num_threads(1)
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    names = set(args.variants.split(',')) if args.variants else None
    results = []
    with tempfile.TemporaryDirectory(dir='/tmp', prefix='sim_overnight_smoke_') as temporary:
        folder = Path(temporary)
        with (ROOT / 'src/waymo_data/full/training_map2_sd_files.pkl').open('rb') as handle:
            files = pickle.load(handle)['files'][:2]
        with (folder / 'train.pkl').open('wb') as handle:
            pickle.dump({'files': files}, handle)
        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
            cfg = compose(config_name='run', overrides=['experiment=init_diffusion_lane_conditioned_frozen_map_current_best'])
        with open_dict(cfg):
            cfg.paths.output_dir = str(folder); cfg.paths.log_dir = str(folder)
            cfg.data.train_batch_size = 2; cfg.data.num_workers = 0; cfg.data.shuffle = False
            cfg.data.scenario_dreamer_train_sample_list = str(folder / 'train.pkl')
        for arm in design_arms():
            if names and arm['name'] not in names:
                continue
            current = copy.deepcopy(cfg)
            with open_dict(current):
                current.model.model_config.decoder.init_diffusion.component_loss = arm['recipe']
                for key, value in arm['config'].items():
                    OmegaConf.update(current, key, value, force_add=False)
            L.seed_everything(4817, workers=True)
            dm = hydra.utils.instantiate(current.data)
            model = hydra.utils.instantiate(current.model, _recursive_=False)
            model.encoder.init_decoder.G1._sample_time = lambda x, agent: torch.full((len(x), 1), .8, device=x.device, dtype=x.dtype)
            trainer = L.Trainer(accelerator='cpu', devices=1, precision='32-true', logger=False,
                max_steps=1, max_epochs=128, limit_val_batches=0, num_sanity_val_steps=0,
                enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False, gradient_clip_val=.5)
            trainer.fit(model, datamodule=dm)
            metrics = {key: float(value) for key, value in trainer.callback_metrics.items()
                       if torch.is_tensor(value) and value.numel() == 1}
            if not metrics or not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
                raise ValueError('Nonfinite/missing smoke loss')
            if torch.cuda.is_initialized() or any(p.requires_grad for p in model.encoder.map_encoder.parameters()):
                raise ValueError('Unexpected CUDA initialization or unfrozen map')
            results.append({'name': arm['name'], 'cpu_only_steps': 1, 'metrics': metrics, 'cuda_initialized': False})
            print('PASS ' + arm['name'], flush=True)
            del trainer, model, dm
    output = ART / 'real_batch_cpu_training_preflight.json'
    if names and output.exists():
        old = json.loads(output.read_text())
        results = [item for item in old['results'] if item['name'] not in names] + results
    save(output, {'results': results, 'scope': 'CPU diagnostic only; excluded from search evidence',
                  'forced_training_t': .8, 'scenes': 2, 'reproduction_tool': str(Path(__file__).relative_to(ROOT))})


if __name__ == '__main__':
    main()
