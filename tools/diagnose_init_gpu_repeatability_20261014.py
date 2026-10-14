"""Same-input GPU repeats, after all owned experiment queues have ended."""
from pathlib import Path
import copy
import fcntl
import hashlib
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import now, read, save, sha
from tools import run_init_expanded_20261014 as search


def digest(items):
    h = hashlib.sha256()
    for name, value in sorted(items.items()):
        h.update(name.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch_geometric.loader import DataLoader
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    from tools.scratch_component_loss_sweep import calibrate
    from src.smart.datasets import MultiDataset
    from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain
    art = search.ART / 'map_architecture'
    calibration = art / 'size_calibration'
    assert not search.engine.shared.alive(read(art / 'budget_handoff_pid.json')['pid'])
    assert not search.engine.shared.alive(read(calibration / 'pid.json')['pid'])
    mask_owner = art / 'mask_followup/pid.json'
    if mask_owner.exists():
        assert not search.engine.shared.alive(read(mask_owner)['pid'])
    with (search.ART / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        search.engine.ART = art
        assert not search.engine.occupied(), 'Native GPU queue still has an owner'
        install_cached_loader(torch)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('medium')
        source = ROOT / 'logs/component_loss_scratch/calibration/calibration_files.pkl'
        with source.open('rb') as f:
            files = pickle.load(f)['files'][:16]
        manifest = art / 'repeatability_train16.pkl'
        with manifest.open('wb') as f:
            pickle.dump({'files': files}, f)
        cfg_file = search.RUNS / 'map_architecture/roadK8_T4817_2048_N4096/roadK8/config.yaml'
        cfg = OmegaConf.load(cfg_file)
        plan = dict(registered_utc=now(), script_sha256=sha(__file__),
                    config_sha256=sha(cfg_file), manifest_sha256=sha(manifest),
                    normalizer_sha256=sha(search.engine.shared.NORMALIZER),
                    repeated_calls=3, seed=26817, batch_size=16,
                    scope='Same initialized current model/input/RNG, no optimizer or sampling. '
                          'Tests GPU within-process forward/backward repeatability only. '
                          'It cannot isolate old/new source effects or explain trained JSD.')
        save(art / 'gpu_repeatability_plan.json', plan)
        torch.manual_seed(4817)
        model = hydra.utils.instantiate(cfg.model, _recursive_=False).cpu()
        calibrate(model, read(search.engine.shared.NORMALIZER))
        model.cuda().train()
        dataset = MultiDataset(cfg.data.train_raw_dir, WaymoTargetBuilderTrain(32),
                               sample_list=str(manifest), sample_list_check_exists=False)
        batch = next(iter(DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False))).cuda()
        records, reference, first_losses = [], None, None
        for i in range(3):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(plan['seed'])
            random.seed(plan['seed'])
            np.random.seed(plan['seed'])
            maps, agents = model.token_processor(copy.deepcopy(batch))
            losses = model.encoder(maps, agents)['initial_logit']
            loss = losses[0] + losses[1]
            loss.backward()
            torch.cuda.synchronize()
            values = torch.stack([x.detach().cpu() for x in losses])
            gradients = {n: p.grad.detach().cpu().clone() for n, p in
                         model.encoder.init_decoder.G1.named_parameters() if p.grad is not None}
            assert torch.isfinite(values).all()
            assert all(torch.isfinite(v).all() for v in gradients.values())
            if reference is None:
                reference, first_losses = gradients, values.clone()
            differences = [(gradients[n] - v).abs() for n, v in reference.items()]
            records.append(dict(repeat=i, losses=values.tolist(),
                                losses_exactly_equal=torch.equal(values, first_losses),
                                gradient_sha256=digest(gradients),
                                gradients_exactly_equal=all(torch.equal(gradients[n], v) for n, v in reference.items()),
                                max_abs_gradient_difference=max(x.max().item() for x in differences),
                                unequal_gradient_elements=sum(int((x != 0).sum()) for x in differences),
                                cpu_rng_sha256=digest({'rng': torch.get_rng_state()}),
                                cuda_rng_sha256=digest({'rng': torch.cuda.get_rng_state()}),
                                numpy_rng_sha256=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                                python_rng_sha256=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest()))
        save(art / 'gpu_repeatability_result.json', dict(
            status='CHECK_COMPLETE', at_utc=now(), plan_sha256=sha(art / 'gpu_repeatability_plan.json'),
            device=torch.cuda.get_device_name(), torch_version=torch.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            float32_matmul_precision=torch.get_float32_matmul_precision(), rows=records,
            forward_backward_exactly_repeatable=all(r['losses_exactly_equal'] and r['gradients_exactly_equal'] for r in records),
            scope=plan['scope']))
        print('GPU_REPEATABILITY_CHECK_COMPLETE', records, flush=True)


if __name__ == '__main__':
    main()
