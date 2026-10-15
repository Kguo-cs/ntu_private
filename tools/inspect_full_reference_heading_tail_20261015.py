"""Describe full50K GT heading tails without fitting or changing metrics."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/sim_mpl_continuous'
import gc
gc.disable()
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import datetime
    import hashlib
    import json
    import numpy as np
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    from src.smart.metrics.real_cache import CachedReferenceStore

    torch.set_num_threads(1)
    install_cached_loader(torch)
    root = Path(__file__).resolve().parents[1]
    base = root / 'output/reports/init_diffusion_continuous_20261014'
    store = CachedReferenceStore(root / 'src/waymo_data/sd_real_metric_cache.sqlite', lru_size=0)
    target = json.loads((root / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    assert store.metadata['eval_set_sha256'] == target['eval_set_sha256']
    # Independently verify the cached vehicle-row0 convention against64 native
    # scenes, whose explicit SMART ego row is last and world transform is saved.
    native = json.loads((base / 'full_sampler_50k_v2/full50000_files.json').read_text())
    maximum_error = 0.
    for i in np.linspace(0, len(native)-1, 64, dtype=int):
        raw = torch.load(root / 'src/waymo_data/full/scenario_dreamer_val' / native[i], map_location='cpu', weights_only=False)
        _, gt = store.get(raw['scenario_dreamer_cache_file'])
        t = int(raw['scene_timestep'])
        ego = raw['agent']['position'][-1, t, :2].double().numpy()
        info = raw['scenario_dreamer']
        offset = ego-np.asarray(info['center_world'])
        angle = float(info['rotation_angle'])
        c, s = np.cos(angle), np.sin(angle)
        position = np.array([c*offset[0]-s*offset[1], s*offset[0]+c*offset[1]])
        error = float(np.linalg.norm(position-gt['vehicles'][0, :2]))
        assert error < .01, (native[i], error)
        maximum_error = max(maximum_error, error)
    chunks, ego_tail, all_tail, all_onroad, empty_onroad_scenes = [], 0, 0, 0, 0
    for i, name in enumerate(store.files):
        _, gt = store.get(name)
        vehicles, lanes = gt['vehicles'], gt['metric_lanes']
        distances = np.linalg.norm(vehicles[:, None, None, :2]-lanes[None], axis=-1).min(axis=(1, 2))
        use = distances <= 1.5
        if not use.any():
            empty_onroad_scenes += 1
            continue
        angles = store.official.get_angular_devs(vehicles[use], lanes)
        all_tail += int((np.abs(angles) >= 175).sum())
        all_onroad += len(angles)
        selected_indices = np.flatnonzero(use)
        ego = selected_indices == 0
        ego_tail += int(((np.abs(angles) >= 175) & ego).sum())
        chunks.append(np.column_stack((angles[~ego], vehicles[use][~ego, 2])))
        if (i+1) % 10000 == 0:
            print('FULL_REFERENCE_SCENES', i+1, flush=True)
    array = np.concatenate(chunks)
    assert all_onroad == target['feature_counts_gt'][2]
    def describe(values):
        tail = np.abs(values[:, 0]) >= 175
        return dict(vehicles=len(values), tail175_count=int(tail.sum()),
            tail175_fraction=float(tail.mean()), within5deg_fraction=float((np.abs(values[:, 0]) < 5).mean()),
            tail_speed_quantiles=np.quantile(values[tail, 1], [0,.25,.5,.75,1]).tolist())
    result = dict(status='COMPLETE_FULL50K_REFERENCE_DIAGNOSTIC',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scenes=len(store.files), eval_set_sha256=store.metadata['eval_set_sha256'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_source_sha256=store.official.source_sha256,
        verified_ego_alignment_scenes=64, ego_coordinate_error_max_m=maximum_error,
        all_onroad_vehicles=all_onroad, all_tail175_count=all_tail, ego_tail175_count=ego_tail,
        empty_onroad_scenes=empty_onroad_scenes,
        non_ego=describe(array), stationary=describe(array[array[:,1] <= .5]),
        moving=describe(array[array[:,1] > .5]),
        limitation='Descriptive held-out GT inspection only. No fitting, objective, sampler or metric change; '
                   'do not use evaluation labels to construct per-scene training targets.')
    output = base / 'full_reference_heading_tail_20261015.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
