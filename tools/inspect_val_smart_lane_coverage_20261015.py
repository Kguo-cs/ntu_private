"""Read-only map geometry audit; no generated model score or loss tuning."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/sim_mpl_continuous')
import gc
gc.disable()
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
from src.smart.metrics.real_cache import CachedReferenceStore
install_cached_loader(torch)
torch.set_num_threads(1)


def rotate(x, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1]),-1)


def segment_distance(position, poly):
    start = poly[:,:-1].reshape(-1,2)
    delta = (poly[:,1:]-poly[:,:-1]).reshape(-1,2)
    length2 = (delta*delta).sum(-1)
    use = length2 > 1e-10
    if not use.any():
        return np.full(len(position), np.inf)
    start, delta, length2 = start[use], delta[use], length2[use]
    diff = position[:,None]-start[None]
    fraction = np.clip((diff*delta).sum(-1)/length2,0,1)
    return np.linalg.norm(diff-fraction[:,:,None]*delta,axis=-1).min(-1)


def main():
    base = ROOT / 'output/reports/init_diffusion_continuous_20261014'
    names = json.loads((base/'full_sampler_50k_v2/full50000_files.json').read_text())[25000:25512]
    config_path = ROOT/('logs/init_diffusion_continuous_20261014/periodic_edge_training_v2/'
                       'control_T4817_S2048_N8192/control/config.yaml')
    cfg = OmegaConf.load(config_path)
    model_half_extent = float(cfg.model.model_config.token_processor.init_map_half_extent)
    store = CachedReferenceStore(ROOT/'src/waymo_data/sd_real_metric_cache.sqlite',lru_size=2)
    rows = []
    for name in names:
        raw = torch.load(ROOT/'src/waymo_data/full/scenario_dreamer_val'/name,
                         map_location='cpu',weights_only=False)
        info, agent = raw['scenario_dreamer'],raw['agent']
        _, gt = store.get(raw['scenario_dreamer_cache_file'])
        vehicles = gt['vehicles']
        t = int(raw['scene_timestep'])
        position = agent['position'][:,t,:2].double().numpy()
        heading = agent['heading'][:,t].double().numpy()
        labels = agent['type'].numpy()
        center = np.asarray(info['center_world'],dtype=float)
        angle = float(info['rotation_angle'])
        reference_xy = rotate(position[labels==0]-center,angle)
        error = np.linalg.norm(reference_xy[:,None]-vehicles[None,:,:2],axis=-1)
        a, b = linear_sum_assignment(error)
        assert len(a)==len(reference_xy)==len(vehicles)
        maximum_error = float(error[a,b].max())
        assert maximum_error < .01, (name,maximum_error)
        ego_reference = rotate(position[-1]-center,angle)
        ego_row = int(np.linalg.norm(vehicles[:,:2]-ego_reference,axis=-1).argmin())
        non_ego = np.arange(len(vehicles)) != ego_row
        poly = raw['map_save']['traj_pos'].double().numpy()
        kind = raw['pt_token']['type'].numpy()
        local = rotate(poly[:,0]-position[-1],-heading[-1])
        lanes = np.isin(kind,[0,1,2])
        cropped = lanes & (np.abs(local)<32).all(-1)
        model_crop = lanes & (np.abs(local)<model_half_extent).all(-1)
        full_poly = rotate(poly[lanes]-center,angle)
        cropped_poly = rotate(poly[cropped]-center,angle)
        model_poly = rotate(poly[model_crop]-center,angle)
        full_d = segment_distance(vehicles[:,:2],full_poly)
        crop_d = segment_distance(vehicles[:,:2],cropped_poly)
        model_d = segment_distance(vehicles[:,:2],model_poly)
        metric_d = np.linalg.norm(vehicles[:,None,:2]-gt['metric_lanes'].reshape(1,-1,2),axis=-1).min(-1)
        rows.append(dict(name=name,vehicles=int(non_ego.sum()),coordinate_error=maximum_error,
            smart_full_onroad=int(((full_d<=1.5)&non_ego).sum()),
            smart_crop32_onroad=int(((crop_d<=1.5)&non_ego).sum()),
            smart_model_crop_onroad=int(((model_d<=1.5)&non_ego).sum()),
            official_onroad=int(((metric_d<=1.5)&non_ego).sum())))
    store.close()
    count=sum(r['vehicles'] for r in rows)
    result=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),scenes=len(rows),vehicles=count,
        coordinate_alignment_max_m=max(r['coordinate_error'] for r in rows),ego_excluded=True,
        model_half_extent_m=model_half_extent,
        model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        coverage={k:sum(r[k] for r in rows)/count for k in
                  ['smart_full_onroad','smart_crop32_onroad','smart_model_crop_onroad','official_onroad']},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
        limitations=['Diagnostic on512 validation caches, not a model evaluation.',
                     'SMART uses unconnected5m token segments; official reference uses compacted lanes resampled100points.',
                     'This does not change training labels, loss weights, or reference metric definitions.'])
    output=base/'val_smart_lane_coverage_20261015_v2.json'
    assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    main()
