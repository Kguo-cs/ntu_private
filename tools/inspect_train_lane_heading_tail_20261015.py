"""TRAIN-only geometric diagnostic, not the official angular metric or a loss."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import gc
gc.disable()
from pathlib import Path
import hashlib
import json
import pickle
import sys
import datetime as dt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
torch.set_num_threads(1)


def main():
    base = ROOT / 'output/reports/init_diffusion_continuous_20261014'
    manifest = ROOT / ('logs/init_diffusion_continuous_20261014/periodic_edge_training_v2/'
                       'control_T4817_S2048_N8192/control/train_files.pkl')
    names = pickle.loads(manifest.read_bytes())['files']
    selected = [names[i] for i in np.linspace(0, len(names)-1, 2048, dtype=int)]
    assert len(set(selected)) == 2048
    raw_root = ROOT / 'src/waymo_data/full/training_map2_sd'
    records, no_lane = [], 0
    vehicle_count, cropped_vehicle_count = 0, 0
    examples = []
    for i, name in enumerate(selected):
        data = torch.load(raw_root / name, map_location='cpu', weights_only=False)
        agent, lane = data['tokenized_agent'], data['tokenized_map']
        positions = agent['initial_pos'].numpy().astype(float)
        heads = agent['initial_heading'].numpy().astype(float)
        velocity = agent['local_vel'].numpy().astype(float)
        labels = agent['type'].numpy()
        # The native TokenProcessor identifies the last cached row as ego.
        use = labels == 0
        use[-1] = False
        pos = lane['position'].numpy().astype(float)
        theta = lane['orientation'].numpy().astype(float)
        kind = lane['type'].numpy()
        offset = pos - positions[-1]
        co, si = np.cos(heads[-1]), np.sin(heads[-1])
        local_map = np.column_stack((co*offset[:,0]+si*offset[:,1],
                                     -si*offset[:,0]+co*offset[:,1]))
        agent_offset = positions - positions[-1]
        local_agent = np.column_stack((co*agent_offset[:,0]+si*agent_offset[:,1],
                                       -si*agent_offset[:,0]+co*agent_offset[:,1]))
        vehicle_count += int(use.sum())
        cropped_vehicle_count += int((use & (np.abs(local_agent) < 32).all(-1)).sum())
        keep = np.isin(kind, [0,1,2]) & (np.abs(local_map) < 32).all(-1)
        if not keep.any():
            no_lane += 1
            continue
        pos, theta = pos[keep], theta[keep]
        local = lane['traj_pos_local'].numpy().astype(float)[keep]
        assert local.shape[1:] == (2,2)
        c, s = np.cos(theta)[:,None], np.sin(theta)[:,None]
        xy = np.stack((local[:,:,0]*c-local[:,:,1]*s,
                       local[:,:,0]*s+local[:,:,1]*c), -1) + pos[:,None]
        poly = np.concatenate((pos[:,None], xy), 1)
        start, end = poly[:,:-1].reshape(-1,2), poly[:,1:].reshape(-1,2)
        direction = end-start
        length2 = (direction**2).sum(-1)
        good = length2 > 1e-10
        start, direction, length2 = start[good], direction[good], length2[good]
        assert len(start), name
        segment_heading = np.arctan2(direction[:,1], direction[:,0])
        rows = np.flatnonzero(use)
        delta = positions[rows,None]-start[None]
        fraction = np.clip((delta*direction[None]).sum(-1)/length2[None],0,1)
        distance = np.linalg.norm(delta-fraction[:,:,None]*direction[None],axis=-1)
        nearest = distance.argmin(-1)
        angle = np.arctan2(np.sin(heads[rows]-segment_heading[nearest]),
                           np.cos(heads[rows]-segment_heading[nearest]))
        for j, row in enumerate(rows):
            if distance[j,nearest[j]] > 1.5:
                continue
            angular = np.arctan2(np.sin(heads[row]-segment_heading),np.cos(heads[row]-segment_heading))
            aligned = np.abs(angular) < np.pi/12
            d_aligned = float(distance[j,aligned].min()) if aligned.any() else None
            item = dict(angle_deg=float(np.degrees(angle[j])),
                        speed=float(np.linalg.norm(velocity[row])),
                        nearest_distance=float(distance[j,nearest[j]]),
                        aligned_lane_distance=d_aligned)
            records.append(item)
            if abs(item['angle_deg']) >= 175 and len(examples) < 12:
                examples.append(dict(cache_file=name, agent_index=int(row), **item))
        if (i+1) % 512 == 0:
            print('TRAIN_SCENES',i+1,'eligible_vehicles',len(records),flush=True)
    def describe(values):
        tail = [r for r in values if abs(r['angle_deg']) >= 175]
        gaps = [r['aligned_lane_distance']-r['nearest_distance'] for r in tail
                if r['aligned_lane_distance'] is not None]
        return dict(vehicles=len(values), tail175_count=len(tail),
                    tail175_fraction=len(tail)/len(values) if values else None,
                    tail_with_aligned_lane=len(gaps),
                    tail_aligned_lane_gap_median=float(np.median(gaps)) if gaps else None,
                    tail_aligned_lane_gap_le_02m=sum(x <= .2 for x in gaps))
    result = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        train_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        selected_filenames_sha256=hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
        scenes=2048, scenes_without_cropped_lanes=no_lane, ego_excluded=True,
        all_non_ego_vehicle_count=vehicle_count, inside_ego_square_vehicle_count=cropped_vehicle_count,
        eligible_proxy_fraction_of_all_vehicles=len(records)/vehicle_count,
        all=describe(records), stationary=describe([r for r in records if r['speed'] <= .5]),
        moving=describe([r for r in records if r['speed'] > .5]), examples=examples,
        limitations=['TRAIN-only2048-scene diagnostic; no held-out agent labels used.',
                     'Nearest reconstructed SMART token segment differs from official compacted/resampled lanes.',
                     'Vehicle positions within1.5m of a segment; this is a geometric proxy, not the official metric.',
                     'No optimizer, model, sampling, or training loss changed.'])
    output = base / 'train_lane_heading_tail_proxy_20261015_v2.json'
    assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['scenes','scenes_without_cropped_lanes','all_non_ego_vehicle_count',
                                         'inside_ego_square_vehicle_count','eligible_proxy_fraction_of_all_vehicles',
                                         'all','stationary','moving']}),flush=True)


if __name__ == '__main__':
    main()
