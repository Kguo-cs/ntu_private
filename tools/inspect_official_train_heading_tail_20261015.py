"""TRAIN-only rare-heading diagnostic with the actual SD metric geometry."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import gc
gc.disable()
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    import datetime
    import hashlib
    import json
    import pickle
    import numpy as np
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    from src.smart.metrics.official_backend import load_official_backend

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=int, default=2048)
    options = parser.parse_args()
    assert options.scenes >= 2048
    install_cached_loader(torch)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    base = root / 'output/reports/init_diffusion_continuous_20261014'
    data_root = root / 'src/waymo_data/scenario_dreamer_ae_preprocess_waymo'
    manifest = data_root / 'train_files.pkl'
    filenames = pickle.loads(manifest.read_bytes())['files']
    full = [n for n in filenames if n.rsplit('_', 2)[-2] == '0']
    selected = [full[i] for i in np.linspace(0, len(full)-1, options.scenes, dtype=int)]
    assert len(set(selected)) == options.scenes
    backend = load_official_backend()
    rows, examples, source_digest = [], [], hashlib.sha256()
    total_vehicles, scenes_without_vehicle = 0, 0
    for index, name in enumerate(selected):
        payload = (data_root / 'train' / name).read_bytes()
        source_digest.update(hashlib.sha256(payload).digest())
        scene = pickle.loads(payload)
        assert int(scene['lg_type']) == 0
        states, types = np.asarray(scene['agent_states']), np.asarray(scene['agent_types'])
        assert types[0].argmax() == 0  # Released preprocessed schema puts ego first.
        scene = dict(scene, agent_states=states[1:], agent_types=types[1:],
                     num_agents=int(scene['num_agents'])-1)
        unified = backend.convert_data_to_unified_format(scene, dataset_name='waymo_gt')
        vehicles = unified['vehicles']
        total_vehicles += len(vehicles)
        if not len(vehicles):
            scenes_without_vehicle += 1
            continue
        lanes = backend.resample_lanes(unified['lanes'], num_points=100)
        distance = np.linalg.norm(vehicles[:, None, None, :2] - lanes[None], axis=-1)
        onroad = distance.min(axis=(1, 2)) <= 1.5
        vehicles, distance = vehicles[onroad], distance[onroad]
        if not len(vehicles):
            continue
        angles = backend.get_angular_devs(vehicles, lanes)
        direction = np.diff(lanes, axis=1)
        lane_heading = np.arctan2(direction[..., 1], direction[..., 0])
        lane_heading = np.concatenate((lane_heading[:, :1], lane_heading), axis=1)
        for j, (vehicle, angle) in enumerate(zip(vehicles, angles)):
            item = dict(angle_deg=float(angle), speed=float(vehicle[2]),
                        length=float(vehicle[5]), width=float(vehicle[6]),
                        nearest_distance=float(distance[j].min()))
            if abs(angle) >= 175:
                theta = np.arctan2(vehicle[4], vehicle[3])
                error = np.arctan2(np.sin(theta-lane_heading), np.cos(theta-lane_heading))
                aligned = np.abs(error) < np.pi/12
                item['aligned_lane_distance'] = float(distance[j][aligned].min()) if aligned.any() else None
                if len(examples) < 12:
                    examples.append(dict(filename=name, onroad_non_ego_vehicle_index=j, **item))
            rows.append(item)
        if (index+1) % 512 == 0:
            print('OFFICIAL_TRAIN_SCENES', index+1, 'onroad_non_ego_vehicles', len(rows), flush=True)

    def describe(values):
        tail = [r for r in values if abs(r['angle_deg']) >= 175]
        gaps = [r['aligned_lane_distance']-r['nearest_distance'] for r in tail
                if r.get('aligned_lane_distance') is not None]
        return dict(vehicles=len(values), tail175_count=len(tail),
            tail175_fraction=len(tail)/len(values) if values else None,
            within5deg_fraction=sum(abs(r['angle_deg']) < 5 for r in values)/len(values) if values else None,
            tail_with_aligned_lane=len(gaps),
            tail_aligned_gap_le_02m=sum(g <= .2 for g in gaps),
            tail_aligned_gap_median=float(np.median(gaps)) if gaps else None,
            tail_speed_quantiles=np.quantile([r['speed'] for r in tail], [0, .25, .5, .75, 1]).tolist() if tail else [])

    result = dict(status='COMPLETE_TRAIN_ONLY_DIAGNOSTIC',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scenes=len(selected), full_lane_training_files=len(full), ego_excluded=True,
        scenes_without_non_ego_vehicles=scenes_without_vehicle,
        all_non_ego_vehicles=total_vehicles, onroad_non_ego_vehicles=len(rows),
        all=describe(rows), stationary=describe([r for r in rows if r['speed'] <= .5]),
        moving=describe([r for r in rows if r['speed'] > .5]), examples=examples,
        official_source_sha256=backend.source_sha256,
        train_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        selected_filenames_sha256=hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
        source_payloads_digest=source_digest.hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Official TRAIN snapshots, not the compact SMART training cache; no per-scene/time join established.',
            'Same local SD lane compaction,100point resampling,onroad threshold and angular metric code.',
            'No held-out labels, fitting, optimizer, model, cache or loss change.',
            'A near opposite-direction lane is a lane-association diagnostic, not proof of a wrong label.'])
    suffix = '' if options.scenes == 2048 else f'_N{options.scenes}'
    output = base / f'official_train_heading_tail_20261015{suffix}.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('scenes','all_non_ego_vehicles','onroad_non_ego_vehicles','all','stationary','moving')}), flush=True)


if __name__ == '__main__':
    main()
