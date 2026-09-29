"""Reproduce map_geometry_stats.json using CPU only.

Run from any directory with the sim environment:
  /home/ke/miniconda3/envs/sim/bin/python /tmp/sim_lat_audit/map_geometry_audit.py

Read-only dataset audit; writes its JSON beside this script. Statistical sample
and RNG seed are identical to the original executed audit. Boolean arrays use
the same generic summary as distances: their mean is the fraction satisfying
the stated condition. Distances between anchors are NOT lateral error bounds.
"""
from pathlib import Path
from collections import Counter, defaultdict
import json
import os
import pickle
import random
import sqlite3
import zlib

import numpy as np
import torch
from scipy.spatial import cKDTree


def summary(values):
    a = np.array(values, dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": len(a), "mean": float(a.mean()),
        "p50": float(np.quantile(a, .5)),
        "p90": float(np.quantile(a, .9)),
        "p95": float(np.quantile(a, .95)),
        "p99": float(np.quantile(a, .99)),
        "gt_0.1_fraction": float((a > .1).mean()),
        "gt_20_fraction": float((a > 20).mean()),
    }


def main():
    os.chdir('/home/ke/code/sim')
    torch.set_num_threads(1)
    base = Path('src/waymo_data/full/scenario_dreamer_val')
    files = random.Random(37).sample(sorted(base.glob('*.pt')), 256)
    con = sqlite3.connect('file:src/waymo_data/sd_real_metric_cache.sqlite?mode=ro', uri=True)
    counts = Counter()
    arr = defaultdict(list)
    scene_flags = Counter()
    vehicle_count = 0
    for f in files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        pos = d['map_save']['traj_pos'][:, 0].numpy().astype(np.float64)
        typ = d['pt_token']['type'].numpy()
        counts.update(typ.tolist())
        sd = d['scenario_dreamer']
        center = sd['center_world'].numpy()
        phi = float(sd['rotation_angle'])
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        loc = (pos - center) @ rot.T
        within = (np.abs(loc) < 32).all(1)
        edge = (typ == 4) | (typ == 5)
        lane = typ < 4
        name = d['scenario_dreamer_cache_file']
        row = con.execute('select payload from scenes where name=?', (name,)).fetchone()
        gt = pickle.loads(zlib.decompress(row[0]))['gt']
        lanes = gt['metric_lanes'].reshape(-1, 2)
        lane_tree = cKDTree(lanes)
        if np.any(edge & within):
            arr['output_roadedge_anchor_distance_to_metric_centerlines_in_fov'].extend(
                lane_tree.query(loc[edge & within])[0].tolist())
        if np.any(lane):
            a_dist = cKDTree(loc[lane]).query(lanes)[0]
            arr['metric_centerline_point_distance_to_input_lane_anchor'].extend(a_dist.tolist())
        else:
            scene_flags['no_input_lane_anchors'] += 1
        # Same geometric candidates as current one-layer map encoder: nearest
        # 30 sources, then 0 < distance < 20m. Tied distances may choose a
        # different order from torch_cluster; ordinary non-tied edges match.
        e_indices = np.flatnonzero(edge)
        source_tree = cKDTree(pos)
        if len(e_indices):
            ed, ei = source_tree.query(pos[e_indices], k=min(30, len(pos)))
            ed, ei = np.atleast_2d(ed), np.atleast_2d(ei)
            include = (ed < 20) & (ed > 0)
            is_lane = lane[ei] & include
            arr['edge_anchor_has_any_centerline_neighbor'].extend(is_lane.any(1).tolist())
            represented = np.unique(ei[is_lane])
            local_lanes = np.flatnonzero(lane & within)
            arr['in_fov_lane_anchor_reaches_any_edge'].extend(
                np.isin(local_lanes, represented).tolist())
            emitted = e_indices[np.linalg.norm(pos[e_indices] - center, axis=1) < 100]
            if len(emitted):
                emit_local = loc[emitted]
                etree = cKDTree(emit_local)
                vehicles = gt['vehicles'][:, :2]
                vehicle_count += len(vehicles)
                _, near_e = etree.query(vehicles, k=min(30, len(emitted)))
                near_e = np.asarray(near_e).reshape(len(vehicles), -1)
                e_row = {idx: j for j, idx in enumerate(e_indices)}
                for vp, near in zip(vehicles, near_e):
                    src = np.unique(np.concatenate([
                        ei[e_row[emitted[j]]][is_lane[e_row[emitted[j]]]] for j in near]))
                    arr['gt_vehicle_has_two_hop_centerline'].append(bool(len(src)))
                    if len(src):
                        arr['gt_vehicle_nearest_two_hop_lane_anchor'].append(
                            float(np.linalg.norm(loc[src] - vp, axis=1).min()))
            else:
                scene_flags['no_output_edges_within100'] += 1
        else:
            scene_flags['no_edges'] += 1
    con.close()
    with Path('src/smart/tokens/map_traj_token5.pkl').open('rb') as h:
        traj = np.asarray(pickle.load(h)['traj_src'])
    idx = np.linspace(0, traj.shape[1] - 1, 3).astype(int)
    qstat = defaultdict(list)
    trfiles = random.Random(37).sample(
        sorted(Path('src/waymo_data/full/training_map2_sd').glob('*.pt')), 128)
    for f in trfiles:
        d = torch.load(f, map_location='cpu', weights_only=False)['tokenized_map']
        local = d['traj_pos_local'].numpy()
        code = traj[d['token_idx'].numpy()][:, idx[1:]]
        delta = code - local
        typ = d['type'].numpy()
        for name, mask in [('all', np.ones(len(typ), dtype=bool)),
                           ('lane', typ < 4), ('roadedge', (typ == 4) | (typ == 5))]:
            qstat[name + '_max_lateral_token_error'].extend(
                np.abs(delta[mask, :, 1]).max(1).tolist())
            qstat[name + '_max_point_token_error'].extend(
                np.linalg.norm(delta[mask], axis=-1).max(1).tolist())
    result = {
        'n_val_scenes': len(files), 'n_gt_vehicles': vehicle_count,
        'map_type_counts': dict(sorted(counts.items())),
        'scene_flags': dict(scene_flags),
        'geometry': {k: summary(v) for k, v in arr.items()},
        'quantization_train_128': {k: summary(v) for k, v in qstat.items()},
    }
    out = Path(__file__).resolve().parent / 'map_geometry_stats.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
