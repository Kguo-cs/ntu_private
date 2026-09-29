"""Reproduce map_token_verify.json using CPU only.

Run with /home/ke/miniconda3/envs/sim/bin/python. Reads existing data and
vocabulary; writes JSON beside this script. The historical 'full'/'short'
keys mean endpoint chord >4.9m / <=4.9m, respectively; 'short' therefore
includes short fragments AND sufficiently curved fragments. Error is local
frame y reconstruction error, not a vehicle's true lane-normal deviation.
"""
from pathlib import Path
from collections import defaultdict
import json
import os
import pickle
import random

import numpy as np
import torch
from scipy.spatial.distance import cdist


def main():
    os.chdir('/home/ke/code/sim')
    torch.set_num_threads(1)
    with Path('src/smart/tokens/map_traj_token5.pkl').open('rb') as h:
        traj = np.asarray(pickle.load(h)['traj_src'])
    i = np.linspace(0, traj.shape[1] - 1, 3).astype(int)
    ref = traj[:, i].reshape(len(traj), -1)
    items = defaultdict(list)
    total = mismatch = 0
    files = random.Random(37).sample(
        sorted(Path('src/waymo_data/full/training_map2_sd').glob('*.pt')), 32)
    for f in files:
        d = torch.load(f, map_location='cpu', weights_only=False)['tokenized_map']
        local = np.concatenate([
            np.zeros((len(d['type']), 1, 2)), d['traj_pos_local'].numpy()], axis=1)
        saved = d['token_idx'].numpy()
        typ = d['type'].numpy()
        cur = cdist(local.reshape(len(local), -1), ref, 'sqeuclidean').argmin(1)
        total += len(saved)
        mismatch += int((cur != saved).sum())
        code = traj[saved][:, i]
        delta = code - local
        full = np.linalg.norm(local[:, -1], axis=1) > 4.9
        for name, mask in [
            ('lane_full', (typ < 4) & full),
            ('lane_short', (typ < 4) & ~full),
            ('edge_full', ((typ == 4) | (typ == 5)) & full),
            ('edge_short', ((typ == 4) | (typ == 5)) & ~full),
        ]:
            items[name].extend(np.abs(delta[mask, :, 1]).max(1).tolist())
    result = {
        'vocab_shape': traj.shape,
        'n_tokens': total,
        'saved_current_argmin_mismatch': mismatch,
        'stratified_max_local_lateral_error': {
            k: {'n': len(v), 'p50': float(np.median(v)),
                'p90': float(np.quantile(v, .9)), 'p95': float(np.quantile(v, .95)),
                'gt0.1_fraction': float((np.array(v) > .1).mean())}
            for k, v in items.items()
        },
    }
    out = Path(__file__).resolve().parent / 'map_token_verify.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
