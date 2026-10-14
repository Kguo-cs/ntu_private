"""Independently recompute completed overnight metrics from native histograms."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now

ART = ROOT / 'output/reports/init_diffusion_overnight_20261014'
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')
FOCUS = ('speed_jsd', 'ang_dev_jsd', 'length_jsd')


def divergence(hist, scale):
    generated, truth = np.asarray(hist['generated'], dtype=np.float64), np.asarray(hist['matched_gt'], dtype=np.float64)
    assert generated.shape == truth.shape and generated.ndim == 1
    assert np.isfinite(generated).all() and np.isfinite(truth).all()
    assert (generated >= 0).all() and (truth >= 0).all() and generated.sum() > 0 and truth.sum() > 0
    p, q = generated / generated.sum(), truth / truth.sum()
    mix = (p + q) / 2
    pr = np.divide(p, mix, out=np.ones_like(p), where=mix > 0)
    qr = np.divide(q, mix, out=np.ones_like(q), where=mix > 0)
    bins = .5 * (xlogy(p, pr) + xlogy(q, qr)) * scale
    return dict(value=float(bins.sum()), generated_mass=p, gt_mass=q, contributions=bins)


def audit_rows(rows):
    checks = 0; cohorts = {}; cells = []; signatures = set()
    for row in rows:
        key = (row['variant'], row['seed'], row['budget'], row['eval_seed'], row['num_samples'])
        assert key not in signatures, key
        signatures.add(key); checks += 1
        report_path = Path(row['metrics_path'])
        hist_path = report_path.parent / 'feature_histograms.json'
        report, hist = read(report_path), read(hist_path)
        assert report['num_samples'] == report['num_gt_samples'] == row['num_samples']; checks += 1
        assert report['reference_mode'] == 'matched' and report['map_source'] == 'reference'; checks += 1
        assert report['unverified_generation_timestep_count'] == 0; checks += 1
        assert report['decoder']['sampling_steps'] == 20; checks += 1
        assert report['decoder']['ema_num_updates'] == row['budget'] and report['decoder']['use_ema']; checks += 1
        assert row['generator_num_parameters'] <= 8100000 and row['map_frozen']; checks += 1
        assert report['runtime']['trainer_precision'] == '32-true'; checks += 1
        assert report['runtime']['float32_matmul_precision'] == 'medium'; checks += 1
        if row['num_samples'] == 50000:
            assert report['full_membership']; checks += 1
        recalculated = {}
        dominant = {}
        for index, metric in enumerate(METRICS):
            result = divergence(hist[metric], 10 if index < 2 else 100)
            value = result['value']; recalculated[metric] = value
            assert math.isclose(value, row[metric], rel_tol=1e-9, abs_tol=1e-10), (key, metric, value, row[metric]); checks += 1
            assert math.isclose(value, report['agent_metrics'][metric], rel_tol=1e-9, abs_tol=1e-10); checks += 1
            cohort = (row['eval_manifest_sha256'], metric)
            if cohort in cohorts:
                assert cohorts[cohort] == hist[metric]['matched_gt']; checks += 1
            else:
                cohorts[cohort] = hist[metric]['matched_gt']
            selected = np.argsort(result['contributions'])[-5:][::-1]
            lo, width = hist[metric]['minimum'], hist[metric]['bin_width']
            dominant[metric] = [dict(lower=float(lo+i*width), upper=float(lo+(i+1)*width),
                 generated_mass=float(result['generated_mass'][i]), gt_mass=float(result['gt_mass'][i]),
                 jsd_contribution=float(result['contributions'][i])) for i in selected]
        collision = 100 * report['num_colliding_vehicles'] / report['num_generated_vehicles']
        assert math.isclose(collision, row['collision_rate'], rel_tol=1e-9, abs_tol=1e-10); checks += 1
        cells.append(dict(key=list(key), metrics=recalculated, collision_rate=collision,
            report_sha256=sha(report_path), histogram_sha256=sha(hist_path), dominant_bins=dominant))
    return dict(status='PASS' if cells else 'NO_COMPLETED_CELLS', audited_utc=now(),
                unique_actual_cells=len(cells), checks=checks, cells=cells,
                scope='Native histogram/count recomputation. Sampling_steps is configured N20, not instrumented NFE evidence.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ART / 'observations.json')
    p.add_argument('--output', type=Path, default=ART / 'native_metric_audit.json')
    args = p.parse_args()
    rows = read(args.input)
    result = audit_rows(rows)
    result['observations_sha256'] = sha(args.input)
    result['auditor_sha256'] = sha(__file__)
    save(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'unique_actual_cells', 'checks')}))


if __name__ == '__main__':
    main()
