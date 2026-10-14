"""Describe GEN histogram variation against an explicitly IID counterfactual."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from scipy.special import xlogy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now
ART = ROOT / 'output/reports/init_diffusion_expanded_20261014'
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd')


def jsd(p, q, scale):
    mixture = (p + q) / 2
    a = np.divide(p, mixture, out=np.ones_like(mixture), where=mixture > 0)
    b = np.divide(q, mixture, out=np.ones_like(mixture), where=mixture > 0)
    return .5 * (xlogy(p, a) + xlogy(q, b)).sum(axis=-1) * scale


def main():
    rows = read(ART / 'observations.json')
    groups = {}
    for row in rows:
        if row['budget'] < 2048:
            continue
        key = (row['variant'], row['seed'], row['budget'], row['num_samples'], row['checkpoint'])
        groups.setdefault(key, []).append(row)
    results = []
    for key, group in sorted(groups.items()):
        by_seed = {r['eval_seed']: r for r in group}
        if set(by_seed) != {2817, 3817} or len(group) != 2:
            continue
        paths = [Path(by_seed[s]['metrics_path']).parent / 'feature_histograms.json' for s in (2817, 3817)]
        hist = [read(path) for path in paths]
        features = {}
        for index, metric in enumerate(METRICS):
            first, second = (h[metric] for h in hist)
            assert first['matched_gt'] == second['matched_gt']
            assert all(first[k] == second[k] for k in ('minimum', 'maximum', 'bin_width'))
            counts = [np.asarray(h['generated'], dtype=np.int64) for h in (first, second)]
            truth = np.asarray(first['matched_gt'], dtype=np.float64)
            n = [int(c.sum()) for c in counts]
            p, q = [c / count for c, count in zip(counts, n)]
            gt = truth / truth.sum()
            scale = 10 if index < 2 else 100
            measured = [float(jsd(mass, gt, scale)) for mass in (p, q)]
            for s, value in zip((2817, 3817), measured):
                assert np.isclose(value, by_seed[s][metric], rtol=1e-9, atol=1e-10)
            # Per-metric reset keeps this descriptive reference stable as arms
            # finish. It never touches the model's training/generation RNG.
            rng = np.random.default_rng(61417 + index)
            base = (p + q) / 2
            a, b = [rng.multinomial(count, base, size=2048) / count for count in n]
            iid_gap = np.abs(jsd(a, gt, scale) - jsd(b, gt, scale))
            iid_pair = jsd(a, b, scale)
            features[metric] = dict(per_GEN=measured, generated_counts=n,
                absolute_metric_gap=abs(measured[0] - measured[1]),
                inter_GEN_jsd=float(jsd(p, q, scale)),
                inter_GEN_total_variation=float(np.abs(p - q).sum() / 2),
                iid_counterfactual_metric_gap_percentiles=np.quantile(iid_gap, [.5, .95, .99]).tolist(),
                iid_counterfactual_inter_GEN_jsd_percentiles=np.quantile(iid_pair, [.5, .95, .99]).tolist())
        results.append(dict(variant=key[0], TRAIN=key[1], updates=key[2], scenes=key[3],
            checkpoint=key[4], GEN=[2817, 3817], features=features,
            histogram_sha256={str(p): sha(p) for p in paths}))
    save(ART / 'generation_variability_diagnostic.json', dict(generated_utc=now(),
        source_sha256=sha(__file__), observations_sha256=sha(ART / 'observations.json'),
        iid_replicates=2048, results=results,
        interpretation='Descriptive variation across the two predeclared GEN seeds. '
        'The multinomial counterfactual assumes independent agents from their mean generated histogram; '
        'actual scene-conditioned agents are dependent and these percentiles are NOT model confidence intervals, '
        'significance tests or an error floor. No pooling is used for native JSD or candidate success. '
        'No data, calibration, sampler or selection rule is changed.'))
    for row in results:
        if row['updates'] >= 6144:
            f = row['features']['speed_jsd']
            print(row['variant'], row['updates'], row['scenes'], 'speed', f)


if __name__ == '__main__':
    main()
