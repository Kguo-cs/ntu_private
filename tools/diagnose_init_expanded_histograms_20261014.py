"""Locate distribution errors without mixing generation seeds before JSD."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_init_overnight_20261014 import divergence
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now

ART = ROOT / 'output/reports/init_diffusion_expanded_20261014'
METRICS = ('speed_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd')


def main():
    rows = read(ART / 'observations.json')
    groups = {}
    for row in rows:
        if row['budget'] < 2048:
            continue
        key = (row['variant'], row['seed'], row['budget'], row['num_samples'])
        groups.setdefault(key, []).append(row)
    results = []
    size_plot = {}
    for (arm, seed, budget, scenes), members in sorted(groups.items()):
        assert len({r['eval_seed'] for r in members}) == len(members)
        histograms = [read(Path(r['metrics_path']).parent / 'feature_histograms.json') for r in members]
        features = {}
        for metric in METRICS:
            per_seed = [divergence(h[metric], 100) for h in histograms]
            for row, value in zip(members, per_seed):
                assert math.isclose(row[metric], value['value'], rel_tol=1e-9, abs_tol=1e-10)
            h = histograms[0][metric]
            for other in histograms[1:]:
                assert other[metric]['matched_gt'] == h['matched_gt']
                assert all(other[metric][k] == h[k] for k in ('minimum', 'maximum', 'bin_width'))
            p = np.mean([v['generated_mass'] for v in per_seed], axis=0)
            q = per_seed[0]['gt_mass']
            contribution = np.mean([v['contributions'] for v in per_seed], axis=0)
            total = float(contribution.sum())

            def interval(lo, hi):
                return dict(range=[h['minimum'] + lo * h['bin_width'],
                                   h['minimum'] + hi * h['bin_width']],
                            generated_fraction=float(p[lo:hi].sum()),
                            gt_fraction=float(q[lo:hi].sum()),
                            jsd_contribution=float(contribution[lo:hi].sum()),
                            fraction_of_jsd=float(contribution[lo:hi].sum() / total) if total else 0.)

            features[metric] = dict(mean_jsd=total,
                top_bins=[interval(int(i), int(i) + 1) for i in np.argsort(contribution)[::-1][:5]])
            if (seed, budget, scenes) == (4817, 2048, 4096) and arm in ('match4', 'prior_speed', 'prior_shape_speed') and metric in ('length_jsd', 'width_jsd'):
                centers = h['minimum'] + (np.arange(len(p)) + .5) * h['bin_width']
                size_plot[arm, metric] = (centers, p, q)
            if metric == 'speed_jsd':
                assert h['minimum'] == 0 and h['maximum'] == 50 and h['bin_width'] == 1
                features[metric]['speed_bands'] = [interval(lo, hi) for lo, hi in
                    ((0, 1), (1, 5), (5, 15), (15, 30), (30, 50))]
        trace_path = Path(members[0]['checkpoint']).parent / 'train_trace.json'
        training = None
        if trace_path.exists():
            trace = read(trace_path)
            assert trace and trace[-1]['step'] == budget - 1
            window = trace[-256:]
            keys = ['optimized_loss', 'train/type_accuracy', 'generator_grad_norm_before_clip']
            keys += ['train/component/' + name + '_loss' for name in
                     ('position', 'heading', 'length', 'width', 'velocity')]
            training = dict(trace_sha256=sha(trace_path), window_start=window[0]['step'],
                window_end_exclusive=budget, means={k:float(np.mean([r[k] for r in window])) for k in keys},
                gradient_clip=members[0]['gradient_clip_val'],
                fraction_clipped=float(np.mean([r['generator_grad_norm_before_clip'] > members[0]['gradient_clip_val'] for r in window])),
                median_gradient_norm=float(np.median([r['generator_grad_norm_before_clip'] for r in window])))
        results.append(dict(arm=arm, seed=seed, budget=budget, scenes=scenes,
                            generation_seeds=[r['eval_seed'] for r in members], features=features,
                            training_last256=training))
    save(ART / 'histogram_diagnostics.json', dict(generated_utc=now(),
        observations_sha256=sha(ART / 'observations.json'),
        scope='Mean of per-GEN JSD bin contributions, not JSD of pooled histograms. '
              'Used to locate errors; neither a new selection score nor proof of training causality. '
              'Training-window losses are descriptive and can change meaning with the source or objective; '
              'gradient clipping scale is not an effective Adam learning-rate multiplier.',
        results=results))
    if len(size_plot) == 6:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
        labels = {'match4': 'Matching4 (Gaussian source)', 'prior_speed': 'TRAIN speed source',
                  'prior_shape_speed': 'TRAIN speed + size source'}
        colors = {'match4': '#237bb8', 'prior_speed': '#299c79', 'prior_shape_speed': '#d25b31'}
        for ax, metric, limits in zip(axes, ('length_jsd', 'width_jsd'), ((3.5, 6.5), (1.7, 2.8))):
            x, _, q = size_plot['match4', metric]
            ax.plot(x, q, color='#24333b', lw=2, label='Matched ground truth')
            for arm in labels:
                x, p, _ = size_plot[arm, metric]
                ax.plot(x, p, color=colors[arm], lw=1.6, label=labels[arm])
            ax.set_xlim(*limits);ax.set_ylim(bottom=0)
            ax.set_xlabel(metric.replace('_jsd', '').title() + ' (m)')
            ax.set_ylabel('Probability per 0.1 m bin')
            ax.grid(alpha=.2)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .91),
                   ncol=2, fontsize=8.2, frameon=False)
        fig.suptitle('Generated vehicle sizes: 2048 updates / TRAIN4817 / 4096 scenes / two GEN', fontsize=9.7)
        fig.text(.5, .01, 'Central size ranges shown; native JSD uses all bins. Curves average normalized per-GEN histograms.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .81))
        fig.savefig(ART / 'generated_size_source_comparison.png', dpi=180)
        plt.close(fig)
    for result in results:
        feature = result['features']['speed_jsd']
        print(result['arm'], result['budget'], 'speed', round(feature['mean_jsd'], 5),
              '15-30 m/s error fraction', round(feature['speed_bands'][3]['fraction_of_jsd'], 3))


if __name__ == '__main__':
    main()
