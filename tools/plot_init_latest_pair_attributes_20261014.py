"""Plot the latest completed raw-model pair; never select an arm by its score."""
import argparse
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_init_overnight_20261014 import divergence
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now

ART = ROOT / 'output/reports/init_diffusion_expanded_20261014'
FEATURES = ('length_jsd', 'width_jsd', 'speed_jsd', 'ang_dev_jsd')
GEN = (2817, 3817)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    assert args.candidate != 'roadK8'
    audit = read(ART / 'native_metric_audit.json')
    assert audit['status'] == 'PASS'
    assert audit['observations_sha256'] == sha(ART / 'observations.json')
    rows = read(ART / 'observations.json')
    arms = ('roadK8', args.candidate)
    groups = {}
    for row in rows:
        if row['variant'] in arms and row['seed'] == 4817 and row['budget'] >= 6144:
            groups.setdefault((row['variant'], row['budget'], row['num_samples']), []).append(row)
    complete = {(budget, scenes) for arm, budget, scenes in groups
                if scenes in (4096, 50000) and all(
                    sorted(r['eval_seed'] for r in groups.get((a, budget, scenes), [])) == list(GEN)
                    for a in arms)}
    if not complete:
        raise RuntimeError('No completed equal-budget, equal-cohort pair for both GEN')
    # Prefer actual full50K evidence, then the largest common training budget.
    budget, scenes = max(complete, key=lambda x: (x[1] == 50000, x[0]))
    selected = {arm: sorted(groups[arm, budget, scenes], key=lambda r: r['eval_seed']) for arm in arms}
    reference = selected['roadK8'][0]
    records, curves, truth = {}, {}, {}
    for arm, pair in selected.items():
        records[arm] = {'features': {}, 'histogram_sha256': []}
        histograms = []
        for row in pair:
            for key in ('eval_manifest_sha256', 'train_order_sha256', 'initial_map_sha256', 'evaluation_weights'):
                assert row[key] == reference[key], (arm, key)
            assert row['evaluation_weights'] == 'ema' and not row.get('postprocessing')
            path = Path(row['metrics_path'])
            report = read(path)
            assert report['decoder']['sampling_steps'] == 20
            assert report['num_samples'] == report['num_gt_samples'] == scenes
            if scenes == 50000:
                assert report['full_membership']
            hp = path.parent / 'feature_histograms.json'
            histograms.append(read(hp))
            records[arm]['histogram_sha256'].append(sha(hp))
        for feature in FEATURES:
            h = histograms[0][feature]
            values = [divergence(hist[feature], 100) for hist in histograms]
            for row, hist, value in zip(pair, histograms, values):
                assert math.isclose(value['value'], row[feature], rel_tol=1e-9, abs_tol=1e-10)
                assert hist[feature]['matched_gt'] == h['matched_gt']
            signature = (h['minimum'], h['bin_width'], h['matched_gt'])
            if feature in truth:
                assert signature == truth[feature]
            truth[feature] = signature
            generated = np.stack([value['generated_mass'] for value in values])
            centers = h['minimum'] + (np.arange(generated.shape[1]) + .5) * h['bin_width']
            curves[arm, feature] = centers, (generated - values[0]['gt_mass']) * 100
            records[arm]['features'][feature] = dict(
                per_GEN=[dict(GEN=seed, jsd=value['value']) for seed, value in zip(GEN, values)],
                mean_jsd=float(np.mean([value['value'] for value in values])))

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/sim_init_pair_plot_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    limits = ((3.5, 7), (1.5, 3.3), (0, 32), (-30, 30))
    for ax, feature, bounds in zip(axes.flat, FEATURES, limits):
        for arm, color in zip(arms, ('#646f79', '#237bb8')):
            x, error = curves[arm, feature]
            ax.plot(x, error.mean(0), color=color, lw=1.4, label=arm)
            ax.fill_between(x, error.min(0), error.max(0), color=color, alpha=.12)
        ax.axhline(0, color='#9ba7ac', lw=.8)
        ax.set_xlim(*bounds)
        ax.grid(alpha=.15)
        unit = 'degrees' if feature == 'ang_dev_jsd' else 'm/s' if feature == 'speed_jsd' else 'm'
        ax.set_xlabel(feature.replace('_jsd', '') + ' (' + unit + ')')
        ax.set_ylabel('Generated minus GT (pp / bin)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f'{budget} updates / TRAIN4817 / {scenes} scenes / GEN2817 and3817', fontsize=12)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=2, fontsize=9, frameon=False)
    fig.text(.5, .01, 'Shading: two-GEN range, not confidence intervals. Native JSD uses all bins; common ranges shown.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .88))
    output = ART / 'paired_latest_attribute_errors.png'
    fig.savefig(output, dpi=170)
    plt.close(fig)
    result = dict(at_utc=now(),source_sha256=sha(__file__),observations_sha256=sha(ART/'observations.json'),
        native_audit_sha256=sha(ART/'native_metric_audit.json'),candidate=args.candidate,TRAIN=4817,
        GEN=list(GEN),updates=budget,scenes=scenes,records=records,image_sha256=sha(output),
        selection_rule='Externally specified candidate; prefer completed full50K pair, then largest common budget. No score-based candidate selection.',
        interpretation='Raw model with N20/EMA. Average per-GEN normalized histograms for visualization; JSD is computed separately per GEN. Shaded range is not a confidence interval. This plot does not prove final128epoch superiority.')
    save(ART/'paired_latest_attribute_diagnostic.json', result)
    print(dict(candidate=args.candidate, updates=budget, scenes=scenes, image=str(output)))


if __name__ == '__main__':
    main()
