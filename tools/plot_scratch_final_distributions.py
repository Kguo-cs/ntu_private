"""Compare saved scratch-run vehicle distributions on identical evaluation scenes.

No model loading or generated-value reconstruction: the curves and approximate
statistics use saved official clipped-bin counts. The reference GT histograms
must be exactly equal across all supplied runs.
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sim_scratch_hist_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_component_loss_distributions import FEATURES, describe, validate_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series', nargs='+', required=True, metavar='LABEL=HISTOGRAM_JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loaded = []
    for entry in args.series:
        label, name = entry.split('=', 1)
        path = Path(name)
        histogram = json.loads(path.read_text())
        metrics = json.loads((path.parent / 'sd_agent_metrics.json').read_text())
        loaded.append((label, path, histogram, metrics))
    if len(loaded) < 2 or len({item[0] for item in loaded}) != len(loaded):
        raise ValueError('Supply at least two distinct labeled generated series')
    identity_fields = ('num_samples', 'num_gt_samples', 'eval_set_sha256', 'map_source',
                       'reference_mode', 'num_gt_vehicles', 'feature_counts_gt')
    identity = {key: loaded[0][3][key] for key in identity_fields}
    for label, _, _, metrics in loaded:
        if {key: metrics[key] for key in identity_fields} != identity:
            raise ValueError(f'Evaluation cohort differs for {label}')
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'legend.fontsize': 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    palette = ['#111111', '#D55E00', '#0072B2', '#009E73']
    rows = []
    for ax, (feature, info) in zip(axes.flat, FEATURES.items()):
        spec = loaded[0][2][feature]
        edges, gt_counts, probability = validate_counts(spec, 'matched_gt', 'reference GT')
        ax.stairs(probability, edges, label='Matched GT', color=palette[0], linewidth=2)
        rows.append(describe(feature, 'Matched GT', loaded[0][1], edges, gt_counts, probability))
        peaks = [probability]
        for index, (label, path, histogram, _) in enumerate(loaded, 1):
            gt_edges, counts, _ = validate_counts(histogram[feature], 'matched_gt', label + ' GT')
            if not np.array_equal(edges, gt_edges) or not np.array_equal(gt_counts, counts):
                raise ValueError(f'Matched GT bins/counts differ for {feature}/{label}')
            gen_edges, counts, probability = validate_counts(histogram[feature], 'generated', label)
            if not np.array_equal(edges, gen_edges):
                raise ValueError(f'Generated bin edges differ for {feature}/{label}')
            ax.stairs(probability, edges, label=label, color=palette[index % len(palette)],
                      linewidth=1.7)
            peaks.append(probability)
            rows.append(describe(feature, label, path, edges, counts, probability))
        limits = info['zoom'] or (edges[0], edges[-1])
        visible = (edges[:-1] < limits[1]) & (edges[1:] > limits[0])
        ax.set_xlim(*limits)
        ax.set_ylim(0, 1.1 * max(float(value[visible].max()) for value in peaks))
        ax.set_xlabel(f"{info['label']} ({info['unit']})")
        ax.set_ylabel('Probability per bin')
        ax.set_title(f"{info['label']}: bin width {spec['bin_width']:g} {info['unit']}")
        ax.grid(alpha=.2, axis='y')
    ax = axes[1, 1]
    first = loaded[0][3]
    gt_fraction = first['feature_counts_gt'][1] / first['num_gt_vehicles']
    coverage = [gt_fraction] + [item[3]['feature_counts_generated'][1] /
                              item[3]['num_generated_vehicles'] for item in loaded]
    labels = ['Matched GT'] + [item[0] for item in loaded]
    counts = [first['num_gt_vehicles']] + [item[3]['num_generated_vehicles'] for item in loaded]
    ax.bar(np.arange(len(labels)), np.asarray(coverage) * 100,
           color=[palette[index % len(palette)] for index in range(len(labels))], width=.55)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=10)
    ax.set_ylim(0, 120)  # Reserve space above GT for the two-line count label.
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('On-road vehicles (%)')
    ax.set_title('Vehicle evaluation coverage')
    for index, (fraction, count) in enumerate(zip(coverage, counts)):
        ax.text(index, fraction * 100 + 2, f'{fraction * 100:.1f}%\nN={count:,}',
                ha='center', va='bottom', fontsize=9)
    ax.grid(alpha=.2, axis='y')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .05))
    fig.suptitle(f"Same {identity['num_samples']:,} scenes; saved official vehicle histograms", y=.995)
    fig.text(.5, .012, 'Counts are clipped to official bounds. Size axes are zoomed without renormalization.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .97))
    fig.savefig(args.output / 'vehicle_distributions_and_coverage.png', dpi=200)
    plt.close(fig)
    (args.output / 'distribution_statistics.json').write_text(
        json.dumps({'cohort': identity, 'statistics': rows,
                    'scope': 'Approximate clipped-bin midpoint statistics; not raw physical values.',
                    'sources': [{'label': label, 'histogram': str(path.resolve()),
                                 'metrics': str((path.parent / 'sd_agent_metrics.json').resolve())}
                                for label, path, _, _ in loaded]}, indent=2) + '\n')
    with (args.output / 'distribution_statistics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == '__main__':
    main()
