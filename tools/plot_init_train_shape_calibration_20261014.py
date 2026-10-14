"""Plot measured complete50K raw/calibrated shape histograms, one fixed GEN."""
import os
os.environ['MPLCONFIGDIR'] = '/tmp/sim_calibrated_shape_plot'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    art = ROOT / 'output/reports/init_diffusion_expanded_20261014/map_architecture/size_calibration'
    path = art / 'full50000_comparison.json'
    comparison = read(path)
    row = next(r for r in comparison['rows'] if r['eval_seed'] == 2817)
    baseline = next(r for r in comparison['raw_rows'] if r['eval_seed'] == 2817)
    calibrated_path = Path(row['metrics_path']).parent / 'feature_histograms.json'
    baseline_path = Path(baseline['metrics_path']).parent / 'feature_histograms.json'
    calibrated, raw = read(calibrated_path), read(baseline_path)
    assert row['num_samples'] == baseline['num_samples'] == 50000
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, name, limits in zip(axes, ('length_jsd', 'width_jsd'), ((2., 9.), (1., 3.3))):
        h, h0 = calibrated[name], raw[name]
        assert h['matched_gt'] == h0['matched_gt']
        x = h['minimum'] + (np.arange(len(h['generated'])) + .5) * h['bin_width']
        for counts, label, color in (
            (h['matched_gt'], 'GT, same50K', '#1b3947'),
            (h0['generated'], f'Raw, JSD={baseline[name]:.4f}', '#b7612d'),
            (h['generated'], f'TRAIN-calibrated, JSD={row[name]:.4f}', '#187f75'),
        ):
            value = np.asarray(counts, dtype=float)
            ax.step(x, value / value.sum(), where='mid', label=label, color=color, linewidth=1.4)
        ax.set_xlim(*limits)
        ax.set_xlabel(name.split('_')[0].capitalize() + ' (m)')
        ax.set_ylabel('Bin probability')
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle('Measured shape marginals:6144 updates, full50K, GEN2817', fontsize=12)
    output = art / 'full50000_shape_histograms.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    save(art / 'full50000_shape_histograms.json', dict(
        at_utc=now(), figure_sha256=sha(output), script_sha256=sha(__file__),
        comparison_sha256=sha(path), histograms_sha256={str(p): sha(p) for p in (calibrated_path, baseline_path)},
        checkpoint=row['checkpoint'], GEN=2817, scenes=50000,
        scope='One prespecifiedGEN. Main-size ranges zoomed for readability; normalized '
              'using all original bins. JSD values use full original official bins. '
              'Measured model+TRAIN-only postprocessing; not improved network learning.'))
    print(output)


if __name__ == '__main__':
    main()
