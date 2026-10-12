"""Render verified frozen native and hybrid inference tradeoffs; never evaluate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011'
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd',
           'width_jsd', 'speed_jsd', 'collision_rate')
LABELS = ('Near', 'Lat', 'Ang', 'Length', 'Width', 'Speed', 'Col%')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_proof(decision_path, protocol_path, verifier):
    decision = json.loads(Path(decision_path).read_text())
    summary_path = Path(decision['grid_summary'])
    if not decision['selection_is_frozen'] or sha(summary_path) != decision['grid_summary_sha256']:
        raise ValueError('Missing immutable completed inference decision')
    grid, proof = verifier(summary_path, protocol_path, main_art=ART)
    if proof != decision['criterion_proof'] or proof['selected_option'] != decision['selected_option']:
        raise ValueError('Frozen inference decision differs from certified actual rows')
    return grid, proof


def main():
    from tools.select_init_diffusion_native_sampler import verify_actual_grid as verify_native
    from tools.select_init_diffusion_hybrid_sampler import verify_actual_grid as verify_hybrid
    native_path = ART / 'frozen_native_sampler.json'
    hybrid_path = ART / 'frozen_hybrid_sampler.json'
    native_grid, native = actual_proof(native_path, ART / 'native_sampler_selection_protocol.json', verify_native)
    hybrid_grid, hybrid = actual_proof(hybrid_path, ART / 'hybrid_sampler_selection_protocol.json', verify_hybrid)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), constrained_layout=True)
    x = np.arange(len(METRICS))
    specs = [('n10_eps005', '10 steps / floor .05 (gates fail)', '#aab6c1'),
             ('n40_eps005', '40 steps / floor .05 (frozen choice)', '#216b94'),
             ('n40_eps0001', '40 steps / floor .0001', '#edac52')]
    native_payload = {}
    for offset, (option, label, color) in enumerate(specs):
        values = [(native['comparisons'][option]['ratios'][k] - 1) * 100 for k in METRICS]
        axes[0].bar(x + (offset - 1) * .23, values, width=.22, label=label, color=color)
        native_payload[option] = dict(zip(METRICS, values))
    axes[0].set_title('Step / denominator-floor comparison: each change relative to fresh 20-step anchors', loc='left', fontsize=11)
    axes[0].legend(loc='upper left', frameon=False, ncol=1, fontsize=9)
    axes[0].set_ylim(-55, 108)
    values = [(hybrid['comparisons']['hybrid_pos0001']['ratios'][k] - 1) * 100 for k in METRICS]
    axes[1].bar(x, values, width=.55, color='#26886f', label='40 steps, position floor .0001; other floors .05')
    for index, value in enumerate(values):
        axes[1].text(index, value - .65 if value < 0 else value + .45, f'{value:+.2f}%',
                     ha='center', va='top' if value < 0 else 'bottom', fontsize=9)
    axes[1].set_ylim(-30, 7)
    axes[1].set_title('Position-only final update: each change relative to new fresh 40-step anchors', loc='left', fontsize=11)
    axes[1].legend(loc='lower right', frameon=False, fontsize=9)
    for axis in axes:
        axis.axhline(0, color='#34414d', linewidth=.8)
        axis.set_xticks(x, LABELS)
        axis.set_ylabel('Metric change (%)\nNegative = better')
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
    figure.suptitle('Actual matched4096 inference experiments | 2 reused TRAIN checkpoints x 2 GEN seeds', fontsize=13)
    output = ART / 'analysis/report_figures'
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / 'sampler_tradeoffs_actual.png', output / 'sampler_tradeoffs_actual.svg']
    figure.savefig(paths[0], dpi=200)
    figure.savefig(paths[1])
    plt.close(figure)
    # Individual panels keep their smallest labels at >=9pt when fitted to
    # the report's 511pt content width. The combined plot stays available.
    for kind in ('steps', 'hybrid'):
        panel, axis = plt.subplots(figsize=(7.05, 4.4), constrained_layout=True)
        if kind == 'steps':
            for offset, (option, label, color) in enumerate(specs):
                axis.bar(x + (offset - 1) * .23,
                         [native_payload[option][key] for key in METRICS],
                         width=.22, label=label, color=color)
            axis.set_title('Step / denominator-floor comparison\nRelative to fresh 20-step anchors',
                           loc='left', fontsize=11)
            axis.legend(loc='upper left', frameon=False, fontsize=9)
            axis.set_ylim(-55, 108)
        else:
            axis.bar(x, values, width=.55, color='#26886f')
            for index, value in enumerate(values):
                axis.text(index, value - .65 if value < 0 else value + .45,
                          f'{value:+.2f}%', ha='center',
                          va='top' if value < 0 else 'bottom', fontsize=9)
            axis.set_title('Position-only final update\nRelative to separate fresh 40-step anchors',
                           loc='left', fontsize=11)
            axis.set_ylim(-30, 7)
            axis.text(.98, .06, '40 steps: position floor .0001; other floors .05',
                      transform=axis.transAxes, ha='right', va='bottom', fontsize=9)
        axis.axhline(0, color='#34414d', linewidth=.8)
        axis.set_xticks(x, LABELS)
        axis.set_ylabel('Metric change (%)\nNegative = better')
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=9)
        for suffix in ('png', 'svg'):
            path = output / f'sampler_{kind}_actual.{suffix}'
            panel.savefig(path, dpi=200)
            paths.append(path)
        plt.close(panel)
    metadata = {'figure_paths': {str(p): sha(p) for p in paths},
                'source_tool_sha256': sha(__file__),
                'native_decision_sha256': sha(native_path), 'hybrid_decision_sha256': sha(hybrid_path),
                'native_grid_rows': len(native_grid['rows']), 'hybrid_grid_rows': len(hybrid_grid['rows']),
                'native_percent_change_vs_fresh20': native_payload,
                'hybrid_percent_change_vs_separate_fresh40': dict(zip(METRICS, values)),
                'phase_baselines_are_different': True, 'new_training_runs': 0,
                'no_cumulative_ratio_multiplication': True, 'torch_imported': 'torch' in sys.modules,
                'GPU_started': False}
    if metadata['torch_imported']:
        raise ValueError('Figure generator must remain CPU metadata-only')
    (output / 'sampler_tradeoffs_actual.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'figure': str(paths[0]), 'native_rows': len(native_grid['rows']),
                      'hybrid_rows': len(hybrid_grid['rows']), 'GPU_started': False}))


if __name__ == '__main__':
    main()
