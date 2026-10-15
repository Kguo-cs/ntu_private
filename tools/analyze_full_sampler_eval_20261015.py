"""Review paired native sampler results without promoting partial evaluations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'output/reports/init_diffusion_continuous_20261014/full_sampler_eval_v3'
JSD = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd',
       'length_jsd', 'width_jsd', 'speed_jsd')
FOCUS = ('speed_jsd', 'ang_dev_jsd', 'length_jsd')
ATTR = ('length_jsd', 'width_jsd', 'speed_jsd')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=ART)
    args = parser.parse_args()
    path = args.artifacts / 'observations.json'
    rows = read(path) if path.exists() else []
    cells = {}
    for row in rows:
        key = row['model'], row['mode'], row['scenes'], row['seed']
        if key in cells:
            raise ValueError(f'Duplicate evaluation cell: {key}')
        verified = read(Path(row['output']) / 'verified.json')
        assert verified['metrics'] == row['metrics']
        assert verified['metrics_recomputed_from_integer_histograms']
        if row['scenes'] == 50000:
            report = read(Path(row['output']) / 'sd_agent_metrics.json')
            target = read(ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json')
            for name in ('num_samples', 'num_gt_samples', 'full_membership', 'eval_set_sha256',
                         'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt'):
                assert report[name] == target[name], name
            assert report['require_full_set']
        cells[key] = row
    comparisons = []
    for (model, mode, scenes, seed), row in cells.items():
        control_mode = 'native24' if mode == 'midpoint24' or model == 'N24match4' else 'native20'
        if mode == control_mode:
            continue
        control = cells.get((model, control_mode, scenes, seed))
        if control is None:
            continue
        assert row['checkpoint_sha256'] == control['checkpoint_sha256']
        assert row['manifest_sha256'] == control['manifest_sha256']
        assert row['matmul_precision'] == control['matmul_precision']
        assert row['source_config_sha256'] == control['source_config_sha256']
        assert row['runner_sha256'] == control['runner_sha256']
        assert row['smart_source_sha256'] == control['smart_source_sha256']
        assert row['sampling_source_sha256_per_batch'] == control['sampling_source_sha256_per_batch']
        a_hist = read(Path(row['output']) / 'feature_histograms.json')
        b_hist = read(Path(control['output']) / 'feature_histograms.json')
        for name in JSD:
            assert a_hist[name]['edges'] == b_hist[name]['edges']
            assert a_hist[name]['matched_gt'] == b_hist[name]['matched_gt']
        ratios = {k: row['metrics'][k] / control['metrics'][k] for k in JSD}
        a_report = read(Path(row['output']) / 'sd_agent_metrics.json')
        b_report = read(Path(control['output']) / 'sd_agent_metrics.json')
        comparisons.append(dict(model=model, mode=mode, control_mode=control_mode,
                                scenes=scenes, seed=seed, ratios=ratios,
                                R_jsd=statistics.mean(ratios.values()),
                                R_focus=statistics.mean(ratios[k] for k in FOCUS),
                                R_attr=statistics.mean(ratios[k] for k in ATTR),
                                vehicle_count_ratio=a_report['num_generated_vehicles'] / b_report['num_generated_vehicles'],
                                onroad_fraction_change=(a_report['feature_counts_generated'][2] / a_report['num_generated_vehicles']
                                    - b_report['feature_counts_generated'][2] / b_report['num_generated_vehicles']),
                                candidate_metrics=row['metrics'], control_metrics=control['metrics'],
                                same_weights_manifest_initial_noise_and_GT=True,
                                candidate_NFE=row['network_evaluations_per_scene'],
                                control_NFE=control['network_evaluations_per_scene']))
    groups = defaultdict(list)
    for row in comparisons:
        groups[row['model'], row['mode'], row['control_mode'], row['scenes']].append(row)
    aggregate = []
    for (model, mode, control_mode, scenes), members in groups.items():
        ratios = {k: statistics.mean(m['ratios'][k] for m in members) for k in JSD}
        complete = {m['seed'] for m in members} == {2817, 3817} and scenes == 4096
        score = dict(model=model, mode=mode, control_mode=control_mode, scenes=scenes,
                     GEN_seeds=sorted(m['seed'] for m in members), ratios=ratios,
                     R_jsd=statistics.mean(ratios.values()),
                     R_focus=statistics.mean(ratios[k] for k in FOCUS),
                     R_attr=statistics.mean(ratios[k] for k in ATTR),
                     dual_GEN_4K_complete=complete,
                     full50000_comparison_available=scenes == 50000,
                     dual_GEN_full50000_complete=(scenes == 50000
                         and {m['seed'] for m in members} == {2817, 3817}),
                     full50000_improvement_proven=False)
        # Screening is a budget decision, never evidence of the final objective.
        score['promising_for_full50000_check'] = (complete and score['R_jsd'] < .99
            and score['R_focus'] < .99 and max(ratios.values()) <= 1.05
            and all(m['R_jsd'] < 1 and m['R_focus'] < 1 for m in members)
            and all(.99 <= m['vehicle_count_ratio'] <= 1.01 for m in members))
        score['attribute_tradeoff_worth_full50000_check'] = (complete and score['R_jsd'] < .9
            and score['R_focus'] < .9 and ratios['length_jsd'] < .8
            and max(ratios[k] for k in FOCUS) < 1.1
            and all(m['R_jsd'] < 1 and m['R_focus'] < 1 for m in members)
            and all(.99 <= m['vehicle_count_ratio'] <= 1.01
                    and m['onroad_fraction_change'] >= -.003 for m in members))
        score['onroad_fraction_changes'] = {m['seed']: m['onroad_fraction_change'] for m in members}
        score['regressing_metrics'] = [k for k, value in ratios.items() if value > 1]
        aggregate.append(score)
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  evaluation_cells=len(rows), paired_comparisons=comparisons,
                  aggregate=aggregate,
                  score_definition='Arithmetic mean of per-GEN native JSD ratios; no pooling of histograms.',
                  limitation='256-scene rows are runtime preflights. Paired4K estimates sampling effects on final weights; '
                  'only actual50000-scene rows prove that scope. Sampling comparisons do not prove altered training performance.')
    (args.artifacts / 'comparisons.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(evaluation_cells=len(rows), aggregate=aggregate), indent=2))


if __name__ == '__main__':
    main()
