"""CPU-only actual-result inventory for the deadline-bounded InitDiffusion search.

Only verified-success protocol roots create observations. GEN repetitions are
averaged within TRAIN before independent TRAIN means/SD. No unfinished budget,
missing GEN, unpaired TRAIN, changed source code or evaluation cohort is padded.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import unittest

if __name__ == '__main__':
    # Report-only isolated process; no collector during native library imports.
    import gc
    gc.disable()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import analyze_frozen_map_search as original
from tools import continue_overnight_init_diffusion as controller
from tools.runtime_completion import verify_completion

DEFAULT_RUNS = ROOT / 'logs/init_diffusion_overnight_20261011'
DEFAULT_ART = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011'
METRICS, ATTR = original.METRICS, original.ATTR
VALUES = METRICS + ('generated_onroad_fraction', 'num_generated_vehicles', 'num_gt_vehicles')


def phase(label):
    return next((name for name in ('screen', 'medium', 'long', 'full50000')
                 if label.startswith(name + '_')), 'other')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return original.digest(value)


def planned_generations(protocol):
    planned = [protocol.get('eval_seed'), protocol.get('extra_eval_seed')]
    seeds = sorted({seed for seed in planned if type(seed) is int})
    if not seeds or type(protocol.get('eval_seed')) is not int:
        raise ValueError('No explicit primary GEN seed in protocol')
    return seeds


def source_method_hash(execution):
    files = execution.get('source_hashes', {})
    relevant = {name: sha for name, sha in files.items()
                if name.startswith('src/smart/') or name == 'configs/model/smart.yaml'}
    if not relevant:
        raise ValueError('Missing recorded model/data source fingerprints')
    return digest(relevant), relevant


def attach_cohorts(observation, protocol, execution, method_sha):
    """Separate mathematical comparison scope from disclosed execution policy.

    Recovery changes only the authenticated loader/GC runtime. Precision,
    metric/helper, scene manifest and model/data source remain semantic scope;
    removing only cyclic_gc_policy must not remove any of those boundaries.
    """
    semantic = {key: value for key, value in observation['cohort'].items()
                if key != 'cyclic_gc_policy'}
    semantic['model_data_source_sha256'] = method_sha
    recovery = execution.get('runtime_recovery', {})
    command = execution.get('command') or []
    runtime = {
        'defer_cyclic_gc': protocol.get('defer_cyclic_gc'),
        'cyclic_gc_policy': protocol.get('cyclic_gc_policy'),
        'young_gc_interval': protocol.get('young_gc_interval'),
        'entrypoint': command[1] if len(command) > 1 else None,
        'recorded_tool_source_hashes': {name: value for name, value in execution.get('source_hashes', {}).items()
                                       if name.startswith('tools/')},
        'runtime_loader_source_hashes': recovery.get('runtime_loader_source_hashes', {}),
        'recovery_wrapper_sha256': recovery.get('wrapper_sha256'),
    }
    observation.update(cohort=semantic, cohort_sha256=digest(semantic), runtime_cohort=runtime,
                       runtime_cohort_sha256=digest(runtime),
                       semantic_cohort_policy='metric_model_data_precision_without_runtime_policy_v2')
    return observation


def discover(runs):
    observations, stages, warnings = [], [], []
    roots = sorted(Path(runs).glob('*/execution.json'))
    for execution_path in roots:
        run = execution_path.parent
        label = run.name
        status = {'label': label, 'run': str(run.resolve()), 'phase': phase(label),
                  'included': False, 'status': 'unreadable'}
        try:
            execution = read(execution_path)
            status['status'] = execution.get('status')
            status['started_unix'] = execution.get('started_unix')
            status['ended_unix'] = execution.get('ended_unix')
            if execution.get('status') != 'success' or execution.get('returncode') != 0:
                status['reason'] = 'Only verified successful completed stages count; partial checkpoints are diagnostic only'
                stages.append(status)
                continue
            protocol = read(run / 'protocol.json')
            spec = execution.get('completion_spec')
            if not isinstance(spec, dict):
                raise ValueError('Missing durable completion specification')
            rows, artifacts = verify_completion(run, Path(runs) / (label + '.log'), spec)
            if artifacts != execution.get('artifacts'):
                raise ValueError('Recorded completion artifact fingerprints changed')
            method_sha, sources = source_method_hash(execution)
            generations = planned_generations(protocol)
            if sorted(spec['generations']) != generations:
                raise ValueError('Completion specification and protocol GEN plans disagree')
            source = run / 'summary.json'
            for row in rows:
                converted = original.enrich(row, protocol, run, source, warnings)
                if converted is None:
                    raise ValueError('Incomplete row rejected by native metadata enrichment')
                converted.update(stage=label, phase=phase(label), planned_generation_seeds=generations,
                                 model_data_source_sha256=method_sha, model_data_source_files=sources,
                                 initial_generator_sha256=row.get('initial_generator_sha256'),
                                 initial_map_sha256=row.get('initial_map_sha256'), map_frozen=row.get('map_frozen'),
                                 resumed_from=row.get('resumed_from'),
                                 actual_train_offset=row.get('actual_train_offset'),
                                 train_manifest_sha256=row.get('train_manifest_sha256'),
                                 defer_cyclic_gc=protocol.get('defer_cyclic_gc'),
                                 cyclic_gc_policy=protocol.get('cyclic_gc_policy'),
                                 peak_observed_rss_kib=execution.get('peak_observed_rss_kib'),
                                 last_observed_rss_kib=execution.get('last_rss_kib'),
                                 stage_execution_source=str(execution_path.resolve()))
                attach_cohorts(converted, protocol, execution, method_sha)
                observations.append(converted)
            status.update(included=True, rows=len(rows), model_data_source_sha256=method_sha,
                          peak_observed_rss_kib=execution.get('peak_observed_rss_kib'),
                          cyclic_gc_policy=protocol.get('cyclic_gc_policy'),
                          scope_scenes=spec['scenes'], train_seed=spec['seed'],
                          planned_generation_seeds=generations,
                          endpoint_actual_exposure=controller.actual_exposure(spec['steps'], spec['batch']))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            status['reason'] = str(error)
            warnings.append(f'{label}: excluded invalid stage: {error}')
            # A stage is atomic evidence, never keep earlier rows after a later
            # row fails metadata validation.
            observations = [row for row in observations if row['stage'] != label]
        stages.append(status)
    return observations, stages, warnings


def seed_means(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[row['seed']].append(row)
    output = {}
    for seed, repetitions in sorted(cells.items()):
        plans = {tuple(row['planned_generation_seeds']) for row in repetitions}
        if len(plans) != 1:
            raise ValueError('A TRAIN cell has conflicting planned GEN sets')
        expected = set(next(iter(plans)))
        observed = [row['eval_seed'] for row in repetitions]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError('A TRAIN cell has missing or duplicate GEN observations')
        if len({(row['budget'], row['train_batch_size']) for row in repetitions}) != 1:
            raise ValueError('A TRAIN cell mixes batch/update methods at one exposure')
        output[seed] = {key: statistics.fmean(row[key] for row in repetitions) for key in VALUES}
        output[seed]['generation_seeds'] = sorted(expected)
    return output


def statistics_by_arm(rows):
    means = seed_means(rows)
    def sample_stats(values):
        result = original.stats(values)
        if result:
            result['sample_sd_defined'] = result['n'] >= 2
            if not result['sample_sd_defined']:
                result['sd'] = None
        return result
    return {'variant': rows[0]['variant'], 'arm_id': rows[0]['arm_id'],
            'actual_exposure': rows[0]['seen_training_examples'],
            'actual_full_data_epochs': rows[0]['seen_training_examples'] / controller.TRAIN_COUNT,
            'batch_update_pairs': sorted({(row['train_batch_size'], row['budget']) for row in rows}),
            'training_seed_ids': sorted(means), 'independent_training_N': len(means),
            'actual_GEN_observations': len(rows),
            'runtime_cohort_sha256_ids': sorted({r['runtime_cohort_sha256'] for r in rows}),
            'runtime_cohorts_by_TRAIN': {str(seed): sorted({r['runtime_cohort_sha256'] for r in rows if r['seed'] == seed}) for seed in means},
            'training_seed_roles': {str(seed): 'adaptive discovery / reused selection' if seed == 4817
                                    else 'held-out frozen-recipe confirmation' if seed == 5817 else 'unspecified'
                                    for seed in means},
            'per_training_seed': {str(seed): value for seed, value in means.items()},
            'metrics_native': {key: sample_stats(value[key] for value in means.values()) for key in VALUES},
            'metrics_raw_jsd_collision_percent': {
                key: sample_stats(value[key] / original.MULTIPLIERS[key] for value in means.values()) for key in METRICS},
            'training_origin': 'evaluation-only checkpoint audit' if all(row['evaluation_only'] for row in rows)
                               else 'fresh G1/optimizer/EMA from frozen AIRL map',
            'initial_generator_hashes_by_TRAIN': {str(seed): sorted({row['initial_generator_sha256']
                                                for row in rows if row['seed'] == seed}) for seed in means}}


def compare(candidate, control):
    candidate_seeds, control_seeds = seed_means(candidate), seed_means(control)
    if set(candidate_seeds) != set(control_seeds):
        return {'ranking_eligible': False, 'reason': 'No complete matched TRAIN seed set; no partial pairing/padding',
                'candidate_training_seeds': sorted(candidate_seeds), 'control_training_seeds': sorted(control_seeds)}
    per_seed = {}
    for seed in candidate_seeds:
        if candidate_seeds[seed]['generation_seeds'] != control_seeds[seed]['generation_seeds']:
            return {'ranking_eligible': False, 'reason': 'GEN seed sets do not match'}
        per_seed[str(seed)] = controller.compare(candidate_seeds[seed], control_seeds[seed])
    candidate_mean = {key: statistics.fmean(value[key] for value in candidate_seeds.values()) for key in VALUES}
    control_mean = {key: statistics.fmean(value[key] for value in control_seeds.values()) for key in VALUES}
    pooled = controller.compare(candidate_mean, control_mean)
    all_seed_gates = all(value['eligible'] for value in per_seed.values())
    pooled.update(ranking_eligible=pooled['eligible'] and all_seed_gates,
                  pooled_gates_pass=pooled['eligible'], each_training_seed_gates_pass=all_seed_gates,
                  per_training_seed=per_seed, independent_training_N=len(per_seed),
                  paired_R_sample_statistics=original.stats(value['R'] for value in per_seed.values()),
                  independent_5817=per_seed.get('5817'),
                  criterion='R is ratio of equal-TRAIN-weighted arm metric means; GEN averaged inside TRAIN')
    return pooled


def make_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['phase'], row['cohort_sha256'], row['seen_training_examples'], row['evaluation_weights'])].append(row)
    result = []
    for signature, group in sorted(grouped.items()):
        arms = defaultdict(list)
        for row in group:
            arms[row['arm_id']].append(row)
        control_arms = [values for values in arms.values() if values[0]['variant'] == 'control']
        if len(control_arms) > 1:
            raise ValueError('Multiple control configurations in one physical cohort/exposure')
        control = control_arms[0] if control_arms else None
        summaries = []
        for name, values in sorted(arms.items()):
            summary = statistics_by_arm(values)
            summary['comparison'] = compare(values, control) if control is not None else None
            summaries.append(summary)
        result.append({'phase': signature[0], 'cohort_sha256': signature[1], 'actual_exposure': signature[2],
                       'actual_full_data_epochs': signature[2] / controller.TRAIN_COUNT,
                       'evaluation_weights': signature[3], 'cohort': group[0]['cohort'],
                       'control_present': control is not None, 'arms': summaries,
                       'runtime_cohort_sha256_ids': sorted({row['runtime_cohort_sha256'] for row in group}),
                       'mixed_execution_runtime_cohorts': len({row['runtime_cohort_sha256'] for row in group}) > 1,
                       'runtime_cohorts': {row['runtime_cohort_sha256']: row['runtime_cohort'] for row in group},
                       'runtime_comparison_note': 'Metric/model/data/precision scope is matched; runtime policy is separately recorded, never claimed identical.'})
    return result


def timing(rows, stages):
    cells = defaultdict(list)
    for row in rows:
        cells[(row['stage'], row['variant'], row['seed'])].append(row)
    costs = []
    for (stage, variant, seed), values in sorted(cells.items()):
        seconds = {row['train_seconds'] for row in values}
        if len(seconds) != 1:
            raise ValueError('Repeated fit timing differs within a stage')
        updates = max(row['budget'] for row in values)
        fit = next(iter(seconds))
        costs.append({'stage': stage, 'variant': variant, 'TRAIN_seed': seed,
                      'batch': values[0]['train_batch_size'], 'fit_seconds_counted_once': fit,
                      'fit_update_budget': updates,
                      'fit_seconds_per_update': fit / updates if fit and not values[0]['evaluation_only'] else None,
                      'eval_seconds_all_actual_GEN_budget_cells': sum(row['eval_seconds'] or 0. for row in values),
                      'num_eval_cells': len(values)})
    return {'per_completed_stage': costs, 'total_fit_seconds': sum(row['fit_seconds_counted_once'] or 0. for row in costs),
            'total_actual_eval_seconds': sum(row['eval_seconds_all_actual_GEN_budget_cells'] for row in costs),
            'note': 'Each reported train_seconds is the same full fit repeated in checkpoint/GEN rows and is counted once, never multiplied by evaluations.'}


def figures(rows, groups, output):
    """Publication-size static charts, with no fabricated missing observations."""
    # Reuse only rendering/font checks from the old plotter; new-stage grouping
    # and statistics above are independent of its old control/stage names.
    from tools import plot_frozen_map_search as drawing
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    drawing.FIGURE_LAYOUTS.clear()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 9, 'svg.fonttype': 'none'})
    metadata = {'scope': 'Verified actual completed stages only; missing stages not padded',
                'files': [], 'heatmaps': [], 'long_curves': [], 'full50000': {'status': 'missing'}}
    for stage in ('screen', 'medium'):
        eligible_groups = [group for group in groups if group['phase'] == stage and group['control_present']]
        cohorts = sorted({group['cohort_sha256'] for group in eligible_groups})
        for cohort_index, cohort in enumerate(cohorts):
            observed = [group for group in eligible_groups if group['cohort_sha256'] == cohort]
            group = max(observed, key=lambda item: item['actual_exposure'])
            arms = [arm for arm in group['arms'] if arm['comparison'] and 'ratios' in arm['comparison']]
            arms.sort(key=lambda arm: (arm['variant'] != 'control', not arm['comparison']['ranking_eligible'],
                                       arm['comparison']['R'], arm['variant']))
            for page_index, offset in enumerate(range(0, len(arms), 8)):
                page = arms[offset:offset + 8]
                matrix = np.asarray([[arm['comparison']['ratios'][key] for key in METRICS]
                                     + [arm['comparison']['R']] for arm in page])
                fig, ax = plt.subplots(figsize=(drawing.FIGURE_WIDTH_IN, 2.7 + .34 * len(page)))
                ax.imshow(matrix, cmap='RdYlGn_r', norm=TwoSlopeNorm(vmin=.5, vcenter=1., vmax=1.5), aspect='auto')
                labels = [drawing.compact(arm['variant'], 18) + (' *' if not arm['comparison']['ranking_eligible'] else '') for arm in page]
                ax.set_yticks(range(len(page)), labels, fontsize=9)
                ax.set_xticks(range(8), ['Near', 'Lat', 'Angle', 'Length', 'Width', 'Speed', 'Col', 'R'], fontsize=9)
                for y, values in enumerate(matrix):
                    for x, value in enumerate(values):
                        ax.text(x, y, f'{value:.3f}', ha='center', va='center', fontsize=9)
                ax.set_title(f'{stage.capitalize()}: actual relative metrics (page {page_index + 1})', fontsize=12)
                fig.text(.5, .052, f'{group["actual_exposure"]:,} TRAIN exposures / {group["actual_full_data_epochs"]:.3f} epochs; lower is better.\n'
                         '* = failed gate/unpaired seeds. Green <1, red >1; colors capped at0.5..1.5.',
                         ha='center', va='bottom', fontsize=9)
                fig.subplots_adjust(left=.30, right=.98, top=.85, bottom=.23)
                name = f'{stage}_ratios_c{cohort_index + 1}_p{page_index + 1}'
                files = drawing.save_figure(fig, output, name)
                metadata['files'].extend(files)
                metadata['heatmaps'].append({'phase': stage, 'cohort_sha256': cohort, 'page': page_index + 1,
                    'actual_exposure': group['actual_exposure'], 'arms': [arm['variant'] for arm in page],
                    'actual_ratios': matrix.tolist(), 'scope_scenes': group['cohort']['num_samples'],
                    'independent_training_N': [arm['independent_training_N'] for arm in page],
                    'actual_endpoint_only': True, 'max_arms_per_page': 8})
    long = [group for group in groups if group['phase'] == 'long']
    for cohort_index, cohort in enumerate(sorted({group['cohort_sha256'] for group in long})):
        observations = [group for group in long if group['cohort_sha256'] == cohort]
        curves = defaultdict(list)
        for group in observations:
            for arm in group['arms']:
                curves[arm['arm_id']].append(arm)
        curves = {name: sorted(points, key=lambda item: item['actual_exposure']) for name, points in curves.items()}
        for metric in (*METRICS, 'independent_training_N'):
            fig, ax = plt.subplots(figsize=(drawing.FIGURE_WIDTH_IN, 4.35))
            data = []
            for index, (name, points) in enumerate(sorted(curves.items())):
                x = [point['actual_full_data_epochs'] for point in points]
                count = [point['independent_training_N'] for point in points]
                batches = sorted({pair[0] for point in points for pair in point['batch_update_pairs']})
                label = drawing.compact(points[0]['variant'], 24) + f' (B{batches[0]}; TRAIN N=' + ','.join(map(str, sorted(set(count)))) + ')'
                if metric == 'independent_training_N':
                    y, sd = count, [None] * len(points)
                else:
                    values = [point['metrics_raw_jsd_collision_percent'][metric] for point in points]
                    y, sd = [value['mean'] for value in values], [value['sd'] for value in values]
                color = drawing.PALETTE[index % len(drawing.PALETTE)]
                ax.plot(x, y, '-o', label=label, color=color, markersize=4.5, linewidth=1.7)
                # Sample SD is across independent TRAIN means. With N=1 it is
                # undefined, so no band/errorbar is rendered at that endpoint.
                defined = [i for i, value in enumerate(sd) if value is not None]
                if defined:
                    ax.errorbar([x[i] for i in defined], [y[i] for i in defined],
                                yerr=[sd[i] for i in defined], fmt='none', color=color, capsize=3, linewidth=1.3)
                data.append({'arm_id': name, 'variant': points[0]['variant'], 'x_actual_epochs': x,
                             'actual_exposures': [point['actual_exposure'] for point in points],
                             'mean': y, 'sample_sd_across_TRAIN': sd, 'independent_TRAIN_N': count,
                             'TRAIN_membership_by_exposure': [point['training_seed_ids'] for point in points],
                             'batch_update_pairs': [point['batch_update_pairs'] for point in points]})
            ax.set_xlabel('Actual TRAIN exposures /493,542 (equivalent full-data epochs)')
            ax.set_ylabel('Independent TRAIN N' if metric == 'independent_training_N' else
                          'Collision rate (%)' if metric == 'collision_rate' else 'Raw JSD')
            ax.set_title('Long training: ' + ('independent TRAIN count' if metric == 'independent_training_N' else drawing.LABELS[metric]), fontsize=12)
            ax.grid(alpha=.2)
            if metric == 'independent_training_N':
                ax.set_yticks(sorted({int(n) for curve in data for n in curve['mean']}))
                ax.set_ylim(.5, max(curve['mean'][-1] for curve in data) + .5)
            ax.legend(loc='best', fontsize=9, frameon=False)
            fig.text(.5, .018, 'GEN 2817/3817 averaged within TRAIN; bars = sample SD across TRAIN, absent at N=1.\n'
                     'TRAIN 4817 adaptive; 5817 independent. N/membership can change with exposure.', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .16, 1, .97))
            name = f'long_c{cohort_index + 1}_' + metric
            files = drawing.save_figure(fig, output, name)
            metadata['files'].extend(files)
            metadata['long_curves'].append({'cohort_sha256': cohort, 'metric': metric,
                                            'curves': data, 'files': files, 'unobserved_points_padded': False})
    metadata['full50000'] = full_histogram_figures(rows, output, drawing)
    metadata['files'].extend(metadata['full50000'].get('files', []))
    metadata['layout_audit'] = dict(drawing.FIGURE_LAYOUTS)
    controller.save(output / 'metadata.json', metadata)
    return metadata


def full_histogram_figures(rows, output, drawing):
    import matplotlib.pyplot as plt
    import numpy as np
    from tools.plot_component_loss_distributions import FEATURES, validate_counts
    observed = [row for row in rows if row['phase'] == 'full50000']
    if len(observed) != 4 or len({row['variant'] for row in observed}) != 2:
        return {'status': 'missing', 'reason': 'Exactly two actual50000 arms x two GEN are required; no missing rows padded',
                'actual_observations': len(observed), 'files': []}
    if len({row['cohort_sha256'] for row in observed}) != 1 or {row['seed'] for row in observed} != {4817}:
        raise ValueError('Full50000 histogram physical cohorts or frozen TRAIN origins differ')
    by_arm = defaultdict(list)
    for row in observed:
        by_arm[row['variant']].append(row)
    for values in by_arm.values():
        seed_means(values)
    arms = sorted(by_arm, key=lambda name: (name != 'control', name))
    loaded, sources = {}, []
    for arm in arms:
        loaded[arm] = []
        for row in sorted(by_arm[arm], key=lambda item: item['eval_seed']):
            path = Path(row['report_source']).parent / 'feature_histograms.json'
            if not path.is_file():
                return {'status': 'missing', 'reason': f'Missing actual histogram {path}', 'files': []}
            loaded[arm].append(read(path))
            sources.append({'arm': arm, 'GEN_seed': row['eval_seed'], 'path': str(path), 'sha256': drawing.sha256(path)})
    files, statistics_metadata = [], []
    for feature, spec in FEATURES.items():
        edges, gt_count, gt_prob = validate_counts(loaded[arms[0]][0][feature], 'matched_gt', 'Matched GT')
        fig, ax = plt.subplots(figsize=(drawing.FIGURE_WIDTH_IN, 4.6))
        ax.stairs(gt_prob, edges, label='Matched GT', color=drawing.PALETTE[0], linewidth=2)
        peaks = [gt_prob]
        for index, arm in enumerate(arms, 1):
            probabilities, counts = [], []
            for histogram in loaded[arm]:
                ref_edges, ref_count, _ = validate_counts(histogram[feature], 'matched_gt', arm + ' GT')
                gen_edges, gen_count, gen_prob = validate_counts(histogram[feature], 'generated', arm)
                if not np.array_equal(ref_edges, edges) or not np.array_equal(ref_count, gt_count) or not np.array_equal(gen_edges, edges):
                    raise ValueError('Full50000 exact matchedGT counts/bin edges differ across arms/GEN')
                probabilities.append(gen_prob); counts.append(gen_count.tolist())
            probability = np.mean(probabilities, axis=0)
            ax.stairs(probability, edges, label=drawing.compact(arm, 24), color=drawing.PALETTE[index], linewidth=1.8)
            peaks.append(probability)
            statistics_metadata.append({'feature': feature, 'arm': arm, 'GEN_seeds': [2817, 3817],
                'mean_normalized_bin_probability_across_GEN': probability.tolist(), 'actual_counts_by_GEN': counts,
                'matchedGT_counts': gt_count.tolist(), 'edges': edges.tolist()})
        limits = spec['zoom'] or (float(edges[0]), float(edges[-1]))
        visible = (edges[:-1] < limits[1]) & (edges[1:] > limits[0])
        ax.set_xlim(*limits)
        ax.set_ylim(0, 1.12 * max(float(probability[visible].max()) for probability in peaks))
        ax.set_title(f'50,000 scenes: {spec["label"]}', fontsize=12)
        ax.set_xlabel(f'{spec["label"]} ({spec["unit"]})'); ax.set_ylabel('Probability per official bin')
        ax.grid(axis='y', alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .075), fontsize=9)
        fig.text(.5, .014, 'Same TRAIN 4817; mean of two normalized GEN histograms (not pooled vehicle counts).\n'
                 'Exact matchedGT bins verified; size axes zoom without renormalization.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .25, 1, .98))
        files.extend(drawing.save_figure(fig, output, 'full50000_' + feature))
    first = observed[0]
    if len({(row['num_gt_vehicles'], row['num_gt_onroad_vehicles']) for row in observed}) != 1:
        raise ValueError('Full50000 physical matchedGT population differs across arms/GEN')
    fractions = [first['gt_onroad_fraction']] + [statistics.fmean(row['generated_onroad_fraction'] for row in by_arm[arm]) for arm in arms]
    counts = [first['num_gt_vehicles']] + [statistics.fmean(row['num_generated_vehicles'] for row in by_arm[arm]) for arm in arms]
    fig, ax = plt.subplots(figsize=(drawing.FIGURE_WIDTH_IN, 4.3))
    ax.bar(range(3), np.asarray(fractions) * 100, color=drawing.PALETTE[:3], width=.58)
    ax.set_xticks(range(3), ['Matched GT'] + [drawing.compact(arm, 22) for arm in arms], fontsize=9)
    ax.set_ylabel('On-road vehicles (%)'); ax.set_ylim(0, 120); ax.set_yticks(range(0, 101, 20))
    ax.set_title('50,000 scenes: physical road coverage / vehicle counts', fontsize=12)
    ax.grid(axis='y', alpha=.2)
    for index, (value, count) in enumerate(zip(fractions, counts)):
        ax.text(index, value * 100 + 2, f'{value * 100:.2f}%\nmean count={count:,.0f}', ha='center', fontsize=9)
    fig.text(.5, .02, 'TRAIN 4817 only; two GEN mean of coverage and evaluated vehicle count.\n'
             'This scope audit adds no independent TRAIN seed and does not reselect.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .98))
    files.extend(drawing.save_figure(fig, output, 'full50000_road_coverage'))
    return {'status': 'actual_complete', 'files': files, 'histogram_sources': sources,
            'bin_statistics': statistics_metadata, 'roadcoverage': fractions, 'mean_vehicle_counts': counts,
            'aggregation': 'Equal GEN means within TRAIN4817; fullset audit only, not independent training confirmation'}


class CohortConsumerTests(unittest.TestCase):
    @staticmethod
    def row(name, runtime, *, precision='32-true', data='same-scenes', source='same-model-source'):
        observation = {'cohort': {'eval_manifest_sha256': data, 'precision': precision,
            'float32_matmul_precision': 'medium', 'metric_backend_sha256': 'same-helper',
            'conditioning': 'same-GT-map/fixed-ego', 'cyclic_gc_policy': runtime},
            'phase': 'long', 'variant': name, 'arm_id': name, 'seen_training_examples': 1024,
            'evaluation_weights': 'ema', 'num_samples': 4096, 'seed': 4817, 'eval_seed': 2817,
            'planned_generation_seeds': [2817], 'budget': 2, 'train_batch_size': 512,
            'initial_generator_sha256': name, 'actual_GEN_observations': 1, 'evaluation_only': False,
            'generated_onroad_fraction': .85, 'num_generated_vehicles': 100, 'num_gt_vehicles': 100,
            **{k: 1. for k in METRICS}}
        observation['cohort']['num_samples'] = 4096
        return attach_cohorts(observation, {'cyclic_gc_policy': runtime, 'young_gc_interval': 128 if runtime == 'gc128' else 0,
                               'defer_cyclic_gc': True}, {'command': ['python', 'frozen.py'], 'source_hashes': {}}, source)

    def test_two_gc_policies_compare_in_one_semantic_scope_but_runtime_is_retained(self):
        old, recovered = self.row('control', 'gc128'), self.row('candidate', 'gc0-cached')
        self.assertEqual(old['cohort_sha256'], recovered['cohort_sha256'])
        self.assertNotEqual(old['runtime_cohort_sha256'], recovered['runtime_cohort_sha256'])
        self.assertEqual(old['runtime_cohort']['young_gc_interval'], 128)
        self.assertEqual(recovered['runtime_cohort']['young_gc_interval'], 0)
        group, = make_groups([old, recovered])
        self.assertTrue(group['mixed_execution_runtime_cohorts'])
        self.assertEqual(len(group['runtime_cohorts']), 2)
        candidate = next(a for a in group['arms'] if a['variant'] == 'candidate')
        self.assertEqual(candidate['comparison']['R'], 1.)
        self.assertTrue(candidate['comparison']['ranking_eligible'])

    def test_precision_scene_manifest_and_model_source_boundaries_still_split(self):
        control = self.row('control', 'gc128')
        changes = (dict(precision='16-mixed'), dict(data='other-scenes'), dict(source='changed-model-source'))
        for change in changes:
            with self.subTest(change=change):
                candidate = self.row('candidate', 'gc0-cached', **change)
                self.assertNotEqual(control['cohort_sha256'], candidate['cohort_sha256'])
                groups = make_groups([control, candidate])
                self.assertEqual(len(groups), 2)
                self.assertFalse(next(g for g in groups if g['arms'][0]['variant'] == 'candidate')['control_present'])

    def test_runtime_helper_and_entrypoint_sha_are_not_lost(self):
        observation = {'cohort': {'precision': '32-true', 'float32_matmul_precision': 'medium'}}
        value = attach_cohorts(observation, {'cyclic_gc_policy': 'gc0', 'young_gc_interval': 0},
            {'command': ['python', 'runtime_entry.py'], 'source_hashes': {'tools/frozen.py': 'frozen-sha'},
             'runtime_recovery': {'runtime_loader_source_hashes': {'cache.py': 'actual-cache-sha'}, 'wrapper_sha256': 'actual-wrapper-sha'}}, 'model-sha')
        self.assertEqual(value['runtime_cohort']['runtime_loader_source_hashes'], {'cache.py': 'actual-cache-sha'})
        self.assertEqual(value['runtime_cohort']['recovery_wrapper_sha256'], 'actual-wrapper-sha')
        self.assertEqual(value['runtime_cohort']['recorded_tool_source_hashes'], {'tools/frozen.py': 'frozen-sha'})
        self.assertNotIn('cyclic_gc_policy', value['cohort'])
        self.assertEqual(value['cohort']['float32_matmul_precision'], 'medium')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=DEFAULT_RUNS)
    parser.add_argument('--output', type=Path, default=DEFAULT_ART / 'analysis')
    parser.add_argument('--figures', action='store_true', help='Render actual-only publication-size PNG/SVG charts')
    parser.add_argument('--self-test', action='store_true', help='CPU semantic/runtime separation regression tests')
    args = parser.parse_args()
    if args.self_test:
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(CohortConsumerTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful() else 1
    rows, stages, warnings = discover(args.runs)
    summary = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'ACTUAL_PARTIAL_SNAPSHOT',
               'scope': 'CPU-only; verified actual observations only; no synthetic/padded missing stages',
               'semantic_cohort_policy': 'metric_model_data_precision_without_runtime_policy_v2',
               'runtime_policy_comparison': 'Recorded separately from semantic scope; mixed runtime policies are disclosed, not claimed equal.',
               'consumer_source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (Path(__file__).resolve(), ROOT / 'tools/runtime_completion.py')},
               'runs': str(args.runs.resolve()), 'actual_observations': len(rows), 'stages': stages,
               'groups': make_groups(rows), 'timing': timing(rows, stages), 'warnings': warnings,
               'deadline_utc': '2026-10-12T02:00:00+00:00',
               'caveats': ['TRAIN4817 is reused for adaptive selection; TRAIN5817 only confirms a frozen recipe.',
                           'Two GEN seeds do not increase independent training N.',
                           'Full50k is a scope audit of seed4817 checkpoints, not a new TRAIN repetition or untouched test.',
                           'Final exposure is chosen from about8.50,10.62,12.75epochs under measured throughput; no128epoch optimum is proven.']}
    selection = DEFAULT_ART / 'final_selection.json'
    if selection.is_file() and args.runs.resolve() == DEFAULT_RUNS.resolve():
        summary['controller_frozen_selection'] = read(selection)
    args.output.mkdir(parents=True, exist_ok=True)
    controller.save(args.output / 'summary.json', summary)
    controller.save(args.output / 'actual_observations.json', rows)
    if args.figures:
        summary['figures'] = figures(rows, summary['groups'], args.output / 'figures')
        controller.save(args.output / 'summary.json', summary)
    flat = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows]
    with (args.output / 'actual_observations.csv').open('w', newline='') as handle:
        fields = sorted({key for row in flat for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps({'actual_observations': len(rows), 'included_stages': sum(stage['included'] for stage in stages),
                      'warnings': warnings, 'output': str(args.output.resolve()), 'torch_imported': 'torch' in sys.modules}))
    return 0


if __name__ == '__main__':
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(result)
