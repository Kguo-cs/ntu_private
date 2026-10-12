"""Deadline-bounded serial GPU search, with fresh generators at every promotion."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf

ART = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011'
RUNS = ROOT / 'logs/init_diffusion_overnight_20261011'
OLD = ROOT / 'docs/experiments/init_diffusion_frozen_map_search_20261011'
PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
PREFIX = 'model.model_config.decoder.init_diffusion.'
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd', 'collision_rate')
DEADLINE = datetime(2026, 10, 12, 1, 35, tzinfo=timezone.utc).timestamp()
TRAIN_COUNT = 493542
ATTR = ('length_jsd', 'width_jsd', 'speed_jsd')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def actual_exposure(steps, batch):
    epochs, remainder = divmod(steps, (TRAIN_COUNT + batch - 1) // batch)
    return epochs * TRAIN_COUNT + remainder * batch


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot aggregate an empty observation set')
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row['seed'], []).append(row)
    for seed, group in by_seed.items():
        if len({row['eval_seed'] for row in group}) != len(group):
            raise ValueError(f'Duplicate GEN observations for TRAIN {seed}')
    keys = METRICS + ('generated_onroad_fraction', 'num_generated_vehicles', 'num_gt_vehicles')
    return {key: sum(sum(r[key] for r in group) / len(group) for group in by_seed.values()) / len(by_seed)
            for key in keys}


def compare(values, baseline):
    if any(not math.isfinite(values[key]) or not math.isfinite(baseline[key])
           or values[key] < 0 or baseline[key] <= 0 for key in METRICS):
        raise ValueError('Metric ratios require finite nonnegative candidates and positive controls')
    ratios = {key: values[key] / baseline[key] for key in METRICS}
    checks = {'onroad': 0 <= values['generated_onroad_fraction'] <= 1
                          and values['generated_onroad_fraction'] >= baseline['generated_onroad_fraction'] - .02,
              'vehicle_ratio': values['num_gt_vehicles'] > 0
                          and .9 <= values['num_generated_vehicles'] / values['num_gt_vehicles'] <= 1.1,
              **{key: ratios[key] <= 1.15 for key in ATTR}}
    return {'metrics': values, 'R': sum(ratios.values()) / len(METRICS),
            'R_attr': sum(ratios[key] for key in ATTR) / len(ATTR), 'ratios': ratios,
            'gate_checks': checks, 'eligible': all(checks.values())}


def source_hashes():
    files = list((ROOT / 'src/smart/diffusion').glob('*.py')) + [ROOT / 'configs/model/smart.yaml',
            ROOT / 'src/smart/model/smart.py', ROOT / 'src/smart/model/smart_gail.py',
            ROOT / 'src/smart/datasets/scalable_dataset.py',
            ROOT / 'src/smart/datamodules/scalable_datamodule.py', ROOT / 'src/smart/tokens/token_processor.py',
            ROOT / 'src/smart/modules/map_decoder.py',
            ROOT / 'tools/scratch_component_loss_sweep.py',
            ROOT / 'tools/overnight_init_diffusion_sweep.py', Path(__file__)]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def process_running(pid):
    """Conservatively prevent handoff while any prior submitting process lives."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_handoff():
    """An explicit terminal-failure handoff never fabricates old completion."""
    completion = OLD / 'queue_complete.json'
    if completion.is_file():
        value = read(completion)
        if value.get('status') in ('COMPLETE', 'COMPLETE_ACTUAL_SEARCH'):
            return {'mode': 'verified_old_queue_completion', 'path': str(completion),
                    'sha256': hashlib.sha256(completion.read_bytes()).hexdigest(), 'old_status': value['status']}
    marker = ART / 'gpu_handoff.json'
    if not marker.is_file():
        return None
    value = read(marker)
    pids = value.get('terminated_pids')
    if (value.get('released') is not True or value.get('old_queue_terminal') is not True
            or value.get('gpu_processes_reaped') is not True
            or not isinstance(value.get('reason'), str) or not value['reason'].strip()
            or not isinstance(pids, list) or not pids
            or any(type(pid) is not int or pid <= 0 for pid in pids)):
        raise RuntimeError('gpu_handoff.json must certify terminal old queue, reaped GPU processes, reason and terminated_pids')
    alive = [pid for pid in pids if process_running(pid)]
    if alive:
        raise RuntimeError(f'GPU handoff refused: prior controller/child PIDs still exist: {alive}')
    return {'mode': 'explicit_terminal_old_queue_handoff', 'path': str(marker),
            'sha256': hashlib.sha256(marker.read_bytes()).hexdigest(),
            'old_status': value.get('old_queue_status', 'PARTIAL_FAILED_RUNTIME'),
            'old_queue_completed': False, 'terminated_pids': pids, 'reason': value['reason']}


def stop_after_failed_screen_control(name, rows):
    if name != 'control' or rows:
        return False
    status = 'DEADLINE_NO_SCREEN_CONTROL' if time.time() >= DEADLINE else 'FAILED_SCREEN_CONTROL'
    save(ART / 'queue_complete.json', {'status': status, 'finished_unix': time.time(),
         'complete_final_matrix': False, 'complete_full50000_audit': False,
         'reason': 'First screening control did not complete; stop immediately instead of training unrankable candidates.',
         'manual_fresh_recovery_required': True})
    save(ART / 'queue_active.json', {'stage': status, 'controller_pid': os.getpid(), 'status': 'TERMINAL_PARTIAL'})
    return True


def make_command(label, arm, *, steps, scenes, seed=4817, checkpoints=(), extra_seed=False, evaluation_checkpoint=None):
    if Path(label).name != label or label in ('', '.', '..'):
        raise ValueError('Stage label must be a safe basename')
    if scenes not in (1024, 4096, 50000):
        raise ValueError('Only declared 1024/4096/50000 evaluation cohorts are supported')
    folder = RUNS / label
    design = ART / 'design' / label
    inputs = {str(design.with_suffix('.recipe.json')): arm['recipe'],
              str(design.with_suffix('.variants.json')):
                  {arm['name']: {'__config': arm['config'], '__lr': arm.get('lr', .0005)}}}
    command = [PYTHON, 'tools/overnight_init_diffusion_sweep.py', '--map-checkpoint', arm['map'],
               '--batch-size', str(arm['batch']), '--scenes', str(scenes), '--eval-batch-size', '128',
               '--workers', '0', '--defer-cyclic-gc', '--young-gc-interval', '128',
               '--seed', str(seed), '--eval-seed', '2817',
               '--normalizer-json', arm['normalizer'], '--evaluation-weights', 'ema',
               '--steps', str(steps), '--checkpoint-steps', ','.join(map(str, checkpoints)),
               '--recipe', str(design.with_suffix('.recipe.json')), '--variant-file', str(design.with_suffix('.variants.json')),
               '--variants', arm['name'], '--output', str(folder), '--allow-multiple-epochs']
    if scenes == 50000:
        # The native runner selects its sorted full physical-cache manifest;
        # require-full-set also verifies the actual matched reference scope.
        command += ['--require-full-set']
    elif scenes == 4096:
        command += ['--eval-manifest', str(OLD / 'confirmation4096_files.json')]
    else:
        subset = read(OLD / 'confirmation4096_files.json')
        subset = subset.get('files') if isinstance(subset, dict) else subset
        if not isinstance(subset, list) or len(subset) != 4096:
            raise ValueError('The fixed 4096 evaluation manifest is incomplete')
        manifest = ART / f'confirmation{scenes}_files.json'
        inputs[str(manifest)] = subset[:scenes]
        command += ['--eval-manifest', str(manifest)]
    if extra_seed:
        command += ['--extra-eval-seed', '3817']
    if evaluation_checkpoint is not None:
        mapping = design.with_suffix('.checkpoints.json')
        inputs[str(mapping)] = {arm['name']: str(evaluation_checkpoint)}
        command[command.index('--variants') + 1] = ''
        command += ['--evaluation-checkpoints', str(mapping)]
    return command, inputs


def validate_rows(rows, *, name, seed, steps, scenes, batch, generations, budgets=None):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Stage summary is empty or malformed')
    budgets = set(budgets or [steps])
    expected = {(budget, seed, gen) for budget in budgets for gen in generations}
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Stage summary has a non-object row')
        cell = (row.get('budget'), row.get('seed'), row.get('eval_seed'))
        if cell in observed or cell not in expected:
            raise ValueError('Unexpected or duplicate budget/TRAIN/GEN cell')
        observed.add(cell)
        if (row.get('variant') != name or row.get('evaluation_weights') != 'ema'
                or row.get('num_samples') != scenes
                or row.get('seen_training_examples') != actual_exposure(cell[0], batch)
                or row.get('batch_size', row.get('train_batch_size')) != batch):
            raise ValueError('Stage row has incompatible variant, scope, weights, batch or actual exposure')
        keys = METRICS + ('generated_onroad_fraction', 'num_generated_vehicles', 'num_gt_vehicles')
        if any(not isinstance(row.get(key), (int, float)) or isinstance(row[key], bool)
               or not math.isfinite(row[key]) or row[key] < 0 for key in keys):
            raise ValueError('Stage row has incomplete or nonfinite metrics/coverage')
        if row['generated_onroad_fraction'] > 1 or row['num_gt_vehicles'] <= 0:
            raise ValueError('Stage row has invalid physical coverage counts')
        if row.get('map_frozen') is not True or row.get('initialization_scope') != 'frozen_pretrained_map_fresh_G1':
            raise ValueError('Stage row does not certify a frozen source map')
        if not row.get('evaluation_only') and (row.get('resumed_from') is not None
                or row.get('optimizer_continued') or row.get('resume_global_step', 0) != 0
                or row.get('actual_train_offset', 0) != 0):
            raise ValueError('A promoted training arm must start with fresh G1/optimizer/EMA at exposure zero')
    if observed != expected:
        raise ValueError('Stage summary is missing planned budget/TRAIN/GEN cells')


def verify_completion(folder, log_path, spec):
    try:
        protocol, rows = read(folder / 'protocol.json'), read(folder / 'summary.json')
        lines = log_path.read_text().splitlines()
        if not isinstance(protocol, dict) or not protocol or not lines or lines[-1].strip() != 'DONE ' + str(folder):
            raise ValueError('Terminal DONE or protocol does not agree with this stage')
        validate_rows(rows, **spec)
        if (protocol.get('reference_mode') != 'matched' or protocol.get('scenes') != spec['scenes']
                or protocol.get('seed') != spec['seed'] or protocol.get('batch_size') != spec['batch']
                or bool(protocol.get('require_full_set')) != (spec['scenes'] == 50000)):
            raise ValueError('Protocol does not certify the requested physical evaluation scope')
        report_paths = set()
        for row in rows:
            report_path = Path(row['metrics_path'])
            report = read(report_path)
            if (report.get('reference_mode') != 'matched' or report.get('num_samples') != spec['scenes']
                    or report.get('num_gt_samples') != spec['scenes']
                    or row.get('map_checkpoint_sha256') != protocol.get('map_checkpoint_sha256')):
                raise ValueError('Per-GEN report does not certify matched physical scope/source map')
            if any(not math.isclose(row[key], report['agent_metrics'][key], rel_tol=1e-10, abs_tol=1e-12)
                   for key in METRICS):
                raise ValueError('Summary metrics disagree with their actual per-GEN report')
            report_paths.add(report_path)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f'{folder.name}: completion artifacts are invalid; manual recovery needed: {error}') from error
    paths = {'protocol': folder / 'protocol.json', 'summary': folder / 'summary.json', 'log': log_path}
    paths.update({f'report:{path.relative_to(folder)}': path for path in sorted(report_paths)})
    for path in sorted(folder.glob('*/config.yaml')) + sorted(folder.glob('*/initialization.json')):
        paths[str(path.relative_to(folder))] = path
    return rows, {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                  for name, path in paths.items()}


def run(label, arm, *, steps, scenes, seed=4817, checkpoints=(), extra_seed=False, evaluation_checkpoint=None):
    """Run one isolated arm; partial/failed outputs are never eligible evidence."""
    folder = RUNS / label
    command, inputs = make_command(label, arm, steps=steps, scenes=scenes, seed=seed,
                                  checkpoints=checkpoints, extra_seed=extra_seed,
                                  evaluation_checkpoint=evaluation_checkpoint)
    spec = {'name': arm['name'], 'seed': seed, 'steps': steps, 'scenes': scenes,
            'batch': arm['batch'], 'generations': [2817, 3817] if extra_seed else [2817],
            'budgets': [steps] if evaluation_checkpoint else sorted(set([steps, *checkpoints]))}
    fingerprints = {'design_inputs': {path: digest(value) for path, value in inputs.items()},
                    'source_hashes': source_hashes(),
                    'external_inputs': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                        for path in (arm['map'], arm['normalizer'])}}
    if '--eval-manifest' in command:
        manifest = command[command.index('--eval-manifest') + 1]
        if manifest not in inputs:
            fingerprints['external_inputs'][manifest] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if evaluation_checkpoint:
        fingerprints['external_inputs'][str(evaluation_checkpoint)] = hashlib.sha256(Path(evaluation_checkpoint).read_bytes()).hexdigest()
    log_path = RUNS / (label + '.log')
    if (folder / 'execution.json').exists():
        previous = read(folder / 'execution.json')
        if previous.get('command') != command or previous.get('fingerprints') != fingerprints or previous.get('arm') != arm:
            raise RuntimeError(f'{label}: command/design/source changed; refusing stale stage reuse')
        if previous.get('status') != 'success' or previous.get('returncode') != 0:
            raise RuntimeError(f'{label}: partial/failed stage; manual checkpoint recovery needed')
        rows, artifacts = verify_completion(folder, log_path, spec)
        if previous.get('artifacts') != artifacts:
            raise RuntimeError(f'{label}: completion artifacts changed; refusing stale stage reuse')
        previous.update(last_verified_unix=time.time())
        save(folder / 'execution.json', previous)
        print('SKIP VERIFIED COMPLETE ' + label, flush=True)
        return rows
    if folder.exists():
        raise RuntimeError(f'{label}: existing output has no verified execution manifest; manual recovery needed')
    if time.time() >= DEADLINE:
        return []
    folder.mkdir(parents=True, exist_ok=False)
    for path, value in inputs.items():
        save(path, value)
    execution = {'label': label, 'arm': arm, 'command': command, 'seed': seed,
                 'started_unix': time.time(), 'status': 'running', 'source_hashes': fingerprints['source_hashes'],
                 'fingerprints': fingerprints, 'completion_spec': spec}
    save(folder / 'execution.json', execution)
    print('START ' + label, flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        execution['child_pid'] = process.pid
        save(folder / 'execution.json', execution)
        save(ART / 'queue_active.json', {'stage': label, 'child_pid': process.pid, 'controller_pid': os.getpid(), 'log': str(log_path)})
        while process.poll() is None:
            try:
                status = Path(f'/proc/{process.pid}/status').read_text()
                rss = next(line for line in status.splitlines() if line.startswith('VmRSS:'))
                rss_kib = int(rss.split()[1])
                execution['peak_observed_rss_kib'] = max(execution.get('peak_observed_rss_kib', 0), rss_kib)
                execution['last_rss_kib'] = rss_kib
                save(folder / 'execution.json', execution)
            except (OSError, StopIteration, ValueError):
                pass
            if time.time() >= DEADLINE:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                execution['status'] = 'deadline_interrupted'
                break
            time.sleep(15)
    execution.update(ended_unix=time.time(), returncode=process.returncode)
    if execution['status'] == 'running':
        execution['status'] = 'completed' if process.returncode == 0 else 'failed'
    save(folder / 'execution.json', execution)
    print(execution['status'].upper() + ' ' + label, flush=True)
    if execution['status'] != 'completed':
        return []
    rows, artifacts = verify_completion(folder, log_path, spec)
    execution.update(status='success', artifacts=artifacts, last_verified_unix=time.time())
    save(folder / 'execution.json', execution)
    return rows


def endpoint(rows):
    return [row for row in rows if row['budget'] == max(r['budget'] for r in rows)] if rows else []


def row_cohort(row):
    report = read(row['metrics_path'])
    stage = next((path for path in Path(row['metrics_path']).parents if (path / 'execution.json').is_file()), None)
    execution = read(stage / 'execution.json') if stage is not None else {}
    sources = execution.get('source_hashes', {})
    model_sources = {key: value for key, value in sources.items()
                     if key.startswith('src/smart/') or key == 'configs/model/smart.yaml'}
    return {key: row.get(key) for key in ('eval_manifest_sha256', 'num_samples', 'evaluation_weights',
            'map_checkpoint_sha256', 'initialization_scope')} | {
        'reference_mode': report.get('reference_mode'),
        'reference_source_manifest_sha256': report.get('reference_source_manifest_sha256'),
        'conditioning': report.get('decoder', {}).get('conditioning'),
        'metric_source': report.get('official_source_sha256'),
        'model_source_hashes': model_sources or None,
    }


def ranking(results, control_name='control', generations=(2817,)):
    if not results.get(control_name):
        return []
    control = endpoint(results[control_name])
    seed = control[0]['seed']
    seen = control[0]['seen_training_examples']
    cohort = row_cohort(control[0])
    for name, rows in results.items():
        if not rows:
            continue
        current = endpoint(rows)
        validate_rows(current, name=name, seed=seed, steps=current[0]['budget'],
                      scenes=control[0]['num_samples'], batch=current[0]['batch_size'], generations=generations)
        if any(row['seen_training_examples'] != seen or row_cohort(row) != cohort for row in current):
            raise ValueError('Cannot rank different actual exposure or physical evaluation cohorts')
    baseline = aggregate(control)
    ranked = [dict(compare(aggregate(endpoint(rows)), baseline), name=name)
              for name, rows in results.items() if rows]
    return sorted(ranked, key=lambda row: (not row['eligible'], row['R'], row['R_attr'], row['name']))


def promote_screen(ranked):
    eligible = [row for row in ranked if row['eligible'] and row['name'] != 'control']
    novel = [row for row in eligible if row['name'] != 'combo_reference']
    chosen = novel[:2]
    if len(chosen) < 2:
        chosen += [row for row in eligible if row['name'] == 'combo_reference'][:2 - len(chosen)]
    exploratory = not chosen
    if exploratory:
        chosen = sorted((row for row in ranked if row['name'] != 'control'),
                        key=lambda row: (row['R'], row['R_attr'], row['name']))[:2]
    return ['control'] + [row['name'] for row in chosen], exploratory


def choose_medium(ranked):
    challengers = [row for row in ranked if row['name'] != 'control']
    eligible = [row for row in challengers if row['eligible']]
    chosen = eligible[0] if eligible else min(challengers, key=lambda row: (row['R'], row['R_attr'], row['name'])) if challengers else None
    return chosen, bool(chosen and not eligible)


def choose_final_budget(medium, challenger, now=None):
    """Protect two fresh TRAIN seeds before increasing their shared exposure."""
    now = time.time() if now is None else now
    rates, evaluation_rates = {}, {}
    for name in ('control', challenger):
        rows = endpoint(medium[name])
        budgets, fits = {row['budget'] for row in rows}, {row['train_seconds'] for row in rows}
        if len(budgets) != 1 or len(fits) != 1 or not all(value > 0 for value in fits):
            raise ValueError('Final budget projection requires complete actual medium fit timing')
        rates[name] = next(iter(fits)) / next(iter(budgets))
        measured = [row['eval_seconds'] / row['num_samples'] for row in medium[name]]
        if not measured or any(not math.isfinite(value) or value <= 0 for value in measured):
            raise ValueError('Final budget projection requires positive actual evaluation timing')
        evaluation_rates[name] = max(measured)
    # Each arm has two TRAIN jobs, three checkpoint budgets and two GENs.
    long_evaluation_seconds = 12 * 4096 * sum(evaluation_rates.values())
    # Audit one frozen TRAIN checkpoint per arm with both GENs. Use observed
    # per-scene costs and allow slower full-set decoding/metric accumulation.
    full_audit_seconds = max(20 * 60, 2 * 50000 * sum(evaluation_rates.values()) * 1.25)
    projections = []
    for control_steps, candidate_steps in ((12288, 24576), (10240, 20480), (8192, 16384)):
        fit_seconds = 2 * (control_steps * rates['control'] + candidate_steps * rates[challenger])
        projected_seconds = (fit_seconds + long_evaluation_seconds) * 1.15 + full_audit_seconds
        projections.append({'control_B512_updates': control_steps, 'candidate_B256_updates': candidate_steps,
                            'actual_exposure_per_TRAIN': actual_exposure(control_steps, 512),
                            'equivalent_full_data_epochs': actual_exposure(control_steps, 512) / TRAIN_COUNT,
                            'pair_two_TRAIN_fit_seconds': fit_seconds,
                            'long_evaluation_seconds': long_evaluation_seconds,
                            'full50k_reserved_seconds': full_audit_seconds,
                            'with_15percent_margin_and_measured_audit_seconds': projected_seconds,
                            'estimated_finish_unix': now + projected_seconds,
                            'fits_gpu_cutoff': now + projected_seconds <= DEADLINE})
    chosen = next((item for item in projections if item['fits_gpu_cutoff']), projections[-1])
    return chosen, {'observed_medium_seconds_per_update': rates,
                    'observed_max_eval_seconds_per_scene': evaluation_rates, 'projections': projections,
                    'decision_unix': now, 'gpu_cutoff_unix': DEADLINE,
                    'chosen': chosen, 'deadline_risk': not chosen['fits_gpu_cutoff'],
                    'rule': 'Highest shared exposure fitting measured two-TRAIN fit and24 final evaluations plus15% margin; full50k reserve=max20min or measured4GEN cost x1.25. If none fit, attempt minimum with explicit deadline risk. Two TRAIN seeds outrank higher exposure.'}


def design_arms():
    from tools.overnight_init_diffusion_sweep import sweep
    def source(label, name):
        folder = ROOT / 'logs/component_loss_frozen_map' / label / name
        cfg = OmegaConf.load(folder / 'config.yaml')
        options = {key: OmegaConf.select(cfg, key) for key in sorted(sweep.SEARCH_CONFIG_KEYS)
                   if OmegaConf.select(cfg, key) is not None}
        return {'name': name, 'recipe': OmegaConf.to_container(cfg.model.model_config.decoder.init_diffusion.component_loss),
                'config': options, 'batch': int(cfg.data.train_batch_size),
                'normalizer': str(ROOT / 'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'),
                'map': str(ROOT / 'src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt')}
    control = source('long_seed817_new_control', 'new_control'); control['name'] = 'control'
    anchor = source('long_seed817_combo_batch256_ego_sincos', 'combo_batch256_ego_sincos'); anchor['name'] = 'combo_reference'
    specs = {
        'speed_p0': {'loss': {'time_power': 0, 'weight': 2.}},
        'speed_l1_p0': {'loss': {'kind': 'l1', 'time_power': 0, 'weight': 2.}},
        'train_features': {'config': {'state_feature_normalization': 'train'}},
        'separate_speed': {'config': {'speed_head_kind': 'separate'}},
        'detached_speed': {'config': {'speed_head_kind': 'separate', 'speed_head_detach_context': True}},
        'softplus_speed': {'config': {'speed_x0_parameterization': 'softplus'}},
        'residual_speed': {'config': {'speed_x0_parameterization': 'time_residual'}},
        'high_noise': {'config': {'time_sample_power': .5}},
        'match_speed4': {'config': {'matching_weights': {'velocity': 4.}}},
        'transport_speed': {'config': {'distribution_loss': {'speed': 2., 'time_min': .5}}},
        'transport_attributes': {'config': {'distribution_loss': {'speed': 2., 'length': 2., 'width': 2., 'time_min': .5}}},
        'depth5': {'config': {'denoiser_num_layers': 5}},
        'width384': {'config': {'denoiser_hidden_dim': 384}},
        'normalized_speed_p0': {'config': {'state_feature_normalization': 'train'}, 'loss': {'time_power': 0, 'weight': 2.}},
        'lane_center_context': {'config': {'map_condition_types': 'lanes_and_edges'}},
        'lane_center_normalized': {'config': {'map_condition_types': 'lanes_and_edges', 'state_feature_normalization': 'train'}},
    }
    arms = [control, anchor]
    for name, spec in specs.items():
        arm = copy.deepcopy(anchor); arm['name'] = name
        arm['config'].update({PREFIX + key: value for key, value in spec.get('config', {}).items()})
        arm['recipe']['components']['velocity'].update(spec.get('loss', {}))
        arms.append(arm)
    return arms


def main():
    ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
    arms = design_arms(); by_name = {arm['name']: arm for arm in arms}
    save(ART / 'search_protocol.json', {'arms': arms, 'start_training_seed': 4817, 'independent_training_seed': 5817,
        'generation_seeds': [2817, 3817], 'screen_exposure': 393216, 'medium_exposure_approx': 2097048,
        'final_exposure_approx': 6291144, 'fresh_G1_at_every_promotion': True, 'frozen_map': True,
        'criterion': 'Mean of seven metric ratios to same-budget control; shape/speed<=1.15; onroad>=control-.02; vehicle counts .9..1.1',
        'deadline_utc': '2026-10-12 01:35:00', 'report_deadline_utc': '2026-10-12 02:00:00',
        'training_order': 'Seeded fixed full order repeated each epoch; no multi-epoch checkpoint continuation',
        'runtime_policy': 'Isolated stages, automatic cyclic GC deferred, imports frozen once, synchronized generation0 every128 TRAIN/test batches; RSS and CUDA allocated/reserved monitored',
        'selection_data_reused': True, 'final128epoch_optimum_proven': False})
    handoff = gpu_handoff()
    while handoff is None:
        if time.time() >= DEADLINE:
            raise RuntimeError('Original queue did not complete before deadline')
        save(ART / 'queue_active.json', {'stage': 'WAIT_ORIGINAL_SERIAL_GPU_QUEUE', 'controller_pid': os.getpid()})
        time.sleep(30)
        handoff = gpu_handoff()
    save(ART / 'accepted_gpu_handoff.json', {**handoff, 'accepted_unix': time.time(),
         'controller_pid': os.getpid(), 'source_hashes': source_hashes()})
    screen = {}
    for arm in arms:
        steps = 768 if arm['batch'] == 512 else 1536
        screen[arm['name']] = run('screen_' + arm['name'], arm, steps=steps, scenes=1024, checkpoints=[steps // 2])
        if stop_after_failed_screen_control(arm['name'], screen[arm['name']]):
            return
        if screen.get('control'):
            save(ART / 'screen_ranking.json', ranking(screen))
    ranked = ranking(screen)
    if not ranked:
        save(ART / 'queue_complete.json', {'status': 'NO_COMPLETE_SCREEN_CONTROL', 'finished_unix': time.time()}); return
    promoted, exploratory_screen = promote_screen(ranked)
    save(ART / 'medium_protocol.json', {'promoted': promoted, 'screen_ranking': ranked,
         'exploratory': exploratory_screen,
         'rule': 'Up to two eligible novel candidates by R, R_attr, name; eligible combo_reference fills remaining slots. If no eligible candidate, explore best R but do not claim improvement; all promotions from scratch.'})
    medium = {}
    for name in promoted:
        arm = by_name[name]; steps = 4096 if arm['batch'] == 512 else 8192
        medium[name] = run('medium_' + name, arm, steps=steps, scenes=4096,
                           checkpoints=[steps // 2], extra_seed=True)
    if not medium.get('control'):
        save(ART / 'queue_complete.json', {'status': 'DEADLINE_NO_MEDIUM_CONTROL', 'finished_unix': time.time()}); return
    ranked_medium = ranking(medium, generations=(2817, 3817))
    chosen_medium, exploratory_medium = choose_medium(ranked_medium)
    if not chosen_medium:
        save(ART / 'queue_complete.json', {'status': 'NO_COMPLETE_MEDIUM_CHALLENGER', 'finished_unix': time.time()}); return
    challenger = chosen_medium['name']
    budget, budget_rule = choose_final_budget(medium, challenger)
    control_final_steps = budget['control_B512_updates']
    candidate_final_steps = budget['candidate_B256_updates']
    final_exposure = budget['actual_exposure_per_TRAIN']
    save(ART / 'frozen_long_protocol.json', {'control': 'control', 'challenger': challenger,
         'train_seeds': [4817, 5817], 'generation_seeds': [2817, 3817], 'medium_ranking': ranked_medium,
         'final_budget_B512': control_final_steps, 'final_budget_B256': candidate_final_steps,
         'actual_final_exposure_per_TRAIN': final_exposure, 'final_budget_rule': budget_rule,
         'exploratory': exploratory_medium,
         'selection': 'Frozen candidate before independent-seed long confirmation', 'full50000_no_reselection': True})
    final = {'control': [], challenger: []}
    interrupted_pair = False
    for seed in (4817, 5817):
        for name in ('control', challenger):
            arm = by_name[name]; steps = control_final_steps if arm['batch'] == 512 else candidate_final_steps
            shared_checkpoints = [control_final_steps // 3, control_final_steps * 2 // 3]
            checkpoints = [step * 512 // arm['batch'] for step in shared_checkpoints]
            rows = run(f'long_seed{seed}_{name}', arm, steps=steps, scenes=4096, seed=seed,
                       checkpoints=checkpoints, extra_seed=True)
            final[name].extend(endpoint(rows))
            save(ART / 'long_completed_rows.json', final)
            # Do not start an unmatched seed or reselect on missing cells.
            if not rows:
                interrupted_pair = True
                break
        if interrupted_pair:
            break
    matrix = [row for rows in final.values() for row in rows]
    cells = {(row['variant'], row['seed'], row['eval_seed']) for row in matrix}
    expected = {(name, seed, gen) for name in final for seed in (4817, 5817) for gen in (2817, 3817)}
    complete_matrix = cells == expected and len(matrix) == len(expected)
    decision = {'complete_matrix': complete_matrix, 'rows': matrix, 'frozen_challenger': challenger,
                'actual_final_exposure_per_TRAIN': final_exposure, 'final_budget_rule': budget_rule,
                'final128epoch_optimum_proven': False}
    if complete_matrix:
        cohorts = {digest(row_cohort(row)) for row in matrix}
        exposures = {row['seen_training_examples'] for row in matrix}
        if len(cohorts) != 1 or exposures != {final_exposure}:
            raise ValueError('Final matrix has different physical cohorts or actual exposures')
        baseline = aggregate(final['control'])
        final_ranked = []
        for name, rows in final.items():
            comparison = dict(compare(aggregate(rows), baseline), name=name)
            per_seed = {str(seed): compare(aggregate([row for row in rows if row['seed'] == seed]),
                              aggregate([row for row in final['control'] if row['seed'] == seed]))
                        for seed in (4817, 5817)}
            comparison.update(per_training_seed=per_seed, aggregate_gates_pass=comparison['eligible'],
                              independent_5817=per_seed['5817'],
                              eligible=comparison['eligible'] and all(item['eligible'] for item in per_seed.values()))
            final_ranked.append(comparison)
        final_ranked.sort(key=lambda row: (not row['eligible'], row['R'], row['R_attr'], row['name']))
        improvements = [row for row in final_ranked if row['eligible'] and row['R'] < 1.]
        decision.update(selected=improvements[0]['name'] if improvements else 'control', ranking=final_ranked,
                        selected_arm=by_name[improvements[0]['name'] if improvements else 'control'])
    save(ART / 'final_selection.json', decision)
    audits = {}
    if complete_matrix:
        for name in ('control', challenger):
            origin = [row for row in final[name] if row['seed'] == 4817 and row['eval_seed'] == 2817][0]
            arm = by_name[name]
            audits[name] = run('full50000_seed4817_' + name, arm, steps=origin['budget'], scenes=50000,
                               seed=4817, extra_seed=True, evaluation_checkpoint=origin['checkpoint'])
        save(ART / 'full50000_audit.json', {'selection_is_frozen': True, 'no_reselection': True,
             'train_seed': 4817, 'generation_seeds': [2817, 3817], 'rows_by_arm': audits,
             'complete': all(len(rows) == 2 for rows in audits.values()),
             'physical_matched_reference': True, 'require_full_set': True})
    audits_complete = bool(audits) and all(len(rows) == 2 for rows in audits.values())
    save(ART / 'queue_complete.json', {'status': 'COMPLETE' if complete_matrix and audits_complete else 'PARTIAL_DEADLINE',
                                     'complete_final_matrix': complete_matrix, 'complete_full50000_audit': audits_complete,
                                     'finished_unix': time.time(), 'selected': decision.get('selected')})
    save(ART / 'queue_active.json', {'stage': 'COMPLETE', 'controller_pid': os.getpid()})
    print('OVERNIGHT SEARCH COMPLETE', flush=True)


if __name__ == '__main__':
    main()
