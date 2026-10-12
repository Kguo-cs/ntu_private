"""Separate hybrid position-denominator inference cohort; existing decisions remain immutable.

Planning imports no torch and reads no checkpoint payload. --execute runs one
isolated native src.run TEST process per cell; it never fits or selects a model.
Full50k requires an explicit frozen sampler decision from the complete grid.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011'
OUTPUT = ROOT / 'logs/init_diffusion_hybrid_sampler_cohort_20261012'
MANIFEST = ROOT / 'docs/experiments/init_diffusion_frozen_map_search_20261011/confirmation4096_files.json'
TRAIN_SEEDS = (4817, 5817)
GEN_SEEDS = (2817, 3817)
OPTIONS = {'n40_eps005': (40, .05), 'hybrid_pos0001': (40, .05)}
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd',
           'width_jsd', 'speed_jsd', 'collision_rate')
GC_ENV = 'SIM_SAMPLER_INFERENCE_DEFER_CYCLIC_GC'
PARITY_ABSOLUTE_TOLERANCE = 1e-6
RUNTIME_POLICY = {
    'isolated_child_per_cell': True, 'automatic_cyclic_gc': 'disabled before native imports',
    'explicit_collection_all_generations': 'disabled until process exit',
    'freeze': 'not used', 'generation_0_collection': 'disabled; actual training collect(0) SIGSEGV observed',
    'scene_loader': 'same public cached pickle_module as recovered main training',
    'environment_guard': GC_ENV + '=1', 'successful_exit': 'flush artifacts/stdout/stderr, then os._exit(0)',
    'failure_exit': 'nonzero; partial artifacts never count as a complete cell',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def files_hash(files):
    return hashlib.sha256(('\n'.join(files) + '\n').encode()).hexdigest()


def runtime_seed(seed):
    import lightning as L
    L.seed_everything(seed, workers=True)


def runtime_seed_value():
    import torch
    return torch.initial_seed()


def runtime_loader_sources():
    paths = (ROOT / 'tools/runtime_cached_torch_loader_sweep.py', ROOT / 'tools/torch_pickle_class_cache.py')
    return {str(path): file_hash(path) for path in paths}


def runtime_loader_stats():
    from tools.torch_pickle_class_cache import cache_stats
    return copy.deepcopy(cache_stats())


def defer_runtime_gc():
    """Child-only guard; no automatic or explicit GC of any generation."""
    require(os.environ.get(GC_ENV) == '1', 'Isolated child GC guard environment is required')
    gc.disable()
    gc.collect = lambda *unused_args, **unused_kwargs: 0


# Reuse the immutable original native hook/factory, with identical restoration and seed reset.
from tools.evaluate_init_diffusion_sampler import InferenceSamplerHookCore, make_inference_hook


def build_eval_config(source, cell, folder, manifest_path):
    """Pure builder: preserve model conditions; clear only startup initializers."""
    cfg = copy.deepcopy(source)
    cfg.update(action='test', ckpt_path=cell['checkpoint'], seed=cell['TRAIN_seed'], logger=False)
    cfg['float32_matmul_precision'] = 'medium'
    cfg['paths'].update(output_dir=str(folder), log_dir=str(folder))
    model = cfg['model']['model_config']
    model['initial_map_checkpoint'] = None
    model.update(sd_reference_mode='matched', sd_require_full_set=cell['scenes'] == 50000,
                 sd_expected_scenes=50000)
    init = model['decoder']['init_diffusion']
    require(init['use_ema'] is True, 'Selected source must use EMA')
    init.update(normalizer_path=None, sampling_steps=cell['steps'])
    cfg['data'].update(test_batch_size=cell['evaluation_batch_size'], num_workers=0, pin_memory=False,
                       persistent_workers=False, scenario_dreamer_eval_set=str(manifest_path))
    cfg['trainer'].update(accelerator='gpu', devices=1, precision='32-true',
                          limit_test_batches=1., logger=False, enable_checkpointing=False)
    cfg['callbacks'] = {'sampler_audit': {
        '_target_': 'tools.hybrid_position_sampler_hook.make_hybrid_hook',
        'hybrid_position': cell['hybrid_position'],
        'seed': cell['GEN_seed'], 'steps': cell['steps'], 'eps': cell['eps'],
        'expected_updates': cell['trained_updates'], 'audit_path': str(folder / 'sampler_audit.json'),
        'expected_batch_size': cell['evaluation_batch_size'],
        'defer_gc': True}}
    require(cfg['action'] == 'test', 'Inference driver may not train')
    return cfg


def verify_main(main_art):
    """Reuse final matrix validation and prior materializer CPU weight evidence."""
    main_art = Path(main_art)
    queue = read(main_art / 'queue_complete.json')
    require(queue.get('status') == 'COMPLETE' and queue.get('complete_final_matrix') is True
            and queue.get('complete_full50000_audit') is True, 'Main search/full50k is not actually complete')
    from tools import materialize_overnight_init_diffusion_best as materializer
    from tools import runtime_completion
    from omegaconf import OmegaConf
    decision, frozen, search = (read(main_art / name) for name in
                                ('final_selection.json', 'frozen_long_protocol.json', 'search_protocol.json'))
    materializer.validate(decision, frozen, search['arms'])
    proof = read(main_art / 'native_best_materialization.json')
    require(proof.get('selected') == decision['selected'] and proof.get('Hydra_composition_verified') is True,
            'Require validated actual native materialization evidence after final selection')
    require(proof['decision']['sha256'] == file_hash(main_art / 'final_selection.json'), 'Materialization decision changed')
    checkpoint_audits = {v['checkpoint']['path']: v for v in proof['source_audits']}
    sources = {}
    for seed in TRAIN_SEEDS:
        rows = [r for r in decision['rows'] if r['variant'] == decision['selected'] and r['seed'] == seed]
        require(len(rows) == 2 and {r['eval_seed'] for r in rows} == set(GEN_SEEDS), 'Missing selected paired GEN cells')
        row = rows[0]
        path = Path(row['checkpoint']).resolve()
        audit = checkpoint_audits.get(str(path))
        require(audit is not None and file_hash(path) == audit['checkpoint']['sha256'], 'Selected checkpoint identity changed')
        cfg_path = path.parent / 'config.yaml'
        require(file_hash(cfg_path) == audit['source_config']['sha256'], 'Selected actual model config changed')
        ex = read(path.parent.parent / 'execution.json')
        actual, artifacts = runtime_completion.verify_completion(path.parent.parent,
            path.parent.parent.with_suffix('.log'), ex['completion_spec'])
        require(ex['status'] == 'success' and ex['returncode'] == 0 and artifacts == ex['artifacts'],
                'Selected training stage evidence changed')
        actual_endpoint = [r for r in actual if r['budget'] == row['budget']]
        require(sorted(actual_endpoint, key=lambda r: r['eval_seed']) == sorted(rows, key=lambda r: r['eval_seed']),
                'Selected decision rows differ from completed actual endpoint')
        for name, wanted in ex['source_hashes'].items():
            if name.startswith('src/smart/') or name in ('configs/model/smart.yaml', 'src/run.py'):
                require(file_hash(ROOT / name) == wanted, 'Frozen model/data source changed: ' + name)
        sources[seed] = {'row': row, 'rows': rows, 'source_config': str(cfg_path), 'source_config_sha256': file_hash(cfg_path),
                         'checkpoint': str(path), 'checkpoint_sha256': audit['checkpoint']['sha256'],
                         'model_source_sha256': ex['source_hashes'],
                         'cfg': materializer.plain(OmegaConf.load(cfg_path))}
    return decision, sources


def make_plan(args):
    from tools.select_init_diffusion_hybrid_sampler import validate_protocol, sha
    protocol_path = Path(args.main_art) / 'hybrid_sampler_selection_protocol.json'
    protocol = validate_protocol(protocol_path)
    decision, sources = verify_main(args.main_art)
    if args.stage == 'grid':
        selected = read(args.manifest)
        files = selected.get('files') if isinstance(selected, dict) else selected
        require(isinstance(files, list) and len(files) == len(set(files)) == 4096
                and all(Path(n).name == n for n in files), 'Require exact unique4096 native filenames')
        require(all(files_hash(files) == v['row']['eval_manifest_sha256'] for v in sources.values()),
                'Inference grid must reuse main matched4096 scene order')
        options, train_seeds = list(OPTIONS), TRAIN_SEEDS
        freeze_evidence = None
    else:
        require(args.frozen_sampler is not None, 'Full50k requires an explicit frozen sampler JSON')
        frozen = read(args.frozen_sampler)
        require(frozen.get('selection_is_frozen') is True and frozen.get('selected_option') in OPTIONS,
                'Invalid frozen inference-only sampler')
        grid_path = Path(frozen['grid_summary'])
        grid = read(grid_path)
        require(file_hash(grid_path) == frozen['grid_summary_sha256'] and grid.get('complete') is True,
                'Frozen sampler must cite an actual complete inference grid')
        require(grid['plan']['selected_model'] == decision['selected'] and
                grid['plan']['main_decision_sha256'] == file_hash(Path(args.main_art) / 'final_selection.json')
                and grid['plan']['driver_sha256'] == file_hash(__file__), 'Frozen sampler source/model cohort changed')
        expected = {(s, g, o) for s in TRAIN_SEEDS for g in GEN_SEEDS for o in OPTIONS}
        require({(v['TRAIN_seed'], v['GEN_seed'], v['option']) for v in grid['rows']} == expected
                and len(grid['rows']) == len(expected), 'Frozen sampler lacks the full8 fresh-native paired cells')
        require(len(grid['plan']['cells']) == len(expected) and grid['plan']['stage'] == 'grid',
                'Frozen sampler does not cite the planned inference grid')
        expected_manifest = sources[TRAIN_SEEDS[0]]['row']['eval_manifest_sha256']
        require(len(grid['plan']['files']) == len(set(grid['plan']['files'])) == 4096
                and files_hash(grid['plan']['files']) == expected_manifest, 'Frozen grid scene cohort changed')
        for row, cell in zip(grid['rows'], grid['plan']['cells']):
            source = sources[cell['TRAIN_seed']]
            require(cell['checkpoint_sha256'] == source['checkpoint_sha256']
                    and cell['source_config_sha256'] == source['source_config_sha256'], 'Frozen grid checkpoint/config changed')
            require((cell['steps'], cell['eps']) == OPTIONS[cell['option']] and cell['scenes'] == 4096
                    and cell['eval_files_sha256'] == expected_manifest, 'Frozen grid sampler/input option changed')
            validate_completed_cell(row, cell, Path(row['metrics_path']).parent, grid['plan'], require_execution=True)
        from tools.select_init_diffusion_hybrid_sampler import verify_actual_grid, rank_actual_rows
        actual_grid, proof = verify_actual_grid(grid_path, protocol_path, main_art=Path(args.main_art))
        require(frozen.get('protocol_sha256') == sha(protocol_path) and frozen.get('criterion_proof') == proof
                and frozen['selected_option'] == proof['selected_option'], 'Frozen native inference choice differs from actual preregistered criterion')
        options, train_seeds = [frozen['selected_option']], tuple(args.full_train_seeds)
        raw = Path(sources[TRAIN_SEEDS[0]]['cfg']['data']['test_raw_dir'])
        files = sorted(p.name for p in raw.iterdir() if p.is_file() and p.suffix in ('.pt', '.pkl'))
        require(len(files) == len(set(files)) == 50000, 'Full50k native input membership/count changed')
        freeze_evidence = {'path': str(Path(args.frozen_sampler).resolve()), 'sha256': file_hash(args.frozen_sampler)}
    cells = []
    for seed in train_seeds:
        src = sources[seed]
        raw = Path(src['cfg']['data']['test_raw_dir'])
        batch_size = src['cfg']['data']['test_batch_size']
        require(type(batch_size) is int and batch_size > 0, 'Main endpoint evaluation batch size is missing/invalid')
        require(all((raw / n).is_file() for n in files), 'Missing exact inference input scene')
        for gen in GEN_SEEDS:
            endpoint = next(r for r in src['rows'] if r['eval_seed'] == gen)
            for option in options:
                steps, eps = OPTIONS[option]
                cells.append({'TRAIN_seed': seed, 'GEN_seed': gen, 'option': option, 'steps': steps,
                    'eps': eps, 'scenes': len(files), 'eval_files_sha256': files_hash(files),
                    'evaluation_batch_size': batch_size, 'main_endpoint_scenes': 4096,
                    'checkpoint': src['checkpoint'], 'checkpoint_sha256': src['checkpoint_sha256'],
                    'source_config': src['source_config'], 'source_config_sha256': src['source_config_sha256'],
                    'main_endpoint_metrics': {name: endpoint[name] for name in METRICS},
                    'trained_updates': src['row']['budget'], 'source_model': decision['selected'],
                    'actual_training_updates_in_this_job': 0, 'hybrid_position': option == 'hybrid_pos0001'})
    # Measure four NEW N40/base.05 anchors before any changed sampler.
    # No old failed-parity result is reused or relabelled as a passing baseline.
    cells.sort(key=lambda cell: (cell['option'] != 'n40_eps005', cell['TRAIN_seed'], cell['GEN_seed'], list(OPTIONS).index(cell['option'])))
    return {'stage': args.stage, 'auxiliary_inference_cohort': True, 'main_selection_unchanged': True,
            'selected_model': decision['selected'], 'main_decision_sha256': file_hash(Path(args.main_art) / 'final_selection.json'),
            'driver_sha256': file_hash(__file__), 'immutable_native_core_sha256': file_hash(ROOT / 'tools/evaluate_init_diffusion_sampler.py'),
            'selection_protocol_sha256': sha(protocol_path), 'fresh_native_baselines': True, 'separate_hybrid_cohort': True, 'native16_selection_unchanged': True,
            'hybrid_hook_sha256': file_hash(ROOT / 'tools/hybrid_position_sampler_hook.py'),
            'original_main_replay_parity_claimed': False, 'original_sampler_status': 'FAILED_DEFAULT20_PARITY',
            'native_runner_sha256': file_hash(ROOT / 'src/run.py'),
            'model_source_sha256': sources[TRAIN_SEEDS[0]]['model_source_sha256'],
            'frozen_sampler': freeze_evidence, 'train_seeds': list(train_seeds), 'generation_seeds': list(GEN_SEEDS),
            'runtime_policy': RUNTIME_POLICY,
            'runtime_loader_source_hashes': runtime_loader_sources(),
            'independent_new_training_runs': 0, 'reused_training_seeds': list(train_seeds),
            'GEN_is_not_independent_TRAIN_N': True, 'files': files, 'cells': cells}


def execute_cell(plan, index, folder):
    # All collection remains off. The cached public pickle_module removes the
    # measured per-file local-class growth; each cell exits independently.
    defer_runtime_gc()
    require(plan['runtime_loader_source_hashes'] == runtime_loader_sources(), 'Runtime loader source changed')
    require(plan['hybrid_hook_sha256'] == file_hash(ROOT / 'tools/hybrid_position_sampler_hook.py'), 'Hybrid hook source changed')
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    from omegaconf import OmegaConf
    from src.run import run
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    cell = plan['cells'][index]
    require(file_hash(cell['checkpoint']) == cell['checkpoint_sha256'], 'Checkpoint changed before native TEST')
    require(file_hash(cell['source_config']) == cell['source_config_sha256'], 'Actual source config changed')
    require(not folder.exists(), 'Cell folder already exists without verified completion; inspect manually')
    folder.mkdir(parents=True)
    manifest = folder / 'eval_files.pkl'
    with manifest.open('wb') as handle:
        pickle.dump({'files': plan['files']}, handle)
    source = OmegaConf.to_container(OmegaConf.load(cell['source_config']), resolve=True)
    cfg = OmegaConf.create(build_eval_config(source, cell, folder, manifest))
    OmegaConf.save(cfg, folder / 'config.yaml')
    started = time.time()
    run(cfg)
    report_path = folder / 'sd_agent_metrics.json'
    report, audit = read(report_path), read(folder / 'sampler_audit.json')
    validate_native_report(report, audit, cell)
    result = {**cell, 'evaluation_only': True, 'evaluation_weights': 'ema',
              'driver_sha256': plan['driver_sha256'], 'native_runner_sha256': plan['native_runner_sha256'],
              'main_decision_sha256': plan['main_decision_sha256'], 'auxiliary_inference_cohort': True,
              'runtime_policy': RUNTIME_POLICY,
              'runtime_loader_source_hashes': plan['runtime_loader_source_hashes'],
              'started_unix': started, 'eval_seconds': time.time() - started,
              'metrics_path': str(report_path), 'report_sha256': file_hash(report_path),
              'sampler_audit_sha256': file_hash(folder / 'sampler_audit.json'),
              **report['agent_metrics'], 'num_generated_vehicles': report['num_generated_vehicles'],
              'num_gt_vehicles': report['num_gt_vehicles'],
              'generated_onroad_fraction': report['feature_counts_generated'][1] / report['num_generated_vehicles'],
              'gt_onroad_fraction': report['feature_counts_gt'][1] / report['num_gt_vehicles']}
    result['baseline_role'] = 'fresh_native_anchor' if cell['option'] == 'n40_eps005' else 'native_comparison_to_fresh_anchor'
    result['main_endpoint_parity_claimed'] = False
    save(folder / 'cell_summary.json', result)
    print('DONE_INFERENCE_ONLY ' + str(folder), flush=True)
    return result


def validate_native_report(report, audit, cell):
    require(audit.get('hybrid_position') is cell['hybrid_position']
            and audit.get('prediction_velocity_wrapper_restored') is True
            and audit.get('GT_postprocessing_added') is False
            and audit.get('nonposition_velocity_preserved') is True and audit.get('x0_preserved') is True
            and audit.get('prediction_velocity_calls') == audit.get('sample_calls', 0) * 40
            and audit.get('lowtime_velocity_calls') == audit.get('sample_calls')
            and audit.get('hybrid_active_calls') == (audit.get('sample_calls') if cell['hybrid_position'] else 0)
            and audit.get('unchanged_EMA_updates') == cell['trained_updates'], 'Actual hybrid TEST-only override not certified')
    require(audit.get('finished') is True and audit.get('sample_calls', 0) > 0
            and audit.get('actual_sampling_steps') == [cell['steps']]
            and audit.get('effective_steps') == cell['steps'] and audit.get('effective_eps') == cell['eps']
            and audit.get('actual_seed_after_reset') == cell['GEN_seed']
            and audit.get('training_executed') is False and audit.get('training_loss_config_unchanged') is True
            and audit.get('EMA_num_updates') == cell['trained_updates']
            and audit.get('completed_test_batches') == math.ceil(cell['scenes'] / cell['evaluation_batch_size'])
            and audit.get('actual_evaluation_batch_size') == cell['evaluation_batch_size']
            and audit.get('runtime_policy') == RUNTIME_POLICY
            and audit.get('runtime_loader_source_hashes') == runtime_loader_sources(), 'Effective sampler/test hook was not certified')
    cache = audit.get('loader_cache_at_test_end', {})
    require(0 < cache.get('classes_created', 0) <= 2 and '_load' in cache.get('cached_loader_kinds', [])
            and cache.get('classes_created', 0) + cache.get('class_reuses', 0) >= cell['scenes'],
            'Public cached scene loader was not actually used for native TEST')
    require(report['reference_mode'] == 'matched' and report['map_source'] == 'reference'
            and report['num_samples'] == report['num_gt_samples'] == cell['scenes']
            and report['unverified_generation_timestep_count'] == 0, 'Native physical metric scope mismatch')
    require(report['require_full_set'] is (cell['scenes'] == 50000), 'Full-set requirement differs from inference stage')
    if cell['scenes'] == 50000:
        require(report['full_membership'] is True, 'Full50k evaluation was incomplete')
    require(report['decoder']['sampling_steps'] == cell['steps'] and report['decoder']['use_ema'] is True
            and report['decoder']['ema_num_updates'] == cell['trained_updates']
            and report['runtime']['torch_initial_seed'] == cell['GEN_seed']
            and report['runtime']['trainer_precision'] == '32-true'
            and report['runtime']['float32_matmul_precision'] == 'medium', 'Native EMA/sampler/seed/precision mismatch')
    require(all(k in report['agent_metrics'] and math.isfinite(report['agent_metrics'][k])
                and report['agent_metrics'][k] >= 0 for k in METRICS), 'Missing/invalid native metrics')
    require(report['num_generated_vehicles'] > 0 and report['num_gt_vehicles'] > 0, 'Empty native vehicle population')


def compare_main_default20(actual, main_endpoint):
    """Numerical reproduction audit, not sampler ranking or new selection."""
    return {'same_physical_scope_and_units': True, 'exact_seven_metrics_identical': all(actual[k] == main_endpoint[k] for k in METRICS),
            'native_absolute_tolerance': PARITY_ABSOLUTE_TOLERANCE,
            'within_absolute_tolerance': all(math.isclose(actual[k], main_endpoint[k], rel_tol=0., abs_tol=PARITY_ABSOLUTE_TOLERANCE) for k in METRICS),
            'delta_native_scaled_units': {k: actual[k] - main_endpoint[k] for k in METRICS},
            'main_endpoint': {k: main_endpoint[k] for k in METRICS},
            'note': 'Actual reproduction check; mismatches require inspection before attributing option effects. No GPU bit identity assumed.'}


def validate_completed_cell(row, cell, folder, plan, *, require_execution):
    """Parent CPU certification, including completion/exit and immutable artifacts."""
    folder = Path(folder)
    require(all(row[k] == v for k, v in cell.items()) and row['driver_sha256'] == plan['driver_sha256']
            and row['native_runner_sha256'] == plan['native_runner_sha256']
            and row['main_decision_sha256'] == plan['main_decision_sha256']
            and row['evaluation_only'] is True and row['evaluation_weights'] == 'ema'
            and row.get('runtime_loader_source_hashes') == plan.get('runtime_loader_source_hashes') == runtime_loader_sources()
            and row['auxiliary_inference_cohort'] is True and row['runtime_policy'] == RUNTIME_POLICY,
            'Existing inference cell metadata changed')
    require(Path(row['metrics_path']).resolve() == (folder / 'sd_agent_metrics.json').resolve(), 'Report path is not cell-local')
    paths = {'report_sha256': folder / 'sd_agent_metrics.json',
             'sampler_audit_sha256': folder / 'sampler_audit.json'}
    require(all(file_hash(path) == row[key] for key, path in paths.items()), 'Inference artifacts changed')
    report, audit = read(paths['report_sha256']), read(paths['sampler_audit_sha256'])
    validate_native_report(report, audit, cell)
    require(all(row[k] == report['agent_metrics'][k] for k in METRICS), 'Summary does not match actual native metrics')
    require(row.get('main_endpoint_parity_claimed') is False, 'New cohort cannot claim old endpoint replay parity')
    require(row['num_generated_vehicles'] == report['num_generated_vehicles'] and row['num_gt_vehicles'] == report['num_gt_vehicles']
            and row['generated_onroad_fraction'] == report['feature_counts_generated'][1] / report['num_generated_vehicles']
            and row['gt_onroad_fraction'] == report['feature_counts_gt'][1] / report['num_gt_vehicles'], 'Actual population counts differ from report')
    if require_execution:
        require(read(folder / 'cell_summary.json') == row, 'Actual hashed cell summary differs from aggregate row')
        execution = read(folder.parent / (folder.name + '.execution.json'))
        log = folder.parent / (folder.name + '.log')
        require(execution.get('status') == 'success' and execution.get('returncode') == 0
                and execution.get('driver_sha256') == plan['driver_sha256']
                and execution.get('runtime_loader_source_hashes') == plan['runtime_loader_source_hashes']
                and execution.get('cell') == cell, 'Cell has no independently successful child execution')
        require(execution['artifacts'] == {str(p): file_hash(p) for p in
                (folder / 'cell_summary.json', *paths.values())}, 'Certified cell artifacts changed')
        require(file_hash(log) == execution['log_sha256'] and
                ('DONE_INFERENCE_ONLY ' + str(folder)) in log.read_text().splitlines(), 'No exact native completion marker')
    return row


def cell_folder(output, cell):
    return Path(output) / f"train{cell['TRAIN_seed']}_gen{cell['GEN_seed']}_{cell['option']}"


def wait_isolated_child(command, log, execution, execution_path, deadline):
    """Always stop our child group on timeout or parent interruption."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(os.environ, **{GC_ENV: '1'}), start_new_session=True)
    execution['child_pid'] = process.pid
    save(execution_path, execution)
    try:
        return process.wait(timeout=max(1, deadline - time.time()))
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                process.wait(timeout=10)
        execution['returncode'] = process.returncode
        execution['child_terminated_after_parent_interruption_or_timeout'] = True
        raise


def parent_sigterm(signum, frame):
    raise KeyboardInterrupt('Parent received SIGTERM; stop isolated child and preserve partial evidence')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-art', type=Path, default=ART)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('grid', 'full'), default='grid')
    parser.add_argument('--frozen-sampler', type=Path)
    parser.add_argument('--full-train-seeds', type=int, choices=TRAIN_SEEDS, nargs='+', default=[4817])
    parser.add_argument('--deadline-utc', default='2026-10-12T01:35:00+00:00')
    parser.add_argument('--execute', action='store_true', help='Otherwise validate/plan only; never wait/autorun')
    parser.add_argument('--cell', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('main_art', 'manifest', 'output', 'frozen_sampler'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    require(len(set(args.full_train_seeds)) == len(args.full_train_seeds), 'Duplicate full TRAIN seeds')
    if args.cell is not None:
        require(args.execute, 'Isolated cell requires explicit --execute')
        defer_runtime_gc()
    plan = make_plan(args)
    output = args.output / args.stage
    if args.cell is not None:
        require(args.execute and 0 <= args.cell < len(plan['cells']), 'Invalid isolated inference cell')
        execute_cell(plan, args.cell, cell_folder(output, plan['cells'][args.cell]))
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').is_file():
        require(read(output / 'plan.json') == plan, 'Existing inference plan differs; use a new output directory')
    else:
        save(output / 'plan.json', plan)
    if not args.execute:
        print(f'PLANNED_ONLY {len(plan["cells"])} inference cells; no torch/checkpoint loading/GPU')
        return plan
    signal.signal(signal.SIGTERM, parent_sigterm)
    deadline = datetime.fromisoformat(args.deadline_utc).timestamp()
    rows = []
    for index, cell in enumerate(plan['cells']):
        folder = cell_folder(output, cell)
        execution_path = output / (folder.name + '.execution.json')
        if folder.exists() or execution_path.exists():
            require((folder / 'cell_summary.json').is_file(), 'Incomplete prior child requires manual inspection/new output')
            row = validate_completed_cell(read(folder / 'cell_summary.json'), cell, folder, plan, require_execution=True)
        else:
            require(time.time() < deadline, 'Sampler deadline reached; incomplete grid cannot be selected')
            command = [sys.executable, str(Path(__file__).resolve()), '--main-art', str(args.main_art),
                       '--manifest', str(args.manifest), '--output', str(args.output), '--stage', args.stage,
                       '--deadline-utc', args.deadline_utc, '--execute', '--cell', str(index)]
            if args.frozen_sampler:
                command += ['--frozen-sampler', str(args.frozen_sampler)]
            command += ['--full-train-seeds', *map(str, args.full_train_seeds)]
            execution = {'command': command, 'cell': cell, 'status': 'running', 'started_unix': time.time(),
                         'driver_sha256': plan['driver_sha256'], 'runtime_policy': RUNTIME_POLICY,
                         'runtime_loader_source_hashes': plan['runtime_loader_source_hashes'],
                         'child_environment': {GC_ENV: '1'}}
            save(execution_path, execution)
            log_path = output / (folder.name + '.log')
            try:
                with log_path.open('w') as log:
                    returncode = wait_isolated_child(command, log, execution, execution_path, deadline)
                execution.update(returncode=returncode, ended_unix=time.time())
                require(returncode == 0, f'Isolated inference child failed: {returncode}')
                require(('DONE_INFERENCE_ONLY ' + str(folder)) in log_path.read_text().splitlines(),
                        'Child returned0 without exact complete native marker')
                row = validate_completed_cell(read(folder / 'cell_summary.json'), cell, folder, plan, require_execution=False)
                execution.update(status='success', log_sha256=file_hash(log_path), artifacts={str(p): file_hash(p) for p in
                    (folder / 'cell_summary.json', folder / 'sd_agent_metrics.json', folder / 'sampler_audit.json')})
                save(execution_path, execution)
                validate_completed_cell(row, cell, folder, plan, require_execution=True)
            except BaseException as error:
                execution.update(status='failed', ended_unix=time.time(), error=repr(error))
                save(execution_path, execution)
                raise
        rows.append(row)
        save(output / 'summary.json', {'complete': len(rows) == len(plan['cells']), 'plan': plan, 'rows': rows,
                                      'main_selection_unchanged': True, 'new_selection_made': False})
    print('COMPLETE_INFERENCE_ONLY ' + str(output / 'summary.json'))
    return rows


if __name__ == '__main__':
    main()
