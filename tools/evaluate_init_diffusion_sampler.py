"""Separate inference-only sampler audit after the completed overnight search.

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
OUTPUT = ROOT / 'logs/init_diffusion_sampler_inference_20261012'
MANIFEST = ROOT / 'docs/experiments/init_diffusion_frozen_map_search_20261011/confirmation4096_files.json'
TRAIN_SEEDS = (4817, 5817)
GEN_SEEDS = (2817, 3817)
OPTIONS = {'n10_eps005': (10, .05), 'n20_eps005': (20, .05),
           'n40_eps005': (40, .05), 'n40_eps0001': (40, 1e-4), 'n80_eps0001': (80, 1e-4)}
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


class InferenceSamplerHookCore:
    """Dependency-light test-only logic; the factory adds Lightning Callback."""

    def __init__(self, seed, steps, eps, expected_updates, audit_path, defer_gc=False, expected_batch_size=128):
        require(type(steps) is int and steps > 0, 'Sampling steps must be a positive integer')
        require(type(seed) is int and 0 < float(eps) <= .05, 'Invalid seed/epsilon')
        require(type(expected_updates) is int and expected_updates > 0, 'Invalid actual EMA update budget')
        self.seed, self.steps, self.eps = seed, steps, float(eps)
        self.expected_updates, self.audit_path = expected_updates, Path(audit_path)
        require(type(expected_batch_size) is int and expected_batch_size > 0, 'Invalid native evaluation batch size')
        self.expected_batch_size = expected_batch_size
        self.defer_gc = defer_gc
        self.record = {'seed_reset_hook': 'on_test_start, after checkpoint restore, before test epoch/first batch',
                       'requested_steps': steps, 'requested_eps': self.eps, 'GEN_seed': seed,
                       'actual_sampling_steps': [], 'sample_calls': 0, 'training_executed': False,
                       'completed_test_batches': 0, 'runtime_policy': RUNTIME_POLICY if defer_gc else 'CPU fixture'}
        self.original_sample = None

    def on_fit_start(self, *args):
        raise ValueError('Sampler override is inference-only; fit is forbidden')

    def on_train_start(self, *args):
        raise ValueError('Sampler override is inference-only; training is forbidden')

    def on_validation_start(self, *args):
        raise ValueError('Use explicit native TEST, not training/validation')

    def on_test_start(self, trainer, pl_module):
        require(str(trainer.state.fn) in ('test', 'TrainerFn.TESTING'), 'Hook is only valid in TEST')
        require(not pl_module.training and trainer.world_size == 1, 'Require eval mode and single-device TEST')
        require(trainer.precision == '32-true', 'Require native float32 TEST')
        require(trainer.datamodule.test_batch_size == self.expected_batch_size, 'Native evaluation batch split changed')
        decoder = pl_module.encoder.init_decoder
        flow = decoder.G1
        require(decoder.use_ema and decoder.ema is not None, 'Restored generator EMA is required')
        require(decoder.ema.num_updates == self.expected_updates, 'Restored EMA budget differs from trained endpoint')
        require(decoder.sampling_steps == self.steps, 'Checkpoint restoration unexpectedly changed sampling steps')
        require(not flow.use_sde, 'This audit is for supervised ODE inference only')
        self.loss_before = copy.deepcopy(decoder.component_loss)
        self.record.update(loaded_sampling_steps=decoder.sampling_steps, loaded_t_eps=float(flow.t_eps),
                           EMA_num_updates=decoder.ema.num_updates, use_ema=decoder.use_ema,
                           actual_evaluation_batch_size=trainer.datamodule.test_batch_size)
        # A plain scalar runtime attribute, absent from state_dict/extra_state.
        # This executes only after native strict load and cannot alter fit loss.
        flow.t_eps = self.eps
        runtime_seed(self.seed)
        self.record.update(actual_seed_after_reset=runtime_seed_value(),
                           effective_steps=decoder.sampling_steps, effective_eps=float(flow.t_eps))
        self.original_sample = flow.sample
        self.sample_was_instance_attribute = 'sample' in flow.__dict__
        self.old_sample_attribute = flow.__dict__.get('sample')

        def observed_sample(*args, **kwargs):
            steps = kwargs.get('steps', args[2] if len(args) > 2 else 20)
            require(steps == self.steps and flow.t_eps == self.eps, 'Effective sampler differs from planned option')
            require(not flow.training, 'Sampling hook cannot run a training forward')
            self.record['sample_calls'] += 1
            if steps not in self.record['actual_sampling_steps']:
                self.record['actual_sampling_steps'].append(steps)
            return self.original_sample(*args, **kwargs)

        flow.sample = observed_sample
        pl_module.video_dir = self.audit_path.parent / 'videos'
        pl_module.video_dir.mkdir(parents=True, exist_ok=True)
        if self.defer_gc:
            require(not gc.isenabled() and os.environ.get(GC_ENV) == '1', 'Deferred child GC guard was not active')
            self.record['runtime_loader_source_hashes'] = runtime_loader_sources()
            self.record['loader_cache_at_test_start'] = runtime_loader_stats()
        save(self.audit_path, self.record)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        require(batch_idx == self.record['completed_test_batches'], 'Unexpected TEST batch ordering/repetition')
        self.record['completed_test_batches'] += 1

    def on_test_end(self, trainer, pl_module):
        require(self.record['sample_calls'] > 0, 'No actual generation was observed')
        require(pl_module.encoder.init_decoder.component_loss == self.loss_before, 'Training loss config changed')
        require(pl_module.encoder.init_decoder.ema.num_updates == self.expected_updates, 'EMA updated during TEST')
        flow = pl_module.encoder.init_decoder.G1
        if self.sample_was_instance_attribute:
            flow.sample = self.old_sample_attribute
        else:
            del flow.sample
        flow.t_eps = self.record['loaded_t_eps']
        self.record['training_loss_config_unchanged'] = True
        self.record['t_eps_restored_after_test'] = float(flow.t_eps)
        self.record['finished'] = True
        if self.defer_gc:
            self.record['loader_cache_at_test_end'] = runtime_loader_stats()
        save(self.audit_path, self.record)


def make_inference_hook(**kwargs):
    import lightning as L
    class Hook(InferenceSamplerHookCore, L.Callback):
        pass
    return Hook(**kwargs)


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
        '_target_': 'tools.evaluate_init_diffusion_sampler.make_inference_hook',
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
                and len(grid['rows']) == len(expected), 'Frozen sampler lacks the full20 paired cells')
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
                    'actual_training_updates_in_this_job': 0})
    # Verify all four default20 repeats before any changed sampler is run.
    # A failed parity check stops execution and blocks interpretation/selection.
    cells.sort(key=lambda cell: (cell['option'] != 'n20_eps005', cell['TRAIN_seed'], cell['GEN_seed'], list(OPTIONS).index(cell['option'])))
    return {'stage': args.stage, 'auxiliary_inference_cohort': True, 'main_selection_unchanged': True,
            'selected_model': decision['selected'], 'main_decision_sha256': file_hash(Path(args.main_art) / 'final_selection.json'),
            'driver_sha256': file_hash(__file__), 'native_runner_sha256': file_hash(ROOT / 'src/run.py'),
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
    if cell['option'] == 'n20_eps005' and cell['scenes'] == 4096:
        result['default20_main_endpoint_comparison'] = compare_main_default20(report['agent_metrics'], cell['main_endpoint_metrics'])
        save(folder / 'default20_main_endpoint_comparison.json', result['default20_main_endpoint_comparison'])
        require(result['default20_main_endpoint_comparison']['within_absolute_tolerance'],
                'Default20 differs from main endpoint beyond1e-6 native units; stop sampler interpretation and inspect RNG/restore')
    save(folder / 'cell_summary.json', result)
    print('DONE_INFERENCE_ONLY ' + str(folder), flush=True)
    return result


def validate_native_report(report, audit, cell):
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
    if cell['option'] == 'n20_eps005' and cell['scenes'] == 4096:
        require(row.get('default20_main_endpoint_comparison') == compare_main_default20(report['agent_metrics'], cell['main_endpoint_metrics']),
                'Default20 main endpoint comparison changed')
        require(row['default20_main_endpoint_comparison']['within_absolute_tolerance'],
                'Default20 failed required native parity tolerance; no inference option interpretation permitted')
    if require_execution:
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
