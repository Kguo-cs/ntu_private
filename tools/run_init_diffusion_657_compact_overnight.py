"""Isolated postgraph compact-model sweep with verified fresh657 controls.

The parent uses only the standard library. Each cell runs the existing verified
training/evaluation stage in a separate process; legacy ART/RUNS are changed
only inside that child. Partial stages never become observations or warm starts.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_init_diffusion_657_iteration import METRICS, read, save, sha, score, now

ART = ROOT / 'output/reports/init_diffusion_657_compact_20261013'
MAIN_ART = ROOT / 'output/reports/init_diffusion_657_edges_20261013'
MAIN_RUNS = ROOT / 'logs/init_diffusion_657_edges_20261013'
RUNS = ROOT / 'logs/init_diffusion_657_compact_20261013'
PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
PREFIX = 'model.model_config.decoder.init_diffusion.'
SOURCE_FILES = (
    'src/smart/diffusion/denoiser.py', 'src/smart/diffusion/initial_diffusion.py',
    'src/smart/diffusion/scale_flow.py', 'src/smart/diffusion/diffusion_utils.py',
    'src/smart/model/smart.py', 'src/smart/model/smart_gail.py',
    'src/smart/tokens/token_processor.py', 'src/smart/metrics/metric_core.py',
    'configs/run.yaml', 'configs/model/smart.yaml', 'configs/model/smart_fast.yaml',
    'configs/experiment/init_bc.yaml', 'configs/experiment/init_diffusion_lane_conditioned.yaml',
    'configs/experiment/init_diffusion_lane_conditioned_mixed_loss.yaml',
    'configs/optimizer/default.yaml', 'tools/overnight_init_diffusion_sweep.py',
    'tools/scratch_component_loss_sweep.py', 'tools/runtime_cached_torch_loader_sweep.py',
    'tools/run_init_diffusion_657_iteration.py', 'tools/run_init_diffusion_657_edges_overnight.py',
)
STOP_REQUESTED = False
ARCHITECTURE_RNG_SCOPE = {
    'shared': ['TRAIN seed', 'seeded TRAIN data order', 'exposure', 'GEN seed', 'scene membership'],
    'fresh_training_seed_location': 'seed_everything before datamodule/model construction',
    'post_G1_fresh_training_rng_reset': False,
    'common_per_step_training_noise_verified': False,
    'limitation': 'Width/depth/MLP/sincos changes initialization RNG consumption. Training noise uses device RNG, but stepwise common noise is not recorded or guaranteed. Both TRAIN confirmations are required.',
    'source': 'tools/scratch_component_loss_sweep.py: seed before construction; on_train_start RNG restoration applies only to resumed checkpoints',
}


def aware_deadline(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Deadline must include a timezone')
    return result.astimezone(timezone.utc).timestamp()


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (ProcessLookupError, FileNotFoundError):
        return False


def gpu_pids():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=True)
    return sorted({int(line.strip()) for line in result.stdout.splitlines()
                   if line.strip().isdigit() and alive(int(line.strip()))})


def source_hashes(plan):
    paths = tuple(dict.fromkeys(SOURCE_FILES + tuple(plan.get('source_files', ()))))
    result = {path: sha(ROOT / path) for path in paths}
    result[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    return result


def normalize_plan(plan):
    """Fail before launch on contradictory sampling or parameter budgets."""
    plan = copy.deepcopy(plan)
    aware_deadline(plan.get('deadline', '2026-10-13T10:00:00+08:00'))
    control = plan.get('control', 'control657')
    variants = plan['variants']
    order = plan.get('order', list(variants))
    if len(order) != len(set(order)) or set(order) != set(variants):
        raise ValueError('Order must contain every variant exactly once')
    if control not in variants:
        raise ValueError('A control variant must be supplied')
    plan['order'] = [control] + [name for name in order if name != control]
    plan['control'] = control
    plan.setdefault('deadline', '2026-10-13T10:00:00+08:00')
    plan.setdefault('medium_candidates', 3)
    plan.setdefault('medium_fallback_candidates', 0)
    plan.setdefault('allow_combo',False)
    plan.setdefault('mandatory_medium',[])
    if any(name not in variants or name == control for name in plan['mandatory_medium']):
        raise ValueError('mandatory_medium must name noncontrol registered variants')
    if len(plan['mandatory_medium']) > plan['medium_candidates']:
        raise ValueError('mandatory_medium exceeds medium_candidates')
    if plan['allow_combo'] and plan.get('combo_anchor') not in variants:
        raise ValueError('allow_combo requires a registered combo_anchor')
    plan.setdefault('long_steps', 10240)
    plan.setdefault('long_seeds', [4817, 5817])
    plan.setdefault('generation_seeds', [2817, 3817])
    plan.setdefault('deadline_grace_seconds', 120)
    plan.setdefault('max_stage_attempts', 2)
    plan.setdefault('tie_relative_R', .02)
    plan.setdefault('contemporaneous_control_phases', ['screen'])
    if plan['long_steps'] != 10240 or plan['long_seeds'] != [4817, 5817]:
        raise ValueError('Registered long confirmation is10240steps, TRAIN4817/5817')
    if plan['generation_seeds'] != [2817, 3817]:
        raise ValueError('Registered GEN seeds are2817/3817')
    if not 1 <= plan['max_stage_attempts'] <= 3:
        raise ValueError('max_stage_attempts must be1..3')
    if not 0 <= plan['deadline_grace_seconds'] <= 300:
        raise ValueError('Deadline grace must be0..300seconds')
    for name, variant in variants.items():
        config = variant.setdefault('__config', {})
        if config.get(PREFIX + 'sampling_steps', 20) != 20:
            raise ValueError(f'{name}: sampling_steps must be20')
        config[PREFIX + 'sampling_steps'] = 20
        hidden = config.get(PREFIX + 'denoiser_hidden_dim', 256)
        depth = config.get(PREFIX + 'denoiser_num_layers', 3)
        metadata = plan.get('variant_metadata', {}).get(name, {})
        count = metadata.get('num_parameters')
        if plan.get('max_num_parameters') is not None:
            if count is None or count > plan['max_num_parameters']:
                raise ValueError(f'{name}: parameter count absent or above budget')
        elif hidden > 256 or depth > 3:
            raise ValueError(f'{name}: expected at most256hidden and3layers unless an explicit parameter budget is supplied')
        metadata.setdefault('denoiser_hidden_dim', hidden)
        metadata.setdefault('denoiser_num_layers', depth)
        plan.setdefault('variant_metadata', {})[name] = metadata
    return plan


def prefer_small(results, plan):
    """Within2% of the best R, prefer the smallest registered network."""
    if not results:
        return None
    best = min(row['R'] for row in results)
    close = [row for row in results if row['R'] <= best * (1 + plan['tie_relative_R'])]
    def key(row):
        metadata = plan['variant_metadata'].get(row['variant'], {})
        return (metadata.get('num_parameters', math.inf),
                metadata.get('denoiser_num_layers', 3), row['R'])
    return min(close, key=key)


def estimate_seconds(spec, timings, metadata):
    """Conservative estimate learned from complete cells, with startup/eval."""
    training = not spec.get('evaluation')
    compatible = [entry for entry in timings if entry['training'] == training]
    if training:
        # Production three-layer runs have approximately0.4s/step. Start with
        #0.55s until measured, then retain20% safety margin and90s startup.
        per_step = max([entry['train_seconds'] / entry['steps'] for entry in compatible] or [.55])
        depth_scale = max(1., metadata.get('denoiser_num_layers', 3) / 3.)
        train_seconds = spec['steps'] * per_step * depth_scale
    else:
        train_seconds = 0.
    eval_per_scene = max([entry['eval_seconds'] / entry['evaluated_scenes']
                          for entry in timings if entry['evaluated_scenes'] > 0] or [.006])
    count = len(spec['gens']) * (1 if not training else len(spec['checkpoints']) + 1)
    return 90 + 1.2 * (train_seconds + eval_per_scene * spec['scenes'] * count)


def load_verified_main_data():
    """Only outer-verified native main-phase observations, with frozen sources."""
    protocol = read(MAIN_ART/'protocol.json')
    for path,digest in protocol['sources'].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError('Main-phase frozen source changed: '+path)
    rows,provenance = [],[]
    for marker_path in sorted(MAIN_RUNS.glob('*/edge_verified_completion.json')):
        folder=marker_path.parent
        marker=read(marker_path)
        contract=read(folder/'edge_contract.json')
        legacy=read(folder/'verified_completion.json')
        if marker['status']!='VERIFIED_COMPLETE' or marker['returncode']!=0:
            continue
        if legacy['status']!='VERIFIED_COMPLETE' or legacy['returncode']!=0:
            raise RuntimeError('Main native execution was not successful')
        if contract['sources'] != protocol['sources']:
            raise RuntimeError('Main cell source differs from final frozen source: '+str(folder))
        if marker['summary_sha256']!=sha(folder/'summary.json') or legacy['summary_sha256']!=sha(folder/'summary.json'):
            raise RuntimeError('Main verified summary changed')
        if marker['contract_sha256']!=sha(folder/'edge_contract.json'):
            raise RuntimeError('Main stage contract changed')
        values=read(folder/'summary.json')
        if validate_observations(values,contract['spec'])!=marker['metrics_reports']:
            raise RuntimeError('Main native metric reports changed')
        rows.extend(values)
        provenance.append({'folder':str(folder),'summary_sha256':sha(folder/'summary.json'),
            'verified_marker_sha256':sha(marker_path),'contract_sha256':sha(folder/'edge_contract.json'),
            'observations':len(values),'runtime_native_sources':protocol['sources']})
    controls=[row for row in rows if row['variant']=='control657']
    if not controls:
        raise RuntimeError('No verified fresh main-phase657 control')
    return rows,controls,provenance


def load_historical_controls():
    # Compatibility name retained for the isolated base driver constructor;
    # this compact runner does not load any historical-control files.
    _,controls,provenance=load_verified_main_data()
    return controls,provenance


def validate_observations(rows, spec):
    for row in rows:
        count = row.get('generator_num_parameters')
        expected_count = spec.get('expected_num_parameters')
        limit = spec.get('max_num_parameters')
        if expected_count is not None and count != expected_count:
            raise RuntimeError('Actual generator parameter count differs from measured plan')
        if limit is not None and (count is None or count > limit):
            raise RuntimeError('Actual generator parameter count exceeds budget or is missing')
        report = read(row['metrics_path'])
        for key,value in spec.get('expected_denoiser_options',{}).items():
            if report['decoder']['denoiser_options'].get(key) != value:
                raise RuntimeError('Actual denoiser setting differs: '+key)
        if report['decoder']['sampling_steps'] != 20:
            raise RuntimeError('Actual native evaluation did not use N20')
        if report['num_samples'] != spec['scenes'] or report['num_gt_samples'] != spec['scenes']:
            raise RuntimeError('GT/generated scene scope differs')
        if row['evaluation_weights'] != 'ema' or row['seed'] != spec['seed']:
            raise RuntimeError('Seed or EMA scope differs')
        if report['runtime']['trainer_precision'] != '32-true':
            raise RuntimeError('Training/evaluation precision differs')
        if spec['scenes'] == 50000 and (not report['require_full_set'] or not report['full_membership']):
            raise RuntimeError('Full evaluation membership was not enforced')
        for key in METRICS:
            if not math.isfinite(row[key]) or row[key] != report['agent_metrics'][key]:
                raise RuntimeError('Summary metrics differ from native scaled metric report: '+key)
        for key,value in spec.get('expected_decoder',{}).items():
            if report['decoder'].get(key) != value:
                raise RuntimeError('Actual decoder/embedding setting differs: '+key)
        expected_recipe=spec.get('expected_component_recipe',{})
        actual_recipe=report['decoder'].get('component_loss',{})
        for key,value in expected_recipe.items():
            if key=='components':
                for name,fields in value.items():
                    if actual_recipe.get('components',{}).get(name) != fields:
                        raise RuntimeError('Actual component loss recipe differs: '+name)
            elif actual_recipe.get(key) != value:
                raise RuntimeError('Actual component loss recipe differs: '+key)
        for key,value in spec.get('expected_native_scope',{}).items():
            if report.get(key) != value:
                raise RuntimeError('Actual metric scope differs: '+key)
        for key,value in spec.get('expected_row_scope',{}).items():
            if row.get(key) != value:
                raise RuntimeError('Actual training scope differs: '+key)
        if spec.get('strict_native_metadata'):
            if report['runtime']['float32_matmul_precision'] != 'medium':
                raise RuntimeError('Actual matmul precision differs')
            if report['runtime']['torch_initial_seed'] != row['eval_seed']:
                raise RuntimeError('Actual generation seed differs')
            if report['unverified_generation_timestep_count'] != 0:
                raise RuntimeError('Native generation timestep membership is unverified')
            if report['decoder']['ema_num_updates'] != row['budget']:
                raise RuntimeError('EMA update count differs from verified exposure')
    if spec.get('strict_native_metadata'):
        budgets=[spec['steps']] if spec.get('evaluation') else spec['checkpoints']+[spec['steps']]
        expected={(spec['variant'],budget,gen) for budget in budgets for gen in spec['gens']}
        actual={(row['variant'],row['budget'],row['eval_seed']) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise RuntimeError('Native stage rows differ from all registered budgets/GEN seeds')
    return [{'path': row['metrics_path'], 'sha256': sha(row['metrics_path'])} for row in rows]


def worker(spec_path):
    """Legacy stage globals are isolated in this process only."""
    spec = read(spec_path)
    from tools import run_init_diffusion_657_iteration as legacy
    legacy.ART, legacy.RUNS = ART, RUNS
    rows = legacy.stage(spec['label'], spec['variants'], spec['steps'], spec['scenes'],
                        seed=spec['seed'], checkpoints=tuple(spec['checkpoints']),
                        gens=tuple(spec['gens']), evaluation=spec.get('evaluation'))
    validate_observations(rows, spec)
    print('EDGE_STAGE_WORKER_SUCCESS', flush=True)


class DeadlineReached(Exception):
    pass


class StageFailed(RuntimeError):
    def __init__(self,message,details=None):
        super().__init__(message)
        self.details = details or {}


class ResearchDriver:
    def __init__(self, plan, plan_path):
        self.plan, self.plan_path = plan, plan_path
        self.deadline = aware_deadline(plan['deadline'])
        self.sources = source_hashes(plan)
        self.controls, control_provenance = load_historical_controls()
        self.repeat_controls = []
        self.anchor_rows = []
        self.variants = copy.deepcopy(plan['variants'])
        self.variant_metadata = copy.deepcopy(plan['variant_metadata'])
        self.rows, self.timings = [], []
        self.failures = read(ART/'failed_stages.json') if (ART/'failed_stages.json').exists() else []
        self.pid = os.getpid()
        protocol = {'created_utc': now(), 'plan_path': str(plan_path), 'plan_sha256': sha(plan_path),
                    'effective_plan': plan, 'sources': self.sources,
                    'control_provenance': control_provenance,
                    'architecture_rng_scope':ARCHITECTURE_RNG_SCOPE,
                    'sampling_steps': 20, 'initialization': 'freshG1;fixedAIRLmap;TRAIN-onlycalibration',
                    'warning': 'Short and10epoch curves do not prove128epoch convergence',
                    'selection': 'Only verified fresh main/registered compact657; no historical fallback. Medium must beat matched main initializer and657 with gates/slope; smallest network within2%R. Long bothTRAIN must beat657 with gates, plus initializer if exact long scope exists; full bothTRAIN separately reported.'}
        existing = ART / 'protocol.json'
        if existing.is_file():
            old = read(existing)
            if old['plan_sha256'] != protocol['plan_sha256'] or old['sources'] != self.sources:
                raise RuntimeError('Frozen plan or source changed; create a separate registered run')
        else:
            save(existing, protocol)
        save(ART / 'variants.json', self.variants)
        save(ART / 'recipe.json', read(ROOT / 'tmp/control657_paired_20261012/recipe.json'))
        save(ART / 'verified_source_control_provenance.json', control_provenance)
        if (ART / 'duration_observations.json').exists():
            self.timings = read(ART / 'duration_observations.json')

    def status(self, status, **details):
        value = {'status': status, 'driver_pid': self.pid,
             'updated_utc': now(), 'deadline': self.plan['deadline'],
             'remaining_seconds': max(0, self.deadline - time.time()), **details}
        save(ART / 'queue_status.json',value)
        save(ART / 'rootdriver_status.json',value)
        save(ART / 'status.json',value)

    def check_frozen(self):
        if source_hashes(self.plan) != self.sources or sha(self.plan_path) != read(ART / 'protocol.json')['plan_sha256']:
            raise RuntimeError('Source or plan changed during execution')

    def wait_serial(self, wait_pid=None):
        while (wait_pid and alive(wait_pid)) or gpu_pids():
            if STOP_REQUESTED or time.time() >= self.deadline:
                raise DeadlineReached('Deadline while waiting for serialGPU')
            self.status('WAITING_FOR_SERIAL_GPU', wait_pid=wait_pid, gpu_pids=gpu_pids())
            time.sleep(5)

    def stage(self, label, variant, steps, scenes, seed=4817, checkpoints=(), gens=(2817,), evaluation=None):
        self.check_frozen()
        spec = {'label': label, 'variants': [] if evaluation else [variant], 'variant': variant,
                'steps': steps, 'scenes': scenes, 'seed': seed,
                'checkpoints': list(checkpoints), 'gens': list(gens), 'evaluation': evaluation,
                'expected_num_parameters':self.variant_metadata.get(variant,{}).get('num_parameters'),
                'max_num_parameters':self.plan.get('max_num_parameters'),
                'expected_denoiser_options': {key[len(PREFIX):]:value
                    for key,value in self.variants.get(variant, {}).get('__config', {}).items()
                    if key in {PREFIX+name for name in (
                        'a2a_radius','a2a_max_num_neighbors','map2agent_radius',
                        'map2agent_max_num_neighbors','denoiser_hidden_dim',
                        'denoiser_num_layers','time_conditioning')}}}
        if hasattr(self,'native_contract'):
            spec.update(self.native_contract(variant))
        # A restarted driver may reuse only a complete, hash-verified same-source
        # cell. Incomplete attempts receive fresh output directories and freshG1.
        label_base, attempt = label, 1
        while True:
            folder = RUNS / spec['label']
            contract_path = folder / 'edge_contract.json'
            marker_path = folder / 'edge_verified_completion.json'
            contract = {'spec': spec, 'plan_sha256': sha(self.plan_path), 'sources': self.sources}
            if marker_path.exists():
                marker = read(marker_path)
                if marker['status'] != 'VERIFIED_COMPLETE' or marker['returncode'] != 0:
                    raise RuntimeError('Completed stage execution was not successful')
                if read(contract_path) != contract or marker['contract_sha256'] != sha(contract_path) or marker['summary_sha256'] != sha(folder / 'summary.json'):
                    raise RuntimeError('Completed stage contract or summary changed')
                rows = read(folder / 'summary.json')
                if validate_observations(rows, spec) != marker['metrics_reports']:
                    raise RuntimeError('Completed metrics report changed')
                self.rows += rows
                save(ART / 'observations.json', self.rows)
                return rows
            if not (folder / 'protocol.json').exists() and not contract_path.exists():
                break
            save(folder / 'abandoned_partial.json', {'abandoned_utc': now(),
                 'reason': 'No outer verified completion; never promoted or resumed', 'next_attempt': attempt + 1})
            attempt += 1
            spec['label'] = f'{label_base}_attempt{attempt}'
        if attempt > self.plan['max_stage_attempts']:
            raise StageFailed('Maximum fresh attempts reached: '+label_base,
                {'label':label_base,'attempts':attempt-1,'partial_stages_usable':False})
        estimated = estimate_seconds(spec, self.timings, self.variant_metadata.get(variant, {}))
        if STOP_REQUESTED or time.time() + estimated > self.deadline:
            self.status('DEADLINE_NO_NEW_STAGE', next_task=spec, estimated_seconds=estimated)
            raise DeadlineReached('Insufficient time for a complete ' + label)
        self.wait_serial()
        if time.time() + estimated > self.deadline:
            raise DeadlineReached('Deadline after waiting forGPU')
        folder.mkdir(parents=True, exist_ok=True)
        save(contract_path, contract)
        save(ART / f"{spec['label']}.spec.json", spec)
        self.status('STARTING_STAGE', active_stage=spec['label'], next_task=spec, estimated_seconds=estimated)
        started = time.monotonic()
        command = [PYTHON, str(Path(__file__).resolve()), '--worker', str(ART / f"{spec['label']}.spec.json")]
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
        with (folder / 'driver_console.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            self.status('RUNNING', active_stage=spec['label'], child_pid=child.pid,
                        next_task=spec, estimated_seconds=estimated)
            while child.poll() is None:
                if time.time() > self.deadline + self.plan['deadline_grace_seconds']:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    self.status('DEADLINE_INCOMPLETE_STAGE', active_stage=spec['label'], partial_stage=str(folder))
                    raise DeadlineReached('Stage exceeded the bounded deadline grace; partial ignored')
                if int(time.monotonic()-started) % 30 < 2:
                    save(ART / 'rootdriver_status.json', {
                        'status':'RUNNING','driver_pid':self.pid,'child_pid':child.pid,
                        'active_stage':spec['label'],'deadline':self.plan['deadline'],
                        'updated_utc':now(),'elapsed_seconds':time.monotonic()-started,
                        'remaining_seconds':max(0,self.deadline-time.time()),'next_task':spec})
                time.sleep(2)
        if child.returncode != 0 or 'EDGE_STAGE_WORKER_SUCCESS' not in (folder / 'driver_console.log').read_text():
            execution_path = folder / 'execution.json'
            inner_returncode = read(execution_path).get('returncode') if execution_path.exists() else None
            save(folder / 'edge_execution.json', {'status':'FAILED','returncode':child.returncode,
                'native_returncode':inner_returncode,'elapsed_seconds':time.monotonic()-started,
                'finished_utc':now(),'metrics_usable':False})
            if (inner_returncode is not None and inner_returncode != 0 or child.returncode in (-11,-7,-9)) and attempt < self.plan['max_stage_attempts']:
                self.status('RETRYING_FRESH_STAGE',failed_stage=spec['label'],
                    native_returncode=inner_returncode, next_task=spec)
                return self.stage(label_base,variant,steps,scenes,seed=seed,
                    checkpoints=checkpoints,gens=gens,evaluation=evaluation)
            raise StageFailed('Stage worker failed: ' + spec['label'],
                {'stage':spec['label'],'variant':variant,'returncode':child.returncode,
                 'native_returncode':inner_returncode,'attempts':attempt,'folder':str(folder),
                 'metrics_usable':False})
        legacy_marker = read(folder / 'verified_completion.json')
        rows = read(folder / 'summary.json')
        if legacy_marker['status'] != 'VERIFIED_COMPLETE' or legacy_marker['returncode'] != 0 or legacy_marker['summary_sha256'] != sha(folder / 'summary.json'):
            raise RuntimeError('Legacy stage was not verified complete')
        self.check_frozen()
        reports = validate_observations(rows, spec)
        record = {'status': 'VERIFIED_COMPLETE', 'finished_utc': now(), 'returncode': child.returncode,
                  'summary_sha256': sha(folder / 'summary.json'), 'metrics_reports': reports,
                  'contract_sha256': sha(contract_path), 'elapsed_seconds': time.monotonic() - started}
        save(marker_path, record)
        duration = {'label': spec['label'], 'training': not bool(evaluation), 'steps': steps,
                    'elapsed_seconds': record['elapsed_seconds'],
                    'train_seconds': max((r['train_seconds'] for r in rows), default=0),
                    'eval_seconds': sum(r['eval_seconds'] for r in rows),
                    'evaluated_scenes': len(rows) * scenes}
        self.timings.append(duration)
        save(ART / 'duration_observations.json', self.timings)
        self.rows += rows
        save(ART / 'observations.json', self.rows)
        self.status('STAGE_COMPLETE', active_stage=spec['label'], observations=len(self.rows))
        return rows

    def comparison(self,values):
        keys = lambda row:(row['seed'],row['eval_seed'],row['seen_training_examples'],row['eval_manifest_sha256'])
        required = {keys(row) for row in values}
        historical_ok = required.issubset({keys(row) for row in self.controls})
        contemporary_ok = required.issubset({keys(row) for row in self.repeat_controls})
        historical = score(values,self.controls) if historical_ok else None
        if contemporary_ok:
            result = score(values,self.repeat_controls)
            result['denominator'] = 'contemporaneous_same_phase657'
            result['historical_657_comparison'] = historical
            return result
        if historical is not None:
            return dict(historical,denominator='historical_exposure_matched657',
                historical_657_comparison=historical)
        raise RuntimeError('No complete matched657 denominator forcomparison')

    def curve(self, name, rows, budgets):
        curves = []
        for budget in budgets:
            values = [r for r in rows if r['variant'] == name and r['budget'] == budget]
            result = {'budget': budget, **self.comparison(values)}
            compatible = [r for r in self.repeat_controls if r['budget'] == budget]
            if compatible:
                result['contemporaneous_control'] = score(values, compatible)
            anchor = [r for r in self.anchor_rows if r['budget'] == budget]
            if anchor:
                result['graph_anchor_comparison'] = score(values,anchor)
            curves.append(result)
        return {'variant': name, 'curve': curves, 'R': curves[-1]['R'],
                'variant_metadata': self.variant_metadata.get(name, {})}

    def adaptive_combo(self,screen):
        if not self.plan['allow_combo']:
            return None
        anchor_name = self.plan['combo_anchor']
        anchor_variant = self.variants[anchor_name]
        anchor_rows = [r for r in screen if r['variant'] == anchor_name and r['budget'] == 768]
        groups = {'a2a':[], 'map2agent':[]}
        graph_keys = {PREFIX+key for key in ('a2a_radius','a2a_max_num_neighbors','map2agent_radius','map2agent_max_num_neighbors')}
        for name in self.plan['order']:
            if name in (self.plan['control'],anchor_name):
                continue
            delta = {key:value for key,value in self.variants[name].get('__config',{}).items()
                     if key in graph_keys and value != anchor_variant.get('__config',{}).get(key)}
            family = self.variant_metadata.get(name,{}).get('family')
            if family is None:
                families = {'a2a' if key.startswith(PREFIX+'a2a_') else 'map2agent' for key in delta}
                family = next(iter(families)) if len(families)==1 else None
            if family not in groups or not delta:
                continue
            candidate_rows = [r for r in screen if r['variant']==name and r['budget']==768]
            if not candidate_rows or not anchor_rows:
                continue
            result = score(candidate_rows,anchor_rows)
            if result['gates_pass'] and result['R'] < 1:
                groups[family].append({'variant':name,'R':result['R'],'delta':delta,'anchor_comparison':result})
        if any(not rows for rows in groups.values()):
            save(ART/'adaptive_registration.json', {'status':'NO_COMBO',
                'reason':'Both single graph families must improve anchor and pass allgates',
                'screen_evidence_sha256':sha(ART/'screen_base.json'),'groups':groups})
            return None
        picked = {family:min(rows,key=lambda r:r['R']) for family,rows in groups.items()}
        variant = copy.deepcopy(anchor_variant)
        for row in picked.values():
            variant['__config'].update(row['delta'])
        name = 'adaptive_graph_combo'
        metadata = copy.deepcopy(self.variant_metadata[anchor_name])
        metadata.update(family='adaptive_graph_combo',parents=[r['variant'] for r in picked.values()],
                        parameter_count_source='Graph radius/neighbour options add no parameters; same measured anchor architecture')
        registration = {'status':'REGISTERED','name':name,'variant':variant,'metadata':metadata,
            'parents':picked,'plan_sha256':sha(self.plan_path),'sources':self.sources,
            'screen_evidence_sha256':sha(ART/'screen_base.json')}
        path = ART/'adaptive_registration.json'
        if path.exists() and read(path) != registration:
            raise RuntimeError('Adaptive registration changed')
        save(path,registration)
        self.variants[name],self.variant_metadata[name] = variant,metadata
        save(ART/'variants.json',self.variants)
        return name

    def record_failure(self,phase,name,error):
        self.failures.append({'phase':phase,'variant':name,'recorded_utc':now(),
            'error':str(error),'details':error.details,'partial_metrics_usable':False})
        save(ART/'failed_stages.json',self.failures)

    def run(self):
        screen = []
        for name in self.plan['order']:
            try:
                values = self.stage('screen_' + name, name, 768, 1024, checkpoints=(384,))
            except StageFailed as error:
                self.record_failure('screen',name,error)
                if name in (self.plan['control'],self.plan.get('combo_anchor')):
                    raise
                continue
            screen += values
            if name == self.plan['control']:
                self.repeat_controls += values
            if name == self.plan.get('combo_anchor'):
                self.anchor_rows += values
            rankings = [self.curve(n, screen, (384, 768)) for n in self.plan['order']
                        if any(row['variant'] == n for row in screen)]
            save(ART / 'screen.json', sorted(rankings, key=lambda r: r['R']))
        save(ART/'screen_base.json',sorted(rankings,key=lambda r:r['R']))
        combo = self.adaptive_combo(screen)
        if combo:
            try:
                screen += self.stage('screen_'+combo,combo,768,1024,checkpoints=(384,))
                rankings.append(self.curve(combo,screen,(384,768)))
                save(ART/'screen.json',sorted(rankings,key=lambda r:r['R']))
            except StageFailed as error:
                self.record_failure('screen',combo,error)
        candidates = sorted([r for r in rankings if r['variant'] != self.plan['control']
                             and r['curve'][-1]['gates_pass']], key=lambda r: r['R'])
        limit = self.plan['medium_candidates'] + self.plan['medium_fallback_candidates']
        selected = list(self.plan['mandatory_medium'])
        selected += [r['variant'] for r in candidates if r['variant'] not in selected][:max(0,limit-len(selected))]
        save(ART / 'medium_selection.json', {'selected': selected, 'created_utc': now(),
                                           'evidence_sha256': sha(ART / 'screen.json')})
        medium = []
        results = []
        if 'medium' in self.plan['contemporaneous_control_phases']:
            self.repeat_controls += self.stage('medium_' + self.plan['control'], self.plan['control'],
                                               4096, 4096, checkpoints=(2048,), gens=(2817, 3817))
        for index, name in enumerate(selected):
            if len(results) >= self.plan['medium_candidates'] and any(r['eligible'] for r in results):
                break
            try:
                values = self.stage('medium_' + name, name, 4096, 4096,
                                    checkpoints=(2048,), gens=(2817, 3817))
            except StageFailed as error:
                self.record_failure('medium',name,error)
                continue
            medium += values
            if name == self.plan.get('combo_anchor'):
                self.anchor_rows += values
            result = self.curve(name, medium, (2048, 4096))
            last, early = result['curve'][-1], result['curve'][0]
            result['eligible'] = last['all_TRAIN_gates_pass'] and result['R'] < 1 and result['R'] <= early['R'] * 1.15
            results.append(result)
            save(ART / 'medium.json', results)
        selection_plan = dict(self.plan,variant_metadata=self.variant_metadata)
        winner_row = prefer_small([r for r in results if r['eligible']], selection_plan)
        if winner_row is None:
            self.status('COMPLETE_NO_PROMOTION', reason='No medium arm meetsR,gates,slope')
            return
        winner = winner_row['variant']
        save(ART / 'long_selection.json', {'winner': winner, 'selected_utc': now(),
                                         'evidence_sha256': sha(ART / 'medium.json')})
        long = []
        for seed in self.plan['long_seeds']:
            if 'long' in self.plan['contemporaneous_control_phases']:
                self.repeat_controls += self.stage(f'long_control{seed}', self.plan['control'],
                    10240, 4096, seed=seed, checkpoints=(3413,6826), gens=(2817,3817))
            long += self.stage(f'long{seed}', winner, 10240, 4096, seed=seed,
                               checkpoints=(3413,6826), gens=(2817,3817))
            result = self.curve(winner, long, (3413,6826,10240))
            save(ART / 'long.json', result)
            if not result['curve'][-1]['all_TRAIN_gates_pass'] or result['R'] >= 1:
                self.status('COMPLETE_NO_PROMOTION', reason='Long confirmation fails', winner=winner)
                return
        # Freeze the winner using4096scene confirmation; full50k is confirmation,
        # never used to select a different candidate after seeing the answer.
        full = []
        for seed in self.plan['long_seeds']:
            # Missing full baseline is evaluated first so an isolated candidate
            # full report cannot be mistaken for an exposure-matched comparison.
            fresh_full = 'full' in self.plan['contemporaneous_control_phases']
            denominator = self.repeat_controls if fresh_full else self.controls
            matched = [r for r in denominator if r['seed'] == seed and r['budget'] == 10240
                       and r['num_samples'] == 50000 and r['eval_seed'] in (2817,3817)]
            if {r['eval_seed'] for r in matched} != {2817,3817}:
                if fresh_full:
                    trained_controls = [r for r in self.repeat_controls if r['seed']==seed
                        and r['budget']==10240 and r['num_samples']==4096]
                    if not trained_controls:
                        raise RuntimeError('Fresh fullcontrol requires verified same-seed longcontrol')
                    checkpoint = Path(trained_controls[-1]['checkpoint'])
                else:
                    checkpoint = ROOT / f'logs/control657_paired_20261012/train{seed}/control657/last.ckpt'
                values = self.stage(f'full_control{seed}', self.plan['control'], 10240,
                    50000, seed=seed, gens=(2817,3817), evaluation={self.plan['control']:str(checkpoint)})
                if fresh_full:
                    self.repeat_controls += values
                else:
                    self.controls += values
            long_row = next(r for r in reversed(long) if r['seed'] == seed and r['budget'] == 10240)
            full += self.stage(f'full{seed}', winner, 10240, 50000, seed=seed,
                               gens=(2817,3817), evaluation={winner:long_row['checkpoint']})
            result = self.comparison(full)
            save(ART / 'full.json', {'winner': winner, **result})
        self.status('COMPLETE', winner=winner, beats657=result['all_TRAIN_gates_pass'] and result['R'] < 1)


def on_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def pair_key(row):
    return (row['seed'],row['eval_seed'],row['seen_training_examples'],row['eval_manifest_sha256'])


def matched_subset(values,baselines):
    required={pair_key(row) for row in values}
    if not values:
        return None
    selected=[row for row in baselines if pair_key(row) in required]
    return selected if required.issubset({pair_key(row) for row in selected}) else None


def default_policy():
    """Only predeclared options; root must save/review this policy beforelaunch."""
    options={
        'compact192L4':({'denoiser_hidden_dim':192,'denoiser_num_layers':4},5837962),
        'compact256L2':({'denoiser_hidden_dim':256,'denoiser_num_layers':2},5844490),
        'compact192L3':({'denoiser_hidden_dim':192,'denoiser_num_layers':3},4576138),
        'sincos_context':({'ego_context_heading_encoding':'sincos'},8084746),
        'mlp_edges':({'edge_embedding_type':'mlp'},7356682),
    }
    option_pool={name:{'__config':{PREFIX+key:value for key,value in delta.items()}}
                 for name,(delta,count) in options.items()}
    option_pool['lane_agentmax05']={'__config':{
        PREFIX+'map_condition_types':'lanes_and_edges',PREFIX+'collision_reduction':'agent_max'},
        'collision':{'weight':.5}}
    pool_metadata={name:{'num_parameters':count,
        'denoiser_hidden_dim':delta.get('denoiser_hidden_dim',256),
        'denoiser_num_layers':delta.get('denoiser_num_layers',3)}
        for name,(delta,count) in options.items()}
    pool_metadata['lane_agentmax05']={'num_parameters':8083978,'denoiser_hidden_dim':256,
        'denoiser_num_layers':3,'untested_hypothesis':True,
        'parameter_count_source':'Map token selection/collision aggregation andweight addzero network parameters'}
    return {'deadline':'2026-10-13T10:00:00+08:00','control':'control657',
        'candidate_order':list(options),'candidate_pool':option_pool,
        'candidate_pool_metadata':pool_metadata,
        'candidate_options':{name:{'__config':{PREFIX+key:value for key,value in delta.items()}}
                             for name,(delta,count) in options.items()},
        'candidate_metadata':{name:{'num_parameters':count,
            'denoiser_hidden_dim':delta.get('denoiser_hidden_dim',256),
            'denoiser_num_layers':delta.get('denoiser_num_layers',3),
            'parameter_count_source':'CPU architecture probes; graph radius/neighbour options do not change parameters'}
            for name,(delta,count) in options.items()},
        'max_num_parameters':8100000,'medium_candidates':2,'deadline_grace_seconds':120,
        'max_stage_attempts':2,'tie_relative_R':.02,
        'ensure_missing_fresh_controls':True,'ensure_missing_anchor_medium':True,
        'full_evaluation_if_time':True,
        'source_artifact_dir':str(MAIN_ART),'source_run_dir':str(MAIN_RUNS),
        'anchor_policy':'LowestR eligible main4096graph arm; fallback norm_time_every_layer with original graph',
        'architecture_rng_scope':copy.deepcopy(ARCHITECTURE_RNG_SCOPE),
        'warning':'CPU initial losses are not learning-quality estimates; short/10epoch results do not prove128epoch convergence; architecture comparisons match TRAIN/data order but do not guarantee identical stepwise training noise'}


def validate_policy(policy):
    aware_deadline(policy['deadline'])
    allowed=default_policy()['candidate_pool']
    order=policy['candidate_order']
    if len(order)!=len(set(order)) or set(order)!=set(policy['candidate_options']):
        raise ValueError('Candidate order must include every option exactly once')
    for name,option in policy['candidate_options'].items():
        if name not in allowed or option!=allowed[name]:
            raise ValueError('Only the predeclared architecture/embedding options are allowed')
        expected=default_policy()['candidate_pool_metadata'][name]['num_parameters']
        if policy['candidate_metadata'][name]['num_parameters']!=expected:
            raise ValueError('Candidate must use its measured CPU parameter count')
        if expected>policy['max_num_parameters']:
            raise ValueError('Candidate is above parameter budget')
    if not 1<=policy['medium_candidates']<=2:
        raise ValueError('Postgraph medium budget allows top1or2 candidates')
    if policy.get('source_artifact_dir',str(MAIN_ART))!=str(MAIN_ART) or policy.get('source_run_dir',str(MAIN_RUNS))!=str(MAIN_RUNS):
        raise ValueError('Use the registered fresh main-phase evidence')
    return copy.deepcopy(policy)


def choose_graph_anchor(rows,controls,variants):
    """Choose only a graph with verified paired4096 curves and declared gates."""
    candidates=[]
    for name,variant in variants.items():
        config=variant.get('__config',{})
        if config.get(PREFIX+'state_feature_normalization')!='train' or config.get(PREFIX+'time_conditioning')!='every_layer':
            continue
        if config.get(PREFIX+'denoiser_hidden_dim',256)!=256 or config.get(PREFIX+'denoiser_num_layers',3)!=3:
            continue
        curve=[]
        for budget in (2048,4096):
            values=[row for row in rows if row['variant']==name and row['budget']==budget
                    and row['seed']==4817 and row['num_samples']==4096]
            if {row['eval_seed'] for row in values}!={2817,3817}:
                break
            baseline=matched_subset(values,controls)
            if baseline is None:
                break
            curve.append({'budget':budget,**score(values,baseline)})
        if len(curve)==2 and curve[-1]['all_TRAIN_gates_pass'] and curve[-1]['R']<1 and curve[-1]['R']<=curve[0]['R']*1.15:
            candidates.append({'variant':name,'R':curve[-1]['R'],'curve':curve})
    if candidates:
        winner=min(candidates,key=lambda row:row['R'])
        return winner['variant'],{'grade':'paired4096eligible','selected':winner,'eligible_graphs':candidates}
    name='norm_time_every_layer'
    if name not in variants or not any(row['variant']==name for row in rows):
        raise RuntimeError('No verified norm_time anchor or eligible graph')
    return name,{'grade':'fallback_original_graph','reason':'No main4096graph meets gates/R/slope; do not claim graph gain'}


def build_effective_plan(policy,rows,controls,variants):
    name,selection=choose_graph_anchor(rows,controls,variants)
    anchor=copy.deepcopy(variants[name])
    effective_variants={'control657':copy.deepcopy(variants['control657']),name:anchor}
    metadata={key:{'num_parameters':8083978,'denoiser_hidden_dim':256,'denoiser_num_layers':3,
                   'parameter_count_source':'Verified fresh main-phase native initialization'}
              for key in effective_variants}
    for candidate in policy['candidate_order']:
        variant=copy.deepcopy(anchor)
        option=policy['candidate_options'][candidate]
        variant['__config'].update(option['__config'])
        for component,fields in option.items():
            if component!='__config':variant.setdefault(component,{}).update(fields)
        effective_variants[candidate]=variant
        metadata[candidate]=copy.deepcopy(policy['candidate_metadata'][candidate])
    plan={'deadline':policy['deadline'],'control':'control657',
        'order':['control657',name]+policy['candidate_order'],'variants':effective_variants,
        'variant_metadata':metadata,'max_num_parameters':policy['max_num_parameters'],
        'medium_candidates':policy['medium_candidates'],'medium_fallback_candidates':0,
        'allow_combo':False,'long_steps':10240,'long_seeds':[4817,5817],
        'generation_seeds':[2817,3817],'contemporaneous_control_phases':[],
        'deadline_grace_seconds':policy['deadline_grace_seconds'],
        'max_stage_attempts':policy['max_stage_attempts'],'tie_relative_R':policy['tie_relative_R'],
        'compact_candidate_order':policy['candidate_order'],'selected_graph_anchor':name,
        'graph_selection':selection,'source_files':list(read(MAIN_ART/'protocol.json')['sources']),
        'control_policy':'Only hash-verified same-source fresh main-phase657 or registered fresh compact-phase657 at matching exposure/seeds/scenes; no historical fallback',
        'ensure_missing_fresh_controls':policy.get('ensure_missing_fresh_controls',False),
        'ensure_missing_anchor_medium':policy.get('ensure_missing_anchor_medium',False),
        'full_evaluation_if_time':policy.get('full_evaluation_if_time',False)}
    return normalize_plan(plan)


class CompactResearchDriver(ResearchDriver):
    def __init__(self,plan,plan_path,source_rows,source_controls):
        super().__init__(plan,plan_path)
        self.repeat_controls=source_controls
        self.controls=source_controls
        self.frozen_recipe=read(ART/'recipe.json')
        native_recipe=read(source_controls[0]['metrics_path'])['decoder']['component_loss']
        for key,value in self.frozen_recipe.items():
            if key=='components':
                for component,fields in value.items():
                    if native_recipe['components'].get(component) != fields:
                        raise RuntimeError('Base recipe differs from verified source657: '+component)
            elif native_recipe.get(key) != value:
                raise RuntimeError('Base recipe differs from verified source657: '+key)
        if sha(ROOT/'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json') != source_controls[0]['calibration_sha256']:
            raise RuntimeError('Current TRAIN calibration differs from source657')
        save(ART/'frozen_data_contract.json',{'base_recipe_sha256':sha(ART/'recipe.json'),
            'calibration_sha256':source_controls[0]['calibration_sha256'],
            'metric_scale_provenance':'Frozen metric_core source; exact native agent_metrics checked for every row'})
        self.anchor_name=plan['selected_graph_anchor']
        self.anchor_rows=[row for row in source_rows if row['variant']==self.anchor_name]
        if not self.timings and (MAIN_ART/'duration_observations.json').exists():
            self.timings=read(MAIN_ART/'duration_observations.json')
        save(ART/'source_observations.json',{'control_policy':plan['control_policy'],
            'fresh_controls':source_controls,'selected_anchor_rows':self.anchor_rows,
            'main_protocol_sha256':sha(MAIN_ART/'protocol.json')})

    def check_frozen(self):
        super().check_frozen()
        if read(ART/'recipe.json') != self.frozen_recipe:
            raise RuntimeError('Registered base recipe changed during execution')

    def native_contract(self,name):
        baseline=self.repeat_controls[0]
        report=read(baseline['metrics_path'])
        decoder=report['decoder']
        flat={key:copy.deepcopy(value) for key,value in decoder.items()
              if key not in ('component_loss','denoiser_options','ema_num_updates','init_map_ema_num_updates')}
        options=copy.deepcopy(decoder['denoiser_options'])
        variant=self.variants[name]
        for path,value in variant.get('__config',{}).items():
            if path.startswith(PREFIX):
                key=path[len(PREFIX):]
                if key in options:options[key]=value
                elif key in flat:flat[key]=value
                else:raise RuntimeError('Config option lacks native metadata: '+key)
        recipe=copy.deepcopy(read(ART/'recipe.json'))
        for component,fields in variant.items():
            if not component.startswith('__'):
                recipe['components'][component].update(fields)
        return {'strict_native_metadata':True,'expected_decoder':flat,
            'expected_denoiser_options':options,'expected_component_recipe':recipe,
            'expected_native_scope':{key:report[key] for key in (
                'metric_scope','generation_task','map_source','reference_mode','backend_scope')},
            'expected_row_scope':{key:baseline[key] for key in (
                'batch_size','calibration_sha256','map_checkpoint_sha256','initial_map_sha256',
                'map_frozen','initialization_scope','lr','gradient_clip_val')}}

    def curve(self,name,rows,budgets):
        # Compare an anchor only when it covers every TRAIN/GEN/exposure scope.
        curves=[]
        for budget in budgets:
            values=[row for row in rows if row['variant']==name and row['budget']==budget]
            point={'budget':budget,**self.comparison(values)}
            anchor=matched_subset(values,self.anchor_rows)
            if anchor is not None:point['graph_anchor_comparison']=score(values,anchor)
            curves.append(point)
        return {'variant':name,'curve':curves,'R':curves[-1]['R'],
                'variant_metadata':self.variant_metadata.get(name,{})}

    def comparison(self,values):
        baseline=matched_subset(values,self.repeat_controls)
        if baseline is None:
            raise RuntimeError('No verified fresh source657 for this exact scope')
        return {**score(values,baseline),'denominator':'same_source_verified_fresh657_main_or_registered_compact'}

    def describe(self,name,rows,budgets):
        result=self.curve(name,rows,budgets)
        result['architecture_rng_scope']=copy.deepcopy(ARCHITECTURE_RNG_SCOPE)
        for point in result['curve']:
            values=[row for row in rows if row['variant']==name and row['budget']==point['budget']]
            anchor=matched_subset(values,self.anchor_rows)
            if anchor is not None:
                baseline=matched_subset(values,self.repeat_controls)
                point['main_initializer_R']=score(anchor,baseline)['R']
                point['comparison_vs_main_initializer']=score(values,anchor)
            else:
                point['main_initializer_R']=None
                point['comparison_vs_main_initializer']=None
        return result

    def per_seed_promotion(self,name,rows,budget,compare_initializer=True):
        checks={}
        values=[row for row in rows if row['variant']==name and row['budget']==budget]
        for seed in sorted({row['seed'] for row in values}):
            subset=[row for row in values if row['seed']==seed]
            if {row['eval_seed'] for row in subset}!={2817,3817}:
                raise RuntimeError('Promotion requires both registered GEN seeds for every TRAIN seed')
            baseline=matched_subset(subset,self.repeat_controls)
            if baseline is None:raise RuntimeError('Missing fresh657 per-TRAIN promotion denominator')
            measured=score(subset,baseline)
            anchor=matched_subset(subset,self.anchor_rows) if compare_initializer else None
            anchor_R=score(anchor,baseline)['R'] if anchor is not None else None
            threshold=min(1.,anchor_R) if anchor_R is not None else 1.
            checks[str(seed)]={'R':measured['R'],'main_initializer_R':anchor_R,
                'threshold':threshold,'gates_pass':measured['all_TRAIN_gates_pass'],
                'passes':measured['all_TRAIN_gates_pass'] and measured['R']<threshold}
        return {'per_TRAIN':checks,'all_TRAIN_pass':bool(checks) and all(check['passes'] for check in checks.values())}

    def controls_available(self,budgets,scenes,seed,gens):
        return all({row['eval_seed'] for row in self.repeat_controls if row['budget']==budget
            and row['num_samples']==scenes and row['seed']==seed and row['eval_seed'] in gens}==set(gens)
            for budget in budgets)

    def ensure_controls(self,label,budgets,steps,scenes,seed,gens):
        if self.controls_available(budgets,scenes,seed,gens):return True
        if not self.plan['ensure_missing_fresh_controls']:return False
        values=self.stage(label,'control657',steps,scenes,seed=seed,
            checkpoints=tuple(b for b in budgets if b!=steps),gens=gens)
        self.repeat_controls+=values;self.controls=self.repeat_controls
        save(ART/'registered_fresh_controls.json',self.repeat_controls)
        return self.controls_available(budgets,scenes,seed,gens)

    def budgets_available(self,budgets,scenes,seed,gens,require_anchor=True):
        for budget in budgets:
            control=[row for row in self.repeat_controls if row['budget']==budget
                and row['num_samples']==scenes and row['seed']==seed and row['eval_seed'] in gens]
            anchor=[row for row in self.anchor_rows if row['budget']==budget
                and row['num_samples']==scenes and row['seed']==seed and row['eval_seed'] in gens]
            if {row['eval_seed'] for row in control}!=set(gens) or (require_anchor and matched_subset(control,anchor) is None):
                return False
        return True

    def run(self):
        screen=[];rankings=[]
        if not self.budgets_available((384,768),1024,4817,(2817,)):
            self.status('COMPLETE_NO_MATCHED_SOURCE_SCOPE',reason='Source fresh657/graph anchor lacks screen384/768 scope')
            return
        for name in self.plan['compact_candidate_order']:
            try:
                screen+=self.stage('screen_'+name,name,768,1024,checkpoints=(384,))
            except StageFailed as error:
                self.record_failure('screen',name,error)
                continue
            rankings.append(self.describe(name,screen,(384,768)))
            save(ART/'screen.json',sorted(rankings,key=lambda row:row['R']))
        candidates=sorted([row for row in rankings if row['curve'][-1]['gates_pass']],key=lambda row:row['R'])
        selected=[row['variant'] for row in candidates[:self.plan['medium_candidates']]]
        save(ART/'medium_selection.json',{'selected':selected,'created_utc':now(),
            'screen_evidence_sha256':sha(ART/'screen.json') if (ART/'screen.json').exists() else None})
        if not selected:
            self.status('COMPLETE_NO_PROMOTION',reason='No compact screen candidate passes native population/attribute/on-road gates')
            return
        self.ensure_controls('registration_medium_control657',(2048,4096),4096,4096,4817,(2817,3817))
        if not self.budgets_available((2048,4096),4096,4817,(2817,3817)) and self.plan['ensure_missing_anchor_medium']:
            self.anchor_rows+=self.stage('registration_medium_anchor',self.anchor_name,4096,4096,
                checkpoints=(2048,),gens=(2817,3817))
            save(ART/'registered_medium_anchor.json',self.anchor_rows)
        if not self.budgets_available((2048,4096),4096,4817,(2817,3817)):
            self.status('COMPLETE_SCREEN_ONLY',reason='No verified paired main657/anchor4096 scope; no cross-budget forecast')
            return
        medium=[];results=[]
        for name in selected:
            try:
                medium+=self.stage('medium_'+name,name,4096,4096,checkpoints=(2048,),gens=(2817,3817))
            except StageFailed as error:
                self.record_failure('medium',name,error)
                continue
            result=self.describe(name,medium,(2048,4096))
            late,early=result['curve'][-1],result['curve'][0]
            threshold=min(1.,late['main_initializer_R'])
            result['eligible']=late['all_TRAIN_gates_pass'] and result['R']<threshold and result['R']<=early['R']*1.15
            result['promotion_R_threshold']=threshold
            results.append(result);save(ART/'medium.json',results)
        winner=prefer_small([row for row in results if row['eligible']],self.plan)
        if winner is None:
            self.status('COMPLETE_NO_PROMOTION',reason='Compact medium does not beat both fresh657 and matched main initializer with gates')
            return
        name=winner['variant'];long=[]
        save(ART/'long_selection.json',{'winner':name,'selected_utc':now(),'evidence_sha256':sha(ART/'medium.json')})
        for seed in (4817,5817):
            self.ensure_controls(f'registration_long_control{seed}',(3413,6826,10240),10240,4096,seed,(2817,3817))
            if not self.budgets_available((3413,6826,10240),4096,seed,(2817,3817),require_anchor=False):
                self.status('COMPLETE_MATCHED_SCOPE_LIMIT',winner=name,
                    verified_TRAIN=sorted({row['seed'] for row in long}),
                    reason=f'No exact verified fresh657 long scope for TRAIN{seed}; no historical fallback')
                return
            long+=self.stage(f'long{seed}',name,10240,4096,seed=seed,checkpoints=(3413,6826),gens=(2817,3817))
            result=self.describe(name,long,(3413,6826,10240))
            promotion=self.per_seed_promotion(name,long,10240)
            result['promotion']=promotion;save(ART/'long.json',result)
            late=result['curve'][-1]
            threshold=min(1.,late['main_initializer_R']) if late['main_initializer_R'] is not None else 1.
            if not promotion['all_TRAIN_pass'] or not late['all_TRAIN_gates_pass'] or result['R']>=threshold:
                self.status('COMPLETE_NO_PROMOTION',winner=name,
                    per_TRAIN_promotion=promotion,
                    reason='Every TRAIN seed and pooled long confirmation must beat fresh657 and each available matched initializer with gates')
                return
        full=[]
        if self.plan['full_evaluation_if_time']:
            for seed in (4817,5817):
                trained_control=[row for row in self.repeat_controls if row['seed']==seed
                    and row['budget']==10240 and row['num_samples']==4096][-1]
                control_checkpoint=trained_control['checkpoint']
                available=[row for row in self.repeat_controls if row['seed']==seed
                    and row['budget']==10240 and row['num_samples']==50000
                    and row['checkpoint']==control_checkpoint]
                if {row['eval_seed'] for row in available}!={2817,3817}:
                    self.repeat_controls+=self.stage(f'full_control{seed}','control657',10240,50000,
                        seed=seed,gens=(2817,3817),evaluation={'control657':control_checkpoint})
                candidate_checkpoint=[row for row in long if row['seed']==seed and row['budget']==10240][-1]['checkpoint']
                full+=self.stage(f'full{seed}',name,10240,50000,seed=seed,gens=(2817,3817),
                    evaluation={name:candidate_checkpoint})
                final=self.describe(name,full,(10240,))
                final['fresh657_promotion']=self.per_seed_promotion(name,full,10240,compare_initializer=False)
                save(ART/'full.json',final)
        main_available=late['main_initializer_R'] is not None
        self.status('COMPLETE',winner=name,beats_fresh657=True,
            beats_main_initializer=True if main_available else None,
            main_initializer_long_comparison_available=main_available,
            beats_fresh657_scope='paired4096_scenes_at10240steps',
            verified_budget=10240,verified_TRAIN=[4817,5817],verified_GEN=[2817,3817],
            full50000_evaluation=bool(full),
            full50000_beats_fresh657=(final['fresh657_promotion']['all_TRAIN_pass'] and final['curve'][-1]['all_TRAIN_gates_pass'] and final['R']<1.) if full else None,
            per_TRAIN_long_promotion=promotion,
            warning='10epoch confirmation does not prove128epoch convergence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,help='Root-reviewed predeclared policy JSON')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--print-default-policy',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker);return
    if args.print_default_policy:
        print(json.dumps(default_policy(),indent=2));return
    if args.plan is None:
        parser.error('--plan is required')
    proposed_path=args.plan.resolve();policy=validate_policy(read(proposed_path))
    if not args.execute:
        print(json.dumps({'predeclared_policy':policy,'adaptive_graph_freeze':'Deferred until mainPID andGPU areidle',
            'artifact_dir':str(ART),'run_dir':str(RUNS),'no_GPU_launch':True},indent=2));return
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'driver.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A compact driver already owns this queue')
        signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
        deadline=aware_deadline(policy['deadline']);proposed_hash=sha(proposed_path)
        wait_pid=args.wait_pid
        if wait_pid is None:
            main_status=MAIN_ART/'rootdriver_status.json'
            wait_pid=read(main_status).get('driver_pid',1419205) if main_status.exists() else 1419205
        while (wait_pid and alive(wait_pid)) or gpu_pids():
            value={'status':'WAITING_FOR_MAIN_PHASE_AND_SERIAL_GPU','driver_pid':os.getpid(),
                'wait_pid':wait_pid,'deadline':policy['deadline'],'updated_utc':now(),'policy_sha256':proposed_hash}
            save(ART/'rootdriver_status.json',value);save(ART/'status.json',value)
            if STOP_REQUESTED or time.time()>=deadline:
                save(ART/'status.json',{'status':'DEADLINE_WHILE_WAITING','finished_utc':now()});return
            time.sleep(5)
        if sha(proposed_path)!=proposed_hash:
            raise RuntimeError('Predeclared policy changed while waiting')
        rows,controls,provenance=load_verified_main_data()
        effective=build_effective_plan(policy,rows,controls,read(MAIN_ART/'variants.json'))
        effective['predeclared_policy_sha256']=proposed_hash
        effective['main_protocol_sha256']=sha(MAIN_ART/'protocol.json')
        effective_path=ART/'effective_plan.json'
        if effective_path.exists() and read(effective_path)!=effective:
            raise RuntimeError('Adaptive compact plan already frozen to different evidence')
        save(effective_path,effective)
        save(ART/'policy_registration.json',{'policy_path':str(proposed_path),'policy_sha256':proposed_hash,
            'main_protocol_sha256':effective['main_protocol_sha256'],'source_cells':provenance,
            'selected_graph_anchor':effective['selected_graph_anchor'],'graph_selection':effective['graph_selection']})
        driver=CompactResearchDriver(effective,effective_path,rows,controls)
        try:
            driver.wait_serial();driver.run()
        except DeadlineReached as error:
            previous=read(ART/'rootdriver_status.json')
            driver.status('DEADLINE_REACHED',reason=str(error),next_task=previous.get('next_task'),
                observations=len(driver.rows),warning='Only complete native stages count; no128epoch convergence claim')
        except Exception as error:
            if isinstance(error,StageFailed):driver.record_failure('required_confirmation','unknown',error)
            driver.status('FAILED_NEEDS_INSPECTION',reason=repr(error),observations=len(driver.rows));raise


if __name__=='__main__':main()
