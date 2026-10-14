"""Paired, deadline-bounded InitDiffusion search ending 2026-10-14 10:00 CST.

Uses the existing native training/evaluation chain and fixed AIRL map. Every
candidate starts from the same seeded G1. Longer stages restore only their own
optimizer/EMA/RNG lineage. W&B is read-only observational evidence.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_wandb_cc_scratch_20261013 as shared
from tools import run_init_diffusion_evening_20261013 as evidence
read, save, sha, now = shared.read, shared.save, shared.sha, shared.now
ART = ROOT / 'output/reports/init_diffusion_overnight_20261014'
RUNS = ROOT / 'logs/init_diffusion_overnight_20261014'
END = '2026-10-14T10:00:00+08:00'
DEADLINE = datetime.fromisoformat(END).timestamp()
P = shared.PREFIX
STOP = False
BASE = 'roadK8'
SCREEN = ['size_residual', 'size_time_wp3', 'match2', 'size_time_width_l1',
          'match4', 'speed_time_residual']


def status(value, **extra):
    save(ART / 'status.json', dict(status=value, updated_utc=now(),
         deadline=END, driver_pid=os.getpid(), **extra))


def sources():
    paths = set(shared.SOURCE_FILES) | {
        'src/smart/diffusion/size_parameterization.py',
        'src/smart/diffusion/sampling_time_grid.py',
        'tools/runtime_checkpointed_sweep.py',
        'tools/run_init_diffusion_evening_20261013.py',
        'tools/run_init_overnight_20261014.py',
    }
    return {p: sha(ROOT / p) for p in sorted(paths)}


def prepare():
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Registered phase already exists')
    prior = ROOT / 'output/reports/init_diffusion_evening_20261013/shape_matching_followup'
    base = copy.deepcopy(read(prior / 'variants.json')['K8_size_residual'])
    base['__config'][P + 'shape_x0_parameterization'] = 'identity'
    variants = {BASE: base}
    deltas = {
        'size_residual': {'__config': {P + 'shape_x0_parameterization': 'residual'}},
        'size_time_wp3': {'__config': {P + 'shape_x0_parameterization': 'time_residual'},
                         'width': {'time_power': 3}},
        'size_time_width_l1': {'__config': {P + 'shape_x0_parameterization': 'time_residual'},
                               'width': {'kind': 'l1'}},
        'speed_time_residual': {'__config': {P + 'speed_x0_parameterization': 'time_residual'}},
    }
    for strength in (2, 4):
        deltas[f'match{strength}'] = {'__config': {P + 'matching_weights': dict(
            position=1., heading=float(strength), length=1., width=1., velocity=float(strength))}}
    for name, delta in deltas.items():
        variants[name] = copy.deepcopy(base)
        variants[name]['__config'].update(delta.get('__config', {}))
        variants[name].update({k: v for k, v in delta.items() if k != '__config'})
    save(ART / 'recipe.json', read(prior / 'recipe.json'))
    save(ART / 'variants.json', variants)
    # Full membership is deterministic and separately hashed; no prefix cap.
    files = sorted(p.name for p in (ROOT / 'src/waymo_data/full/scenario_dreamer_val').glob('*.pt'))
    assert len(files) == len(set(files)) == 50000
    save(ART / 'full50000.json', files)
    plan = dict(registered_utc=now(), deadline=END,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=sources(),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in
                (shared.MAP, shared.NORMALIZER, shared.TRAIN_MANIFEST, shared.EVAL_MANIFEST)},
        local_input_hashes={n: sha(ART / n) for n in ('recipe.json', 'variants.json', 'full50000.json')},
        screen_order=[BASE] + SCREEN, screen_steps=4096, longer_steps=[12288, 16384],
        second_TRAIN=5817, first_TRAIN=4817, GEN=[2817, 3817],
        primary='R_focus = mean paired speed/angle/length JSD ratios',
        secondary='R_jsd six JSD; collision reported but never selected on',
        screen_selection='Top one, or top two only if>=30000 seconds remain, by R_focus among width<=1.20, each focus<=1.15, onroad decline<=2pp and vehicle count within10%; if none, best R_focus is explicitly exploratory, not eligible. Reserve second TRAIN and full50K time.',
        promotion='Require R_focus<=.97 at both12288 and16384, every focus<=1.10, width<=1.15, native gates; second TRAIN and full50K confirmation, no final128epoch claim.',
        parameters=8083978, parameter_cap=8100000, sampling_steps=20, guidance=False,
        batch_size=512, quick_workers=0, precision='32-true', matmul='medium',
        map_frozen=True, fresh_G1_optimizer_EMA=True,
        training_order='One seeded full TRAIN order repeated each epoch; full recipe reshuffles. Disclose proxy limitation.',
        safety='Ignore only ToDesk_Session<=1GiB display; never signal external processes. Deadline terminates only child process groups created by this driver.',
        evidence='Completed native stages only; partial/error stages preserved and excluded. Source/inputs frozen for each registered phase.',
        residual_tests='10 CPU unit tests pass; identity leaves parameters and RNG unchanged',
        full_reference='configs/metrics/iu3fa4gx_full50000.json; unequal-budget reference ratios are descriptive only')
    save(ART / 'protocol.json', plan)
    save(ART / 'observations.json', [])
    status('REGISTERED', candidates=SCREEN)


def frozen():
    plan = read(ART / 'protocol.json')
    assert sources() == plan['source_hashes'], 'Source changed: register a new phase'
    for p, value in plan['inputs'].items():
        assert sha(ROOT / p) == value, p
    for p, value in plan['local_input_hashes'].items():
        assert sha(ART / p) == value, p


def occupied():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader'], capture_output=True, text=True, check=True)
    other, display = [], []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 3 or not row[0].strip().isdigit():
            raise RuntimeError('Unrecognized GPU process record')
        pid, name, memory = int(row[0]), row[1].strip(), int(row[2].strip().split()[0])
        record = dict(pid=pid, name=name, memory_MiB=memory)
        (display if name == '/opt/todesk/bin/ToDesk_Session' and memory <= 1024 else other).append(record)
    save(ART / 'gpu_occupancy.json', dict(checked_utc=now(), display=display, other=other))
    return other


def verify(rows, spec, folder):
    import yaml
    assert len(rows) == 2 and {r['eval_seed'] for r in rows} == {2817, 3817}
    cfg = yaml.safe_load((folder / spec['arm'] / 'config.yaml').read_text())
    mc = cfg['model']['model_config']; d = mc['decoder']['init_diffusion']
    variant = read(ART / 'variants.json')[spec['arm']]
    recipe = read(ART / 'recipe.json')
    for key, value in variant.items():
        if not key.startswith('__'):
            recipe['components'][key].update(value)
    assert d['component_loss'] == recipe
    assert d['sampling_steps'] == 20 and not d.get('sampling_time_grid')
    assert not d.get('sampling_exact_x0') and not d.get('collision_guidance')
    assert mc['initial_map_checkpoint'] == str(shared.MAP) and cfg['ckpt_path'] is None
    for key, value in variant['__config'].items():
        if key.startswith(P):
            assert d[key[len(P):]] == value, key
    for row in rows:
        assert row['variant'] == spec['arm'] and row['budget'] == spec['steps']
        assert row['seed'] == spec['seed'] and row['num_samples'] == spec['scenes']
        assert row['map_frozen'] and row['evaluation_weights'] == 'ema'
        assert row['generator_num_parameters'] == row['generator_trainable_parameters'] == 8083978
        assert row['seen_training_examples'] == row['budget'] // 964 * 493542 + row['budget'] % 964 * 512
        report = read(row['metrics_path']); actual = report['decoder']
        assert report['num_samples'] == report['num_gt_samples'] == spec['scenes']
        assert report['reference_mode'] == 'matched' and report['map_source'] == 'reference'
        assert report['unverified_generation_timestep_count'] == 0
        assert actual['sampling_steps'] == 20 and actual['use_ema'] and actual['ema_num_updates'] == spec['steps']
        assert report['runtime']['trainer_precision'] == '32-true'
        assert report['runtime']['float32_matmul_precision'] == 'medium'
        assert all(actual['fix_ego_' + k] for k in ('position', 'heading', 'shape', 'type'))
        assert not actual['fix_ego_velocity']
        assert all(math.isfinite(row[k]) and row[k] == report['agent_metrics'][k] for k in shared.METRICS)
        if not spec.get('evaluation'):
            assert row['optimizer_continued'] == bool(spec.get('resume'))
            assert row['resume_global_step'] == spec.get('resume_step', 0)
            assert row['resumed_from'] == spec.get('resume')
    if not spec.get('evaluation'):
        trace = read(folder / spec['arm'] / 'train_trace.json')
        assert [r['step'] for r in trace] == list(range(spec.get('resume_step', 0), spec['steps']))
        assert all(r['map_grad_norm_before_clip'] == 0 for r in trace)


def record_rows(values):
    rows = read(ART / 'observations.json')
    key = lambda r: (r['variant'], r['seed'], r['budget'], r['eval_seed'], r['num_samples'])
    keys = {key(r) for r in rows}
    assert not any(key(r) in keys for r in values), 'Duplicate experimental cell'
    save(ART / 'observations.json', rows + values)
    refresh()


def run(arm, seed, steps, scenes=4096, resume=None, resume_step=0, evaluation=False, attempt_label=''):
    rows = read(ART / 'observations.json')
    done = [r for r in rows if (r['variant'], r['seed'], r['budget'], r['num_samples']) == (arm, seed, steps, scenes)]
    if done:
        assert len(done) == 2
        return done
    estimate = (steps - resume_step) * .44 + 150 if not evaluation else scenes / 4096 * 75 + 90
    if STOP or DEADLINE - time.time() < estimate + 120:
        status('NO_TIME_FOR_STAGE', arm=arm, steps=steps, scenes=scenes)
        return None
    frozen()
    while occupied():
        status('WAITING_FOR_GPU', arm=arm)
        if STOP or DEADLINE - time.time() < estimate + 120:
            return None
        time.sleep(30)
    label = f'{arm}_T{seed}_{steps}_N{scenes}' + attempt_label
    spec = dict(arm=arm, seed=seed, steps=steps, scenes=scenes, resume=resume,
                resume_step=resume_step, evaluation=evaluation, label=label)
    folder = RUNS / label
    folder.mkdir(parents=True, exist_ok=False)
    variant_path = ART / (label + '.variants.json')
    save(variant_path, {arm: read(ART / 'variants.json')[arm]})
    cmd = [shared.PYTHON, str(ROOT / 'tools/runtime_checkpointed_sweep.py'),
        '--map-checkpoint', str(shared.MAP), '--batch-size', '512',
        '--scenes', str(scenes), '--eval-batch-size', '128', '--workers', '0',
        '--defer-cyclic-gc', '--young-gc-interval', '0', '--allow-multiple-epochs',
        '--seed', str(seed), '--eval-seed', '2817', '--extra-eval-seed', '3817',
        '--normalizer-json', str(shared.NORMALIZER), '--evaluation-weights', 'ema',
        '--steps', str(steps), '--checkpoint-steps', '',
        '--recipe', str(ART / 'recipe.json'), '--variant-file', str(variant_path),
        '--variants', '' if evaluation else arm, '--output', str(folder), '--eval-manifest',
        str(ART / 'full50000.json' if scenes == 50000 else shared.EVAL_MANIFEST)]
    if scenes == 50000:
        cmd.append('--require-full-set')
    if resume:
        path = ART / (label + '.checkpoint.json'); save(path, {arm: resume})
        cmd += ['--evaluation-checkpoints' if evaluation else '--resume-map', str(path)]
        if not evaluation:
            cmd.append('--resume-cyclic-epochs')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/sim_overnight_20261014_mpl', PYTHONFAULTHANDLER='1', SIM_SWEEP_DEFER_CYCLIC_GC='1')
    record = dict(spec=spec, command=cmd, started_utc=now(), protocol_sha256=sha(ART / 'protocol.json'))
    start = time.monotonic()
    with (folder / 'console.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(folder / 'execution.json', dict(record, child_pid=child.pid))
        while child.poll() is None:
            status('RUNNING', stage=spec, child_pid=child.pid, elapsed_seconds=time.monotonic()-start)
            if STOP or time.time() >= DEADLINE:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                status('PARTIAL_EXCLUDED', stage=spec)
                return None
            time.sleep(10)
    record.update(returncode=child.returncode, finished_utc=now(), elapsed_seconds=time.monotonic()-start)
    save(folder / 'execution.json', record)
    if child.returncode or 'RUNTIME_LOADER_SUCCESS' not in (folder / 'console.log').read_text():
        status('FAILED_STAGE', stage=spec, returncode=child.returncode)
        return None
    values = read(folder / 'summary.json')
    verify(values, spec, folder); frozen()
    save(folder / 'verified_completion.json', dict(record, status='VERIFIED_COMPLETE', summary_sha256=sha(folder / 'summary.json')))
    record_rows(values)
    print('COMPLETE ' + label, flush=True)
    return values


def refresh():
    rows = read(ART / 'observations.json'); out = []
    keys = sorted({(r['variant'], r['seed'], r['budget'], r['num_samples']) for r in rows if r['variant'] != BASE})
    for arm, seed, budget, scenes in keys:
        a = [r for r in rows if (r['variant'], r['seed'], r['budget'], r['num_samples']) == (arm, seed, budget, scenes)]
        b = [r for r in rows if (r['variant'], r['seed'], r['budget'], r['num_samples']) == (BASE, seed, budget, scenes)]
        if len(a) != 2 or len(b) != 2:
            continue
        score = evidence.score(a, b)
        score['eligible'] = (score['gates_pass'] and score['R_focus'] <= .97
            and all(score['ratios'][k] <= 1.10 for k in evidence.FOCUS))
        out.append(dict(arm=arm, seed=seed, budget=budget, scenes=scenes, reference=BASE, **score))
    save(ART / 'comparisons.json', dict(generated_utc=now(), comparisons=out,
        limitation='Paired short training; fixed repeated TRAIN order, not final128epoch evidence.'))
    return out


def checkpoint(arm, seed, step):
    rows = read(ART / 'observations.json')
    return next(r['checkpoint'] for r in rows if (r['variant'], r['seed'], r['budget']) == (arm, seed, step))


def wandb_context():
    path = ART / 'wandb/latest_analysis.json'
    return dict(path=str(path), sha256=sha(path), snapshot=read(path)['inventory_fetched_at_utc']) if path.exists() else None


def execute():
    global STOP
    ART.mkdir(parents=True, exist_ok=True)
    lock = (ART / 'driver.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def halt(*unused):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, halt); signal.signal(signal.SIGINT, halt)
    for arm in [BASE] + SCREEN:
        if run(arm, 4817, 4096) is None:
            return
    comparisons = refresh()
    early = [r for r in comparisons if r['budget'] == 4096 and r['seed'] == 4817]
    eligible = [r for r in early if r['ratios']['width_jsd'] <= 1.20
        and all(r['ratios'][k] <= 1.15 for k in evidence.FOCUS)
        and r['gates']['onroad'] and r['gates']['vehicle_count']]
    count = 2 if DEADLINE - time.time() >= 30000 else 1
    selected = [r['arm'] for r in sorted(eligible or early, key=lambda r: r['R_focus'])[:count]]
    save(ART / 'long_selection.json', dict(at_utc=now(), selected=selected, evidence=early,
        exploratory_fallback=not bool(eligible), wandb=wandb_context()))
    # Give the baseline identical optimizer boundaries to each candidate.
    for step, previous in ((12288, 4096), (16384, 12288)):
        for arm in [BASE] + selected:
            if run(arm, 4817, step, resume=checkpoint(arm, 4817, previous), resume_step=previous) is None:
                return
    later = [r for r in refresh() if r['budget'] == 16384 and r['seed'] == 4817]
    best = min(later, key=lambda r: r['R_focus'])['arm']
    both = [r for r in refresh() if r['arm'] == best and r['budget'] in (12288, 16384)]
    save(ART / 'second_seed_selection.json', dict(at_utc=now(), arm=best,
         both_budgets_eligible=len(both) == 2 and all(r['eligible'] for r in both), evidence=both, wandb=wandb_context()))
    for arm in [BASE, best]:
        if run(arm, 5817, 6826) is None:
            return
    # Same-budget second TRAIN is interpreted alongside the longer first seed.
    for arm in [BASE, best]:
        if run(arm, 4817, 6826, resume=checkpoint(arm, 4817, 4096), resume_step=4096) is None:
            return
    for arm in [BASE, best]:
        if run(arm, 4817, 16384, scenes=50000, resume=checkpoint(arm, 4817, 16384), evaluation=True) is None:
            return
    comparisons = refresh()
    proof = [r for r in comparisons if r['arm'] == best and
             ((r['seed'] == 4817 and r['budget'] in (12288, 16384)) or r['budget'] == 6826)]
    promoted = len(proof) == 5 and all(r['eligible'] for r in proof)
    save(ART / 'recommendation.json', dict(generated_utc=now(), candidate=best,
        promoted=promoted, evidence=proof, full128epoch_verified=False,
        default='init_diffusion_657_n20_road_k8_full', wandb=wandb_context()))
    status('PLANNED_EXPERIMENTS_COMPLETE', observations=len(read(ART / 'observations.json')),
        full128epoch_verified=False, monitoring_until=END)


def watch():
    from tools import watch_wandb_evening_20261013 as watcher
    watcher.ART = ART
    while time.time() < DEADLINE:
        try:
            result = watcher.snapshot()
            save(ART / 'wandb/watcher_status.json', dict(status='SNAPSHOT_COMPLETE', updated_utc=now(),
                inventory_utc=result['inventory_fetched_at_utc'], pid=os.getpid(), deadline=END))
        except Exception as exc:
            save(ART / 'wandb/watcher_status.json', dict(status='READ_ONLY_REFRESH_FAILED',
                 error_type=type(exc).__name__, updated_utc=now(), pid=os.getpid(), deadline=END))
        until = min(DEADLINE, time.time() + 1800)
        while time.time() < until:
            time.sleep(min(10, until - time.time()))
    save(ART / 'wandb/watcher_status.json', dict(status='DEADLINE_COMPLETE', updated_utc=now(), pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute', 'watch', 'refresh'])
    args = parser.parse_args()
    globals()[args.mode]()
