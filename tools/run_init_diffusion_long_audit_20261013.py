"""Amended evening experiment: scope audit and longer paired map ablations.

The user reported poor long-run dbf/de results. Preserve the completed short
trials, withdraw lane promotion, and test the original arms beyond the early
EMA transient. This is a same-arm continuation, never a candidate warm start.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_init_diffusion_evening_20261013 as evidence
from tools import run_init_diffusion_657_iteration as native
from tools import run_wandb_cc_scratch_20261013 as shared

read, save, sha, now = native.read, native.save, native.sha, native.now
OLD = ROOT / 'output/reports/init_diffusion_evening_20261013/recovery'
ART = ROOT / 'output/reports/init_diffusion_evening_20261013/long_audit'
RUNS = ROOT / 'logs/init_diffusion_evening_20261013/long_audit'
DEADLINE = datetime.fromisoformat('2026-10-13T22:30:00+08:00').timestamp()
STOP = False


def sources():
    return {**evidence.sources(), str(Path(__file__).relative_to(ROOT)): sha(__file__)}


def status(value, **extra):
    save(ART / 'status.json', dict(status=value, updated_utc=now(),
                                 driver_pid=os.getpid(), **extra))


def prepare():
    assert not (ART / 'protocol.json').exists()
    save(ART / 'recipe.json', read(OLD / 'recipe.json'))
    save(ART / 'variants.json', {k: read(OLD / 'variants.json')[k]
                               for k in ('fresh657', 'freshK8', 'lane_only')})
    rows = read(OLD / 'observations.json')
    seed = 4817
    cp = {}
    for name, budget in [('fresh657', 10240), ('freshK8', 2048), ('lane_only', 6826)]:
        cp[name] = next(r['checkpoint'] for r in rows
                        if (r['variant'], r['seed'], r['budget']) == (name, seed, budget))
    small_cp = {name: next(r['checkpoint'] for r in rows
                          if (r['variant'], r['seed'], r['budget']) == (name, seed, 4096))
                for name in ('fresh657', 'lane_only')}
    # Production validation has no explicit sample list: MultiDataset sorts
    # cache basenames, then Lightning takes its first N batches.
    files = sorted(p.name for p in (ROOT / 'src/waymo_data/full/scenario_dreamer_val').glob('*.pt'))
    cohorts = {}
    for n in (128, 256, 4096):
        path = ART / f'sorted{n}.json'
        save(path, files[:n]); cohorts[str(n)] = str(path)
    random = ROOT / 'tmp/control657_paired_20261012/eval4096_files.json'
    stages = [dict(label=f'scope_sorted{n}', evaluation=small_cp, scenes=n,
                   manifest=cohorts[str(n)], seed=seed, steps=4096, checkpoints=[])
              for n in (128, 256, 4096)]
    # The target is ~17 actual epochs, rather than another 2k/4k proxy. Earlier
    # prefixes remain recorded; every continuation restores its own optimizer,
    # EMA, scheduler and RNG and does not replay the first resumed epoch.
    for name, start in [('fresh657', 10240), ('freshK8', 2048), ('lane_only', 6826)]:
        stages.append(dict(label=f'long_{name}_{seed}', arm=name, seed=seed,
                           steps=16384, resume={name: cp[name]}, resume_step=start,
                           checkpoints=[b for b in (8192, 12288) if b > start],
                           scenes=4096, manifest=str(random)))
    imports = {str(OLD.relative_to(ROOT) / 'observations.json'): sha(OLD / 'observations.json')}
    for path in list(cp.values()) + list(small_cp.values()):
        for p in (Path(path), Path(path).parent / 'config.yaml',
                  Path(path).parent / 'initialization.json'):
            imports[str(p.relative_to(ROOT))] = sha(p)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              (shared.MAP, shared.NORMALIZER, shared.TRAIN_MANIFEST, random)}
    plan = dict(registered_utc=now(), deadline='2026-10-13T22:30:00+08:00',
                reason='User reported poor long-run dbf/de W&B results. Withdraw lane full-run recommendation; replace queued lane variants with longer original-arm comparisons.',
                source_hashes=sources(), imported_sources=imports, inputs=inputs,
                local_input_hashes={n: sha(ART / n) for n in ('recipe.json', 'variants.json', 'sorted128.json', 'sorted256.json', 'sorted4096.json')},
                stages=stages, sampling_steps=20, parameter_cap=8100000,
                TRAIN=[seed], GEN=[2817, 3817], primary='mean of paired speed, angle, length JSD ratios',
                recommendation_rule='A short winner is not a full-training recommendation. Require improvement at both12288 and16384, no focused component more than10% worse and no onroad drop>2pp; otherwise no promotion. Single TRAIN and repeated epoch order are disclosed.',
                frozen_map=True, same_arm_continuation=True,
                resume_numerics='Optimizer/EMA/scheduler/RNG restored; prior proof shows small backend numerical drift, not bitwise identity.',
                full128epoch_verified=False, validation_scope_audit='Same saved weights and GENs on sorted128, sorted256 and sorted4096, compared with existing random4096. No cross-cohort method ranking.',
                abandoned_unstarted=['lane_speed_head32', 'lane_map64', 'lane_focus_combo'],
                optional='If budget remains, expand all three endpoint models to16384 scenes and audit128 vs4096 on those same endpoint weights; do not alter sampling or model midway.')
    save(ART / 'protocol.json', plan)
    save(ART / 'observations.json', [])
    status('REGISTERED', stages=len(stages))


def frozen(plan):
    assert sources() == plan['source_hashes'], 'Source changed during the registered phase'
    for table in ('inputs', 'imported_sources'):
        for path, digest in plan[table].items(): assert sha(ROOT / path) == digest, path
    for path, digest in plan['local_input_hashes'].items(): assert sha(ART / path) == digest, path


def verify(rows, spec):
    import yaml
    names = list(spec.get('evaluation', {})) or [spec['arm']]
    if spec.get('evaluation'):
        expected = {(n, spec['steps'], g) for n in names for g in shared.GENS}
    else:
        expected = {(spec['arm'], b, g) for b in spec['checkpoints'] + [spec['steps']] for g in shared.GENS}
    assert {(r['variant'], r['budget'], r['eval_seed']) for r in rows} == expected
    assert len(rows) == len(expected)
    for r in rows:
        report = read(r['metrics_path']); d = report['decoder']
        assert report['num_samples'] == report['num_gt_samples'] == spec['scenes']
        assert report['reference_mode'] == 'matched' and report['map_source'] == 'reference'
        assert report['unverified_generation_timestep_count'] == 0
        assert d['sampling_steps'] == 20 and d['ema_num_updates'] == r['budget'] and d['use_ema']
        assert r['generator_num_parameters'] == 8083978 and r['map_frozen']
        assert r['evaluation_weights'] == 'ema' and r['seed'] == spec['seed']
        assert report['runtime']['trainer_precision'] == '32-true'
        assert report['runtime']['float32_matmul_precision'] == 'medium'
        if not spec.get('evaluation'):
            assert r['optimizer_continued'] and r['resume_cyclic_epochs']
            assert r['resumed_from'] == spec['resume'][r['variant']]
            assert r['resume_global_step'] == spec['resume_step']
            assert r['seen_training_examples'] == r['budget']//964*493542 + r['budget']%964*512
            trace = read(RUNS / spec['label'] / r['variant'] / 'train_trace.json')
            assert len(trace) == spec['steps']-spec['resume_step']
            assert trace[0]['step'] == spec['resume_step'] and trace[-1]['step'] == spec['steps']-1
            assert all(t['map_grad_norm_before_clip'] == 0 for t in trace)
        cfg = yaml.safe_load((RUNS / spec['label'] / r['variant'] / 'config.yaml').read_text())
        actual = cfg['model']['model_config']['decoder']['init_diffusion']
        assert not actual.get('sampling_exact_x0') and not actual.get('collision_guidance')


def run_stage(spec, plan):
    frozen(plan)
    if shared.gpu_pids(): raise RuntimeError('Another GPU process is active')
    folder = RUNS / spec['label']; folder.mkdir(parents=True, exist_ok=False)
    variants = {spec['arm']: read(ART / 'variants.json')[spec['arm']]} if 'arm' in spec else {}
    variant_path = ART / (spec['label'] + '.variants.json'); save(variant_path, variants)
    cmd = [shared.PYTHON, str(ROOT / 'tools/runtime_cached_torch_loader_sweep.py'),
           '--map-checkpoint', str(shared.MAP), '--batch-size', '512',
           '--scenes', str(spec['scenes']), '--eval-batch-size', '128', '--workers', '0',
           '--defer-cyclic-gc', '--young-gc-interval', '0', '--allow-multiple-epochs',
           '--seed', str(spec['seed']), '--eval-seed', '2817', '--extra-eval-seed', '3817',
           '--normalizer-json', str(shared.NORMALIZER), '--evaluation-weights', 'ema',
           '--steps', str(spec['steps']), '--checkpoint-steps', ','.join(map(str, spec['checkpoints'])),
           '--recipe', str(ART / 'recipe.json'), '--variant-file', str(variant_path),
           '--variants', ','.join(variants), '--output', str(folder), '--eval-manifest', spec['manifest']]
    for key, flag in [('evaluation', '--evaluation-checkpoints'), ('resume', '--resume-map')]:
        if key in spec:
            path = ART / (spec['label'] + '.' + key + '.json'); save(path, spec[key]); cmd += [flag, str(path)]
    if 'resume' in spec: cmd += ['--resume-cyclic-epochs']
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONFAULTHANDLER='1', SIM_SWEEP_DEFER_CYCLIC_GC='1')
    started = time.monotonic()
    record = dict(spec=spec, command=cmd, started_utc=now(), protocol_sha256=sha(ART / 'protocol.json'))
    with (folder / 'console.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            status('RUNNING', stage=spec, child_pid=child.pid,
                   elapsed_seconds=time.monotonic()-started, remaining_seconds=DEADLINE-time.time())
            if STOP or time.time() >= DEADLINE:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                status('PARTIAL_EXCLUDED', stage=spec); return None
            time.sleep(10)
    record['returncode'] = child.returncode
    save(folder / 'execution.json', record)
    if child.returncode or 'RUNTIME_LOADER_SUCCESS' not in (folder / 'console.log').read_text():
        status('FAILED_STAGE', stage=spec, returncode=child.returncode); return None
    rows = read(folder / 'summary.json'); verify(rows, spec); frozen(plan)
    record.update(status='VERIFIED_COMPLETE', finished_utc=now(), summary_sha256=sha(folder / 'summary.json'))
    save(folder / 'verified_completion.json', record)
    print('COMPLETE ' + spec['label'], flush=True)
    return rows


def comparisons(rows):
    results = []
    for budget in (12288, 16384):
        base = [r for r in rows if r['variant'] == 'fresh657' and r['budget'] == budget and r['num_samples'] == 4096 and not r['evaluation_only']]
        for arm in ('freshK8', 'lane_only'):
            a = [r for r in rows if r['variant'] == arm and r['budget'] == budget and r['num_samples'] == 4096 and not r['evaluation_only']]
            if len(a) == len(base) == 2:
                results.append(dict(arm=arm, budget=budget, reference='fresh657', **evidence.score(a, base)))
    save(ART / 'comparisons.json', dict(generated_utc=now(), comparisons=results,
                                     caveat='One TRAIN, two GEN; fixed repeated training order. Not a full128epoch result.'))


def execute():
    global STOP
    def halt(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, halt); signal.signal(signal.SIGINT, halt)
    plan = read(ART / 'protocol.json'); rows = read(ART / 'observations.json')
    for spec in plan['stages']:
        if STOP: break
        updates = spec['steps']-spec.get('resume_step', spec['steps'])
        if DEADLINE-time.time() < updates*.45+180:
            status('NO_TIME_FOR_STAGE', next=spec); break
        values = run_stage(spec, plan)
        if values is None: return
        rows += values; save(ART / 'observations.json', rows); comparisons(rows)
    else:
        # Expanded validation on all models avoids selecting another method
        # simply because it overfits a small validation prefix.
        endpoints = [r for r in rows if r['budget'] == 16384]
        cp = {n: next(r['checkpoint'] for r in endpoints if r['variant'] == n)
              for n in ('fresh657', 'freshK8', 'lane_only')}
        for n, manifest in [(128, ART / 'sorted128.json'), (4096, ART / 'sorted4096.json'),
                            (16384, ROOT / 'tmp/control657_paired_20261012/eval16384_files.json')]:
            if DEADLINE-time.time() < n/4096*150+120: break
            spec = dict(label=f'endpoint_scope{n}', evaluation=cp, scenes=n,
                        manifest=str(manifest), seed=4817, steps=16384, checkpoints=[])
            values = run_stage(spec, plan)
            if values is None: return
            rows += values; save(ART / 'observations.json', rows)
    status('COMPLETE', observations=len(rows), full128epoch_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    execute() if args.execute else prepare()
