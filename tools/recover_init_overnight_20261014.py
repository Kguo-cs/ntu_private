"""Preserve failed native runs and continue in paired <=2048-update processes.

No model/data/optimizer objective changes. Baseline and the first residual use
their verified2048 prefixes, so process boundaries match across both arms.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_init_overnight_20261014 as base

read, save, sha, now = base.read, base.save, base.sha, base.now
OLD, OLD_RUNS = base.ART, base.RUNS
ART, RUNS = OLD / 'recovery', OLD_RUNS / 'recovery'
SOURCE_FN, FROZEN_FN, RUN_FN = base.sources, base.frozen, base.run


def sources():
    return {**SOURCE_FN(), str(Path(__file__).relative_to(ROOT)): sha(__file__)}


def metadata(path):
    code = '''import gc,os,json,sys;gc.disable()
import torch
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
c=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
m=c["scratch_sweep"];e=c["state_dict"]["encoder.init_decoder._extra_state"]["ema"]
assert c["global_step"]==m["saved_global_step"]==e["num_updates"]
assert c["optimizer_states"] and c["lr_schedulers"] and c["scratch_sweep_rng"]
print(json.dumps(dict(step=c["global_step"],seed=m["seed"],map=m["map_checkpoint_sha256"],initial_generator=m["initial_generator_sha256"])),flush=True)
os._exit(0)
'''
    result = subprocess.run([base.shared.PYTHON, '-c', code, str(path)], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Invalid recovery checkpoint ' + str(path))
    return json.loads(result.stdout.strip().splitlines()[-1])


def prepare():
    assert not (ART / 'protocol.json').exists()
    failed = read(OLD / 'status.json')
    assert failed['status'] == 'FAILED_STAGE' and failed['returncode'] == -11
    plan = copy.deepcopy(read(OLD / 'protocol.json'))
    changed = [p for p, digest in plan['source_hashes'].items() if sha(ROOT / p) != digest]
    assert changed == ['tools/run_init_overnight_20261014.py'], changed
    for name in ('recipe.json', 'variants.json', 'full50000.json'):
        save(ART / name, read(OLD / name))
    bootstrap = {}
    imports = [OLD / 'protocol.json', OLD / 'observations.json',
        OLD_RUNS / 'size_residual_T4817_4096_N4096/execution.json']
    for arm in (base.BASE, 'size_residual'):
        folder = OLD_RUNS / f'{arm}_T4817_4096_N4096' / arm
        checkpoint = folder / 'step_2048.ckpt'
        meta = metadata(checkpoint)
        assert meta['step'] == 2048 and meta['seed'] == 4817
        assert meta['map'] == plan['inputs'][str(base.shared.MAP.relative_to(ROOT))]
        trace_path = folder / ('train_trace.json' if arm == base.BASE else 'recovery_trace.json')
        trace = [r for r in read(trace_path) if r['step'] < 2048]
        assert [r['step'] for r in trace] == list(range(2048))
        save(ART / (arm + '.bootstrap_trace.json'), trace)
        bootstrap[arm] = dict(checkpoint=str(checkpoint), step=2048, metadata=meta,
            original_trace=str(trace_path), retained_updates=2048)
        imports += [checkpoint, trace_path, folder / 'config.yaml', folder / 'initialization.json']
    assert bootstrap[base.BASE]['metadata']['initial_generator'] == bootstrap['size_residual']['metadata']['initial_generator']
    save(ART / 'bootstrap.json', bootstrap)
    plan.update(registered_utc=now(), source_hashes=sources(),
        prior_protocol_sha256=sha(OLD / 'protocol.json'),
        imported_sources={str(p.relative_to(ROOT)): sha(p) for p in imports},
        local_input_hashes={n: sha(ART / n) for n in ('recipe.json', 'variants.json', 'full50000.json',
            'bootstrap.json', base.BASE + '.bootstrap_trace.json', 'size_residual.bootstrap_trace.json')},
        recovery='Native segfault at optimizer.zero_grad near3424. Fixed <=2048 update child stages for all remaining arms; baseline/residual both use their verified2048 prefixes. Max3 attempts/chunk, retry exact same start. Preserve failures; discarded updates do not count.',
        source_change='Only supervisor optional attempt directory label; no model/loss/data code change since registered original phase.',
        bootstrap_exclusion='Old residual2048..3424 discarded to align resume boundary with baseline. Old baseline4096 continuous result remains separate observational evidence.',
        numerical_caveat='Optimizer/EMA/scheduler/RNG restored, but nondeterministic backend means no bitwise equivalence claim.',
        direct_control='New chunked roadK8, not the previous continuous4096 result.')
    save(ART / 'protocol.json', plan)
    save(ART / 'observations.json', [])
    save(ART / 'recovery_ledger.json', [])
    base.status('REGISTERED_RECOVERY')


def frozen():
    FROZEN_FN()
    for p, digest in read(ART / 'protocol.json')['imported_sources'].items():
        assert sha(ROOT / p) == digest, p


def run_chunk(arm, seed, target, scenes, resume, start, evaluation):
    for attempt in range(1, 4):
        suffix = f'_attempt{attempt}'
        value = RUN_FN(arm, seed, target, scenes=scenes, resume=resume,
            resume_step=start, evaluation=evaluation, attempt_label=suffix)
        state = read(ART / 'status.json')
        ledger = read(ART / 'recovery_ledger.json')
        ledger.append(dict(at_utc=now(), arm=arm, seed=seed, target=target, scenes=scenes,
            from_step=start, resume=resume, attempt=attempt,
            status='COMPLETE' if value is not None else state['status'],
            retained_updates=0 if value is None or evaluation else target-start,
            discarded_failed_updates_excluded=True))
        save(ART / 'recovery_ledger.json', ledger)
        if value is not None:
            return value
        if base.STOP or state['status'] != 'FAILED_STAGE':
            return None
    base.status('FAILED_AFTER_THREE_ATTEMPTS', arm=arm, target=target)
    return None


def run(arm, seed, steps, scenes=4096, resume=None, resume_step=0, evaluation=False):
    existing = [r for r in read(ART / 'observations.json') if
        (r['variant'], r['seed'], r['budget'], r['num_samples']) == (arm, seed, steps, scenes)]
    if existing:
        assert len(existing) == 2
        return existing
    if evaluation:
        return run_chunk(arm, seed, steps, scenes, resume, resume_step, True)
    bootstrap = read(ART / 'bootstrap.json')
    if resume is None and seed == 4817 and arm in bootstrap:
        resume = bootstrap[arm]['checkpoint']; resume_step = bootstrap[arm]['step']
    value = None
    while resume_step < steps:
        target = min(steps, resume_step + 2048)
        # Existing same-arm intermediate checkpoints are retained rather than
        # counting a replay as another independent observation.
        value = [r for r in read(ART / 'observations.json') if
            (r['variant'], r['seed'], r['budget'], r['num_samples']) == (arm, seed, target, scenes)]
        if not value:
            value = run_chunk(arm, seed, target, scenes, resume, resume_step, False)
        if value is None:
            return None
        assert len(value) == 2
        resume = value[0]['checkpoint']; resume_step = target
    return value


base.ART, base.RUNS, base.sources, base.frozen, base.run = ART, RUNS, sources, frozen, run
base.wandb_context = lambda: dict(path=str(OLD / 'wandb/latest_analysis.json'),
    sha256=sha(OLD / 'wandb/latest_analysis.json'),
    snapshot=read(OLD / 'wandb/latest_analysis.json')['inventory_fetched_at_utc'])

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'execute'])
    args = p.parse_args()
    prepare() if args.mode == 'prepare' else base.execute()
