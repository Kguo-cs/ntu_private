"""Explicit adaptive phase after the registered overnight screen has stopped.

Retains all completed native cells, adds residual+matching2, and budgets up to
two longer-budget candidates using measured runtimes. Does not stop any process.
The prior driver must release its lock before preparation is allowed.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import recover_init_overnight_20261014 as recovery

base = recovery.base
read, save, sha, now = base.read, base.save, base.sha, base.now
OLD, OLD_RUNS = recovery.ART, recovery.RUNS
ART, RUNS = OLD.parent / 'adaptive_long', OLD_RUNS.parent / 'adaptive_long'
ORIGINAL_SOURCES = recovery.sources
COMBO = 'size_skip_match2'
COMBO_CONFIG = ROOT / 'configs/experiment/init_diffusion_657_n20_road_k8_size_skip_match2.yaml'


def sources():
    return {**ORIGINAL_SOURCES(), str(Path(__file__).relative_to(ROOT)): sha(__file__),
            str(COMBO_CONFIG.relative_to(ROOT)): sha(COMBO_CONFIG),
            'tools/handoff_init_overnight_20261014.py': sha(ROOT / 'tools/handoff_init_overnight_20261014.py'),
            'configs/experiment/init_diffusion_657_n20_road_k8_size_time_width_l1.yaml': sha(ROOT / 'configs/experiment/init_diffusion_657_n20_road_k8_size_time_width_l1.yaml')}


def configure():
    recovery.ART, recovery.RUNS = ART, RUNS
    base.ART, base.RUNS, base.sources = ART, RUNS, sources


def measured_runtime():
    rates, overheads, evals, provenance = [], [], [], {}
    for path in sorted(OLD_RUNS.glob('*/execution.json')):
        record = read(path)
        if record.get('returncode') != 0 or record['spec'].get('evaluation'):
            continue
        spec = record['spec']
        rows = read(path.parent / 'summary.json')
        updates = spec['steps'] - spec.get('resume_step', 0)
        duration = rows[0]['train_seconds']
        rates.append(duration / updates)
        overheads.append(record['elapsed_seconds'] - duration)
        evals.extend(r['eval_seconds'] for r in rows)
        provenance[str(path.relative_to(ROOT))] = sha(path)
    if not rates:
        raise ValueError('Need completed native stages to estimate remaining work')
    # Bound well above the measured median (.355s/update and ~43s overhead).
    return dict(seconds_per_update=max(.38, max(rates) * 1.05),
        stage_overhead=max(60., max(overheads) * 1.10),
        eval4096_seconds=max(evals) * 1.10,
        median_seconds_per_update=statistics.median(rates),
        median_stage_overhead=statistics.median(overheads),
        source_hashes=provenance)


def estimate_confirmation(candidate_count, runtime):
    """Budget all confirmation work after 4096-step screening, with reserve.

    First TRAIN: baseline plus candidates to16384, with boundaries at12288.
    Second TRAIN: baseline and the selected candidate to6826. First TRAIN
    also gets a paired6826 point, then baseline/winner get2GEN full50K.
    Imported intermediate checkpoints can reduce the actual cost.
    """
    if candidate_count not in (1, 2):
        raise ValueError('One or two longer-budget candidates are supported')
    updates = (candidate_count + 1) * 12288 + 2 * 6826 + 2 * 682
    train_stages = (candidate_count + 1) * 6 + 2 * 4 + 2
    full_evaluation = 4 * (50000 / 4096) * runtime['eval4096_seconds']
    return (updates * runtime['seconds_per_update']
        + (train_stages + 2) * runtime['stage_overhead'] + full_evaluation + 900.)


def choose_long_candidates(comparisons, remaining_seconds, runtime):
    screen = [r for r in comparisons if r['seed'] == 4817 and r['budget'] == 4096
              and r['scenes'] == 4096]
    if not screen:
        raise ValueError('No completed same-budget screening comparisons')
    eligible = [r for r in screen if r['ratios']['width_jsd'] <= 1.20
        and all(r['ratios'][k] <= 1.15 for k in base.evidence.FOCUS)
        and r['gates']['onroad'] and r['gates']['vehicle_count']]
    candidates = sorted(eligible or screen, key=lambda r: (r['R_focus'], r['arm']))
    estimates = {n: estimate_confirmation(n, runtime) for n in (1, 2)}
    count = 2 if len(candidates) >= 2 and remaining_seconds >= estimates[2] else 1
    return dict(selected=[r['arm'] for r in candidates[:count]],
        exploratory_fallback=not bool(eligible), evidence=screen,
        eligible_arms=[r['arm'] for r in eligible], remaining_seconds=remaining_seconds,
        estimated_confirmation_seconds=estimates,
        full_confirmation_budget_expected=remaining_seconds >= estimates[count])


def prepare():
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Adaptive phase is already registered')
    # Kernel lock remains authoritative across PID namespaces. Do not infer a
    # stopped driver merely from a stale status file or missing sandbox PID.
    with (OLD / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = read(OLD / 'protocol.json')
        for key in ('source_hashes', 'inputs', 'imported_sources'):
            for path, digest in prior[key].items():
                assert sha(ROOT / path) == digest, path
        for name, digest in prior['local_input_hashes'].items():
            assert sha(OLD / name) == digest, name
        rows = read(OLD / 'observations.json')
        for arm in [base.BASE] + base.SCREEN:
            selected = [r for r in rows if (r['variant'], r['seed'], r['budget'], r['num_samples'])
                        == (arm, 4817, 4096, 4096)]
            assert len(selected) == 2 and {r['eval_seed'] for r in selected} == {2817, 3817}, arm
        if base.occupied():
            raise RuntimeError('GPU still occupied; no active job is stopped by this tool')
        for name in prior['local_input_hashes']:
            save(ART / name, read(OLD / name))
        variants = read(ART / 'variants.json')
        variants[COMBO] = copy.deepcopy(variants['size_residual'])
        variants[COMBO]['__config'][base.P + 'matching_weights'] = dict(
            position=1., heading=2., length=1., width=1., velocity=2.)
        save(ART / 'variants.json', variants)
        runtime = measured_runtime()
        save(ART / 'runtime_budget.json', runtime)
        imports = dict(prior['imported_sources'])
        for name in ('protocol.json', 'observations.json', 'status.json', 'recovery_ledger.json'):
            path = OLD / name
            imports[str(path.relative_to(ROOT))] = sha(path)
        imports.update(runtime['source_hashes'])
        inputs = list(prior['local_input_hashes']) + ['runtime_budget.json']
        plan = copy.deepcopy(prior)
        plan.update(registered_utc=now(), source_hashes=sources(),
            parent_protocol_sha256=sha(OLD / 'protocol.json'), imported_sources=imports,
            local_input_hashes={name: sha(ART / name) for name in inputs},
            adaptive_reason='Completed size residual and matching2 show complementary focus gains. W&B rankings reverse at later epochs; use measured runtime to confirm up to2 candidates beyond the early screen.',
            source_change='No production/model/data/loss change since recovery phase. New combination is only a registered variant override; supervisor policy extends screening and candidate count.',
            added_screen=COMBO, source_control=base.BASE,
            direct_control='Retained chunked roadK8 lineage, extended at the same boundaries as candidates.',
            selection='Same width/focus/onroad/count screen guards; top2 only if measured conservative runtime leaves15min reserve for complete confirmation; otherwise top1. No winner picked by collision.',
            imported_observations='Completed cells retained once, not new independent repeats. Any terminated partial old stage remains excluded.',
            full128epoch_verified=False)
        save(ART / 'protocol.json', plan)
        save(ART / 'observations.json', rows)
        save(ART / 'recovery_ledger.json', [])
        base.refresh()
        base.status('REGISTERED_ADAPTIVE_LONG')


def execute():
    lock = (ART / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def halt(*unused):
        base.STOP = True
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    base.frozen()
    if recovery.run(COMBO, 4817, 4096) is None:
        return
    selection = choose_long_candidates(base.refresh(), base.DEADLINE - time.time(),
                                       read(ART / 'runtime_budget.json'))
    save(ART / 'long_selection.json', dict(at_utc=now(), wandb=base.wandb_context(), **selection))
    selected = selection['selected']
    for step, previous in ((12288, 4096), (16384, 12288)):
        for arm in [base.BASE] + selected:
            if recovery.run(arm, 4817, step, resume=base.checkpoint(arm, 4817, previous),
                            resume_step=previous) is None:
                return
    comparisons = base.refresh()
    final = [r for r in comparisons if r['seed'] == 4817 and r['budget'] == 16384
             and r['scenes'] == 4096 and r['arm'] in selected]
    confirmed = [r for r in final if r['eligible'] and any(
        v['arm'] == r['arm'] and v['seed'] == 4817 and v['budget'] == 12288
        and v['scenes'] == 4096 and v['eligible'] for v in comparisons)]
    best = min(confirmed or final, key=lambda r: r['R_focus'])['arm']
    save(ART / 'second_seed_selection.json', dict(at_utc=now(), arm=best,
        both_budgets_eligible=any(r['arm'] == best for r in confirmed),
        evidence=final, wandb=base.wandb_context()))
    for arm in [base.BASE, best]:
        if recovery.run(arm, 5817, 6826) is None:
            return
    for arm in [base.BASE, best]:
        if recovery.run(arm, 4817, 6826, resume=base.checkpoint(arm, 4817, 4096),
                        resume_step=4096) is None:
            return
    for arm in [base.BASE, best]:
        if recovery.run(arm, 4817, 16384, scenes=50000,
                        resume=base.checkpoint(arm, 4817, 16384), evaluation=True) is None:
            return
    proof = [r for r in base.refresh() if r['arm'] == best and
        ((r['seed'] == 4817 and r['budget'] in (12288, 16384)) or r['budget'] == 6826)]
    assert len(proof) == 5, 'Need both long budgets, both same-budget TRAIN seeds and full50K'
    save(ART / 'recommendation.json', dict(generated_utc=now(), candidate=best,
        promoted=all(r['eligible'] for r in proof), evidence=proof,
        full128epoch_verified=False, default='init_diffusion_657_n20_road_k8_full',
        wandb=base.wandb_context()))
    base.status('PLANNED_EXPERIMENTS_COMPLETE', full128epoch_verified=False,
                observations=len(read(ART / 'observations.json')), monitoring_until=base.END)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute'])
    args = parser.parse_args()
    configure()
    globals()[args.mode]()
