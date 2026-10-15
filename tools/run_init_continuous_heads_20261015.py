"""Queue a head-factorization comparison after the registered geometry search."""
from __future__ import annotations
import copy
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_init_continuous_geometry_20261014 as geometry
search = geometry.search
ART = ROOT / 'output/reports/init_diffusion_continuous_20261014/heads_v2'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/heads_v2'
STOP = False


def halt(*unused):
    global STOP
    STOP = True
    search.engine.STOP = True


def sources():
    return {**geometry.sources(), 'tools/run_init_continuous_heads_20261015.py': search.sha(__file__)}


def main():
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    geometry.configure()
    parent = geometry.ART
    parent_pid = search.read(parent.parent / 'local_driver.json')['pid']
    ART.mkdir(parents=True, exist_ok=True)
    while search.engine.shared.alive(parent_pid):
        if STOP:
            return
        search.save(ART / 'queue_status.json', dict(status='WAITING_FOR_GEOMETRY', parent_pid=parent_pid))
        time.sleep(10)
    proof = search.read(parent.parent / 'head_factorization_preflight.json')
    assert proof['same_initial_G1_tensors'] and proof['same_post_init_RNG']
    protocol = search.read(parent / 'protocol.json')
    changed = [n for n, value in protocol['source_hashes'].items() if search.sha(ROOT / n) != value]
    assert changed == ['tools/overnight_init_diffusion_sweep.py'], changed
    helper_proof = search.read(parent.parent / 'runner_head_options_fix/proof.json')
    assert helper_proof['allowlist_only_change']
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Head phase already registered')
    base = copy.deepcopy(search.read(parent / 'variants.json')['roadK8'])
    variants = {'roadK8': base, 'heads4': copy.deepcopy(base)}
    variants['heads4']['__config'][search.P + 'denoiser_num_heads'] = 4
    search.save(ART / 'variants.json', variants)
    for name in ('recipe.json', 'full50000.json'):
        search.save(ART / name, search.read(parent / name))
    plan = copy.deepcopy(protocol)
    plan.update(phase='HEAD_FACTORIZATION_V2_ALLOWLIST_REPAIR', source_hashes=sources(),
                local_input_hashes={n: search.sha(ART / n) for n in ('variants.json', 'recipe.json', 'full50000.json')},
                screen=['roadK8', 'heads4'],
                control_reuse='Completed2048/6144 geometry matching4 control, same source/data/recipe hashes; no fresh control retraining.',
                initialization_proof=proof, registered_utc=search.now(),
                runner_allowlist_amendment=helper_proof,
                excluded_phase='heads_v1 failed before model initialization; no ranking results.')
    search.save(ART / 'protocol.json', plan)
    controls = [r for r in search.read(parent / 'observations.json')
                if r['variant'] == 'roadK8' and r['budget'] in (2048, 6144)]
    assert len(controls) == 4
    search.save(ART / 'observations.json', controls)
    search.ART, search.RUNS = ART, RUNS
    search.sources = sources
    search.configure()
    rows = search.engine.run('heads4', 4817, 2048)
    if rows is None:
        return
    assert all(r['initial_generator_sha256'] == controls[0]['initial_generator_sha256'] for r in rows)
    if search.engine.run('heads4', 4817, 6144,
                         resume=search.engine.checkpoint('heads4', 4817, 2048),
                         resume_step=2048) is None:
        return
    search.engine.status('HEADS_FINITE_PLAN_COMPLETE',
                         next='Review matched6144 results before remote deployment; no automatic promotion.')


if __name__ == '__main__':
    main()
