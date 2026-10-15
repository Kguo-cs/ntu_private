"""Fresh paired denoiser-neighborhood search for the ongoing user goal."""
from __future__ import annotations
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_init_expanded_20261014 as search

ART = ROOT / 'output/reports/init_diffusion_continuous_20261014/geometry_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/geometry_v1'
ARMS = ['map2a_k12_r40', 'map2a_k16_r20', 'a2a_r20']
ORIGINAL_SOURCES = search.sources


def sources():
    values = ORIGINAL_SOURCES()
    for name in ('tools/run_init_continuous_geometry_20261014.py',
                 'src/smart/modules/map_decoder.py', 'src/smart/utils/preprocess.py'):
        values[name] = search.sha(ROOT / name)
    return values


def configure():
    search.ART, search.RUNS = ART, RUNS
    # No user deadline. The finite stages below remain interruptible by SIGTERM.
    search.END = '2099-01-01T00:00:00+08:00'
    search.DEADLINE = 4070880000.0
    search.sources = sources
    search.configure()


def prepare():
    if (ART / 'protocol.json').exists():
        raise RuntimeError('Phase already registered; do not overwrite it')
    prior = ROOT / 'output/reports/init_diffusion_expanded_20261014'
    base = copy.deepcopy(search.read(prior / 'variants.json')['match4'])
    variants = {'roadK8': base}
    deltas = {
        'map2a_k12_r40': {'map2agent_max_num_neighbors': 12, 'map2agent_radius': 40.},
        'map2a_k16_r20': {'map2agent_max_num_neighbors': 16, 'map2agent_radius': 20.},
        'a2a_r20': {'a2a_radius': 20.},
    }
    for arm, changes in deltas.items():
        variants[arm] = copy.deepcopy(base)
        variants[arm]['__config'].update({search.P + k: v for k, v in changes.items()})
    search.save(ART / 'variants.json', variants)
    for filename in ('recipe.json', 'full50000.json'):
        search.save(ART / filename, search.read(prior / filename))
    protocol = copy.deepcopy(search.read(prior / 'protocol.json'))
    protocol.update(phase='CONTINUOUS_GEOMETRY_V1', registered_utc=search.now(),
                    deadline=None, user_scope='Continue until user requests stop',
                    source_hashes=sources(), control='matching4; roadK8 is only its internal arm name',
                    screen=['roadK8'] + ARMS, screen_steps=2048,
                    midpoint_steps=6144, longer_steps=None,
                    second_TRAIN=None, inference='Train/eval N20 without guidance; N24 assessed separately.',
                    selection='Advance best R_focus to6144 as exploratory only. Require a second TRAIN and full50K before promoting.',
                    adaptive_reason='Remote experiments cover depth and LR schedule. Test denoiser neighborhood locality without changing encoder weights, dimensions, loss or parameter count.')
    protocol['local_input_hashes'] = {n: search.sha(ART / n)
                                    for n in ('variants.json', 'recipe.json', 'full50000.json')}
    search.save(ART / 'protocol.json', protocol)
    search.save(ART / 'observations.json', [])


def execute():
    import fcntl
    with (ART / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm in ['roadK8'] + ARMS:
            if search.engine.run(arm, 4817, 2048) is None:
                return
        comparisons = search.engine.refresh()
        winner = min(comparisons, key=lambda r: r['R_focus'])['arm']
        search.save(ART / 'midpoint_selection.json', dict(
            selected=winner, evidence=comparisons, full_training_verified=False,
            caveat='Single TRAIN,4096 scenes; geometry shortcuts can reverse at later epochs.'))
        for arm in ('roadK8', winner):
            if search.engine.run(arm, 4817, 6144,
                                 resume=search.engine.checkpoint(arm, 4817, 2048),
                                 resume_step=2048) is None:
                return
        search.engine.status('GEOMETRY_FINITE_PLAN_COMPLETE',
                             next='Agent reviews midpoint and chooses full50K/second TRAIN; ongoing goal remains active.')


if __name__ == '__main__':
    configure()
    prepare()
    execute()
