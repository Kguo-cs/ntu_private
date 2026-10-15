"""Finite paired objective screen, queued after the owned longer K8 trial."""
from pathlib import Path
import copy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_periodic_edge_training_20261015 as engine


def main():
    art = engine.BASE / 'shared_heading_velocity_v1'
    runs = ROOT / 'logs/init_diffusion_continuous_20261014/shared_heading_velocity_v1'
    art.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    assert not (art / 'protocol.json').exists()
    assert engine.read(engine.BASE / 'shared_heading_velocity_proof_20261015.json')['status'] == 'PASS'
    predecessor = engine.BASE / 'periodic_edge_12288_ecore_v4'
    engine.ART, engine.RUNS = art, runs
    engine.RUNNER = ROOT / 'tools/shared_heading_velocity_20261015.py'
    engine.ARMS = {'omega025': 0, 'omega10': 0}
    source = engine.BASE / 'periodic_edge_long_v1'
    for name in ('recipe.json', 'eval8192.json'):
        engine.save(art / name, engine.read(source / name))
    base = engine.read(source / 'variants.json')['control']
    variants = {}
    for arm, weight in [('omega025', .025), ('omega10', .1)]:
        variant = copy.deepcopy(base)
        variant['__config'].update({engine.P + 'heading_objective': 'angular_velocity',
                                   engine.P + 'heading_x0_loss': 'vector_mse'})
        variant['heading'] = dict(kind='mse', time_power=0, weight=weight, scale='none')
        variants[arm] = variant
    engine.save(art / 'variants.json', variants)
    frozen = engine.source_hashes()
    for name in ('tools/shared_heading_velocity_20261015.py',
                 'tools/runtime_isolated_cache_recovery_20261015.py',
                 'tools/runtime_tensor_cache_recovery_20261015.py',
                 'tools/runtime_periodic_edge_sweep_20261015.py',
                 'tools/tensor_cache_numpy_worker_20261015.py',
                 str(Path(__file__).relative_to(ROOT))):
        frozen[name] = engine.hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    engine.save(art / 'protocol.json', dict(registered_utc=engine.now(), source_hashes=frozen,
        predecessor=str(predecessor), TRAIN=4817, GEN=[2817, 3817], NFE=24,
        from_scratch=True, fixed_map=True, parameters=8083978, heading_weights=[.025, .1],
        heading_objective='Native direct angular-velocity MSE, shared existing state-head row3.',
        control_reuse='Ordinary-edge same-seed/order/weights 2048 and 6144-step controls.',
        budgets=[16, 2048, 6144], screen_scenes=8192,
        continuation_gate='Both GEN angle ratios<1, mean angle<.95, mean speed/length/width<=1.15, '
                          'nearest/lateral<=1.2; same initial tensors, order, exposure and GT histograms.',
        limitation='Finite screening only. A winner still requires another TRAIN seed and complete50K; '
                   'never stop a remote full job from these quick scores.'))
    while True:
        state = engine.read(predecessor / 'status.json')['status']
        if state == 'PERIODIC_12288_RECOVERY_COMPLETE':
            break
        if state.startswith('FAILED'):
            engine.save(art / 'status.json', dict(status='WAITING_FOR_NATIVE_FAILURE_REVIEW',
                                                predecessor_status=state))
            return
        engine.save(art / 'status.json', dict(status='QUEUED_AFTER_OWNED_K8_CONTINUATION'))
        time.sleep(30)

    def verify():
        for name, digest in frozen.items():
            assert engine.hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

    controls = [r for r in engine.read(source / 'observations.json') if r['variant'] == 'control']
    assert {r['budget'] for r in controls} == {6144}
    controls += [r for r in engine.read(engine.BASE / 'periodic_edge_training_v2/observations.json')
                 if r['variant'] == 'control' and r['budget'] == 2048]
    rows, checkpoints, comparisons = list(controls), {}, []
    for arm in engine.ARMS:
        verify()
        engine.stage(arm, 16, scenes=256)
        verify()
        observed, checkpoint = engine.stage(arm, 2048)
        rows.extend(observed)
        checkpoints[arm] = checkpoint
        comparison = engine.compare(rows, arm, 2048)
        comparisons.append(comparison)
        engine.save(art / 'observations.json', rows)
        engine.save(art / 'comparisons.json', comparisons)
    eligible = [r for r in comparisons if
        all(p['ratios']['ang_dev_jsd'] < 1 for p in r['paired']) and
        r['ratios']['ang_dev_jsd'] < .95 and
        max(r['ratios'][k] for k in ('speed_jsd', 'length_jsd', 'width_jsd')) <= 1.15 and
        max(r['ratios'][k] for k in ('nearest_dist_jsd', 'lat_dev_jsd')) <= 1.2]
    engine.save(art / 'selection.json', dict(eligible=eligible, full_training_success_unproven=True))
    if eligible:
        arm = min(eligible, key=lambda r: r['R_focus'])['arm']
        verify()
        observed, _ = engine.stage(arm, 6144, resume=checkpoints[arm])
        rows.extend(observed)
        comparisons.append(engine.compare(rows, arm, 6144))
        engine.save(art / 'observations.json', rows)
        engine.save(art / 'comparisons.json', comparisons)
    verify()
    engine.save(art / 'status.json', dict(status='SHARED_HEADING_VELOCITY_SCREEN_COMPLETE',
        completed_utc=engine.now(), next='Review actual curves and faults; no automatic deployment.'))


if __name__ == '__main__':
    main()
