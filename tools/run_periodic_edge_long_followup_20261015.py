"""Explicit6144-step follow-up of the strong-angle K8 short-budget result."""
from pathlib import Path
import copy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_periodic_edge_training_20261015 as engine


def main():
    parent = engine.ART
    assert engine.read(parent / 'status.json')['status'] == 'PERIODIC_TRAINING_FINITE_PLAN_COMPLETE'
    result = next(r for r in engine.read(parent / 'comparisons.json') if r['arm'] == 'harmonic8')
    assert result['ratios']['ang_dev_jsd'] < .5 and result['R_focus'] < .9
    assert result['ratios']['length_jsd'] < 1 and result['ratios']['width_jsd'] < 1
    engine.ART = engine.BASE / 'periodic_edge_long_v1'
    engine.RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/periodic_edge_long_v1'
    engine.ARMS = {'control': 0, 'harmonic8': 8}
    engine.ART.mkdir(parents=True, exist_ok=True)
    engine.RUNS.mkdir(parents=True, exist_ok=True)
    assert not (engine.ART / 'protocol.json').exists()
    for name in ('recipe.json', 'eval8192.json'):
        engine.save(engine.ART / name, engine.read(parent / name))
    variants = engine.read(parent / 'variants.json')
    engine.save(engine.ART / 'variants.json', {k: copy.deepcopy(variants[k]) for k in engine.ARMS})
    predecessor = engine.read(engine.BASE / 'deep_native_seed0_driver_pid.json')['pid']
    frozen = engine.source_hashes()
    frozen[str(Path(__file__).relative_to(ROOT))] = engine.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine.save(engine.ART / 'protocol.json', dict(registered_utc=engine.now(), evidence=result,
        source_hashes=frozen, predecessor_pid=predecessor, steps=6144, TRAIN=4817, GEN=[2817,3817], NFE=24,
        decision='Explicit follow-up outside automatic speed<=1.10 gate: K8 angle improves~66%, '
                 'both size metrics improve, speed worsens~12%. Measure later equal-budget curves '
                 'before choosing any full training recipe.',
        continuation='Restore2048 optimizer/scheduler/EMA/RNG and full training order.',
        control_policy='Reuse verified6144 ordinary-edge control if identity study produced it; otherwise train it once.',
        limitation='No full128epoch quality claim or automatic server deployment.'))
    while engine.alive(predecessor) or engine.occupied():
        engine.save(engine.ART / 'status.json', dict(status='WAITING_FOR_OWNED_STUDIES_AND_GPU'))
        time.sleep(10)
    def verify_sources():
        for k, v in frozen.items():
            assert engine.hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v, k
    verify_sources()
    originals = engine.read(parent / 'observations.json')
    rows = []
    other = engine.BASE / 'heading_identity_training_v1/observations.json'
    previous = engine.read(other) if other.exists() else []
    controls = [r for r in previous if r['variant'] == 'control' and r['budget'] == 6144]
    if controls:
        assert len(controls) == 2 and {r['eval_seed'] for r in controls} == {2817,3817}
        assert all(r['seed'] == 4817 and r['num_samples'] == 8192 for r in controls)
        rows.extend(controls)
        engine.save(engine.ART / 'control_reuse.json', dict(source=str(other), rows=controls))
    else:
        checkpoint = Path(next(r['checkpoint'] for r in originals if r['variant'] == 'control'))
        controls, _ = engine.stage('control', 6144, resume=checkpoint)
        rows.extend(controls)
    verify_sources()
    checkpoint = Path(next(r['checkpoint'] for r in originals if r['variant'] == 'harmonic8'))
    candidate, _ = engine.stage('harmonic8', 6144, resume=checkpoint)
    rows.extend(candidate)
    verify_sources()
    engine.save(engine.ART / 'observations.json', rows)
    comparison = engine.compare(rows, 'harmonic8', 6144)
    engine.save(engine.ART / 'comparison.json', comparison)
    engine.save(engine.ART / 'status.json', dict(status='PERIODIC_6144_PAIRED_CHECK_COMPLETE', completed_utc=engine.now(),
        next='Review late speed/shape tradeoff and choose second TRAIN/full50K if warranted; do not infer final quality.'))


if __name__ == '__main__':
    main()
