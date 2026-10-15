"""Paired initialization study after the owned periodic geometry experiments."""
from pathlib import Path
import copy
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_periodic_edge_training_20261015 as engine


def compare(rows, budget):
    paired = []
    controls = {r['eval_seed']: r for r in rows if r['variant'] == 'control' and r['budget'] == budget}
    candidates = [r for r in rows if r['variant'] == 'identity' and r['budget'] == budget]
    assert len(controls) == len(candidates) == 2
    for row in candidates:
        control = controls[row['eval_seed']]
        for k in ('initial_map_sha256', 'train_order_sha256', 'eval_manifest_sha256', 'seen_training_examples',
                  'generator_num_parameters', 'seed'):
            assert row[k] == control[k], k
        ah = engine.read(Path(row['metrics_path']).parent / 'feature_histograms.json')
        bh = engine.read(Path(control['metrics_path']).parent / 'feature_histograms.json')
        assert all(ah[k]['matched_gt'] == bh[k]['matched_gt'] for k in engine.JSD)
        paired.append(dict(GEN=row['eval_seed'], ratios={k: row[k] / control[k] for k in engine.JSD}))
    ratios = {k: statistics.mean(r['ratios'][k] for r in paired) for k in engine.JSD}
    return dict(budget=budget, ratios=ratios, paired=paired,
                R_jsd=statistics.mean(ratios.values()),
                R_focus=statistics.mean(ratios[k] for k in ('ang_dev_jsd', 'speed_jsd', 'length_jsd')),
                full128epoch_unproven=True)


def main():
    parent = engine.ART
    engine.ART = engine.BASE / 'heading_identity_training_v1'
    engine.RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/heading_identity_training_v1'
    engine.RUNNER = ROOT / 'tools/runtime_heading_identity_sweep_20261015.py'
    engine.ARMS = {'control': 0, 'identity': 0}  # Both retain the ordinary edge encoder.
    engine.ART.mkdir(parents=True, exist_ok=True)
    engine.RUNS.mkdir(parents=True, exist_ok=True)
    assert not (engine.ART / 'protocol.json').exists()
    assert engine.read(engine.BASE / 'heading_identity_init_proof.json')['status'] == 'PASS'
    predecessor = engine.read(engine.BASE / 'periodic_edge_training_driver_pid.json')['pid']
    frozen = engine.source_hashes()
    for p in (Path(__file__), ROOT / 'tools/heading_identity_initialization_20261015.py'):
        frozen[str(p.relative_to(ROOT))] = engine.hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ('recipe.json', 'eval8192.json'):
        engine.save(engine.ART / name, engine.read(parent / name))
    base = engine.read(parent / 'variants.json')['control']
    engine.save(engine.ART / 'variants.json', {k: copy.deepcopy(base) for k in ('control', 'identity')})
    engine.save(engine.ART / 'protocol.json', dict(registered_utc=engine.now(), source_hashes=frozen,
        predecessor_pid=predecessor, TRAIN=4817, GEN=[2817,3817], NFE=24, steps=[2048,6144],
        objective_architecture_sampler_unchanged=True, inference_and_checkpoint_layout_unchanged=True,
        control_reuse='Completed ordinary-edge periodic-phase control at identical budget/cohort/order.',
        initialization='Only final state-head weight/bias rows2/3 become zero/[1,0]. CPU and runtime non-heading hash proofs.',
        limitation='Only initialization changes. Short curves remain proxies; final quality requires full training.'))
    while engine.alive(predecessor) or engine.occupied():
        engine.save(engine.ART / 'status.json', dict(status='WAITING_FOR_OWNED_PERIODIC_STUDY_AND_GPU'))
        time.sleep(10)
    def verify_sources():
        for name, expected in frozen.items():
            assert engine.hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

    verify_sources()
    previous = engine.read(parent / 'observations.json')
    rows = [r for r in previous if r['variant'] == 'control' and r['budget'] == 2048]
    assert len(rows) == 2
    assert all(r['num_samples'] == 8192 and r['seed'] == 4817 for r in rows)
    candidate, candidate_path = engine.stage('identity', 2048)
    verify_sources()
    rows.extend(candidate)
    engine.save(engine.ART / 'observations.json', rows)
    first = compare(rows, 2048)
    comparisons = [first]
    engine.save(engine.ART / 'comparisons.json', comparisons)
    if first['R_focus'] < .99 and first['ratios']['ang_dev_jsd'] < 1:
        control6144 = [r for r in previous if r['variant'] == 'control' and r['budget'] == 6144]
        if not control6144:
            control2048 = Path(rows[0]['checkpoint'])
            control6144, _ = engine.stage('control', 6144, resume=control2048)
        verify_sources()
        candidate6144, _ = engine.stage('identity', 6144, resume=candidate_path)
        verify_sources()
        rows.extend(control6144 + candidate6144)
        engine.save(engine.ART / 'observations.json', rows)
        comparisons.append(compare(rows, 6144))
        engine.save(engine.ART / 'comparisons.json', comparisons)
    engine.save(engine.ART / 'status.json', dict(status='HEADING_IDENTITY_FINITE_PLAN_COMPLETE',
        completed_utc=engine.now(), next='Assess paired curves and actual initialization audit; do not infer final quality.'))


if __name__ == '__main__':
    main()
