"""Explicit complete50K follow-up for the attribute-improving epoch111 EMA."""
from pathlib import Path
import hashlib
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_late_checkpoint_midpoint_20261015 as study
from tools.run_wandb_cc_scratch_20261013 import alive


def main():
    original = study.ART
    rows = study.read(original / 'comparisons.json')
    evidence = [r for r in rows if r['snapshot'] == 'roadK8_e111']
    assert {r['seed'] for r in evidence} == {2817, 3817}
    assert all(r['ratios']['speed_jsd'] < 1 and r['ratios']['length_jsd'] < 1
               and r['ratios']['width_jsd'] < 1 for r in evidence)
    study.ART = study.BASE / 'late_checkpoint_full_v1'
    study.RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/late_checkpoint_full_v1'
    study.ART.mkdir(parents=True, exist_ok=True)
    study.RUNS.mkdir(parents=True, exist_ok=True)
    assert not (study.ART / 'protocol.json').exists()
    predecessor = study.read(study.BASE / 'native_midpoint_seed0_driver_pid.json')['pid']
    frozen = study.hashes()
    study.save(study.ART / 'sampler.json', study.read(original / 'sampler.json'))
    study.save(study.ART / 'protocol.json', dict(registered_utc=study.now(), source_hashes=frozen,
        snapshot='roadK8_e111', evidence=evidence, predecessor_pid=predecessor, scenes=50000,
        GEN=[2817, 3817], NFE=24,
        decision='Agent follow-up outside the angular screening gate: speed/length/width improved '
                 'in both GENs while angle changed in opposite directions. Earlier grid study showed '
                 'small-cohort angular ranking can reverse; measure exact50K at low cost instead of promoting.',
        control='Verified fresh full50K power1.2 final127 controls; identical code/config/manifest/noise required.',
        limitation='Existing checkpoint selection only; no inferred recipe improvement or seed selection.'))
    while alive(predecessor) or study.occupied():
        study.save(study.ART / 'status.json', dict(status='WAITING_FOR_NATIVE_SEED0_AND_GPU'))
        time.sleep(10)
    controls = study.read(study.BASE / 'midpoint_grid_full_v1/observations.json')
    results, comparisons = [], []
    fixed = study.read(ROOT / 'configs/metrics/iu3fa4gx_full50000.json')['agent_metrics']
    target = study.read(ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json')['targets']
    for seed in (2817, 3817):
        assert study.hashes() == frozen
        row = study.evaluate('roadK8', 'roadK8_e111', 50000, seed)
        control = next(r for r in controls if r['power'] == 1.2 and r['seed'] == seed)
        comparison = study.compare(row, control)
        row['target_pass'] = {k: row['metrics'][k] < v for k, v in target.items()}
        row['R_jsd_fixed657'] = statistics.mean(row['metrics'][k] / fixed[k] for k in study.JSD)
        row['R_attr_fixed657'] = statistics.mean(row['metrics'][k] / fixed[k]
                                                for k in ('length_jsd', 'width_jsd', 'speed_jsd'))
        results.append(row)
        comparisons.append(comparison)
        study.save(study.ART / 'observations.json', results)
        study.save(study.ART / 'comparisons.json', comparisons)
    study.save(study.ART / 'status.json', dict(status='LATE_EMA_FULL50K_COMPLETE', completed_utc=study.now(),
        next='Review all seven thresholds separately per GEN before any checkpoint recommendation.'))


if __name__ == '__main__':
    main()
