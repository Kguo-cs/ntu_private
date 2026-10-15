"""Check the common seed0 through run.py after the owned checkpoint screen."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save
from tools.run_full_sampler_eval_20261015 import occupied
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'production_midpoint_seed0'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/production_midpoint_seed0'


def main():
    ART.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'protocol.json').exists()
    predecessor = read(BASE / 'late_checkpoint_driver_pid.json')['pid']
    save(ART / 'protocol.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
         seed=0, predecessor_pid=predecessor, scenes=50000, NFE=24,
         scope='Common seed0; standard run.py, workers8, complete official membership and fixed R reference.',
         limitation='This complements pre-registered GEN2817/3817; no favorable seed selection for quality claims.'))
    while alive(predecessor) or occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_OWNED_CHECKPOINT_STUDY_AND_GPU'))
        time.sleep(10)
    results = []
    target = read(ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json')
    for label, experiment in [('roadK8', 'init_diffusion_657_n24_road_k8_midpoint_eval'),
                              ('N24match4', 'init_diffusion_657_n24_road_k8_match4_midpoint_eval')]:
        folder = RUNS / label
        assert not folder.exists()
        log = RUNS / (label + '.log')
        cmd = [sys.executable, '-u', 'src/run.py', 'experiment=' + experiment, 'optimizer=default',
               'seed=0', 'data.num_workers=8', 'hydra.run.dir=' + str(folder)]
        with log.open('x') as out:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'WANDB_MODE': 'disabled', 'MPLCONFIGDIR': '/tmp/sim_mpl_continuous'})
            save(ART / 'status.json', dict(status='RUNNING', model=label, child_pid=child.pid, log=str(log)))
            rc = child.wait()
        if rc or not (folder / 'sd_agent_metrics.json').exists():
            save(ART / 'status.json', dict(status='FAILED_REQUIRES_REVIEW', model=label, returncode=rc, log=str(log)))
            return
        report = read(folder / 'sd_agent_metrics.json')
        for k in ('num_samples', 'num_gt_samples', 'full_membership', 'eval_set_sha256',
                  'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt'):
            assert report[k] == target[k], k
        assert report['require_full_set']
        results.append(dict(model=label, seed=0, report=str(folder / 'sd_agent_metrics.json'),
                            metrics=report['agent_metrics'], decoder=report['decoder'], full_scope_verified=True))
        save(ART / 'observations.json', results)
        print('NATIVE_SEED0_COMPLETE ' + label + ' ' + json.dumps(report['agent_metrics']), flush=True)
    save(ART / 'status.json', dict(status='NATIVE_SEED0_FULL50K_COMPLETE',
         completed_utc=datetime.now(timezone.utc).isoformat(), next='Compare seeds and assess remaining angle margin.'))


if __name__ == '__main__':
    main()
