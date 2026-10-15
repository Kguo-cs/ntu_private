"""One standard run.py seed0 full50K check after the owned training trials."""
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save
from tools.run_full_sampler_eval_20261015 import occupied
from tools.run_late_checkpoint_midpoint_20261015 import now


def main():
    base = ROOT / 'output/reports/init_diffusion_continuous_20261014'
    art = base / 'deep_native_seed0'
    folder = ROOT / 'logs/init_diffusion_continuous_20261014/deep_native_seed0'
    art.mkdir(parents=True, exist_ok=True)
    assert not (art / 'protocol.json').exists() and not folder.exists()
    predecessor = read(base / 'heading_identity_training_driver_pid.json')['pid']
    save(art / 'protocol.json', dict(registered_utc=now(), predecessor_pid=predecessor, seed=0,
        checkpoint_epoch=79, checkpoint_step=77120, training_complete=False, NFE=24, scenes=50000,
        experiment='init_diffusion_657_n24_road_k8_match2_d5_midpoint_eval', entrypoint='src/run.py'))
    while alive(predecessor) or occupied():
        save(art / 'status.json', dict(status='WAITING_FOR_OWNED_TRAINING_STUDIES_AND_GPU'))
        time.sleep(10)
    log = folder.parent / (folder.name + '.log')
    cmd = [sys.executable, '-u', 'src/run.py', 'experiment=init_diffusion_657_n24_road_k8_match2_d5_midpoint_eval',
           'optimizer=default', 'seed=0', 'data.num_workers=8', 'hydra.run.dir=' + str(folder)]
    with log.open('x') as out:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, 'WANDB_MODE': 'disabled', 'MPLCONFIGDIR': '/tmp/sim_mpl_continuous'})
        save(art / 'status.json', dict(status='RUNNING', child_pid=child.pid, log=str(log)))
        rc = child.wait()
    if rc or not (folder / 'sd_agent_metrics.json').exists():
        save(art / 'status.json', dict(status='FAILED_REQUIRES_REVIEW', returncode=rc, log=str(log)))
        return
    r = read(folder / 'sd_agent_metrics.json')
    t = read(ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json')
    for k in ('num_samples', 'num_gt_samples', 'full_membership', 'eval_set_sha256',
              'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt'):
        assert r[k] == t[k], k
    assert r['decoder']['ema_num_updates'] == 77120
    assert r['decoder']['sampling_network_evaluations'] == 24 and r['decoder']['sampling_integrator'] == 'midpoint'
    save(art / 'result.json', dict(metrics=r['agent_metrics'], target_pass={k:r['agent_metrics'][k]<v
        for k,v in t['targets'].items()}, report=str(folder / 'sd_agent_metrics.json'), strict_scope_verified=True))
    save(art / 'status.json', dict(status='DEEP_NATIVE_SEED0_FULL50K_COMPLETE', completed_utc=now()))


if __name__ == '__main__':
    main()
