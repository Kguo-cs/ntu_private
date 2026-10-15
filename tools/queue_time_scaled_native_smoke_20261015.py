"""Run one finite GPU smoke after the registered local sampler study finishes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'time_scaled_native_worker_gpu_smoke_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/time_scaled_native_worker_gpu_smoke_v1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    overlap = '--resume-after-primary-screen' in sys.argv
    ART.mkdir(exist_ok=overlap)
    RUNS.mkdir(exist_ok=overlap)
    if overlap:
        assert (ART / 'scheduling_amendment.json').exists()
        assert not (RUNS / 'smoke2').exists()
    original = json.loads((BASE / 'native_worker_heading_preflight_v1/driver.json').read_text())
    variant = json.loads((BASE / 'shared_heading_velocity_isolated_v4/omega025_T4817_S2048_N8192.variant.json').read_text())
    save(ART / 'variants.json', variant)
    command = original['command'][:]
    replacements = {'--heading-readout': 'time_scaled', '--steps': '2',
                    '--variant-file': str(ART / 'variants.json'), '--variants': 'omega025',
                    '--output': str(RUNS / 'smoke2')}
    for flag, value in replacements.items():
        command[command.index(flag) + 1] = value
    tracked = [ROOT / 'tools/runtime_native_worker_heading_20261015.py',
               ROOT / 'tools/time_scaled_heading_full/model.py']
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    protocol = ART / ('protocol_headroom_amendment.json' if overlap else 'protocol.json')
    save(protocol, dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        command=command, source_hashes=frozen, quality_evidence=False,
        prerequisite=('Primary screen terminal success; at most one CUDA application and8.5GiB free'
                      if overlap else 'deep_final_timegrid_v1 FINITE_STUDY_COMPLETE, including all spawned children exited'),
        failure_rule='Do not retry native failures automatically. Preserve logs/checkpoint.',
        changes='Two fresh GPU steps with time-scaled scalar omega readout; same directomega objective.'))
    while True:
        failure = BASE / 'deep_final_timegrid_v1/failure.json'
        if failure.exists():
            raise RuntimeError('Predecessor study failed; review before allocating GPU')
        if overlap:
            status = json.loads((BASE / 'shared_heading_velocity_isolated_v4/status.json').read_text())
            apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            free = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if status['status'] == 'SHARED_HEADING_VELOCITY_SCREEN_COMPLETE' and len(apps) <= 1 and free >= 8704:
                break
        else:
            status = json.loads((BASE / 'deep_final_timegrid_v1/status.json').read_text())
            if status['status'] == 'FINITE_STUDY_COMPLETE':
                break
        save(ART / 'status.json', dict(status='WAITING_FOR_VERIFIED_GPU_HEADROOM'
                                     if overlap else 'WAITING_FOR_FINITE_SAMPLER_STUDY'))
        time.sleep(30)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in frozen.items())
    with (RUNS / 'console.log').open('x') as handle:
        child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
        save(ART / 'status.json', dict(status='GPU_SMOKE_RUNNING', pid=child.pid))
        rc = child.wait()
    assert rc == 0, f'Smoke exit {rc}; inspect preserved log'
    import torch
    checkpoint = torch.load(RUNS / 'smoke2/omega025/last.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == 2
    extra = checkpoint['state_dict']['encoder.init_decoder._extra_state']
    assert extra['ema']['num_updates'] == 2
    assert extra['denoiser_options']['heading_velocity_parameterization'] == 'time_scaled'
    assert extra['denoiser_options']['heading_velocity_readout'] == 'shared_state_row3'
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values()
               if isinstance(v, torch.Tensor))
    save(ART / 'status.json', dict(status='GPU_SMOKE_PASS', actual_steps=2, ema_updates=2,
         checkpoint=str(RUNS / 'smoke2/omega025/last.ckpt'), quality_evidence=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc)))
        raise
