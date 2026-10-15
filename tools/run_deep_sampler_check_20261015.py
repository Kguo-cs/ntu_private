"""Equal24-call sampler check on the genuine ongoing five-layer epoch79."""
from pathlib import Path
import os
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import alive, read, save
from tools.run_full_sampler_eval_20261015 import occupied
from tools.run_late_checkpoint_midpoint_20261015 import hashes, now, JSD

BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'deep_sampler_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/deep_sampler_v1'
SNAP = BASE / 'full_checkpoint_snapshots/deep_sampler_snapshots_20261015/deep_h2v2_e79'
RUNNER = ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'


def evaluate(mode, scenes, seed):
    label = f'deep_e79_{mode}_N{scenes}_GEN{seed}'
    folder = RUNS / label
    assert not folder.exists()
    names = read(BASE / 'full_sampler_50k_v2/full50000_files.json')
    if scenes != 50000:
        names = names[25000:25000 + scenes]
    manifest = ART / (label + '.manifest.json')
    save(manifest, names)
    while occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_GPU', label=label))
        time.sleep(10)
    cmd = [sys.executable, '-u', str(RUNNER), '--checkpoint', str(SNAP / 'model.ckpt'),
           '--source-config', str(SNAP / 'config.yaml'), '--sampler', str(ART / (mode + '.json')),
           '--manifest', str(manifest), '--output', str(folder), '--seed', str(seed),
           '--integration', 'midpoint' if mode == 'midpoint24' else 'native']
    if scenes == 50000:
        cmd.append('--full-set')
    log = RUNS / (label + '.log')
    with log.open('x') as out:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'WANDB_MODE': 'disabled', 'MPLCONFIGDIR': '/tmp/sim_mpl_continuous'})
        save(ART / 'status.json', dict(status='RUNNING', label=label, child_pid=child.pid, log=str(log)))
        rc = child.wait()
    if rc or not (folder / 'verified.json').exists():
        save(ART / 'status.json', dict(status='FAILED_REQUIRES_REVIEW', label=label, returncode=rc, log=str(log)))
        raise RuntimeError('Review strict loading/evaluation failure before retry')
    r = read(folder / 'verified.json')
    assert set(r['observed_denoiser_calls_per_sample_batch']) == {24}
    r.update(model='deep_h2v2_e79', mode=mode, scenes=scenes, seed=seed, training_complete=False,
             checkpoint_origin='Genuine ongoing full-run checkpoint at epoch79 of128; training incomplete.')
    save(folder / 'verified.json', r)
    print('COMPLETE ' + label, flush=True)
    return r


def analyze(rows):
    result = subprocess.run([sys.executable, 'tools/analyze_full_sampler_eval_20261015.py',
                             '--artifacts', str(ART)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return read(ART / 'comparisons.json')


def main():
    ART.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'protocol.json').exists()
    predecessor = read(BASE / 'late_checkpoint_full_driver_pid.json')['pid']
    frozen = hashes()
    save(ART / 'native24.json', dict(sampling_steps=24,
         sampling_time_grid=dict(kind='power', power=1.2, terminal_time=.05)))
    save(ART / 'midpoint24.json', dict(sampling_steps=12,
         sampling_time_grid=dict(kind='power', power=1.2, terminal_time=.1)))
    save(ART / 'protocol.json', dict(registered_utc=now(), snapshot=read(SNAP / 'provenance.json'),
        source_hashes=frozen, predecessor_pid=predecessor, GEN=[2817, 3817], screen_scenes=8192,
        NFE=24, last_network_time=.05,
        decision='Measure midpoint on the five-layer h2/v2 model before judging the architecture from Euler curves.',
        full_gate='Both GEN focus/JSD improve; mean focus<.95, JSD<.98, focus components each<=1.05; density/onroad guards.',
        limitation='Epoch79 is an ongoing checkpoint; no final128epoch or training-causal claim.'))
    while alive(predecessor) or occupied():
        save(ART / 'status.json', dict(status='WAITING_FOR_OWNED_FULL50K_CHECKS_AND_GPU'))
        time.sleep(10)
    rows = []
    for scenes, seeds in [(256, [2817]), (8192, [2817, 3817])]:
        for seed in seeds:
            for mode in ('native24', 'midpoint24'):
                assert hashes() == frozen
                rows.append(evaluate(mode, scenes, seed))
                save(ART / 'observations.json', rows)
    comparisons = analyze(rows)
    paired = [r for r in comparisons['paired_comparisons'] if r['scenes'] == 8192]
    assert len(paired) == 2
    ratios = {k: statistics.mean(r['ratios'][k] for r in paired) for k in JSD}
    eligible = (statistics.mean(ratios[k] for k in ('speed_jsd', 'ang_dev_jsd', 'length_jsd')) < .95
                and statistics.mean(ratios.values()) < .98
                and max(ratios[k] for k in ('speed_jsd', 'ang_dev_jsd', 'length_jsd')) <= 1.05
                and all(r['R_focus'] < 1 and r['R_jsd'] < 1 and .995 <= r['vehicle_count_ratio'] <= 1.005
                        and r['onroad_fraction_change'] >= -.003 for r in paired))
    save(ART / 'selection.json', dict(eligible=eligible, ratios=ratios, full50000_unproven=True))
    if eligible:
        for seed in (2817, 3817):
            for mode in ('native24', 'midpoint24'):
                assert hashes() == frozen
                rows.append(evaluate(mode, 50000, seed))
                save(ART / 'observations.json', rows)
        analyze(rows)
    save(ART / 'status.json', dict(status='DEEP_SAMPLER_FINITE_PLAN_COMPLETE', completed_utc=now(),
         next='Assess actual50K if available; retain ongoing training until final evidence is available.'))


if __name__ == '__main__':
    main()
