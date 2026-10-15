"""Run a bounded native8 readout or circular matching comparison."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'native_floor02_prefix8_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/native_floor02_prefix8_v1'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('floor02', 'arc', 'omega05', 'empirical_speed', 'rectified_speed'), default='floor02')
    args = parser.parse_args()
    arc = args.variant == 'arc'
    omega05 = args.variant == 'omega05'
    empirical = args.variant == 'empirical_speed'
    rectified = args.variant == 'rectified_speed'
    source_ablation = empirical or rectified
    global ART, RUNS
    if arc or omega05 or source_ablation:
        ART = BASE / ('native_' + args.variant + '_prefix8_v1')
        RUNS = ROOT / 'logs/init_diffusion_continuous_20261014' / ART.name
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    proof_name = ('rectified_speed_native_cpu_proof_20261015.json' if rectified else 'empirical_speed_native_cpu_proof_20261015.json' if empirical else 'omega05_cpu_preflight_20261015.json' if omega05 else
        'native_circular_matching_cpu_proof_20261015.json' if arc else 'time_scaled_heading_floor02_cpu_proof_20261015.json')
    proof = json.loads((BASE / proof_name).read_text())
    circular_proof = None
    if arc or omega05 or source_ablation:
        assert proof['status'] == ('RECTIFIED_SPEED_NATIVE_FULL_CONFIG_CPU_PASS_NOT_TRAINED' if rectified else 'EMPIRICAL_SPEED_NATIVE_FULL_CONFIG_CPU_PASS_NOT_TRAINED' if empirical else 'OMEGA05_NATIVE_FULL_CONFIG_CPU_PASS_NOT_DEPLOYED' if omega05 else 'NATIVE_CIRCULAR_MATCHING_CPU_PROOF_PASS')
        if rectified:
            roundtrip = json.loads((BASE / 'rectified_speed_full_checkpoint_roundtrip_20261015.json').read_text())
            assert roundtrip['status'] == 'RECTIFIED_SPEED_NATIVE_MODEL_CHECKPOINT_ROUNDTRIP_PASS'
            assert sha(ROOT / 'tools/rectified_speed_full/model.py') == roundtrip['source_factory_sha256']
        circular_proof = json.loads((BASE / 'native_circular_matching_cpu_proof_20261015.json').read_text())
        assert circular_proof['legacy_frozen_initial_tensors_RNG_noise_losses_gradients_exact']
        for name, digest in circular_proof['source_hashes'].items():
            assert sha(ROOT / name) == digest
        regression = json.loads((BASE / 'native_circular_regression_baseline_comparison_20261015.json').read_text())
        assert regression['same_failure_error_ids'] and not regression['additional_failures']
    else:
        assert proof['status'] == 'PASS' and proof['time_floor'] == .2
    for name, digest in proof['source_hashes'].items():
        assert sha(ROOT / name) == digest
    control = json.loads((BASE / 'native_prefix8_pair_v1/observations.json').read_text())[0]
    assert control['arm'] == 'raw025' and control['training_audit']['actual_step'] == 7712
    old_root = Path(json.loads((BASE / 'native_prefix8_pair_v1/protocol.json').read_text())['source_snapshot'])
    old_manifest = json.loads((old_root / 'SOURCE_MANIFEST.json').read_text())
    core = {p: h for p, h in old_manifest.items() if p.startswith('src/smart/') or p == 'src/run0.py'}
    changed_core = {p for p, h in core.items() if sha(ROOT / p) != h}
    allowed_core = {p for p in circular_proof['source_hashes'] if p.startswith('src/smart/')} if circular_proof else set()
    assert changed_core <= allowed_core, 'Unproved native training core changes'
    snapshot = ROOT / '.codex_runs' / ('native_' + args.variant + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    snapshot.mkdir()
    for directory in ('src/smart', 'src/utils', 'src/my_data_process', 'configs', 'tools'):
        shutil.copytree(ROOT / directory, snapshot / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for source in (ROOT / 'src').glob('*.py'):
        shutil.copy2(source, snapshot / 'src' / source.name)
    normalizer = Path('docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json')
    (snapshot / normalizer).parent.mkdir(parents=True)
    shutil.copy2(ROOT / normalizer, snapshot / normalizer)
    if empirical:
        source_dir = Path('docs/experiments/init_diffusion_train_attribute_source_full_20261015')
        shutil.copytree(ROOT / source_dir, snapshot / source_dir)
    for directory in ('logs', 'output', 'videos', 'src/logs', 'src/waymo_data'):
        if (ROOT / directory).exists():
            (snapshot / directory).symlink_to(ROOT / directory, target_is_directory=True)
    manifest = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*') if p.is_file() and not p.is_symlink()}
    save(snapshot / 'SOURCE_MANIFEST.json', manifest)
    wait_pid = 534875 if rectified else 530269 if empirical else 521457 if omega05 else 517664 if arc else 515096
    predecessor = 'omega025_e104_grid_screen_v1' if rectified else 'omega025_e96_confirmation_v1' if empirical else 'native_arc_prefix8_v1' if omega05 else 'native_floor02_prefix8_v1' if arc else 'shape4_e64_sampler_review_v2'
    required_status = 'FIXED104_NFE24_GRID_SCREEN_COMPLETE' if rectified else 'OMEGA025_E96_CONFIRMATION_COMPLETE' if empirical else 'NATIVE_ARC_PREFIX8_COMPLETE' if omega05 else 'NATIVE_FLOOR02_PREFIX8_COMPLETE' if arc else 'SHAPE4_E64_SAMPLER_REVIEW_COMPLETE'
    limits = dict(ang_dev_jsd=.8 if omega05 else .95 if arc else .5, length_jsd=1.05,
        speed_jsd=1.05 if arc or omega05 else 1.1, width_jsd=1.05 if arc or omega05 else 1.1,
        lat_dev_jsd=1.1, nearest_dist_jsd=1.1)
    if source_ablation:
        limits.update(ang_dev_jsd=1.1, length_jsd=1.05, speed_jsd=.85, width_jsd=1.1)
    save(ART / 'protocol.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_snapshot=str(snapshot), source_manifest_sha256=sha(snapshot / 'SOURCE_MANIFEST.json'),
        variant=args.variant, changed_core_files=sorted(changed_core),
        wait_for_pid=wait_pid, required_status=required_status,
        epochs=8, optimizer_schedule_epochs=128, steps=7712, batch=512, workers=8,
        parameters=8083978, NFE=24, scalar_readout_floor=None if arc or omega05 or source_ablation else .2,
        heading_loss_weight=.05 if omega05 else .025,
        heading_matching_geometry='arc' if arc else 'legacy',
        attribute_source_sha256=proof.get('source_sha256') if empirical else None,
        objective=('Only clamp physicalGaussian source speedat0. Same matchingformula/losses, but different bridgepairings/targets. No empiricaltable or extraRNG.' if rectified else 'Only replace Gaussian speed source with unconditional TRAIN empirical CDF; same matching formula/losses, but different bridge pairings andtargets.' if empirical else 'Only heading MSE weight.025 to.05; same readout, source and matching.' if omega05 else 'Same component loss forms and weights; circular source assignment changes per-agent bridge pairings/targets.' if arc else 'Unchanged omega MSE, weight.025, time_power0; all other losses and matching unchanged.'),
        controls='Completed raw025 native8; require identical initial parameters/map, epoch order/LR/exposure and CPU RNG. Arc may change CUDA RNG consumption through graph topology; record it explicitly.',
        early_screen_gate=limits,
        no_automatic_full128_promotion=True, no_automatic_failure_retries=True,
        rationale=('Empiricalspeed native8 improves speed32% andwidth19% butregresses length8% andlateral12%, failingitsgate. Disentangle nonnegative bridge states from a fullTRAIN-CDF prior; keepidenticalgate and neverautomaticallypromote.' if rectified else 'Test whether a nonnegative source with TRAIN stationary/low-speed mass eases speed generation; no eval attributes, no added parameters, no inference-only source swap.' if empirical else 'Separate the early heading-weight effect using the same physicalGPU and initialization. Full.05 is already independently registered based on mature.025/.1 evidence; this short result does not select that deployment.' if omega05 else 'Test TRAIN-calibrated circular assignment without extra model parameters; preserve whole-row source marginals, not per-row targets.' if arc else 'Test a5x instead of20x maximum scalar amplification.')))
    while alive(wait_pid):
        save(ART / 'status.json', dict(status='WAITING_FOR_REGISTERED_LIVE_STUDY', pid=wait_pid))
        time.sleep(30)
    assert json.loads((BASE / predecessor / 'status.json').read_text())['status'] == required_status
    while subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip():
        time.sleep(30)
    initial = control['training_audit']
    for phase in ('smoke', 'train8'):
        assert all(sha(snapshot / p) == h for p, h in manifest.items())
        output = RUNS / phase
        audit = ART / phase
        output.mkdir(); audit.mkdir()
        command = ['taskset', '-c', '16-27', sys.executable, '-u',
            str(snapshot / 'tools/run_native_prefix_training_20261015.py'),
            '--gpu-memory-fraction', '.5', '--audit-output', str(audit),
            '--expected-initial-g1-sha256', initial['initial_G1_sha256'],
            '--expected-initial-map-sha256', initial['initial_map_sha256'],
            'experiment=init_diffusion_omega025_rectified_speed_native8' if rectified else 'experiment=init_diffusion_omega025_empirical_speed_native8' if empirical else 'experiment=init_diffusion_omega05_native8' if omega05 else 'experiment=init_diffusion_omega025_arc_native8' if arc else 'experiment=init_diffusion_omega025_timescaled_floor02_native8', 'optimizer=default',
            'hydra.run.dir=' + str(output), 'logger.wandb.name=init_diffusion_native8_' + args.variant + '_20261015',
            'logger.wandb.group=continuous_native8_20261015',
            'logger.wandb.tags=[' + args.variant + ',from_scratch,lr128,full50k]']
        if phase == 'smoke':
            command += ['trainer.max_epochs=1', 'trainer.limit_train_batches=2', 'trainer.limit_val_batches=0']
        with (output / 'console.log').open('x') as handle:
            child = subprocess.Popen(command, cwd=snapshot, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                WANDB_MODE='disabled' if phase == 'smoke' else 'online', MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
            save(ART / 'status.json', dict(status='RUNNING_' + phase.upper(), pid=child.pid, log=str(output / 'console.log')))
            result = child.wait()
        assert result == 0, f'{phase}: exit{result}; preserve without retry'
        recorded = json.loads((audit / 'training_audit.json').read_text())
        expected = 2 if phase == 'smoke' else 7712
        assert recorded['status'] == 'NATIVE_PREFIX_COMPLETE' and recorded['actual_step'] == recorded['EMA_updates'] == expected
        assert recorded['frozen_map_unchanged']
        checkpoints = list(output.glob('bc/*/checkpoints/last.ckpt'))
        assert len(checkpoints) == 1
        gc.disable()
        import torch
        torch.set_num_threads(1)
        checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
        extra = checkpoint['state_dict']['encoder.init_decoder._extra_state']
        assert checkpoint['global_step'] == expected
        if rectified:
            assert extra['denoiser_options']['attribute_source'] == dict(kind='physical_rectified_gaussian_speed', speed=True, shape=False)
            assert extra['denoiser_options']['heading_velocity_readout'] == 'shared_state_row3'
            assert extra['component_loss']['components']['heading']['weight'] == .025
        elif empirical:
            source_options = extra['denoiser_options']['attribute_source']
            assert source_options == dict(source_sha256=proof['source_sha256'], knots=149659, speed=True, shape=False)
            assert extra['denoiser_options']['heading_velocity_readout'] == 'shared_state_row3'
            assert extra['component_loss']['components']['heading']['weight'] == .025
            values = checkpoint['state_dict']['encoder.init_decoder.G1.model.attribute_source.quantile_values']
            assert values.shape == (149659, 3) and torch.isfinite(values).all()
        elif omega05:
            assert extra['denoiser_options']['heading_velocity_readout'] == 'shared_state_row3'
            assert extra['denoiser_options'].get('heading_matching_geometry','legacy') == 'legacy'
            assert extra['component_loss']['components']['heading']['weight'] == .05
        else:
            assert (extra['denoiser_options']['heading_matching_geometry'] == 'arc' if arc else extra['denoiser_options']['heading_velocity_time_floor'] == .2)
        assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values() if torch.is_tensor(t))
        del checkpoint
        if phase == 'train8':
            fields = ('initial_G1_sha256', 'initial_map_sha256', 'epochs', 'final_CPU_RNG_sha256')
            for field in fields + (() if arc or omega05 or source_ablation else ('final_CUDA_RNG_sha256',)):
                assert recorded[field] == initial[field], f'Raw8 control audit differs:{field}'
            report = json.loads((output / 'sd_agent_metrics.json').read_text())
            target = json.loads((ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
            for key in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256',
                        'official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):
                assert report[key] == target[key]
            assert report['decoder']['sampling_network_evaluations'] == 24
            ratios = {key: report['agent_metrics'][key] / control['metrics'][key] for key in target['targets']}
            save(ART / 'decision.json', dict(ratios_to_raw8=ratios,
                registered_early_gate_pass=all(ratios[k] <= v for k, v in limits.items()),
                full128_quality_unproven=True, no_automatic_full_deployment=True,
                actual_native_control_order_LR_initialization_CPU_RNG_verified=True,
                final_CUDA_RNG_matches_control=recorded['final_CUDA_RNG_sha256'] == initial['final_CUDA_RNG_sha256'],
                checkpoint=str(checkpoints[0]), metrics=report['agent_metrics']))
    save(ART / 'status.json', dict(status='NATIVE_RECTIFIED_SPEED_PREFIX8_COMPLETE' if rectified else 'NATIVE_EMPIRICAL_SPEED_PREFIX8_COMPLETE' if empirical else 'NATIVE_OMEGA05_PREFIX8_COMPLETE' if omega05 else 'NATIVE_ARC_PREFIX8_COMPLETE' if arc else 'NATIVE_FLOOR02_PREFIX8_COMPLETE'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc)))
        raise
