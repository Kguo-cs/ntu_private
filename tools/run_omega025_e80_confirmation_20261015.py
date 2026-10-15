"""Two independent-GEN full50K confirmations of genuine omega025 weights."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
ART = BASE / 'omega025_e80_confirmation_v1'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/omega025_e80_confirmation_v1'
SNAPSHOT = BASE / 'full_checkpoint_snapshots/omega025_e80_20261015'
RUNNER = ROOT / 'tools/eval_full_checkpoint_midpoint_20261015.py'


def save(path, value):
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(value, indent=2) + '\n')
    partial.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def alive(pid):
    try:
        return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def hashes(include_snapshot=True):
    files = [Path(__file__), RUNNER, ROOT / 'tools/runtime_cached_torch_loader_sweep.py',
        ROOT / 'tools/torch_pickle_class_cache.py', ROOT / 'tools/shared_heading_full/model.py',
        ROOT / 'tools/shared_heading_velocity_20261015.py',
        ROOT / 'tools/fetch_final_depth5_checkpoint_20261015.py',
        ROOT / 'configs/metrics/iu3fa4gx_full50000.json',
        ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json',
        *sorted((ROOT / 'src/smart').rglob('*.py'))]
    if include_snapshot:
        files += [SNAPSHOT / 'config.yaml', SNAPSHOT / 'provenance.json']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare_portable_config(snapshot, art):
    """Relocate storage paths while preserving model and sampler settings."""
    import yaml
    from omegaconf import OmegaConf
    cfg = yaml.safe_load((snapshot / 'original_config.yaml').read_text())
    def relocate(value):
        if isinstance(value, str):
            return value.replace('/home/ke/keguo/sim', str(ROOT))
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value
    cfg = relocate(cfg)
    cfg['paths'].update(root_dir=str(ROOT / 'src'), cache_root=str(ROOT / 'src/waymo_data/full'),
        output_dir=str(art / 'portable_output'), log_dir=str(art / 'portable_output'))
    for key, split in [('train_raw_dir','training_map2_sd'), ('val_raw_dir','scenario_dreamer_val'), ('test_raw_dir','scenario_dreamer_val')]:
        cfg['data'][key] = str(ROOT / 'src/waymo_data/full' / split)
    cfg['data']['scenario_dreamer_train_sample_list'] = str(ROOT / 'src/waymo_data/full/training_map2_sd_files.pkl')
    cfg['data']['scenario_dreamer_eval_set'] = None
    cfg['model']['model_config']['initial_map_checkpoint'] = None
    OmegaConf.register_new_resolver('sim_root', lambda: str(ROOT), replace=True)
    portable = OmegaConf.create(cfg)
    OmegaConf.resolve(portable)
    OmegaConf.save(portable, snapshot / 'config.yaml', resolve=True)
    assert cfg['model']['model_config']['decoder']['init_diffusion']['sampling_integrator'] == 'midpoint'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('e80', 'e96', 'e104', 'final128'), default='e80')
    parser.add_argument('--holdout', action='store_true', help='Predeclared unseenGEN5817/6817 for pinned104; wait for empirical-speed native8.')
    args = parser.parse_args()
    assert not args.holdout or args.profile == 'e104'
    final = args.profile == 'final128'
    selected = args.profile in ('e96', 'e104')
    epoch = {'e80':79,'e96':95,'e104':103,'final128':127}[args.profile]
    global ART, RUNS, SNAPSHOT
    if final:
        ART = BASE / 'omega025_final128_confirmation_v1'
        RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/omega025_final128_confirmation_v1'
        SNAPSHOT = BASE / 'full_checkpoint_snapshots/omega025_final128_20261015'
    elif selected:
        ART = BASE / ('omega025_' + args.profile + ('_holdout_confirmation_v1' if args.holdout else '_confirmation_v1'))
        RUNS = ROOT / 'logs/init_diffusion_continuous_20261014' / ART.name
        SNAPSHOT = BASE / ('full_checkpoint_snapshots/omega025_' + args.profile + '_20261015')
    ART.mkdir(exist_ok=False)
    RUNS.mkdir(exist_ok=False)
    step = (epoch + 1) * 964
    wait_pid = 530964 if args.holdout else 521457 if final else 529103 if args.profile == 'e96' else 522231 if selected else 509864
    stage = args.profile + ('_holdout' if args.holdout else '')
    generation_seeds = [5817, 6817] if args.holdout else [2817, 3817]
    if final:
        from tools.fetch_final_depth5_checkpoint_20261015 import fetch
        frozen_code = hashes(include_snapshot=False)
        save(ART / 'registration.json', dict(registered_utc=now(), profile=stage,
            source_hashes=frozen_code,
            checkpoint='Genuine completed128epoch omega025, epoch127/step123392/EMA123392; never pick an intermediate checkpoint by its GEN results.',
            GEN=[2817,3817], functional_only_GEN=8817, batch_size=128,
            matmul_precision='medium', trainer_precision='32-true', NFE=24,
            integrator='native_configured_midpoint', power=1.2, terminal_time=.1,
            collision_guidance=0, same_full50K=True, wait_for_local_arc_pid=wait_pid,
            acceptance='Both independent GENs must pass all seven absolute SD/VW target thresholds with one unchanged final checkpoint.',
            no_automatic_training_promotion=True, no_automatic_failure_retries=True))
        while True:
            try:
                metadata = fetch('omega025_final128')
            except subprocess.TimeoutExpired:
                save(ART / 'status.json', dict(status='WAITING_FOR_SAME_REMOTE_PROCESS_OBSERVATION', updated_utc=now()))
                time.sleep(30)
                continue
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 255:
                    raise
                save(ART / 'status.json', dict(status='WAITING_FOR_SSH_OBSERVATION', updated_utc=now()))
                time.sleep(30)
                continue
            if metadata['status'] == 'GENUINE_FINAL_CHECKPOINT_READY':
                break
            save(ART / 'status.json', dict(status=metadata['status'], pid=metadata.get('pid'), updated_utc=now()))
            time.sleep(30)
        assert hashes(include_snapshot=False) == frozen_code, 'Evaluation source changed while waiting'
        prepare_portable_config(SNAPSHOT, ART)
    elif selected:
        metadata = json.loads((SNAPSHOT / 'provenance.json').read_text())
        assert metadata['epoch'] == epoch and metadata['global_step'] == metadata['ema_updates'] == step
        assert hashlib.sha256((SNAPSHOT / 'model.ckpt').read_bytes()).hexdigest() == metadata['sha256']
        assert metadata['preserved_report_archive_sha256']
        if args.holdout:
            pinned = json.loads((BASE / 'best_verified_model_20261015.json').read_text())
            assert pinned['checkpoint_sha256'] == metadata['sha256'] and pinned['actual_epoch'] == 103
            assert (SNAPSHOT / 'config.yaml').exists()
        else:
            prepare_portable_config(SNAPSHOT, ART)
    else:
        proof = json.loads((BASE / 'omega025_e80_cpu_preflight_20261015.json').read_text())
        assert proof['status'] == 'OMEGA025_REAL80_NATIVE_CPU_STRICT_LOAD_AND_NATIVE_FULL50K_ALL7_PASS'
    frozen = hashes()
    protocol = dict(registered_utc=now(), profile=stage, checkpoint_epoch=epoch, checkpoint_step=step,
        checkpoint_sha256=json.loads((SNAPSHOT / 'provenance.json').read_text())['sha256'],
        source_hashes=frozen, GEN=generation_seeds, functional_only_GEN=8817,
        batch_size=128, matmul_precision='medium', trainer_precision='32-true',
        sampling=dict(integrator='native_configured_midpoint', intervals=12, NFE=24,
            power=1.2, terminal_time=.1, collision_guidance=0),
        selection=('Genuine final128epoch checkpoint selected before observing its independent-GEN results.' if final else 'Epoch96 has a strongnative attribute andsixJSD score. Compare both fixedGENs to verified104 before selection; never merge checkpoints or cherry-pick GEN.' if args.profile == 'e96' else 'Epoch104 has the lowestnative angle in the completed128epoch run andbetter nativeattrs than final128. Evaluate both fixedGENs before considering selection; never merge checkpoints or cherry-pick GEN.' if selected else 'Snapshot chosen because native80epoch full50K first passes all7; no independent-GEN results observed before registration.'),
        acceptance_gate='Both pre-registered independent GENs pass all seven exact-scope SD/VW thresholds using this single unchanged checkpoint.',
        no_metric_or_weight_tuning=True, no_automatic_failure_retries=True,
        wait_for_live_local_pair_pid=wait_pid,
        limitation=('Actual final128epoch weights.' if final else f'Actual{epoch+1}epoch weights from a completed128epoch training job, not final128weights.' if selected else 'Intermediate80epoch weights, not final128epoch.') + ' One training seed; independent training-seed replication remains unproved. GTego and SMARTmap conditioning differ from SD/VW; comparison is the agreed repository metric benchmark.')
    if args.holdout:
        protocol.update(selection='Pinned104checkpoint fixed before observing anyGEN5817/6817 results; two extra generation seeds test selection on the earlier2817/3817 pair. Never change checkpoint, conditions or thresholds after seeing these results.',
            heldout_generation_seeds=True, previously_used_GEN=[2817,3817],
            no_automatic_winner_replacement=True)
    save(ART / 'protocol.json', protocol)
    while alive(wait_pid):
        save(ART / 'status.json', dict(status='WAITING_FOR_LIVE_NATIVE8_STAGE', pid=wait_pid, updated_utc=now()))
        time.sleep(30)
    pair = json.loads((BASE / ('native_empirical_speed_prefix8_v1' if args.holdout else 'native_arc_prefix8_v1' if final else 'omega025_e104_confirmation_v1' if args.profile == 'e96' else 'omega025_final128_confirmation_v1' if selected else 'native_prefix8_pair_v1') / 'status.json').read_text())
    assert pair['status'] == ('NATIVE_EMPIRICAL_SPEED_PREFIX8_COMPLETE' if args.holdout else 'NATIVE_ARC_PREFIX8_COMPLETE' if final else 'OMEGA025_E104_CONFIRMATION_COMPLETE' if args.profile == 'e96' else 'OMEGA025_FINAL128_CONFIRMATION_COMPLETE' if selected else 'NATIVE8_PAIRED_STUDY_COMPLETE'), 'Review predecessor failure before starting next GPU stage'
    names = json.loads((BASE / 'full_sampler_50k_v2/full50000_files.json').read_text())
    assert len(names) == len(set(names)) == 50000
    target = json.loads((ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    sampler = ART / 'native_sampler.json'
    save(sampler, dict(sampling_steps=12, sampling_time_grid=dict(kind='power', power=1.2, terminal_time=.1)))
    rows = []
    for count, seed in [(256, 8817), *[(50000, seed) for seed in generation_seeds]]:
        assert hashes() == frozen, 'Source changed; preserve and review, no retry'
        while subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True).strip():
            save(ART / 'status.json', dict(status='WAITING_FOR_FREE_LOCAL_GPU', updated_utc=now()))
            time.sleep(30)
        label = f'omega025_{stage}_nativeMP24_N{count}_GEN{seed}'
        manifest = ART / (label + '.manifest.json')
        save(manifest, names if count == 50000 else names[35000:35256])
        output = RUNS / label
        command = ['taskset', '-c', '16-31', sys.executable, '-u', str(RUNNER),
            '--checkpoint', str(SNAPSHOT / 'model.ckpt'), '--source-config', str(SNAPSHOT / 'config.yaml'),
            '--sampler', str(sampler), '--manifest', str(manifest), '--output', str(output),
            '--integration', 'native', '--batch-size', '128', '--eval-matmul-precision', 'medium',
            '--seed', str(seed)]
        if count == 50000:
            command += ['--full-set', '--r-reference', str(ROOT / 'configs/metrics/iu3fa4gx_full50000.json')]
        log = RUNS / (label + '.log')
        with log.open('x') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                WANDB_MODE='disabled', MPLCONFIGDIR='/tmp/sim_mpl_continuous'))
            save(ART / 'status.json', dict(status='RUNNING', label=label, pid=child.pid,
                log=str(log), updated_utc=now()))
            code = child.wait()
        assert code == 0, f'{label}: exit{code}; preserve failure without automatic retry'
        assert hashes() == frozen
        row = json.loads((output / 'verified.json').read_text())
        assert row['checkpoint_step'] == step and row['effective_integrator'] == 'midpoint'
        assert report_step_matches(output, step)
        assert set(row['observed_denoiser_calls_per_sample_batch']) == {24}
        report = json.loads((output / 'sd_agent_metrics.json').read_text())
        if count == 50000:
            scope = ('num_samples', 'num_gt_samples', 'full_membership', 'map_source', 'eval_set_sha256',
                'official_source_sha256', 'reference_source_manifest_sha256', 'feature_counts_gt')
            assert all(report[key] == target[key] for key in scope)
            assert report['agent_metrics']['R_scope_matches_reference'] == 1
            assert all(key in report['agent_metrics'] for key in ['R', 'R_jsd', 'R_attr'])
            row['target_ratios'] = {k: row['metrics'][k] / value for k, value in target['targets'].items()}
            row['passed_targets'] = [k for k, value in row['target_ratios'].items() if value < 1]
        rows.append(row)
        save(ART / 'observations.json', rows)
    full = rows[1:]
    comparison = None
    reference_family = 'omega025_e104_confirmation_v1' if args.profile == 'e96' else 'omega025_final128_confirmation_v1'
    if selected and not args.holdout:
        previous = {r['seed']: r for r in json.loads((BASE / reference_family / 'observations.json').read_text()) if r['scenes'] == 50000}
        assert all(r['sampling_source_sha256_per_batch'] == previous[r['seed']]['sampling_source_sha256_per_batch'] for r in full), 'Review different initialsource draws; do not claim paired comparison'
        comparison = [dict(seed=r['seed'],source_noise_exactly_matched=True,
            metric_ratios={k:r['metrics'][k]/previous[r['seed']]['metrics'][k] for k in target['targets']},
            R_jsd_ratio=r['metrics']['R_jsd']/previous[r['seed']]['metrics']['R_jsd'],
            R_attr_ratio=r['metrics']['R_attr']/previous[r['seed']]['metrics']['R_attr']) for r in full]
    save(ART / 'decision.json', dict(completed_utc=now(), same_checkpoint_two_independent_GEN_full50K=True,
        both_GEN_pass_all7=all(len(r['passed_targets']) == 7 for r in full),
        final128epoch_result=final, selected_epoch=epoch,
        comparison_reference_family=reference_family if selected and not args.holdout else None,
        comparison_to_reference=comparison,no_automatic_winner_replacement=True,
        heldout_generation_seeds=args.holdout,
        independent_training_seed_unproven=True,
        results=[dict(seed=r['seed'], metrics=r['metrics'], passed_targets=r['passed_targets']) for r in full]))
    save(ART / 'status.json', dict(status='OMEGA025_E104_HOLDOUT_CONFIRMATION_COMPLETE' if args.holdout else 'OMEGA025_FINAL128_CONFIRMATION_COMPLETE' if final else 'OMEGA025_E96_CONFIRMATION_COMPLETE' if args.profile == 'e96' else 'OMEGA025_E104_CONFIRMATION_COMPLETE' if selected else 'OMEGA025_E80_CONFIRMATION_COMPLETE', completed_utc=now()))


def report_step_matches(output, step):
    report = json.loads((output / 'sd_agent_metrics.json').read_text())
    return report['decoder']['ema_num_updates'] == step


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if ART.exists():
            save(ART / 'failure.json', dict(status='STOPPED_REQUIRES_REVIEW', error=str(exc), updated_utc=now()))
        raise
