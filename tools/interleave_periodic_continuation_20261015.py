"""Hand off only the queue controller; finish its live child, then pair budgets."""
from pathlib import Path
import os
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_periodic_edge_training_20261015 as engine

BASE = engine.BASE
OLD = BASE / 'periodic_edge_12288_ecore_v4'
ART = BASE / 'periodic_edge_12288_interleaved_v5'
RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/periodic_edge_12288_interleaved_v5'


def verify_adopted(stage):
    state = engine.read(stage / 'status.json')
    child = state['child_pid']
    while engine.alive(child):
        time.sleep(10)
    log = Path(state['log'])
    folder = log.with_suffix('')
    assert 'RUNTIME_LOADER_SUCCESS' in log.read_text(), 'Live child did not finish successfully'
    rows = engine.read(folder / 'summary.json')
    assert len(rows) == 2 and {r['eval_seed'] for r in rows} == {2817, 3817}
    for row in rows:
        assert row['seed'] == 4817 and row['num_samples'] == 8192
        assert row['map_frozen'] and row['evaluation_weights'] == 'ema'
        report = engine.read(Path(row['metrics_path']))
        assert report['num_samples'] == report['num_gt_samples'] == 8192
        decoder = report['decoder']
        assert decoder['sampling_steps'] == 12 and decoder['sampling_integrator'] == 'midpoint'
        assert decoder['sampling_network_evaluations'] == 24
        assert decoder['ema_num_updates'] == row['budget']
        assert Path(row['checkpoint']).is_file()
    assert all(r['map_grad_norm_before_clip'] == 0
               for r in engine.read(folder / rows[0]['variant'] / 'train_trace.json'))
    engine.save(folder / 'verified_training.json', dict(status='PASS', adopted_after_queue_handoff=True,
        steps=rows[0]['budget'], frozen_map_verified=True, full_training_complete=False))
    return rows


def main():
    ART.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    assert not (ART / 'protocol.json').exists()
    old_protocol = engine.read(OLD / 'protocol.json')
    frozen = dict(old_protocol['source_hashes'])
    frozen[str(Path(__file__).relative_to(ROOT))] = engine.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def verify():
        for name, digest in frozen.items():
            assert engine.hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    verify()
    parent = engine.read(BASE / 'periodic_edge_12288_ecore_v4_pid.json')['pid']
    assert engine.alive(parent)
    command = Path(f'/proc/{parent}/cmdline').read_bytes().decode().split('\0')
    assert any(x.endswith('recover_periodic_edge_12288_20261015.py') for x in command)
    assert 'periodic_edge_12288_ecore_v4' in command
    paused = False
    try:
        os.kill(parent, signal.SIGSTOP)
        paused = True
        old_state = engine.read(OLD / 'status.json')
        assert old_state['status'] == 'RUNNING_BOUNDED_RECOVERY'
        stage = Path(old_state['stage'])
        child_state = engine.read(stage / 'status.json')
        child = child_state['child_pid']
        if engine.alive(child):
            assert os.getsid(child) == child, 'Training child must have an independent session'
        engine.save(ART / 'protocol.json', dict(registered_utc=engine.now(), source_hashes=frozen,
            original_protocol=str(OLD / 'protocol.json'), old_controller_pid=parent,
            adopted_child=child_state, adopted_stage=str(stage),
            execution_order='7168control/K8,8192control/K8,...,12288control/K8; skip verified completed stages.',
            TRAIN=4817, GEN=[2817,3817], NFE=24, target_steps=12288,
            model_loss_data_optimizer_rng_unchanged=True,
            checkpoint_policy='Continue actual own checkpoints; no restarts from scratch or new exposure claims.',
            reason='Obtain paired equal-budget curves before a remoteGPU becomes free.',
            limitation='Queue order only. Native failures require review; no automatic training retry.'))
        # Terminate only the sleeping queue controller. Its independently
        # sessioned GPU child receives no signal and finishes its registered job.
        os.kill(parent, signal.SIGTERM)
        os.kill(parent, signal.SIGCONT)
        paused = False
    finally:
        if paused:
            os.kill(parent, signal.SIGCONT)
    engine.save(OLD / 'status.json', dict(status='CONTINUATION_HANDOFF_ACTIVE', successor=str(ART),
                                         child_pid=child, previous_state=old_state))
    engine.save(ART / 'status.json', dict(status='WAITING_FOR_UNINTERRUPTED_ADOPTED_CHILD', child_pid=child))
    try:
        adopted = verify_adopted(stage)
        verify()
        source = BASE / 'periodic_edge_long_v1'
        rows = engine.read(OLD / 'observations.json') if (OLD / 'observations.json').exists() else []
        keys = {(r['variant'],r['budget'],r['eval_seed']) for r in rows}
        rows += [r for r in adopted if (r['variant'],r['budget'],r['eval_seed']) not in keys]
        initial = engine.read(source / 'observations.json')
        checkpoints = {}
        for arm in ('control','harmonic8'):
            available = [r for r in initial+rows if r['variant'] == arm]
            latest = max(available, key=lambda r:r['budget'])
            checkpoints[arm] = Path(latest['checkpoint'])
        engine.RUNNER = ROOT / 'tools/runtime_isolated_cache_recovery_20261015.py'
        engine.ARMS = {'control':0,'harmonic8':8}
        for name in ('recipe.json','variants.json','eval8192.json'):
            engine.save(ART / name, engine.read(source / name))
        comparisons = []
        for target in range(7168,12289,1024):
            for arm in ('control','harmonic8'):
                if len([r for r in rows if r['variant']==arm and r['budget']==target]) == 2:
                    continue
                verify()
                label = f'{arm}_to{target}'
                engine.ART, engine.RUNS = ART/'stages'/label, RUNS/'stages'/label
                engine.ART.mkdir(parents=True, exist_ok=True)
                engine.RUNS.mkdir(parents=True, exist_ok=True)
                for name in ('recipe.json','variants.json','eval8192.json'):
                    engine.save(engine.ART/name, engine.read(ART/name))
                engine.save(ART/'status.json',dict(status='RUNNING_INTERLEAVED_CONTINUATION',arm=arm,
                    target_steps=target,stage=str(engine.ART),resume=str(checkpoints[arm])))
                observed, checkpoints[arm] = engine.stage(arm,target,resume=checkpoints[arm])
                verify()
                rows += observed
                engine.save(ART/'observations.json',rows)
            comparisons.append(engine.compare(rows,'harmonic8',target))
            engine.save(ART/'comparisons.json',comparisons)
        final = dict(status='PERIODIC_12288_RECOVERY_COMPLETE',completed_utc=engine.now(),
            successor=str(ART),next='Review paired curves, secondTRAIN and full50K before recommendation.')
        engine.save(ART/'status.json',final)
        engine.save(OLD/'status.json',final)  # Release the already queued next study.
    except Exception:
        failure = dict(status='FAILED_REQUIRES_REVIEW',successor=str(ART),
                       reason='Preserve child outputs/checkpoints; no automatic retry.')
        engine.save(ART/'status.json',failure)
        engine.save(OLD/'status.json',failure)
        raise


if __name__ == '__main__':
    main()
