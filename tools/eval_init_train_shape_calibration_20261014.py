"""Bounded TRAIN-only size-transport pilot after all registered GPU work.

Never fit on evaluation labels. The selected training checkpoint is unchanged;
report results explicitly as model plus physical-box postprocessing.
"""
from __future__ import annotations
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import continue_init_seed_reference_20261014 as phase
from tools import score_init_sd_vw_targets_20261014 as targets
search=phase.search;engine=search.engine
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART/'size_calibration'
RUNS=search.RUNS/'size_calibration'
STOP=False


def status(name,**extra):save(ART/'status.json',dict(status=name,at_utc=now(),pid=os.getpid(),**extra))


def register():
    ART.mkdir(parents=True,exist_ok=True)
    assert read(ART/'native_output_parity.json')['status']=='PASS'
    assert read(ART/'unit_tests.json')['status']=='PASS'
    plan=dict(registered_utc=now(),deadline=search.END,
        sources={str(p.relative_to(ROOT)):sha(p) for p in (
            Path(__file__).resolve(),ROOT/'tools/run_init_train_shape_calibration_20261014.py',
            ROOT/'tools/init_train_shape_calibration.py',ROOT/'tests/test_init_train_shape_calibration.py')},
        upstream_protocol_sha256=sha(search.ART/'protocol.json'),
        test_proof_sha256={name:sha(ART/name) for name in ('native_output_parity.json','unit_tests.json')},
        priority='After main training, sampler and long-confirmation processes are terminal; one serial GPU child at a time.',
        checkpoint='Largest verified full50K N20 budget of the preselected long_candidate; no selection by calibrated metrics.',
        fit='8192 original TRAIN calibration scenes, GENERATED TRAIN seed6817, batch128, N20 uniform, no guidance, EMA. Separately select positive finite non-ego type0 vehicles in generated/GT TRAIN data. No VAL labels or metric cache read during fit.',
        transport='Independent length/width log-space monotone piecewise linear quantile mapping,257 knots q=.001..999. Equal source knots average their target knots. Continuous tail extrapolation with slope in[0,4]. Invalid generated sizes, fixed ego and nonvehicles unchanged. No stochastic jitter or VAL optimization.',
        evaluation='Original native metrics; real physical output boxes changed before downstream evaluation. Same4096 scenes/GEN2817/3817 as raw N20, full50K only if bothGEN improve length and width, preserve passed absolute thresholds, and every non-size JSD ratio<=1.03.',
        fit_checkpoint_fence='Checkpoint SHA256, TRAIN manifest hashes, normalizer, sampler, EMA, collection tensors and code hashes saved. Refit on TRAIN for a different checkpoint or sampler.',
        time_budget='Launch only with>=1800seconds left; stop child at deadline-900seconds to reserve report/QA. No effect on previous queue or candidate rankings.',
        caveat='Postprocessing diagnostic, not proof of better learning or128epoch performance. Validation reused for screening; full50K is not an untouched test set.')
    path=ART/'protocol.json'
    if path.exists():
        old=read(path);assert {k:v for k,v in old.items() if k!='registered_utc'}=={k:v for k,v in plan.items() if k!='registered_utc'}
        return old
    save(path,plan);return plan


def alive(path,needle):
    if not path.exists():return False
    pid=read(path)['pid']
    if not engine.shared.alive(pid):return False
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert needle in cmd,(pid,cmd)
    return True


def child(command,folder,marker):
    if STOP or search.remaining()<1000:return False
    assert not engine.occupied(),'GPU already occupied'
    folder.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MPLCONFIGDIR':'/tmp/sim_train_shape_mpl','PYTHONFAULTHANDLER':'1','SIM_SWEEP_DEFER_CYCLIC_GC':'1'}
    with (folder/'console.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(folder/'execution.json',dict(command=command,pid=process.pid,started_utc=now()))
        while process.poll() is None:
            if STOP or search.remaining()<=900:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                save(folder/'excluded.json',dict(reason='STOP_OR_REPORT_TIME_RESERVE',at_utc=now()));return False
            time.sleep(10)
    if process.returncode or marker not in (folder/'console.log').read_text():
        save(folder/'excluded.json',dict(reason='NATIVE_ERROR',returncode=process.returncode,at_utc=now()));return False
    return True


def evaluate(arm,budget,checkpoint,calibration,scenes):
    folder=RUNS/f'{arm}_{budget}_calibrated_N{scenes}'
    paths=ART/f'{arm}_{budget}_checkpoint.json';save(paths,{arm:checkpoint})
    command=[engine.shared.PYTHON,str(ROOT/'tools/run_init_train_shape_calibration_20261014.py'),
        'eval','--calibration',str(calibration),'--map-checkpoint',str(engine.shared.MAP),
        '--batch-size','512','--scenes',str(scenes),'--eval-batch-size','128','--workers','0',
        '--defer-cyclic-gc','--young-gc-interval','0','--allow-multiple-epochs',
        '--seed','4817','--eval-seed','2817','--extra-eval-seed','3817',
        '--normalizer-json',str(engine.shared.NORMALIZER),'--evaluation-weights','ema',
        '--steps',str(budget),'--checkpoint-steps','','--variants','',
        '--evaluation-checkpoints',str(paths),'--output',str(folder),
        '--eval-manifest',str(search.ART/'full50000.json' if scenes==50000 else engine.shared.EVAL_MANIFEST)]
    if scenes==50000:command.append('--require-full-set')
    status('EVALUATING',arm=arm,budget=budget,scenes=scenes)
    if not child(command,folder,'TRAIN_SHAPE_EVALUATION_SUCCESS'):return None
    rows=read(folder/'summary.json')
    assert len(rows)==2 and {r['eval_seed'] for r in rows}=={2817,3817}
    for r in rows:
        metric=read(r['metrics_path'])
        assert r['num_samples']==metric['num_samples']==metric['num_gt_samples']==scenes
        assert metric['reference_mode']=='matched' and r['evaluation_only']
        assert r['budget']==budget and metric['decoder']['sampling_steps']==20
        assert metric['decoder']['ema_num_updates']==budget and r['evaluation_weights']=='ema'
        assert metric['postprocessing']['calibration_sha256']==sha(calibration)
        assert all(math.isfinite(r[k]) and r[k]==metric['agent_metrics'][k] for k in engine.shared.METRICS)
    save(folder/'verified.json',dict(status='PASS',at_utc=now(),summary_sha256=sha(folder/'summary.json')))
    return rows


def main():
    global STOP
    plan=register();phase.configure()
    def stop(*unused):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not STOP and search.remaining()>1800:
        live={name:alive(search.ART/path,needle) for name,path,needle in (
            ('training','seed_reference_phase_pid.json','tools/continue_init_seed_reference_20261014.py'),
            ('sampler','sampler_all_seven_pid.json','tools/eval_init_expanded_samplers_20261014.py'),
            ('long','long_confirmation/pid.json','tools/extend_init_long_confirmation_20261014.py'))}
        if not any(live.values()):break
        status('WAITING_FOR_REGISTERED_GPU_PHASES',live=live,remaining_seconds=search.remaining());time.sleep(40)
    else:status('NO_TIME_OR_STOPPED');return
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    assert sha(search.ART/'protocol.json')==plan['upstream_protocol_sha256']
    assert read(search.ART/'samplers/status.json')['status']=='COMPLETE'
    with (search.ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);engine.frozen()
        arm=read(search.ART/'long_candidate.json')['candidate']
        observations=read(search.ART/'observations.json')
        eligible=[r for r in observations if (r['variant'],r['seed'],r['num_samples'])==(arm,4817,50000)]
        assert eligible,'Need verified full50K raw N20 evaluation'
        budget=max(r['budget'] for r in eligible)
        raw_full=[r for r in eligible if r['budget']==budget]
        raw_quick=[r for r in observations if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,budget,4096)]
        assert len(raw_full)==len(raw_quick)==2
        checkpoint=raw_full[0]['checkpoint'];before=sha(checkpoint)
        assert all(r['checkpoint']==checkpoint for r in raw_full+raw_quick)
        selection=dict(at_utc=now(),arm=arm,budget=budget,checkpoint=checkpoint,checkpoint_sha256=before,
            raw_full_rows=raw_full,raw_quick_rows=raw_quick,remaining_seconds=search.remaining())
        save(ART/'selection.json',selection)
        fit_folder=ART/f'{arm}_{budget}_fit'
        log_folder=RUNS/f'{arm}_{budget}_fit'
        command=[engine.shared.PYTHON,str(ROOT/'tools/run_init_train_shape_calibration_20261014.py'),
            'fit','--checkpoint',checkpoint,'--output',str(fit_folder)]
        status('FITTING_ON_TRAIN',arm=arm,budget=budget)
        if not child(command,log_folder,'TRAIN_SHAPE_COLLECTION_SUCCESS'):status('FIT_FAILED');return
        calibration=fit_folder/'calibration.json'
        assert sha(checkpoint)==before and read(calibration)['provenance']['checkpoint_sha256']==before
        quick=evaluate(arm,budget,checkpoint,calibration,4096)
        if quick is None:status('QUICK_EVALUATION_FAILED');return
        score=search.paired_score(quick,raw_quick)
        checks=[]
        for row in quick:
            base=next(r for r in raw_quick if r['eval_seed']==row['eval_seed'])
            checks.append(dict(GEN=row['eval_seed'],size_improved=all(row[k]<base[k] for k in ('length_jsd','width_jsd')),
                passed_thresholds_preserved=all(not (base[k]<limit and row[k]>=limit) for k,limit in targets.TARGETS.items()),
                other_jsds_preserved=all(row[k]<=1.03*base[k] for k in ('ang_dev_jsd','speed_jsd','lat_dev_jsd','nearest_dist_jsd'))))
        promote=all(all(v for k,v in c.items() if k!='GEN') for c in checks)
        save(ART/'quick_comparison.json',dict(at_utc=now(),score=score,rows=quick,raw_rows=raw_quick,checks=checks,promote_full50K=promote))
        if promote:
            full=evaluate(arm,budget,checkpoint,calibration,50000)
            if full is None:status('FULL_EVALUATION_FAILED');return
            save(ART/'full50000_comparison.json',dict(at_utc=now(),score=search.paired_score(full,raw_full),
                rows=full,raw_rows=raw_full,per_GEN_targets=[dict(GEN=r['eval_seed'],**targets.score(r)) for r in full],
                method='model plus TRAIN-only physical-size postprocessing',full128epoch_verified=False))
        assert sha(checkpoint)==before
        status('COMPLETE',arm=arm,budget=budget,full50K_evaluated=promote,remaining_seconds=search.remaining())


if __name__=='__main__':
    if '--register-only' in sys.argv:register()
    else:main()
