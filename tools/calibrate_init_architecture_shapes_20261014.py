"""Optional TRAIN-only size diagnostic after the architecture full50K pair."""
from pathlib import Path
import fcntl
import os
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_architecture_axes_20261014 as arch
from tools import confirm_init_architecture_budget_20261014 as confirmation
from tools import eval_init_train_shape_calibration_20261014 as calibration
from tools import score_init_sd_vw_targets_20261014 as targets
read,save,sha,now=arch.read,arch.save,arch.sha,arch.now
ART=arch.ART/'size_calibration'
RUNS=arch.RUNS/'size_calibration'
engine=arch.engine

def register():
    ART.mkdir(parents=True,exist_ok=True)
    original=arch.PARENT/'size_calibration'
    plan=dict(registered_utc=now(),deadline=arch.previous.END,source_sha256=sha(__file__),
        upstream_protocol_sha256=sha(arch.ART/'protocol.json'),
        sources={p:sha(ROOT/p) for p in ('tools/run_init_train_shape_calibration_20261014.py',
            'tools/init_train_shape_calibration.py','tests/test_init_train_shape_calibration.py',
            'tools/eval_init_train_shape_calibration_20261014.py')},
        preflight_proofs={p:sha(original/p) for p in ('unit_tests.json','native_output_parity.json')},
        priority='After architecture paired6144 full50K is verified and confirmation process terminal; no simultaneous GPU jobs.',
        checkpoint='Preselected map_architecture candidate at6144; no selection using calibration outcomes.',
        fit='8192 original TRAIN calibration scenes, GENERATED TRAIN seed6817, batch128, EMA/N20/no guidance. Positive finite non-ego type0 vehicles. No evaluation labels.',
        mapping='Previously checked independent257-knot monotone log-quantile length/width maps; deterministic continuous tails, ego/nonvehicles unchanged.',
        evaluation='4096 bothGEN first. Both size JSDs improve perGEN, every previously passed threshold retained, other four JSD ratios<=1.03 -> evaluate complete50K.',
        budget='Launch only with>=1800seconds left; native child stop at22:15 to reserve report/QA.',
        interpretation='Physical-size postprocessing diagnostic, not evidence of improved model learning. Raw outputs remain separately recorded.')
    path=ART/'protocol.json'
    if path.exists():assert {k:v for k,v in read(path).items() if k!='registered_utc'}=={k:v for k,v in plan.items() if k!='registered_utc'}
    else:save(path,plan)
    return plan

def main():
    plan=register()
    pid=read(arch.ART/'budget_handoff_pid.json')['pid']
    while engine.shared.alive(pid) and arch.previous.remaining()>1800:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'tools/confirm_init_architecture_budget_20261014.py' in command
        save(ART/'status.json',dict(status='WAITING_FOR_ARCHITECTURE_CONFIRMATION',at_utc=now(),pid=os.getpid(),upstream_pid=pid))
        time.sleep(30)
    if engine.shared.alive(pid) or arch.previous.remaining()<1800:
        save(ART/'status.json',dict(status='NO_TIME_FOR_COMPLETE_CALIBRATION',at_utc=now()));return
    if read(arch.ART/'status.json')['status']!='ARCHITECTURE_PLAN_COMPLETE':
        save(ART/'status.json',dict(status='UPSTREAM_INCOMPLETE',at_utc=now()));return
    assert sha(__file__)==plan['source_sha256']
    assert sha(arch.ART/'protocol.json')==plan['upstream_protocol_sha256']
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    with (arch.PARENT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not engine.occupied()
        protocol=read(arch.ART/'protocol.json')
        assert protocol['source_hashes']==confirmation.sources()
        engine.ART,engine.RUNS=arch.ART,arch.RUNS
        engine.DEADLINE,engine.END=arch.previous.DEADLINE,arch.previous.END
        calibration.ART,calibration.RUNS=ART,RUNS
        calibration.search.ART=arch.ART
        def stop(*unused):calibration.STOP=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        arm=read(arch.ART/'selection.json')['candidate']
        observations=read(arch.ART/'observations.json')
        raw_quick=[r for r in observations if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,6144,4096)]
        raw_full=[r for r in observations if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,6144,50000)]
        assert len(raw_quick)==len(raw_full)==2
        checkpoint=raw_full[0]['checkpoint'];before=sha(checkpoint)
        assert all(r['checkpoint']==checkpoint for r in raw_quick+raw_full)
        save(ART/'selection.json',dict(at_utc=now(),arm=arm,budget=6144,checkpoint=checkpoint,checkpoint_sha256=before,
            raw_quick_rows=raw_quick,raw_full_rows=raw_full))
        fit=ART/f'{arm}_6144_fit'
        command=[engine.shared.PYTHON,'tools/run_init_train_shape_calibration_20261014.py','fit','--checkpoint',checkpoint,'--output',str(fit)]
        calibration.status('FITTING_ON_TRAIN',arm=arm,budget=6144)
        if not calibration.child(command,RUNS/f'{arm}_6144_fit','TRAIN_SHAPE_COLLECTION_SUCCESS'):
            calibration.status('FIT_FAILED');return
        artifact=fit/'calibration.json'
        assert sha(checkpoint)==before and read(artifact)['provenance']['checkpoint_sha256']==before
        quick=calibration.evaluate(arm,6144,checkpoint,artifact,4096)
        if quick is None:calibration.status('QUICK_EVALUATION_FAILED');return
        checks=[]
        for row in quick:
            base=next(r for r in raw_quick if r['eval_seed']==row['eval_seed'])
            checks.append(dict(GEN=row['eval_seed'],size_improved=all(row[k]<base[k] for k in ('length_jsd','width_jsd')),
                passed_thresholds_preserved=all(not(base[k]<limit and row[k]>=limit) for k,limit in targets.TARGETS.items()),
                other_jsds_preserved=all(row[k]<=1.03*base[k] for k in ('ang_dev_jsd','speed_jsd','lat_dev_jsd','nearest_dist_jsd'))))
        promote=all(all(v for k,v in x.items() if k!='GEN') for x in checks)
        save(ART/'quick_comparison.json',dict(at_utc=now(),score=arch.previous.paired_score(quick,raw_quick),rows=quick,
            raw_rows=raw_quick,checks=checks,promote_full50K=promote))
        if promote:
            full=calibration.evaluate(arm,6144,checkpoint,artifact,50000)
            if full is None:calibration.status('FULL_EVALUATION_FAILED');return
            save(ART/'full50000_comparison.json',dict(at_utc=now(),score=arch.previous.paired_score(full,raw_full),rows=full,
                raw_rows=raw_full,per_GEN_targets=[dict(GEN=r['eval_seed'],**targets.score(r)) for r in full],
                method='model plus TRAIN-only physical-size postprocessing',full128epoch_verified=False))
        assert sha(checkpoint)==before
        calibration.status('COMPLETE',arm=arm,budget=6144,full50K_evaluated=promote)

if __name__=='__main__':
    if '--register-only' in sys.argv:register()
    else:main()
