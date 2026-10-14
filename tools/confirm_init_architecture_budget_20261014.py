"""Finish the live native screen, then reserve the evening for a paired midpoint."""
from pathlib import Path
import fcntl
import inspect
import os
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_architecture_axes_20261014 as arch
engine=arch.engine
read,save,sha,now=arch.read,arch.save,arch.sha,arch.now
ART=arch.ART

def sources():
    return {**arch.sources(),'tools/confirm_init_architecture_budget_20261014.py':sha(__file__)}

def main():
    original=read(ART/'pid.json')['pid']
    status=read(ART/'status.json')
    assert status['status']=='RUNNING' and status['stage']['arm']=='denoise_f32',status
    child=status['child_pid']
    for pid,needle in ((original,'tools/run_init_architecture_axes_20261014.py'),
                       (child,'tools/runtime_init_architecture_axes_20261014.py')):
        assert engine.shared.alive(pid)
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert needle in cmd
    save(ART/'budget_handoff.json',dict(at_utc=now(),original_coordinator=original,native_child=child,
        source_sha256=sha(__file__),reason='Reserve paired6144 and complete50K before22:30. Current frequency screen finishes unchanged; remaining fresh lane/head screens are deferred.',
        native_training_untouched=True,original_protocol_sha256=sha(ART/'protocol.json')))
    os.kill(original,signal.SIGSTOP)
    try:
        while engine.shared.alive(child):
            save(ART/'budget_handoff_status.json',dict(status='WAITING_FOR_UNINTERRUPTED_NATIVE_CHILD',at_utc=now(),child_pid=child))
            time.sleep(10)
        # The original coordinator records this finished child normally, then
        # its STOP flag prevents another native child from being launched.
        os.kill(original,signal.SIGTERM)
    finally:
        if engine.shared.alive(original):os.kill(original,signal.SIGCONT)
    while engine.shared.alive(original):time.sleep(5)
    assert not engine.occupied()
    with (arch.PARENT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(ART/'protocol.json')
        assert old['source_hashes']==arch.sources()
        archive=ART/'registrations/before_budget_confirmation';archive.mkdir(parents=True,exist_ok=False)
        save(archive/'protocol.json',old)
        old.update(phase='ARCHITECTURE_PAIRED_BUDGET_CONFIRMATION',source_hashes=sources(),
                   previous_protocol_sha256=sha(archive/'protocol.json'),
                   scheduler_scope='No change to model, loss, native runner, data, seeds or ranking. Defer remaining fresh screens to guarantee the existing paired midpoint/full50K budget.')
        save(ART/'protocol.json',old)
        engine.ART,engine.RUNS=ART,arch.RUNS
        engine.DEADLINE,engine.END=arch.previous.DEADLINE,arch.previous.END
        engine.sources,engine.verify=sources,arch.verify
        engine.evidence.score=arch.previous.paired_score
        runner=inspect.getsource(engine.run)
        key="'tools/runtime_checkpointed_sweep.py'";assert runner.count(key)==1
        exec(compile(runner.replace(key,"'tools/runtime_init_architecture_axes_20261014.py'"),str(__file__),'exec'),engine.__dict__)
        def halt(*unused):engine.STOP=True
        signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
        scores=[x for x in engine.refresh() if x['budget']==2048 and x['scenes']==4096]
        guarded=[x for x in scores if x['ratios']['speed_jsd']<=1.15 and x['ratios']['width_jsd']<=1.15
                 and x['gates']['onroad'] and x['gates']['vehicle_count']]
        ranked=sorted(guarded or scores,key=lambda x:(max(x['ratios']['ang_dev_jsd'],x['ratios']['length_jsd']),x['R_focus'],x['R_jsd']))
        assert ranked
        winner=ranked[0]['arm']
        save(ART/'selection.json',dict(at_utc=now(),candidate=winner,screen=ranked,exploratory_fallback=not bool(guarded),
            deferred_arms=['lane_only_queries','denoise_h4d32'],reason='Paired midpoint and50K confirmation take priority over additional short screens.'))
        old_full=[x for x in read(arch.PARENT/'observations.json') if x['num_samples']==50000]
        full_cost=max(450.,2.4*max(x['eval_seconds'] for x in old_full)+90)
        needed=2*(6144-2048)*.44+300+2*full_cost+1200
        if engine.DEADLINE-time.time()<needed:
            engine.status('SCREEN_ONLY_NO_TIME_FOR_COMPLETE_CONFIRMATION',needed_seconds=needed);return
        for arm in ('roadK8',winner):
            if engine.run(arm,4817,6144,resume=engine.checkpoint(arm,4817,2048),resume_step=2048) is None:return
        for arm in ('roadK8',winner):
            if engine.run(arm,4817,6144,scenes=50000,resume=engine.checkpoint(arm,4817,6144),evaluation=True) is None:return
        engine.status('ARCHITECTURE_PLAN_COMPLETE',candidate=winner,remaining_seconds=engine.DEADLINE-time.time(),
            next='Audit and report; optional TRAIN-only calibration if time permits.')
        save(ART/'budget_handoff_status.json',dict(status='COMPLETE',at_utc=now(),candidate=winner))

if __name__=='__main__':main()
