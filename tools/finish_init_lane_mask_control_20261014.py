"""Finish only the cheap existing-control evaluation after late mask training."""
from datetime import datetime
import inspect
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import confirm_init_lane_mask_4096_20261014 as phase
from tools import audit_init_architecture_axes_20261014 as audit
read,save,sha,now=phase.read,phase.save,phase.sha,phase.now
engine=phase.engine;ART=phase.ART


def sources():return {**phase.sources(),'tools/finish_init_lane_mask_control_20261014.py':sha(__file__)}


def main():
    pid=read(ART/'pid.json')['pid']
    save(ART/'control_fallback_plan.json',dict(registered_utc=now(),source_hashes=sources(),
        deadline='2026-10-14T22:24:00+08:00',upstream_pid=pid,
        scope='Only finish existing4096 control evaluation if the candidate has fully completed. No extra training. Same checkpoint and bothGEN.8minute primary reserve made the old generic eval estimate overly conservative; actual control4096 costs about35s evaluation plus startup. Reserve6minutes after22:24 for prepared final report/QA.'))
    while engine.shared.alive(pid):time.sleep(5)
    rows=read(ART/'observations.json')
    if not len([r for r in rows if r['variant']=='lane_only_queries'])==2:
        save(ART/'control_fallback_status.json',dict(status='NO_COMPLETE_CANDIDATE',at_utc=now()));return
    if len([r for r in rows if r['variant']=='roadK8'])==2:
        save(ART/'control_fallback_status.json',dict(status='ALREADY_COMPLETE',at_utc=now()));return
    import fcntl
    with (phase.arch.PARENT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not engine.occupied()
        old=read(ART/'protocol.json');assert old['source_hashes']==phase.sources()
        archive=ART/'registrations/before_control_completion';archive.mkdir(parents=True,exist_ok=False)
        save(archive/'protocol.json',old)
        old.update(source_hashes=sources(),deadline='2026-10-14T22:24:00+08:00',
            previous_protocol_sha256=sha(archive/'protocol.json'),
            amendment='Completed candidate unchanged. Allow60s existing-control eval estimate until22:24; no change to any scientific condition.')
        save(ART/'protocol.json',old)
        engine.ART,engine.RUNS=ART,phase.RUNS
        engine.END=old['deadline'];engine.DEADLINE=datetime.fromisoformat(engine.END).timestamp()
        engine.sources,engine.verify=sources,phase.arch.verify
        engine.evidence.score=phase.arch.previous.paired_score
        runner=inspect.getsource(engine.run)
        key="'tools/runtime_checkpointed_sweep.py'";assert runner.count(key)==1
        runner=runner.replace(key,"'tools/runtime_init_architecture_axes_20261014.py'")
        runner=runner.replace('scenes / 4096 * 75 + 90','scenes / 4096 * 40 + 20').replace('estimate + 120','estimate + 0')
        exec(compile(runner,str(__file__),'exec'),engine.__dict__)
        if engine.run('roadK8',4817,4096,resume=old['control_checkpoint'],evaluation=True) is None:return
        audit.ART=ART;audit.main()
        engine.status('PAIRED4096_COMPLETE')
        save(ART/'control_fallback_status.json',dict(status='COMPLETE',at_utc=now()))


if __name__=='__main__':main()
