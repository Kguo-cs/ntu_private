"""Wait for all original screens, then hand off only this search's supervisor.

No external training process is signalled. Check the recorded PID's exact
command and creation time again immediately before a graceful SIGTERM.
"""
from pathlib import Path
import fcntl
import os
import signal
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import extend_init_overnight_20261014 as extension

base = extension.base
ART = extension.OLD.parent
read, save, now = extension.read, extension.save, extension.now
EXPECTED = [base.shared.PYTHON, 'tools/recover_init_overnight_20261014.py', 'execute']


def main():
    with (ART/'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (extension.ART/'protocol.json').exists()
        old_pid = read(extension.OLD/'status.json')['driver_pid']
        old = psutil.Process(old_pid)
        assert old.cmdline() == EXPECTED, 'Only our exact recovery supervisor may be stopped'
        created = old.create_time()
        while time.time() < base.DEADLINE:
            rows = read(extension.OLD/'observations.json')
            missing = []
            for arm in [base.BASE]+base.SCREEN:
                found=[r for r in rows if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,4096,4096)]
                if len(found)!=2 or {r['eval_seed'] for r in found}!={2817,3817}: missing.append(arm)
            save(ART/'handoff_status.json',dict(status='WAITING_FOR_SCREENS' if missing else 'SCREENS_COMPLETE',
                updated_utc=now(),pid=os.getpid(),old_pid=old_pid,old_created=created,missing=missing))
            if not missing: break
            if not old.is_running() or old.status()==psutil.STATUS_ZOMBIE:
                raise RuntimeError('Old supervisor stopped before screens completed; manual review required')
            time.sleep(10)
        else:
            return
        assert old.create_time()==created and old.cmdline()==EXPECTED
        old.send_signal(signal.SIGTERM)
        # Its own SIGTERM handler reaps only its child process group.
        old.wait(timeout=45)
        extension.configure()
        extension.prepare()
        from tools import watch_init_overnight_results_20261014 as audit
        audit.ART=extension.ART
        audit.refresh()
        for name,target in [('current_status.json','status.json'),('current_results.json','live_results.json'),('current_audit.json','native_metric_audit.json')]:
            tmp=ART/(name+'.tmp');tmp.symlink_to('adaptive_long/'+target);os.replace(tmp,ART/name)
        save(ART/'active_phase.json',dict(updated_utc=now(),phase='adaptive_long',status='adaptive_long/status.json',results='adaptive_long/live_results.json',audit='adaptive_long/native_metric_audit.json',reason='All original screens completed. Add size-skip+matching2 and budget up to2 longer candidates using measured runtimes.',deadline_local=base.END))
        save(ART/'handoff_status.json',dict(status='EXECUTING_ADAPTIVE_LONG',updated_utc=now(),pid=os.getpid(),old_pid=old_pid,old_created=created,old_terminal_confirmed=True,source_sha256=extension.sha(__file__)))
        # An independent audit thread touches only report files, not training.
        import threading
        def audit_worker():
            audit.main()
        threading.Thread(target=audit_worker,daemon=True).start()
        extension.execute()
        audit.refresh()
        save(ART/'handoff_status.json',dict(status='ADAPTIVE_EXECUTE_RETURNED',updated_utc=now(),pid=os.getpid(),old_pid=old_pid,current_status=read(extension.ART/'status.json')))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        save(ART/'handoff_status.json',dict(status='HANDOFF_FAILED',updated_utc=now(),pid=os.getpid(),error_type=type(exc).__name__,error=str(exc)))
        raise
