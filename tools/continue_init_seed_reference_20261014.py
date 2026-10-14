"""Adopt a completed native stage after a coordinator-only validation repair."""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import resume_init_epochfix_20261014 as phase
search=phase.search
engine=search.engine
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART
BASE_CONFIGURE=phase.configure


def sources():
    return {**phase.sources(),'tools/continue_init_seed_reference_20261014.py':sha(__file__)}


def configure():
    BASE_CONFIGURE()
    search.sources=engine.sources=sources


def main():
    archive=ART/'registrations/pre_second_seed_reference_fix'
    old=read(archive/'epochfix_phase_pid.json')
    while engine.shared.alive(old['pid']):
        cmd=Path(f"/proc/{old['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
        assert 'tools/resume_init_epochfix_20261014.py execute' in cmd
        if search.remaining()<300:return
        time.sleep(10)
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(ART/'protocol.json')
        assert plan['phase']=='CYCLIC_RESUME_EXACT_EPOCH_REPLAY'
        changed={p for p,h in plan['source_hashes'].items() if sha(ROOT/p)!=h}
        assert changed=={'tools/resume_init_epochfix_20261014.py'},changed
        assert sha(archive/'resume_init_epochfix_20261014.py')==plan['source_hashes']['tools/resume_init_epochfix_20261014.py']
        proof=read(ART/'second_seed_reference_fix.json')
        assert proof['status']=='PASS' and proof['source_sha256']==sha(ROOT/'tools/resume_init_epochfix_20261014.py')
        plan.update(phase='CYCLIC_RESUME_WITH_SECOND_TRAIN_REFERENCE',phase_registered_utc=now(),
            previous_protocol_sha256=sha(archive/'protocol.json'),source_hashes=sources(),
            reference_fix='For affine candidates, verify raw initialization against same-TRAIN matching4 when present, otherwise same-TRAIN roadK8. Both have the same G1 initial tensors; reject missing references. Required because optional second-TRAIN queue includes roadK8 and the winner, not necessarily matching4.',
            source_change_scope='Coordinator verifier only. No training, model, data, metric, sampler, budget or ranking change. Let the current native job finish, then adopt it under the new source registration.')
        path=ART/'second_seed_reference_fix.json';plan['inputs'][str(path.relative_to(ROOT))]=sha(path)
        save(ART/'protocol.json',plan)
        configure();engine.frozen();adopted=[]
        for folder in sorted(search.RUNS.iterdir()):
            if not (folder/'summary.json').exists() or (folder/'excluded.json').exists():continue
            if not (folder/'execution.json').exists():continue
            rows=read(folder/'summary.json')
            if not rows:continue
            r=rows[0];key=(r['variant'],r['seed'],r['budget'],r['num_samples'])
            if any((x['variant'],x['seed'],x['budget'],x['num_samples'])==key for x in read(ART/'observations.json')):continue
            if 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():continue
            record=read(folder/'execution.json');phase.verify(rows,record['spec'],folder)
            engine.record_rows(rows);adopted.append(str(folder))
            save(folder/'verified_completion.json',dict(status='VERIFIED_AFTER_COORDINATOR_ONLY_FIX',
                at_utc=now(),summary_sha256=sha(folder/'summary.json'),training_source_unchanged=True))
        save(ART/'second_seed_reference_adopted.json',dict(at_utc=now(),folders=adopted))
        engine.status('REGISTERED_SECOND_TRAIN_REFERENCE',adopted=adopted)
        pending=ART/'long_confirmation'
        assert not engine.shared.alive(read(pending/'pid.json')['pid'])
        for name in ('protocol.json','status.json','pid.json'):
            path=pending/name
            if path.exists():shutil.copy2(path,archive/('long_confirmation_'+name))
        (pending/'protocol.json').unlink(missing_ok=True)
    command=[engine.shared.PYTHON,'tools/extend_init_long_confirmation_20261014.py']
    with (pending/'console_seed_reference.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(pending/'pid.json',dict(pid=child.pid,command=command,started_utc=now()))
    # execute() reconfigures its engine; route that call through this phase's
    # complete source registration while preserving the execution plan.
    phase.configure=configure
    phase.execute()


if __name__=='__main__':main()
