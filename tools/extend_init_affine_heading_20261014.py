"""One evidence-directed interaction screen, without changing model source."""
from __future__ import annotations
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import extend_init_shape_affine_20261014 as parent
previous=parent.previous
search=parent.search
engine=search.engine
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART
ARM='match4_affine_heading_p1'


def sources():
    return {**parent.sources(),'tools/extend_init_affine_heading_20261014.py':sha(__file__)}


def verify(rows,spec,folder):
    parent.verify(rows,spec,folder)
    if spec['arm']==ARM:
        control=next(r for r in read(ART/'observations.json') if r['variant']=='match4' and r['seed']==spec['seed'])
        for row in rows:
            assert row['initial_generator_sha256']==control['initial_generator_sha256']
            assert row['generator_num_parameters']==8083978


def configure():
    parent.configure()
    search.sources=engine.sources=sources
    engine.verify=verify


def main():
    old=read(ART/'shape_affine_phase_pid.json')
    while engine.shared.alive(old['pid']):
        command=Path(f"/proc/{old['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
        assert 'tools/extend_init_shape_affine_20261014.py' in command
        if search.remaining()<300:return
        time.sleep(10)
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(ART/'protocol.json');archive=ART/'registrations/pre_affine_heading_interaction'
        assert plan['phase']=='ALL_SEVEN_TRAIN_AFFINE_SIZE_OUTPUT'
        assert plan['source_hashes']==parent.sources(),'No production/training source changes are part of this phase.'
        variants=read(ART/'variants.json');before=read(archive/'variants.json')
        expected=copy.deepcopy(before['match4_shape_affine']);expected['heading']={'time_power':1}
        assert variants=={**before,ARM:expected}
        assert sha(archive/'variants.json')==plan['local_input_hashes']['variants.json']
        plan.update(phase='AFFINE_HEADING_INTERACTION',phase_registered_utc=now(),source_hashes=sources(),
            previous_protocol_sha256=sha(archive/'protocol.json'),
            interaction_reason='At2048, heading-p1 improves angle but worsens sizes; TRAIN-affine size output improves both sizes and angle and reduces width-gradient dominance in a128scene CPU TRAIN probe. Test the combination directly; do not assume additive effects.',
            extension_scope='One additional2048-update arm using only existing configuration options. Same midpoint/long/full50K selection rules, no new model parameters or production-code changes.')
        plan['screen']=list(dict.fromkeys(plan['screen']+[ARM]))
        plan['new_hypotheses'][ARM]='TRAIN-affine length/width x0 output plus heading time_power1; complete2x2 interaction with existing match4, shape_affine and heading_p1 arms.'
        plan['local_input_hashes']['variants.json']=sha(ART/'variants.json')
        save(ART/'protocol.json',plan);configure();engine.frozen()
        adopted=[]
        for folder in sorted(search.RUNS.iterdir()):
            if not (folder/'summary.json').exists() or (folder/'excluded.json').exists():continue
            rows=read(folder/'summary.json')
            if not rows or not (folder/'execution.json').exists():continue
            row=rows[0];key=(row['variant'],row['seed'],row['budget'],row['num_samples'])
            if any((r['variant'],r['seed'],r['budget'],r['num_samples'])==key for r in read(ART/'observations.json')):continue
            if 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():continue
            record=read(folder/'execution.json');verify(rows,record['spec'],folder)
            engine.record_rows(rows);adopted.append(str(folder))
            save(folder/'verified_completion.json',dict(status='VERIFIED_CONFIG_ONLY_PHASE_BOUNDARY',
                at_utc=now(),summary_sha256=sha(folder/'summary.json'),unchanged_production_sources=True))
        save(ART/'affine_heading_adopted.json',dict(at_utc=now(),folders=adopted))
        engine.status('REGISTERED_AFFINE_HEADING_INTERACTION',adopted=adopted)
    # Restart only the idle future confirmation coordinator with the new source
    # registration. It must wait for this driver and the existing sampler.
    command=[engine.shared.PYTHON,'tools/extend_init_long_confirmation_20261014.py']
    pending=ART/'long_confirmation'
    with (pending/'console.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(pending/'pid.json',dict(pid=child.pid,command=command,started_utc=now(),
        purpose='Wait for interaction phase and samplers, then equal-budget longer confirmation.'))
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if engine.run(ARM,4817,2048) is None:return
    previous.execute()


if __name__=='__main__':main()
