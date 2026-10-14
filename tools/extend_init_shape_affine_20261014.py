"""Add an opt-in TRAIN-affine size output at a verified native job boundary."""
from __future__ import annotations
import copy
import fcntl
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import continue_init_allseven_20261014 as previous
search=previous.search
engine=search.engine
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART
NATIVE_VERIFY=engine.verify


def sources():
    return {**previous.sources(),'tools/extend_init_shape_affine_20261014.py':sha(__file__)}


def verify(rows,spec,folder):
    previous.parity.verify(rows,spec,folder)
    if spec['arm']=='match4_shape_affine':
        control=next(r for r in read(ART/'observations.json') if r['variant']=='match4' and r['seed']==spec['seed'])
        for row in rows:
            assert row['initial_generator_sha256']==control['initial_generator_sha256']
            assert row['generator_num_parameters']==8083978


def configure():
    previous.configure()
    search.sources=engine.sources=sources
    engine.verify=verify


def main():
    old=read(ART/'all_seven_driver_pid.json')
    while engine.shared.alive(old['pid']):
        command=Path(f"/proc/{old['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
        assert 'tools/continue_init_allseven_20261014.py execute' in command,command
        if search.remaining()<300:return
        time.sleep(10)
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(ART/'protocol.json')
        assert plan['phase']=='ALL_SEVEN_SD_VW_TARGETS'
        changed={p for p,h in plan['source_hashes'].items() if sha(ROOT/p)!=h}
        assert changed=={'src/smart/diffusion/denoiser.py','src/smart/diffusion/size_parameterization.py'},changed
        archive=ART/'registrations/pre_shape_train_affine'
        for name in changed:
            assert sha(archive/Path(name).name)==plan['source_hashes'][name]
        proof=read(ART/'shape_affine/integration_preflight.json')
        assert proof['status']=='PASS' and proof['tests']==14
        assert all(proof[k] for k in ('default_tensor_values_unchanged','default_rng_unchanged','state_dict_keys_unchanged'))
        for p,h in proof['source_sha256'].items():assert sha(ROOT/p)==h
        variants=read(ART/'variants.json')
        variants['match4_shape_affine']=copy.deepcopy(variants['match4'])
        variants['match4_shape_affine']['__config'][search.P+'shape_x0_parameterization']='train_affine'
        save(ART/'variants.json',variants)
        plan.update(phase='ALL_SEVEN_TRAIN_AFFINE_SIZE_OUTPUT',phase_registered_utc=now(),source_hashes=sources(),
            previous_protocol_sha256=sha(archive/'protocol.json'),
            adaptive_reason='Current continuous inputs and losses use TRAIN scale, but size x0 head emits raw meters. Test an affine output parameterization mu_TRAIN + sigma_TRAIN * raw for length and width; preserve physical x0 target, source, MSE, matching and sampler.',
            extension_scope='One2048-update screen; same6144/14336 promotion rules and all-seven full50K criteria. No additional parameters or random draws. No evaluation distribution is fitted.',
            default_compatibility=proof)
        plan['screen']=list(dict.fromkeys(plan['screen']+['match4_shape_affine']))
        plan['new_hypotheses']['match4_shape_affine']='Predict standardized clean length/width and decode with saved TRAIN statistics; numerical conditioning hypothesis, not a claim of improved generated marginals.'
        plan['local_input_hashes']['variants.json']=sha(ART/'variants.json')
        p=ART/'shape_affine/integration_preflight.json';plan['inputs'][str(p.relative_to(ROOT))]=sha(p)
        save(ART/'protocol.json',plan)
        configure();engine.frozen()
        # The current child ran with the previous default source already loaded.
        # Its completed results remain valid under the exact default-path proof.
        adopted=[]
        for folder in sorted(search.RUNS.iterdir()):
            if not (folder/'summary.json').exists() or (folder/'excluded.json').exists():continue
            values=read(folder/'summary.json')
            if not values or not (folder/'execution.json').exists():continue
            row=values[0]
            key=(row['variant'],row['seed'],row['budget'],row['num_samples'])
            if any((r['variant'],r['seed'],r['budget'],r['num_samples'])==key for r in read(ART/'observations.json')):continue
            if 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():continue
            record=read(folder/'execution.json');verify(values,record['spec'],folder)
            engine.record_rows(values);adopted.append(str(folder))
            save(folder/'verified_completion.json',dict(status='VERIFIED_DEFAULT_PATH_AFTER_OPT_IN_AFFINE',
                at_utc=now(),summary_sha256=sha(folder/'summary.json'),default_compatibility=proof))
        save(ART/'shape_affine/adopted_completed_stages.json',dict(at_utc=now(),folders=adopted))
        engine.status('REGISTERED_SHAPE_AFFINE',adopted=adopted)
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if engine.run('match4_shape_affine',4817,2048) is None:return
    previous.execute()


if __name__=='__main__':main()
