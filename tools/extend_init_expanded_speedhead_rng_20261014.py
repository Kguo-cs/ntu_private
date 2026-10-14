"""Re-run the optional speed head after preserving full-Flow initialization."""
from __future__ import annotations
import copy
import fcntl
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import extend_init_expanded_speedhead_20261014 as previous
search=previous.search
read,save,sha,now=search.read,search.save,search.sha,search.now
NATIVE_VERIFY=search.verify
SCREEN=['match4','match_v4h2_s2','prior_speed','prior_shape_speed',
        'match4_h320d4','match4_map64','match4_attrdist','match4_speedhead_paired']


def sources():
    return {**previous.sources(), 'tools/extend_init_expanded_speedhead_rng_20261014.py':sha(__file__)}


def verify(rows,spec,folder):
    NATIVE_VERIFY(rows,spec,folder)
    if spec['arm']=='match4_speedhead_paired':
        control=next(r for r in read(search.ART/'observations.json')
                     if r['variant']==search.BASE and r['seed']==spec['seed'])
        for row in rows:
            assert row['initial_shared_generator_sha256']==control['initial_generator_sha256']
            assert row['generator_num_parameters']==8150539


def configure():
    search.SCREEN=SCREEN
    search.sources=sources
    search.configure()
    search.engine.verify=verify


def main():
    art=search.ART
    old_pid=read(art/'speedhead_phase_pid.json')['pid']
    while search.engine.shared.alive(old_pid):
        if search.remaining()<180:return
        time.sleep(10)
    with (art/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(art/'protocol.json')
        changed=[name for name,value in old['source_hashes'].items() if sha(ROOT/name)!=value]
        assert changed==['src/smart/diffusion/scale_flow.py'],changed
        archive=art/'registrations/pre_speed_head_rng_fix'
        assert sha(archive/'scale_flow.py')==old['source_hashes'][changed[0]]
        proof=read(art/'speedhead_initialization/after.json')
        assert proof['status']=='PASS' and all(proof[k] for k in (
            'default_tensor_values_unchanged','default_rng_unchanged',
            'separate_shared_tensors_equal','separate_rng_equal'))
        variants=read(art/'variants.json')
        variants['match4_speedhead_paired']=copy.deepcopy(variants['match4_speedhead'])
        save(art/'variants.json',variants)
        plan=copy.deepcopy(old)
        plan.update(phase='SPEED_HEAD_FULL_FLOW_INIT_PARITY',phase_registered_utc=now(),
            source_hashes=sources(),screen=[search.BASE]+SCREEN,
            previous_protocol_sha256=sha(archive/'protocol.json'),
            adaptive_reason='Before reading the speed-head generation results, its initial shared tensor hash differed from matching4. CPU reproduction isolated2 SD time-MLP weight tensors and RNG drift caused by the outer Flow initializer. Isolate the entire optional head initialization; default shared-head tensors/RNG remain exact. Re-run a fresh speed-head arm with strict native shared-hash equality.',
            initialization_compatibility=proof,
            excluded_arm='match4_speedhead: valid legacy-init trial retained outside selection; not the intended exact shared-initialization comparison.')
        plan['local_input_hashes']['variants.json']=sha(art/'variants.json')
        plan['inputs'][str((art/'speedhead_initialization/after.json').relative_to(ROOT))]=sha(art/'speedhead_initialization/after.json')
        save(art/'protocol.json',plan)
        configure()
        old_rows=read(art/'observations.json')
        excluded=[r for r in old_rows if r['variant']=='match4_speedhead']
        save(art/'observations.json',[r for r in old_rows if r['variant']!='match4_speedhead'])
        for folder in sorted(search.RUNS.iterdir()):
            if not (folder/'summary.json').exists() or not (folder/'execution.json').exists():continue
            values=read(folder/'summary.json')
            if not values:continue
            r=values[0]
            if r['variant']=='match4_speedhead':
                record=read(folder/'execution.json')
                assert 'RUNTIME_LOADER_SUCCESS' in (folder/'console.log').read_text()
                NATIVE_VERIFY(values,record['spec'],folder)
                excluded=values
                save(folder/'excluded.json',dict(reason='FULL_FLOW_SHARED_INITIALIZATION_MISMATCH',
                    at_utc=now(),valid_native_results=True,replacement='match4_speedhead_paired',
                    evidence=str(art/'speedhead_initialization/before.json')))
                continue
            if any((v['variant'],v['seed'],v['budget'],v['num_samples'])==
                   (r['variant'],r['seed'],r['budget'],r['num_samples']) for v in read(art/'observations.json')):continue
            if (folder/'excluded.json').exists():continue
            if 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():continue
            record=read(folder/'execution.json')
            verify(values,record['spec'],folder)
            search.engine.record_rows(values)
            save(folder/'verified_completion.json',dict(status='VERIFIED_DEFAULT_PATH_AFTER_OPTIONAL_HEAD_FIX',
                at_utc=now(),summary_sha256=sha(folder/'summary.json'),default_compatibility=proof))
        save(art/'speedhead_initialization/unpaired_native_results.json',
             dict(excluded_from_selection=True,reason=plan['excluded_arm'],rows=excluded))
        search.engine.refresh()
        search.engine.status('REGISTERED_PAIRED_SPEED_HEAD',screen=SCREEN)
    search.execute()


if __name__=='__main__':main()
