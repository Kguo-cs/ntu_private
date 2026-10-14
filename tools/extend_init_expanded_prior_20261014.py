"""Register the TRAIN attribute-prior hypothesis at a completed native boundary."""
from __future__ import annotations
import copy
import fcntl
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_expanded_20261014 as search
read,save,sha,now=search.read,search.save,search.sha,search.now
ORIGINAL_SOURCES=search.sources
SCREEN=['match4','match_v4h2_s2','prior_speed','prior_shape_speed',
        'match4_h320d4','match4_map64','match4_attrdist','match4_speedhead']
TABLE=ROOT/'docs/experiments/init_diffusion_attribute_prior_20261014/train_quantiles.pt'


def sources():
    return {**ORIGINAL_SOURCES(),**{name:sha(ROOT/name) for name in (
        'src/smart/diffusion/attribute_source.py','tools/extend_init_expanded_prior_20261014.py',
        'tools/probe_init_attribute_prior_20261014.py')}}


def configure():
    search.SCREEN=SCREEN
    search.sources=sources
    search.configure()


def main():
    art=search.ART
    pid=read(art/'clean_retry_pid.json')['pid']
    while search.engine.shared.alive(pid):
        if search.remaining()<180:return
        time.sleep(10)
    with (art/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(art/'protocol.json')
        changed=[p for p,h in old['source_hashes'].items() if sha(ROOT/p)!=h]
        assert set(changed)=={'src/smart/diffusion/initial_diffusion.py',
            'src/smart/diffusion/scale_flow.py','tools/overnight_init_diffusion_sweep.py'},changed
        proof=read(art/'attribute_prior/integration_preflight.json')
        assert proof['default_hash_identical'] and proof['prior_keeps_all_learned_parameter_values']
        archive=art/'registrations/pre_attribute_prior'
        archive.mkdir(parents=True,exist_ok=False)
        for name in ('protocol.json','variants.json','observations.json','comparisons.json'):
            if (art/name).exists():shutil.copy2(art/name,archive/name)
        variants=read(art/'variants.json')
        for name,shape in (('prior_speed',False),('prior_shape_speed',True)):
            variants[name]=copy.deepcopy(variants['match4'])
            variants[name]['__config'][search.P+'attribute_source']=dict(
                path=str(TABLE),sha256=sha(TABLE),knots=2049,speed=True,shape=shape)
        save(art/'variants.json',variants)
        plan=copy.deepcopy(old)
        plan.update(phase='TRAIN_ATTRIBUTE_PRIOR',phase_registered_utc=now(),
            source_hashes=sources(),screen=[search.BASE]+SCREEN,
            integration_pilots=dict(arms=['prior_speed','prior_shape_speed'],steps=64,scenes=4096,
                                    purpose='Native checkpoint/EMA/train/eval verification only; not used for ranking'),
            previous_protocol_sha256=sha(archive/'protocol.json'),
            adaptive_reason='TRAIN-only probe:35.3% vehicle speeds<0.2m/s, Gaussian physical speed21.4% negative and only2% nearzero. Test scalar-speed source and speed+shape source separately without changing target/loss. Reallocate depth5 and separate-head trial budget; keep16M model, map64, matching and marginal-loss trials.',
            default_compatibility=proof,source_implementation='Opt-in empirical inverseCDF consumes no extra RNG, adds checkpointed buffers but no parameters, uses no evaluation GT attributes/types. Disabled defaults preserve exact generator tensor hash and learned parameters.',
            membership='Inherited completed controls keep their original source hashes; default-path compatibility checked. Do not call repeated early runs independent training seeds.')
        for path in (TABLE,TABLE.parent/'provenance.json',art/'attribute_prior/integration_preflight.json'):
            plan['inputs'][str(path.relative_to(ROOT))]=sha(path)
        plan['local_input_hashes']['variants.json']=sha(art/'variants.json')
        save(art/'protocol.json',plan)
        configure()
        rows=read(art/'observations.json')
        for arm in (search.BASE,'match4'):
            if any(r['variant']==arm and r['seed']==4817 and r['budget']==2048 for r in rows):continue
            folder=search.RUNS/f'{arm}_T4817_2048_N4096'
            if not (folder/'summary.json').exists():continue
            if 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():continue
            values=read(folder/'summary.json');record=read(folder/'execution.json')
            try:search.verify(values,record['spec'],folder)
            except (ValueError,AssertionError,UnicodeError) as exc:
                save(folder/'excluded.json',dict(reason='NATIVE_INTEGRITY_CHECK_FAILED',error_type=type(exc).__name__,at_utc=now()))
                continue
            search.engine.record_rows(values)
            save(folder/'verified_completion.json',dict(status='VERIFIED_PREVIOUS_DEFAULT_SOURCE',
                at_utc=now(),summary_sha256=sha(folder/'summary.json'),default_compatibility=proof))
        search.engine.status('REGISTERED_ATTRIBUTE_PRIOR_PHASE',screen=SCREEN)
    for arm in (search.BASE,'match4'):
        if not any(r['variant']==arm and r['seed']==4817 and r['budget']==2048 for r in read(art/'observations.json')):
            if search.engine.run(arm,4817,2048,attempt_label='_prior_phase_retry1') is None:return
    for arm in ('prior_speed','prior_shape_speed'):
        if search.engine.run(arm,4817,64) is None:return
    search.execute()


if __name__=='__main__':main()
