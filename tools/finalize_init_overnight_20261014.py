"""Complete only the failed50K verification of already-trained overnight weights.

Training/search stopped at the registered10:00 deadline. This separate phase
repairs an evaluation-only CLI naming error; it cannot request training updates.
"""
import argparse
import copy
from datetime import datetime,timedelta,timezone
import fcntl
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import recover_init_overnight_20261014 as recovery
base=recovery.base
read,save,sha,now=base.read,base.save,base.sha,base.now
PARENT=ROOT/'output/reports/init_diffusion_overnight_20261014/adaptive_long'
ART=PARENT.parent/'final_verification'
RUNS=ROOT/'logs/init_diffusion_overnight_20261014/final_verification'
SOURCE=recovery.sources


def sources():
    result=SOURCE()
    for name in ('tools/extend_init_overnight_20261014.py','tools/handoff_init_overnight_20261014.py',
                 'tools/finalize_init_overnight_20261014.py',
                 'configs/experiment/init_diffusion_657_n20_road_k8_size_skip_match2.yaml',
                 'configs/experiment/init_diffusion_657_n20_road_k8_size_time_width_l1.yaml'):
        result[name]=sha(ROOT/name)
    return result


def configure():
    recovery.ART,recovery.RUNS=ART,RUNS
    base.ART,base.RUNS,base.sources=ART,RUNS,sources
    if (ART/'protocol.json').exists():
        base.END=read(ART/'protocol.json')['verification_deadline']
        base.DEADLINE=datetime.fromisoformat(base.END).timestamp()


def prepare():
    assert not (ART/'protocol.json').exists()
    with (PARENT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=read(PARENT/'protocol.json')
        changed=[name for name,digest in prior['source_hashes'].items() if sha(ROOT/name)!=digest]
        assert changed==['tools/run_init_overnight_20261014.py'],changed
        archive=ART.parent/'launcher_before_eval_fix.py'
        assert sha(archive)==prior['source_hashes'][changed[0]]
        for name,digest in prior['inputs'].items():assert sha(ROOT/name)==digest,name
        for name,digest in prior['imported_sources'].items():assert sha(ROOT/name)==digest,name
        for name,digest in prior['local_input_hashes'].items():
            assert sha(PARENT/name)==digest,name
            save(ART/name,read(PARENT/name))
        rows=read(PARENT/'observations.json')
        for arm in (base.BASE,'size_time_width_l1'):
            cells=[r for r in rows if (r['variant'],r['seed'],r['budget'],r['num_samples'])==(arm,4817,16384,4096)]
            assert len(cells)==2 and {r['eval_seed'] for r in cells}=={2817,3817}
        if base.occupied():raise RuntimeError('GPU occupied; no external process is stopped')
        plan=copy.deepcopy(prior)
        plan.update(registered_utc=now(),source_hashes=sources(),
            original_search_deadline=prior['deadline'],
            verification_deadline=(datetime.now(timezone.utc)+timedelta(minutes=35)).isoformat(),
            phase='evaluation_only_after_search',optimization_updates_allowed=0,
            source_change='For evaluation-only commands, --variants is empty, avoiding duplicate training/evaluation names. No model/loss/data/sampler change.',
            post_deadline_reason='Repair failed50K verification of the two already-trained16384 checkpoints. No new training or candidate search.',
            parent_protocol_sha256=sha(PARENT/'protocol.json'),
            local_input_hashes={name:sha(ART/name) for name in prior['local_input_hashes']})
        plan['imported_sources'].update({str(p.relative_to(ROOT)):sha(p) for p in (archive,PARENT/'protocol.json',PARENT/'observations.json',PARENT/'status.json')})
        save(ART/'protocol.json',plan);save(ART/'observations.json',rows);save(ART/'recovery_ledger.json',[])
        configure();base.refresh();base.status('REGISTERED_EVALUATION_ONLY',optimization_updates_allowed=0)


def execute():
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def halt(*args):base.STOP=True
        signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
        base.frozen()
        for arm in (base.BASE,'size_time_width_l1'):
            if recovery.run(arm,4817,16384,scenes=50000,
                    resume=base.checkpoint(arm,4817,16384),evaluation=True) is None:return
        proof=[r for r in base.refresh() if r['arm']=='size_time_width_l1' and
            ((r['seed']==4817 and r['budget'] in (12288,16384)) or r['budget']==6826)]
        assert len(proof)==5
        save(ART/'recommendation.json',dict(generated_utc=now(),candidate='size_time_width_l1',
            promoted=all(r['eligible'] for r in proof),evidence=proof,full128epoch_verified=False,
            default='init_diffusion_657_n20_road_k8_full',optimization_updates_in_phase=0,
            reason='Longer and second-seed tests outweigh early4096 ranking.50K verification completed separately after correcting the launcher.'))
        from tools import watch_init_overnight_results_20261014 as audit
        audit.ART=ART;audit.refresh()
        base.status('VERIFICATION_COMPLETE',optimization_updates_allowed=0,actual_cells=len(read(ART/'observations.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','execute']);args=parser.parse_args()
    configure();globals()[args.mode]()
