"""Add the already implemented small speed head at a completed job boundary."""
from __future__ import annotations
import copy
import fcntl
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import extend_init_expanded_prior_20261014 as previous
search=previous.search
read,save,sha,now=search.read,search.save,search.sha,search.now


def sources():
    return {**previous.sources(), 'tools/extend_init_expanded_speedhead_20261014.py':sha(__file__)}


def configure():
    search.SCREEN=previous.SCREEN
    search.sources=sources
    search.configure()


def main():
    art=search.ART
    old_pid=read(art/'attribute_phase_pid.json')['pid']
    while search.engine.shared.alive(old_pid):
        if search.remaining()<180:return
        time.sleep(10)
    with (art/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(art/'protocol.json')
        changed=[name for name,value in old['source_hashes'].items() if sha(ROOT/name)!=value]
        assert changed==['tools/extend_init_expanded_prior_20261014.py'],changed
        archive=art/'registrations/pre_speed_head'
        assert sha(archive/'prior_phase_driver.py')==old['source_hashes'][changed[0]]
        assert sha(archive/'variants.json')==sha(art/'variants.json')
        proof=next(r for r in read(art/'architecture_preflight.json') if r['arm']=='match4_speedhead')
        assert proof['generator_params']==8150539 and proof['map_frozen']
        plan=copy.deepcopy(old)
        plan.update(phase='SPEED_HEAD_ADDITION',phase_registered_utc=now(),source_hashes=sources(),
            screen=[search.BASE]+previous.SCREEN,
            previous_protocol_sha256=sha(archive/'protocol.json'),
            adaptive_reason='Full-model expansion at2048 updates worsened focus JSD versus matching4; empirical speed/shape source lowered regression loss without improving generation. Test the pre-existing separate scalar-speed head (+66561 parameters) as a targeted capacity ablation. This is a hypothesis, not a claim that shared gradients caused the errors.',
            extension_scope='Add one2048-update screen, eligible for the same6144/14336 promotion rules. No change to model/loss/data/evaluator sources or previous variant definitions.',
            scope_proof=proof)
        save(art/'protocol.json',plan)
        configure()
        for folder in sorted(search.RUNS.iterdir()):
            if not (folder/'summary.json').exists() or not (folder/'execution.json').exists():continue
            values=read(folder/'summary.json')
            if not values:continue
            r=values[0]
            if any((v['variant'],v['seed'],v['budget'],v['num_samples'])==
                   (r['variant'],r['seed'],r['budget'],r['num_samples']) for v in read(art/'observations.json')):continue
            if (folder/'excluded.json').exists():continue
            if 'RUNTIME_LOADER_SUCCESS' not in (folder/'console.log').read_text():continue
            record=read(folder/'execution.json')
            search.verify(values,record['spec'],folder)
            search.engine.record_rows(values)
            save(folder/'verified_completion.json',dict(status='VERIFIED_AFTER_CONTROLLER_ONLY_AMENDMENT',
                at_utc=now(),summary_sha256=sha(folder/'summary.json'),changed_controller_sources=changed))
        search.engine.status('REGISTERED_SPEED_HEAD_ADDITION',screen=previous.SCREEN)
    search.execute()


if __name__=='__main__':main()
