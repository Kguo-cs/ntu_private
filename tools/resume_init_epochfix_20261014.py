"""Replay resumed stages after proving exact multi-epoch continuation."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
import shutil
import signal
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import extend_init_affine_heading_20261014 as previous
from tools import audit_init_expanded_schedule_20261014 as schedule
search=previous.search
engine=search.engine
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART
ARCHIVE=ART/'registrations/pre_cyclic_resume_fix'


def sources():
    return {**previous.sources(),**{p:sha(ROOT/p) for p in (
        'tools/resume_init_epochfix_20261014.py',
        'tools/audit_init_expanded_schedule_20261014.py',
        'tests/test_scratch_cyclic_resume.py')}}


def initialization_reference(rows, seed):
    # Matching4 and roadK8 have identical initial G1 tensor layouts/values for
    # the same TRAIN seed. The second-seed queue guarantees roadK8 first, but
    # does not necessarily train an extra matching4 arm.
    for arm in ('match4', search.BASE):
        for row in rows:
            if row['variant'] == arm and row['seed'] == seed:
                return row
    raise ValueError(f"No initialization reference exists for TRAIN{seed}")


def verify(rows,spec,folder):
    if spec['arm'] in ('match4_shape_affine', 'match4_affine_heading_p1'):
        previous.previous.parity.verify(rows,spec,folder)
        control=initialization_reference(read(ART/'observations.json'),spec['seed'])
        for row in rows:
            assert row['initial_generator_sha256']==control['initial_generator_sha256']
            assert row['generator_num_parameters']==8083978
    else:
        previous.verify(rows,spec,folder)
    if not spec.get('evaluation'):
        audit=schedule.audit_row(rows[0])
        save(folder/'training_schedule_audit.json',dict(status='PASS',at_utc=now(),audit=audit))


def configure():
    previous.configure()
    search.sources=engine.sources=sources
    engine.verify=verify


def register():
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(ART/'protocol.json')
        assert plan['phase']=='AFFINE_HEADING_INTERACTION',plan['phase']
        changes={p for p,h in plan['source_hashes'].items() if sha(ROOT/p)!=h}
        assert changes=={'tools/scratch_component_loss_sweep.py'},changes
        assert sha(ARCHIVE/'scratch_component_loss_sweep.py')==plan['source_hashes']['tools/scratch_component_loss_sweep.py']
        proof=read(ART/'cyclic_resume_regression.json')
        assert proof['status']=='PASS'
        for p,h in proof['source_sha256'].items():assert sha(ROOT/p)==h,p
        rows=read(ART/'observations.json')
        excluded=[r for r in rows if r['optimizer_continued'] and not r['evaluation_only']]
        assert {(r['variant'],r['budget']) for r in excluded}=={('roadK8',6144)}
        excluded_folders=[]
        for folder in search.RUNS.iterdir():
            if not (folder/'execution.json').exists():continue
            record=read(folder/'execution.json');spec=record.get('spec',{})
            if spec.get('resume') and not spec.get('evaluation'):
                save(folder/'excluded.json',dict(at_utc=now(),reason='CYCLIC_RESUME_LOADER_NOT_RELOADED',
                    status='INVALID_RESUMED_TRAINING',replacement='Replay from valid2048 checkpoint with epochfix runner.',
                    retained_raw_evidence=True))
                excluded_folders.append(str(folder))
        save(ARCHIVE/'excluded_rows.json',dict(at_utc=now(),rows=excluded,folders=excluded_folders))
        for name in ('generation_variability_diagnostic.json','histogram_diagnostics.json',
                     'recovery_2560_proof.json','all_seven_target_progress.json','training_schedule_audit.json',
                     'parent_comparisons.json','affine_heading_result.json'):
            if (ART/name).exists():shutil.copy2(ART/name,ARCHIVE/name)
        save(ART/'observations.json',[r for r in rows if r not in excluded])
        issue=read(ART/'cyclic_resume_issue.json')
        issue.update(status='FIX_REGRESSION_PASSED_NATIVE_REPLAY_PENDING',fixed_utc=now(),
            root_cause='Wrong singular Trainer attribute reload_dataloaders_every_n_epoch was silently unused; first suffix repeated. Set constructor plural reload_dataloaders_every_n_epochs=1. Reuse unconsumed reload iterator to preserve CPU RNG and clear only parent fit-loop restart flag to avoid duplicate epoch scheduler stepping.',
            regression=str(ART/'cyclic_resume_regression.json'),excluded_folders=excluded_folders,
            fresh2048_screens_unchanged=True,replay_from_updates=2048)
        save(ART/'cyclic_resume_issue.json',issue)
        plan.update(phase='CYCLIC_RESUME_EXACT_EPOCH_REPLAY',phase_registered_utc=now(),
            source_hashes=sources(),previous_protocol_sha256=sha(ARCHIVE/'protocol.json'),
            resume_bug=str(ART/'cyclic_resume_issue.json'),resume_proof=str(ART/'cyclic_resume_regression.json'),
            interruption_recovery='Superseded: previous2560 proof checked state hashes but did not prove later epoch data traversal. All affected resumed stages are excluded.',
            extension_scope='Model/loss/evaluator sources unchanged. Repair native quick-runner continuation, retain all valid fresh2048 results and preselected midpoint arms, replay resumed stages from2048. Require per-epoch loader lengths and every-update LR audit before adopting results.')
        old_proof=str((ART/'recovery_2560_proof.json').relative_to(ROOT))
        plan['inputs'].pop(old_proof,None)
        for p in (ART/'cyclic_resume_regression.json',ARCHIVE/'excluded_rows.json'):
            plan['inputs'][str(p.relative_to(ROOT))]=sha(p)
        save(ART/'protocol.json',plan)
        configure();engine.frozen();previous.previous.refresh();schedule.main()
        engine.status('REGISTERED_EPOCH_REPLAY',excluded_cells=len(excluded),
            retained_fresh_cells=len(rows)-len(excluded),midpoint=read(ART/'midpoint_selection.json')['selected'])


def execute():
    configure()
    with (ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def halt(*unused):engine.STOP=True
        signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
        selected=read(ART/'midpoint_selection.json')['selected']
        assert selected==['match4','match4_shape_affine','match_v4h2_s2']
        for arm in [search.BASE]+selected:
            if engine.run(arm,4817,6144,resume=engine.checkpoint(arm,4817,2048),
                          resume_step=2048,attempt_label='_epochfix') is None:return
        order,fallback=previous.previous.ranked(6144)
        selected=['match4']+[r['arm'] for r in order if r['arm']!='match4'][:1]
        save(ART/'long_selection.json',dict(at_utc=now(),selected=selected,evidence=order,exploratory_fallback=fallback))
        for arm in [search.BASE]+selected:
            if engine.run(arm,4817,14336,resume=engine.checkpoint(arm,4817,6144),resume_step=6144) is None:return
        order,fallback=previous.previous.ranked(14336);winner=order[0]['arm']
        save(ART/'long_candidate.json',dict(at_utc=now(),candidate=winner,evidence=order,
            exploratory_fallback=fallback,full128epoch_verified=False,
            all_seven_target_progress=str(ART/'all_seven_target_progress.json')))
        for arm in (search.BASE,winner):
            if engine.run(arm,4817,14336,scenes=50000,
                          resume=engine.checkpoint(arm,4817,14336),evaluation=True) is None:return
        if search.remaining()>11000:
            for arm in (search.BASE,winner):
                if engine.run(arm,5817,4096) is None:return
            for arm in (search.BASE,winner):
                if engine.run(arm,4817,4096,resume=engine.checkpoint(arm,4817,2048),resume_step=2048) is None:return
        engine.status('TRAINING_PLAN_COMPLETE',winner=winner,remaining_seconds=search.remaining(),
            next='Same-checkpoint samplers and equal-budget longer confirmation if time permits.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','execute']);args=p.parse_args()
    register() if args.mode=='register' else execute()
