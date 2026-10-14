"""Use remaining evening time for a longer paired confirmation, after samplers.

No new architecture screening. Preserve the registered14336-step winner and
extend it and roadK8 equally, then evaluate each checkpoint on the full50K.
"""
from __future__ import annotations
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import continue_init_seed_reference_20261014 as phase
search=phase.search
engine=search.engine
read,save,sha,now=search.read,search.save,search.sha,search.now
ART=search.ART/'long_confirmation'
START=14336
BUDGETS=(24576,22528,20480,18432)


def status(name,**extra):
    save(ART/'status.json',dict(status=name,at_utc=now(),pid=os.getpid(),**extra))


def register():
    ART.mkdir(parents=True,exist_ok=True)
    plan=dict(registered_utc=now(),deadline=search.END,source_sha256=sha(__file__),
        upstream_protocol_sha256=sha(search.ART/'protocol.json'),
        prerequisites=['Owned training phase is terminal after TRAINING_PLAN_COMPLETE.',
            'Registered same-checkpoint sampler phase completed, with its process terminal.'],
        pair='roadK8 and the candidate selected at14336 by the existing angle/length bottleneck rule; no reselection from2048 alone.',
        allowed_total_updates=list(BUDGETS),start_updates=START,
        budget_rule='Largest listed budget fitting conservative .44seconds/update for BOTH arms,150seconds quick eval/startup per arm, '
            '2*(50000/4096*75+90)seconds full50K evaluation, plus900seconds report/QA reserve.',
        evaluation='N20, no guidance, both predeclared GEN2817/3817, full50000 matched scenes at the final equal training budget.',
        scope='Additional from-scratch trajectory continuation with optimizer/scheduler/EMA/RNG preserved; not fine-tuning a fully trained generator. '
            'Full50K evaluation is on the validation cohort used throughout; not a held-out test or proof of128epoch superiority.',
        stop='Do not launch unless the entire pair plus full50K and report reserve fits. No partial or unequal-budget promotion.')
    p=ART/'protocol.json'
    if p.exists():
        old=read(p)
        assert {k:v for k,v in old.items() if k!='registered_utc'}=={k:v for k,v in plan.items() if k!='registered_utc'}
    else:save(p,plan)
    return read(p)


def live_owned(metadata,needle):
    pid=read(metadata)['pid']
    if not engine.shared.alive(pid):return False
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert needle in cmd,(pid,cmd)
    return True


def main():
    plan=register();phase.configure()
    def stop(*unused):engine.STOP=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    while search.remaining()>180 and not engine.STOP:
        train_live=live_owned(search.ART/'seed_reference_phase_pid.json','tools/continue_init_seed_reference_20261014.py')
        sampler_live=live_owned(search.ART/'sampler_all_seven_pid.json','tools/eval_init_expanded_samplers_20261014.py')
        sampler=read(search.ART/'samplers/status.json')
        if not train_live and not sampler_live:
            if sampler['status']!='COMPLETE':
                status('UPSTREAM_NOT_COMPLETE',sampler_status=sampler['status']);return
            break
        status('WAITING_FOR_REGISTERED_PHASES',training_alive=train_live,sampler_alive=sampler_live,
            sampler_status=sampler['status'],remaining_seconds=search.remaining())
        time.sleep(40)
    else:
        status('NO_TIME_OR_STOPPED');return
    assert sha(__file__)==plan['source_sha256']
    assert sha(search.ART/'protocol.json')==plan['upstream_protocol_sha256']
    with (search.ART/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine.frozen()
        winner=read(search.ART/'long_candidate.json')['candidate']
        assert winner!=search.BASE
        remaining=search.remaining()
        costs={step:2*(step-START)*.44+300+2*(50000/4096*75+90)+900 for step in BUDGETS}
        eligible=[step for step in BUDGETS if costs[step]<=remaining]
        if not eligible:
            status('NO_TIME_FOR_COMPLETE_PAIR',estimated_seconds=costs,remaining_seconds=remaining);return
        budget=eligible[0];pair=(search.BASE,winner)
        save(ART/'selection.json',dict(at_utc=now(),pair=pair,updates=budget,
            estimated_seconds=costs[budget],remaining_seconds=remaining,
            source_selection_sha256=sha(search.ART/'long_candidate.json'),
            checkpoints={arm:engine.checkpoint(arm,4817,START) for arm in pair}))
        for arm in pair:
            status('TRAINING',arm=arm,updates=budget)
            if engine.run(arm,4817,budget,resume=engine.checkpoint(arm,4817,START),resume_step=START) is None:
                status('INCOMPLETE_TRAINING_PAIR',arm=arm,updates=budget);return
        for arm in pair:
            status('FULL50K',arm=arm,updates=budget)
            if engine.run(arm,4817,budget,scenes=50000,
                          resume=engine.checkpoint(arm,4817,budget),evaluation=True) is None:
                status('INCOMPLETE_FULL50K_PAIR',arm=arm,updates=budget);return
        engine.refresh()
        status('COMPLETE',pair=pair,updates=budget,remaining_seconds=search.remaining())
        engine.status('EXTENDED_CONFIRMATION_COMPLETE',pair=pair,updates=budget)


if __name__=='__main__':
    if '--register-only' in sys.argv:register()
    else:main()
