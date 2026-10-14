"""Conditional late TRAIN screen of lane-only queries after the verified pair."""
from datetime import datetime
from pathlib import Path
import fcntl
import inspect
import os
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_init_architecture_axes_20261014 as arch
from tools import confirm_init_architecture_budget_20261014 as confirmation
from tools import audit_init_architecture_axes_20261014 as audit
from tools import score_init_sd_vw_targets_20261014 as targets
read, save, sha, now = arch.read, arch.save, arch.sha, arch.now
ART = arch.ART / 'mask_followup'
RUNS = arch.RUNS / 'mask_followup'
END = '2026-10-14T22:12:00+08:00'
DEADLINE = datetime.fromisoformat(END).timestamp()
engine = arch.engine


def sources():
    return {**confirmation.sources(), 'tools/continue_init_map_mask_20261014.py': sha(__file__)}


def main():
    ART.mkdir(parents=True, exist_ok=True)
    plan = dict(registered_utc=now(), deadline=END, source_hashes=sources(),
                upstream_protocol_sha256=sha(arch.ART / 'protocol.json'),
                arm='lane_only_queries', steps=2048, TRAIN=4817, GEN=[2817, 3817], scenes=4096,
                choice='Fixed before any mask-training result; no outcome-based selection.',
                scope='Fresh G1/optimizer/EMA. Sole change: road_edges -> lanes_only. '
                      'Reuse the exact same-source-phase2048 roadK8 control, not a newly '
                      'trained control. No mid/full/secondTRAIN superiority claim.',
                priority='Only after paired6144/full50K and optional TRAIN-size calibration terminate normally.',
                reserve='No new stage unless the conservative estimate fits by22:12. Reserve18minutes for report/QA; no simultaneous native GPU jobs.')
    save(ART / 'queue_plan.json', plan)
    owners = [read(arch.ART / 'budget_handoff_pid.json')['pid'],
              read(arch.ART / 'size_calibration/pid.json')['pid']]
    needed = 2048 * .44 + 150 + 120
    while any(engine.shared.alive(pid) for pid in owners):
        if DEADLINE - time.time() < needed:
            save(ART / 'status.json', dict(status='NO_TIME_AFTER_PRIORITY_QUEUES', at_utc=now()))
            return
        save(ART / 'status.json', dict(status='WAITING_FOR_PRIORITY_QUEUES', at_utc=now(), pid=os.getpid(), upstream_pids=owners))
        time.sleep(30)
    assert read(arch.ART / 'status.json')['status'] == 'ARCHITECTURE_PLAN_COMPLETE'
    if DEADLINE - time.time() < needed:
        save(ART / 'status.json', dict(status='NO_TIME_FOR_COMPLETE_MASK_SCREEN', at_utc=now()))
        return
    assert sources() == plan['source_hashes']
    assert sha(arch.ART / 'protocol.json') == plan['upstream_protocol_sha256']
    with (arch.PARENT / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not engine.occupied()
        rows = read(arch.ART / 'observations.json')
        control = [r for r in rows if (r['variant'], r['seed'], r['budget'], r['num_samples']) == ('roadK8', 4817, 2048, 4096)]
        assert len(control) == 2
        for name in ('variants.json', 'recipe.json', 'full50000.json'):
            save(ART / name, read(arch.ART / name))
        save(ART / 'observations.json', control)
        save(ART / 'protocol.json', dict(phase='LATE_MAP_MASK_SCREEN', source_hashes=sources(),
             inputs={str(p): sha(p) for p in (engine.shared.MAP, engine.shared.NORMALIZER, engine.shared.EVAL_MANIFEST, engine.shared.TRAIN_MANIFEST)},
             local_input_hashes={p: sha(ART / p) for p in ('variants.json', 'recipe.json', 'full50000.json')},
             queue_plan_sha256=sha(ART / 'queue_plan.json'), reused_control_rows=control))
        engine.ART, engine.RUNS = ART, RUNS
        engine.DEADLINE, engine.END = DEADLINE, END
        engine.sources, engine.verify = sources, arch.verify
        engine.evidence.score = arch.previous.paired_score
        runner = inspect.getsource(engine.run)
        key = "'tools/runtime_checkpointed_sweep.py'"
        assert runner.count(key) == 1
        exec(compile(runner.replace(key, "'tools/runtime_init_architecture_axes_20261014.py'"), str(__file__), 'exec'), engine.__dict__)
        def stop(*unused):
            engine.STOP = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        values = engine.run('lane_only_queries', 4817, 2048)
        if values is None:
            return
        audit.ART = ART
        audit.main()
        save(ART / 'result.json', dict(at_utc=now(), raw_rows=values, control_rows=control,
            score=arch.previous.paired_score(values, control),
            new_actual_evaluation_cells=2, reused_control_cells=2,
            per_GEN_targets=[dict(GEN=r['eval_seed'], **targets.score(r)) for r in values],
            interpretation=plan['scope'], full128epoch_verified=False))
        engine.status('MASK_SCREEN_COMPLETE')


if __name__ == '__main__':
    main()
