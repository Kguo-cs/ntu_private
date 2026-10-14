"""Equal4096 mask confirmation using an already audited control snapshot."""
from datetime import datetime
from pathlib import Path
import fcntl
import inspect
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_init_architecture_axes_20261014 as arch
from tools import confirm_init_architecture_budget_20261014 as prior
from tools import audit_init_architecture_axes_20261014 as audit
read,save,sha,now=arch.read,arch.save,arch.sha,arch.now
ART=arch.ART/'mask_midpoint';RUNS=arch.RUNS/'mask_midpoint'
engine=arch.engine


def sources():
    return {**prior.sources(),'tools/confirm_init_lane_mask_4096_20261014.py':sha(__file__)}


def main():
    ART.mkdir(parents=True,exist_ok=True)
    control=arch.RUNS/'roadK8_T4817_6144_N4096/roadK8/step_4096.ckpt'
    candidate=read(arch.ART/'mask_followup/result.json')['raw_rows'][0]['checkpoint']
    assert control.is_file()
    with (arch.PARENT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not engine.occupied()
        assert read(arch.ART/'protocol.json')['source_hashes']==prior.sources()
        for name in ('variants.json','recipe.json','full50000.json','prepared_preflight.json'):
            save(ART/name,read(arch.ART/name))
        save(ART/'observations.json',[])
        save(ART/'protocol.json',dict(phase='LANE_MASK_PAIRED4096',registered_utc=now(),
            source_hashes=sources(),inputs={str(p):sha(p) for p in (engine.shared.MAP,engine.shared.NORMALIZER,engine.shared.TRAIN_MANIFEST,engine.shared.EVAL_MANIFEST)},
            local_input_hashes={p:sha(ART/p) for p in ('variants.json','recipe.json','full50000.json')},
            control_checkpoint=str(control),control_checkpoint_sha256=sha(control),
            candidate_parent=str(candidate),candidate_parent_sha256=sha(candidate),
            scope='Control step4096 saved within the independently audited6144 training stage. Candidate resumes its own2048 optimizer/EMA/RNG to4096. Same4096 cohort and bothGEN. No full50K or128epoch claim.',
            deadline='2026-10-14T22:22:00+08:00',
            scheduling='Measured lane-only2048 train rate is approximately0.335s/update. Use0.35s/update plus45s startup/eval estimate; deadline already reserves8minutes for prepared report/QA. Partial stages remain excluded.'))
        engine.ART,engine.RUNS=ART,RUNS
        engine.END='2026-10-14T22:22:00+08:00'
        engine.DEADLINE=datetime.fromisoformat(engine.END).timestamp()
        engine.sources,engine.verify=sources,arch.verify
        engine.evidence.score=arch.previous.paired_score
        runner=inspect.getsource(engine.run)
        key="'tools/runtime_checkpointed_sweep.py'";assert runner.count(key)==1
        runner=runner.replace(key,"'tools/runtime_init_architecture_axes_20261014.py'")
        assert '(steps - resume_step) * .44 + 150' in runner
        runner=runner.replace('(steps - resume_step) * .44 + 150','(steps - resume_step) * .35 + 45')
        runner=runner.replace('estimate + 120','estimate + 0')
        exec(compile(runner,str(__file__),'exec'),engine.__dict__)
        # Start the only training stage first; the control evaluation is cheap
        # and its state is already available without another training job.
        if engine.run('lane_only_queries',4817,4096,resume=candidate,resume_step=2048) is None:return
        if engine.run('roadK8',4817,4096,resume=str(control),evaluation=True) is None:return
        audit.ART=ART;audit.main()
        engine.status('PAIRED4096_COMPLETE')


if __name__=='__main__':main()
