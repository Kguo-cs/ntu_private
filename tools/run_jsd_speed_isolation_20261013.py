"""Registered adaptive follow-up to a completed marginal-loss failure.

Keeps the original15:00 deadline. Stop the just-started longer-budget job,
exclude its incomplete output, isolate speed marginal loss and type grouping.
The original round2 registration and complete observations remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_jsd_round2_20261013 as core

OLD=ROOT/"output/reports/init_diffusion_jsd_round2_20261013"
ART=OLD/"speed_isolation"
RUNS=ROOT/"logs/init_diffusion_jsd_speed_isolation_20261013"
OLD_CORE_FILE=Path(core.__file__).resolve()
original_sources=core.sources
core.ART,core.RUNS=ART,RUNS
core.__file__=str(Path(__file__).resolve())
def sources():
    return {**original_sources(),str(OLD_CORE_FILE.relative_to(ROOT)):core.sha(OLD_CORE_FILE)}
core.sources=sources


def prepare():
    assert not (ART/"protocol.json").exists()
    old=core.read(OLD/"variants.json")
    variants={name:copy.deepcopy(old[name]) for name in old}
    variants["speed_only"]=copy.deepcopy(old["lane_attr_ot"])
    variants["speed_only"]["__config"][core.P+"distribution_loss"].update(length=0.,width=0.)
    variants["speed_pooled"]=copy.deepcopy(variants["speed_only"])
    variants["speed_pooled"]["__config"][core.P+"distribution_loss"]["by_type"]=False
    core.save(ART/"variants.json",variants)
    core.save(ART/"recipe.json",core.read(OLD/"recipe.json"))
    observations=core.read(OLD/"observations.json")
    oldplan=core.read(OLD/"protocol.json")
    plan={"registered_utc":core.now(),"deadline":core.DEADLINE,
        "adaptive_reason":"Joint attribute marginal loss improved speed but harmed width (TRAIN5817 step4096: speed ratio.7415, width2.0806, R_jsd.9567). Isolate speed, then type-balanced versus pooled matching. Incomplete long-stage excluded.",
        "decision_evidence_sha256":core.sha(OLD/"comparisons.json"),
        "previous_registration_sha256":core.sha(OLD/"protocol.json"),
        "order":[{"arm":"speed_only","seed":5817,"steps":4096,"checkpoints":[2048]},
                 {"arm":"speed_pooled","seed":5817,"steps":2048,"checkpoints":[]}],
        "selection":"Do not cross-rank pooled2048 against4096. Optional expanded validation picks smallest eligible TRAIN5817/4096 R_jsd among speed_only, lane_only and prior lane_agentmax05. No forced new-method winner.",
        "original15h_deadline_preserved":True,"fresh_G1_optimizer_EMA":True,
        "sampling_steps":20,"parameter_cap":8100000,"actual_parameters":8083978,
        "GEN":[2817,3817],"scenes":4096,"precision":"32-true","matmul":"medium",
        "guidance":"OFF","primary":"six-JSD R; collision auxiliary",
        "inputs":oldplan["inputs"],"source_hashes":sources(),
        "imported_sources":{str(p.relative_to(ROOT)):core.sha(p) for p in
            [OLD/"observations.json",OLD/"comparisons.json",OLD/"protocol.json"]},
        "local_input_hashes":{p:core.sha(ART/p) for p in ("recipe.json","variants.json")},
        "no6826_or128epoch_completion_claim":True}
    core.save(ART/"protocol.json",plan)
    core.save(ART/"observations.json",observations)
    core.refresh(observations)
    return plan


def execute(plan):
    lock=(ART/"driver.lock").open("a");fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    def halt(_number,_frame):core.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    rows=core.read(ART/"observations.json")
    for spec in copy.deepcopy(plan["order"]):
        if core.remaining()<.41*spec["steps"]+120:
            core.status("NO_TIME_FOR_NEW_STAGE",next=spec);return
        values=core.run_stage(spec,plan)
        if values is None:return
        rows.extend(values);core.save(ART/"observations.json",rows);core.refresh(rows)
    if core.remaining()>=360:
        eligible=[r for r in core.read(ART/"comparisons.json")["comparisons"]
            if r["budget"]==4096 and r["TRAIN_seeds"]==[5817]
            and r["arm"] in ("speed_only","lane_only","lane_agentmax05")
            and r["gates_pass"] and r["R_jsd"]<1]
        if eligible:
            winner=min(eligible,key=lambda r:r["R_jsd"])
            cells=[r for r in rows if r["variant"] in (winner["arm"],"fresh657")
                and r["seed"]==5817 and r["budget"]==4096]
            checkpoints={r["variant"]:r["checkpoint"] for r in cells}
            assert len(checkpoints)==2
            core.save(ART/"expanded_selection.json",{"selected_utc":core.now(),"comparison":winner,
                "source_sha256":core.sha(ART/"comparisons.json")})
            spec={"arm":winner["arm"],"seed":5817,"steps":4096,"checkpoints":[],
                "scenes":16384,"evaluation":checkpoints}
            values=core.run_stage(spec,plan)
            if values is None:return
            for r in values:assert r["num_samples"]==16384 and r["evaluation_only"] and r["evaluation_weights"]=="ema"
            result=core.score([r for r in values if r["variant"]==winner["arm"]],
                              [r for r in values if r["variant"]=="fresh657"])
            core.save(ART/"expanded_result.json",{"arm":winner["arm"],"budget":4096,"scenes":16384,**result})
            core.save(ART/"expanded_observations.json",values)
    core.status("COMPLETE",observations=len(rows),remaining_seconds=core.remaining())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:core.worker(args.worker)
    elif args.execute:execute(core.read(ART/'protocol.json'))
    else:
        plan=prepare();print(plan['registered_utc'],plan['deadline'])
