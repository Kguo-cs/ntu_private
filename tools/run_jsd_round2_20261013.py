"""JSD-first round2: controlled lane ablation, marginal attributes, longer budget.

All new training starts with a fresh G1/optimizer/EMA. Only completed native
stages enter comparisons; imported controls are immutable, hash-recorded, and
must match exposure, initialization and scene membership before scoring.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_jsd_priority_20261013 as previous
from tools import run_wandb_cc_scratch_20261013 as shared
from tools import run_init_diffusion_657_iteration as native

read, save, sha, now = shared.read, shared.save, shared.sha, shared.now
ART = ROOT / "output/reports/init_diffusion_jsd_round2_20261013"
RUNS = ROOT / "logs/init_diffusion_jsd_round2_20261013"
DEADLINE = "2026-10-13T15:00:00+08:00"
P = shared.PREFIX
STOP = False


def recipe(arm):
    result = read(ART / "recipe.json")
    for name, change in read(ART / "variants.json")[arm].items():
        if not name.startswith("__"):
            result["components"][name].update(change)
    return result


def sources():
    files = shared.SOURCE_FILES + ["src/smart/modules/map_decoder.py",
        "src/smart/metrics/relative_score.py", "tools/run_jsd_priority_20261013.py",
        str(Path(__file__).relative_to(ROOT))]
    return {p:sha(ROOT/p) for p in files}


def prepare():
    assert not (ART / "protocol.json").exists(), "Use a fresh registered round"
    variants = read(previous.ART / "variants.json")
    selected = {name:copy.deepcopy(variants[name]) for name in
                ("fresh657", "freshK8", "lane_only", "lane_agentmax05")}
    selected["lane_attr_ot"] = copy.deepcopy(selected["lane_agentmax05"])
    selected["lane_attr_ot"]["__config"][P+"distribution_loss"] = {
        "speed":.25, "length":.1, "width":.1,
        "time_min":.5, "kind":"mse", "by_type":True}
    save(ART / "variants.json", selected)
    save(ART / "recipe.json", read(previous.ART / "recipe.json"))
    imported = read(previous.ART / "observations.json")
    old_controls_path = ROOT / "logs/control657_paired_20261012/train4817/summary.json"
    old_controls = [copy.deepcopy(r) for r in read(old_controls_path) if r["budget"] in (4096,6826)]
    # Existing4096 control is absent for TRAIN4817 in the preceding round.
    for row in old_controls:
        row["variant"] = "fresh657"
    imported.extend(old_controls)
    for row in imported:
        report = read(row["metrics_path"])
        assert row["num_samples"] == report["num_samples"] == 4096
        assert row["evaluation_weights"] == "ema" and report["decoder"]["sampling_steps"] == 20
        assert report["decoder"]["ema_num_updates"] == row["budget"]
        assert all(math.isfinite(row[k]) for k in shared.METRICS)
    plan = {"registered_utc":now(), "deadline":DEADLINE,
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "order":[{"arm":"lane_only","seed":5817,"steps":4096,"checkpoints":[2048]},
                 {"arm":"lane_attr_ot","seed":5817,"steps":4096,"checkpoints":[2048]}],
        "adaptive_long":{"steps":6826,"checkpoints":[4096],"seed":4817,
            "selection":"From newly completed TRAIN5817/4096 arms lane_only/lane_attr_ot and prior lane_agentmax05: smallest R_jsd among coverage/count/attribute gates passing. Baseline is allowed to win; no forced improvement claim."},
        "optional_expanded_validation":{"scenes":16384,"minimum_seconds_remaining":360,
            "selection":"Best eligible completed comparison at largest new budget; same checkpoints, matched657, both GEN; search4096 retained, not independent test."},
        "primary":"six-JSD candidate/matched657 ratio mean; collision auxiliary",
        "GEN":[2817,3817],"scenes":4096,"batch_size":512,"parameters":8083978,
        "parameter_cap":8100000,"sampling_steps":20,"guidance":"OFF",
        "fresh_initialization":True,"frozen_pretrained_map":True,
        "quick_workers":0,"precision":"32-true","matmul":"medium",
        "full_training_profile":"8workers; validate every8epochs",
        "source_hashes":sources(),
        "inputs":{p:sha(ROOT/p) for p in [str(shared.MAP.relative_to(ROOT)),
            str(shared.NORMALIZER.relative_to(ROOT)),str(shared.TRAIN_MANIFEST.relative_to(ROOT)),
            str(shared.EVAL_MANIFEST.relative_to(ROOT))]},
        "local_input_hashes":{name:sha(ART/name) for name in ("recipe.json","variants.json")},
        "imported_sources":{str(p.relative_to(ROOT)):sha(p) for p in [previous.ART/"observations.json",
            previous.ART/"protocol.json",old_controls_path,old_controls_path.parent/"verified_completion.json"]},
        "imported_control_scope":"Resolved config, initialization/data/map/normalizer hashes and native histograms must pair; earlier source differences are score logging, OFF-guidance code, constructor defaults, overridden run experiment. No unchanged-source claim.",
        "no_final128epoch_claim":True}
    save(ART/"protocol.json",plan)
    save(ART/"observations.json",imported)
    refresh(imported)
    return plan


def frozen(plan):
    assert sources() == plan["source_hashes"], "Source changed during experiment"
    for table in ("inputs","imported_sources"):
        for p,digest in plan[table].items():assert sha(ROOT/p)==digest
    for name,digest in plan["local_input_hashes"].items():assert sha(ART/name)==digest


def setup_native():
    native.ART,native.RUNS=ART,RUNS
    previous.ART,previous.RUNS=ART,RUNS
    previous.effective_recipe=recipe


def worker(specfile):
    setup_native()
    spec=read(specfile)
    if spec.get("evaluation"):
        rows=native.stage(spec["label"],[],spec["steps"],spec["scenes"],seed=spec["seed"],
            gens=tuple(shared.GENS),evaluation=spec["evaluation"])
    else:
        rows=native.stage(spec["label"],[spec["arm"]],spec["steps"],4096,
            seed=spec["seed"],checkpoints=tuple(spec["checkpoints"]),gens=tuple(shared.GENS))
        previous.verify(rows,spec)
    print("ROUND2_NATIVE_SUCCESS",flush=True)


def score(candidate,controls):
    for row in candidate:
        base=next(c for c in controls if (c["seed"],c["budget"],c["eval_seed"],c["eval_manifest_sha256"])==
                  (row["seed"],row["budget"],row["eval_seed"],row["eval_manifest_sha256"]))
        assert row["initial_generator_sha256"]==base["initial_generator_sha256"]
        a=read(Path(row["metrics_path"]).parent/"feature_histograms.json")
        b=read(Path(base["metrics_path"]).parent/"feature_histograms.json")
        assert all(a[k]["matched_gt"]==b[k]["matched_gt"] for k in shared.METRICS[:6])
    result=native.score(candidate,controls)
    for r in [result,*result["per_TRAIN"].values()]:
        r["R_jsd"]=statistics.fmean(r["ratios"][k] for k in shared.METRICS[:6])
        r["R_geometry"]=statistics.fmean(r["ratios"][k] for k in shared.METRICS[:3])
    return result


def refresh(rows):
    controls=[r for r in rows if r["variant"]=="fresh657"]
    results=[]
    for arm in sorted({r["variant"] for r in rows}-{"fresh657"}):
        for budget in sorted({r["budget"] for r in rows if r["variant"]==arm}):
            for seed in sorted({r["seed"] for r in rows if r["variant"]==arm and r["budget"]==budget}):
                cells=[r for r in rows if (r["variant"],r["budget"],r["seed"])==(arm,budget,seed)]
                if len(cells)!=2 or not all(any((c["seed"],c["budget"],c["eval_seed"])==
                    (r["seed"],r["budget"],r["eval_seed"]) for c in controls) for r in cells):continue
                results.append({"arm":arm,"budget":budget,**score(cells,controls)})
    save(ART/"comparisons.json",{"generated_utc":now(),"primary":"R_jsd","comparisons":results})


def status(value,**kw):
    save(ART/"status.json",{"status":value,"updated_utc":now(),"driver_pid":os.getpid(),**kw})


def remaining():return datetime.fromisoformat(DEADLINE).timestamp()-time.time()


def run_stage(spec,plan):
    frozen(plan)
    if shared.gpu_pids():
        status("SKIPPED_GPU_OCCUPIED",stage=spec,gpu_pids=shared.gpu_pids());return None
    spec["label"]=f"train{spec['seed']}_{spec['arm']}_{spec['steps']}" if not spec.get("evaluation") else "expanded"
    folder=RUNS/spec["label"]
    assert not folder.exists(),"Inspect incomplete stage before retry"
    folder.mkdir(parents=True)
    path=ART/(spec["label"]+".spec.json");save(path,spec)
    started=time.monotonic()
    with (folder/"driver_console.log").open("w") as log:
        child=subprocess.Popen([shared.PYTHON,str(Path(__file__)),"--worker",str(path)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            status("RUNNING",stage=spec,child_pid=child.pid,elapsed_seconds=time.monotonic()-started)
            if STOP or remaining()<=0:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                status("PARTIAL_EXCLUDED",stage=spec);return None
            time.sleep(10)
    if child.returncode!=0 or "ROUND2_NATIVE_SUCCESS" not in (folder/"driver_console.log").read_text():
        status("FAILED_STAGE",stage=spec,returncode=child.returncode);return None
    frozen(plan)
    marker=read(folder/"verified_completion.json")
    assert marker["status"]=="VERIFIED_COMPLETE" and marker["summary_sha256"]==sha(folder/"summary.json")
    rows=read(folder/"summary.json")
    if not spec.get("evaluation"):
        setup_native(); previous.verify(rows,spec)
    save(folder/"round2_verified_completion.json",{"status":"VERIFIED_COMPLETE","finished_utc":now(),
        "protocol_sha256":sha(ART/"protocol.json"),"summary_sha256":sha(folder/"summary.json"),
        "elapsed_seconds":time.monotonic()-started,"native_marker_sha256":sha(folder/"verified_completion.json")})
    return rows


def execute(plan):
    global STOP
    lock=(ART/"driver.lock").open("a");fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    def halt(_signum,_frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    rows=read(ART/"observations.json")
    for spec in copy.deepcopy(plan["order"]):
        if remaining()<.43*spec["steps"]+120:status("NO_TIME_FOR_NEW_STAGE",next=spec);return
        values=run_stage(spec,plan)
        if values is None:return
        rows.extend(values);save(ART/"observations.json",rows);refresh(rows)
    eligible=[r for r in read(ART/"comparisons.json")["comparisons"] if r["budget"]==4096
        and r["TRAIN_seeds"]==[5817] and r["arm"] in ("lane_only","lane_attr_ot","lane_agentmax05")
        and r["gates_pass"] and r["R_jsd"]<1]
    if not eligible:status("NO_ELIGIBLE_LONG_CANDIDATE");return
    winner=min(eligible,key=lambda r:r["R_jsd"])
    save(ART/"long_selection.json",{"selection_utc":now(),"arm":winner["arm"],"comparison":winner,
        "comparisons_sha256":sha(ART/"comparisons.json")})
    spec={"arm":winner["arm"],"seed":4817,"steps":6826,"checkpoints":[4096]}
    if remaining()<.43*6826+180:status("NO_TIME_FOR_LONG",selected=spec);return
    values=run_stage(spec,plan)
    if values is None:return
    rows.extend(values);save(ART/"observations.json",rows);refresh(rows)
    if remaining()>=360:
        completed=score([r for r in rows if r["variant"]==winner["arm"] and r["seed"]==4817 and r["budget"]==6826],
                        [r for r in rows if r["variant"]=="fresh657"])
        if completed["gates_pass"] and completed["R_jsd"]<1:
            cells=[r for r in rows if r["variant"] in (winner["arm"],"fresh657") and r["seed"]==4817 and r["budget"]==6826]
            checkpoints={r["variant"]:r["checkpoint"] for r in cells}
            assert len(checkpoints)==2
            spec={"arm":winner["arm"],"seed":4817,"steps":6826,"checkpoints":[],"scenes":16384,"evaluation":checkpoints}
            values=run_stage(spec,plan)
            if values is None:return
            for r in values:
                assert r["num_samples"]==16384 and r["evaluation_only"] and r["evaluation_weights"]=="ema"
            result=score([r for r in values if r["variant"]==winner["arm"]],[r for r in values if r["variant"]=="fresh657"])
            save(ART/"expanded_result.json",{"arm":winner["arm"],"budget":6826,"scenes":16384,**result})
            save(ART/"expanded_observations.json",values)
    status("COMPLETE",observations=len(rows),remaining_seconds=remaining())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--worker",type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.execute:execute(read(ART/"protocol.json"))
    else:
        plan=prepare();print(json.dumps({"registered":str(ART/"protocol.json"),"deadline":plan["deadline"]}))
