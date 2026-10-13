"""Optional 16,384-scene native confirmation within the existing deadline.

Selection is registered before the candidate's new metrics are available. This
expands validation membership; it is not an independent unseen test set.
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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_jsd_priority_20261013 as primary
from tools import run_init_diffusion_657_iteration as legacy
from tools import run_wandb_cc_scratch_20261013 as original
read, save, sha, now = original.read, original.save, original.sha, original.now
ART = ROOT / "output/reports/wandb_cc_20261013/expanded"
RUNS = ROOT / "logs/wandb_cc_jsd_expanded_20261013"
SCENES = 16384
MANIFEST = ROOT / "tmp/control657_paired_20261012/eval16384_files.json"
FULL_MANIFEST = ROOT / "logs/init_diffusion_657_edges_20261013/full_control4817/eval_files.txt"


def prepare():
    names = FULL_MANIFEST.read_text().splitlines()
    search_names = read(original.EVAL_MANIFEST)
    if isinstance(search_names, dict):
        search_names = search_names["files"]
    assert len(names) == len(set(names)) == 50000
    assert set(search_names).issubset(names)
    search_set = set(search_names)
    selected = search_names + [name for name in names if name not in search_set][:SCENES-len(search_names)]
    assert len(selected) == SCENES and set(search_names).issubset(selected)
    if MANIFEST.exists():
        assert read(MANIFEST) == selected
    else:
        save(MANIFEST, selected)
    for name, value in (("recipe.json", read(primary.ART / "recipe.json")), ("variants.json", {})):
        if (ART / name).exists():
            assert read(ART / name) == value
        else:
            save(ART / name, value)
    plan = {"registered_utc": now(), "deadline": original.DEADLINE, "scenes": SCENES,
        "sampling_steps": 20, "guidance": "OFF", "GEN": original.GENS,
        "minimum_seconds_remaining": 360,
        "selection": "Among completed current paired comparisons with R_jsd<1 and all gates passing: largest training budget first, then smallest R_jsd; expand its one-TRAIN candidate and exact current657 control together.",
        "comparison_metric": "six-JSD R; collision auxiliary",
        "expanded_validation_not_independent_test": True,
        "membership": "Original ordered4096 search scenes, then first12288 other names from the verified full50k manifest",
        "overlap_with_search_scenes": len(set(selected) & set(search_names)),
        "full_manifest_sha256": sha(FULL_MANIFEST), "eval_manifest_sha256": sha(MANIFEST),
        "ordered_names_sha256": __import__("hashlib").sha256(("\n".join(selected)+"\n").encode()).hexdigest(),
        "source_hashes": {**primary.source_hashes(), "tools/run_jsd_expanded_eval_20261013.py":sha(Path(__file__))}}
    if (ART / "protocol.json").exists():
        old = read(ART / "protocol.json")
        plan["registered_utc"] = old["registered_utc"]
        assert old == plan
    else:
        save(ART / "protocol.json", plan)
    return plan


def worker(spec_file):
    spec = read(spec_file)
    legacy.ART, legacy.RUNS = ART, RUNS
    rows = legacy.stage("confirmation", [], spec["budget"], SCENES,
                        seed=spec["seed"], gens=tuple(original.GENS), evaluation=spec["checkpoints"])
    assert len(rows) == 4
    print("EXPANDED_NATIVE_SUCCESS", flush=True)


def execute(plan):
    lock = (ART / "driver.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(value, **kw):
        save(ART / "status.json", {"status":value,"updated_utc":now(),"driver_pid":os.getpid(),**kw})
    terminal = {"COMPLETE", "NO_NEW_STAGE_BEFORE_DEADLINE", "NO_TIME_AFTER_GPU_WAIT"}
    while True:
        if original.deadline_timestamp()-time.time() < plan["minimum_seconds_remaining"]:
            status("NOT_RUN_INSUFFICIENT_TIME")
            return
        state = read(primary.ART / "status.json")["status"]
        if state in terminal:
            break
        if state in ("FAILED_STAGE", "PARTIAL_STAGE_EXCLUDED"):
            status("NOT_RUN_PRIMARY_FAILED")
            return
        status("WAITING_FOR_TRAINING_GRID")
        time.sleep(10)
    if original.gpu_pids():
        status("NOT_RUN_GPU_OCCUPIED", gpu_pids=original.gpu_pids())
        return
    eligible = [r for r in read(primary.ART / "comparisons.json")["comparisons"]
                if r["R_jsd"] < 1 and r["all_TRAIN_gates_pass"] and len(r["TRAIN_seeds"])==1]
    if not eligible:
        status("NOT_RUN_NO_ELIGIBLE_CANDIDATE")
        return
    winner = min(eligible, key=lambda r:(-r["budget"],r["R_jsd"]))
    rows = read(primary.ART / "observations.json")
    chosen = [r for r in rows if r["variant"] in ("fresh657",winner["arm"])
              and r["seed"]==winner["TRAIN_seeds"][0] and r["budget"]==winner["budget"]]
    assert len(chosen)==4
    checkpoints={r["variant"]:r["checkpoint"] for r in chosen}
    spec={"budget":winner["budget"],"seed":winner["TRAIN_seeds"][0],"checkpoints":checkpoints,
          "selection_comparisons_sha256":sha(primary.ART/"comparisons.json"),
          "checkpoint_sha256":{k:sha(p) for k,p in checkpoints.items()}}
    save(ART / "selection.json", spec)
    for path,digest in plan["source_hashes"].items():
        assert sha(ROOT/path)==digest
    assert sha(MANIFEST)==plan["eval_manifest_sha256"]
    command=[original.PYTHON,str(Path(__file__).resolve()),"--worker",str(ART/"selection.json")]
    start=time.monotonic()
    with (ART/"driver_console.log").open("w") as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            status("RUNNING",child_pid=child.pid,elapsed_seconds=time.monotonic()-start)
            if time.time()>=original.deadline_timestamp():
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                status("PARTIAL_EXCLUDED")
                return
            time.sleep(10)
    assert child.returncode==0 and "EXPANDED_NATIVE_SUCCESS" in (ART/"driver_console.log").read_text()
    folder=RUNS/"confirmation"
    native=read(folder/"verified_completion.json")
    assert native["status"]=="VERIFIED_COMPLETE" and native["returncode"]==0
    assert native["summary_sha256"]==sha(folder/"summary.json")
    values=read(folder/"summary.json")
    for row in values:
        report=read(row["metrics_path"])
        assert row["num_samples"]==report["num_samples"]==report["num_gt_samples"]==SCENES
        assert row["evaluation_only"] and row["evaluation_weights"]=="ema"
        assert row["generator_num_parameters"]==8083978 and row["map_frozen"]
        assert row["eval_manifest_sha256"]==plan["ordered_names_sha256"]
        assert report["reference_mode"]=="matched" and report["map_source"]=="reference"
        assert report["unverified_generation_timestep_count"]==0
        assert report["decoder"]["sampling_steps"]==20
        assert report["decoder"]["ema_num_updates"]==winner["budget"]
        assert report["runtime"]["torch_initial_seed"]==row["eval_seed"]
        assert report["runtime"]["trainer_precision"]=="32-true"
        assert report["runtime"]["float32_matmul_precision"]=="medium"
        for key in original.METRICS:
            assert math.isfinite(row[key]) and row[key]==report["agent_metrics"][key]
    candidate=[r for r in values if r["variant"]==winner["arm"]]
    control=[r for r in values if r["variant"]=="fresh657"]
    result=legacy.score(candidate,control)
    result.update(R_jsd=sum(result["ratios"][k] for k in original.METRICS[:6])/6,
                  R_geometry=sum(result["ratios"][k] for k in original.METRICS[:3])/3)
    for path,digest in plan["source_hashes"].items():assert sha(ROOT/path)==digest
    for name,path in checkpoints.items():assert sha(path)==spec["checkpoint_sha256"][name]
    save(ART/"result.json",{"arm":winner["arm"],"budget":winner["budget"],"scenes":SCENES,
         "overlap_with_search_scenes":plan["overlap_with_search_scenes"],**result})
    save(ART/"observations.json",values)
    save(ART/"verified_completion.json",{"status":"VERIFIED_COMPLETE","returncode":0,"finished_utc":now(),
         "protocol_sha256":sha(ART/"protocol.json"),"result_sha256":sha(ART/"result.json"),
         "native_reports":{r["metrics_path"]:sha(r["metrics_path"]) for r in values}})
    status("COMPLETE",elapsed_seconds=time.monotonic()-start)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--worker",type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:
        plan=prepare()
        if args.execute:execute(plan)
        else:print(json.dumps({"status":"REGISTERED_NOT_RUN","protocol":str(ART/"protocol.json")}))
