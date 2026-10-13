"""Deadline-bounded length-only residual after width regression was observed."""
import argparse
import copy
import csv
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_init_shape_matching_20261013 as prior
core=prior.core;read,save,sha,now=prior.read,prior.save,prior.sha,prior.now
OLD=prior.ART;ART=OLD/'length_only';RUNS=prior.RUNS/'length_only'
original_sources=prior.sources
BUDGET=3413  # Existing matched roadK8 control; fit/evaluate before22:30.


def sources():return {**original_sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}


def competing_gpu_pids():
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader'],capture_output=True,text=True,check=True)
    occupied=[];display=[]
    for values in csv.reader(query.stdout.splitlines()):
        if len(values)!=3 or not values[0].strip().isdigit():raise RuntimeError('Unknown GPU process record')
        pid,name,mem=values;pid=int(pid);name=name.strip();memory=int(mem.strip().split()[0])
        if name=='/opt/todesk/bin/ToDesk_Session' and memory<=1024:
            display.append(dict(pid=pid,name=name,memory_MiB=memory));continue
        occupied.append(pid)
    save(ART/'gpu_occupancy.json',dict(checked_utc=now(),noncompeting_display=display,competing=occupied,
                                    policy='Never signal any display or external task; known ToDesk<=1GiB is background display, other processes block fit.'))
    return sorted(occupied)


def prepare():
    assert not (ART/'protocol.json').exists()
    rows=read(OLD/'observations.json');assert len(rows)==2 and all(r['variant']=='K8_size_residual' for r in rows)
    complete=read(prior.RUNS/'K8_size_residual_T4817_4096/verified_completion.json')
    assert complete['status']=='VERIFIED_COMPLETE'
    base=copy.deepcopy(read(OLD/'variants.json')['K8_size_residual'])
    base['__config'][core.shared.PREFIX+'shape_x0_parameterization']='length_time_residual'
    save(ART/'variants.json',dict(K8_length_residual=base))
    for name in ('recipe.json','controls.json'):save(ART/name,read(OLD/name))
    plan=copy.deepcopy(read(OLD/'protocol.json'))
    plan.update(registered_utc=now(),source_hashes=sources(),
                local_input_hashes={n:sha(ART/n) for n in ('recipe.json','variants.json','controls.json')},
                initial_order=['K8_length_residual'],
                reason='Both-size residual improved focus21% but worsenedwidth37%. Narrow output change tolength; preserve width identity and originalx0 loss.',
                previous_completed_observations_sha256=sha(OLD/'observations.json'),
                budget=BUDGET,remaining_runtime_rule='Use existing same-budget3413 control. Start if>=budget*.375+45; hardstop22:30. Reserve final QA; partial stages never count.',
                gpu_guard='Read-only nvidia-smi descriptor policy avoids kill(pid,0) PermissionError; only known low-memory ToDesk display is ignored; no external process is killed.')
    save(ART/'protocol.json',plan);save(ART/'observations.json',[]);core.status('REGISTERED')


def execute():
    def halt(*unused):core.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    if core.DEADLINE-time.time()<BUDGET*.375+45:core.status('NO_TIME_FOR_FRESH_STAGE');return
    plan=read(ART/'protocol.json')
    spec=dict(arm='K8_length_residual',seed=4817,steps=BUDGET,scenes=4096,
              manifest=str(core.shared.EVAL_MANIFEST),checkpoints=[],label=f'K8_length_residual_T4817_{BUDGET}')
    rows=core.run_stage(spec,plan)
    if rows is None:return
    save(ART/'observations.json',rows);prior.prior.refresh(rows)
    core.status('COMPLETE_LENGTH_HYPOTHESIS',observations=len(rows),full128epoch_verified=False)


prior.prior.ART,prior.prior.RUNS=ART,RUNS
prior.prior.evidence.ART,prior.prior.evidence.RUNS=ART,RUNS
prior.prior.runtime.ART,prior.prior.runtime.RUNS=ART,RUNS
core.ART,core.RUNS,core.sources,core.shared.gpu_pids=ART,RUNS,sources,competing_gpu_pids
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    execute() if args.execute else prepare()
