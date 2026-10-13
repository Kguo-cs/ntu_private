"""Checkpointed recovery of the registered longer map-condition audit."""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_init_diffusion_long_audit_20261013 as core
read,save,sha,now=core.read,core.save,core.sha,core.now
OLD=core.ART;ART=OLD/'recovery';RUNS=core.RUNS/'recovery'
original_sources=core.sources


def sources():
    return {**original_sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__),
            'tools/runtime_checkpointed_sweep.py':sha(ROOT/'tools/runtime_checkpointed_sweep.py')}


# The recorded command is changed in place before spawning, so execution.json
# identifies the actual checkpointed entry point. Native forward/data stay in
# the frozen original modules; only process-local logging/recovery saves differ.
RealPopen=subprocess.Popen
class CheckpointedPopen(RealPopen):
    def __init__(self,cmd,*args,**kwargs):
        if isinstance(cmd,list) and len(cmd)>1 and cmd[1]==str(ROOT/'tools/runtime_cached_torch_loader_sweep.py'):
            cmd[1]=str(ROOT/'tools/runtime_checkpointed_sweep.py')
        super().__init__(cmd,*args,**kwargs)
subprocess.Popen=CheckpointedPopen
core.ART,core.RUNS=ART,RUNS;core.sources=sources


def prepare():
    assert not (ART/'protocol.json').exists()
    plan=read(OLD/'protocol.json')
    save(ART/'recipe.json',read(OLD/'recipe.json'));save(ART/'variants.json',read(OLD/'variants.json'))
    for n in (128,256,4096):save(ART/f'sorted{n}.json',read(OLD/f'sorted{n}.json'))
    rows=read(OLD/'observations.json');save(ART/'observations.json',rows)
    stages=[]
    for spec in plan['stages']:
        if spec.get('evaluation'):continue
        start=spec['resume_step']
        for target in (8192,12288,16384):
            if target<=start:continue
            stages.append(dict(arm=spec['arm'],seed=spec['seed'],steps=target,
                               initial_resume=spec['resume'],initial_resume_step=start,
                               scenes=4096,manifest=spec['manifest'],checkpoints=[],
                               label=f"long_{spec['arm']}_{spec['seed']}_{target}"))
    plan.update(registered_utc=now(),source_hashes=sources(),stages=stages,
                previous_protocol_sha256=sha(OLD/'protocol.json'),previous_failed_execution_sha256=sha(core.RUNS.parent/'long_fresh657_4817/execution.json'),
                recovery_policy='Every512 optimizer updates save checkpoint+trace. Restart a failed isolated process only from a validated same-arm checkpoint; preserve failures and account for every retained update. Max5 attempts per stage. Config.yaml once, metrics CSV flush without repeated YAML serialization. No objective/model changes.',
                local_input_hashes={n:sha(ART/n) for n in ('recipe.json','variants.json','sorted128.json','sorted256.json','sorted4096.json')})
    plan['imported_sources'][str((OLD/'observations.json').relative_to(ROOT))]=sha(OLD/'observations.json')
    save(ART/'protocol.json',plan);core.comparisons(rows);core.status('REGISTERED')


def valid_checkpoint(path):
    code="""import gc,json,os,sys;gc.disable()
import torch
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
p=sys.argv[1];c=torch.load(p,map_location='cpu',weights_only=False)
assert c['scratch_sweep']['saved_global_step']==c['global_step']
assert c['scratch_sweep_rng'] and c['optimizer_states'] and c['lr_schedulers']
print(json.dumps({'step':c['global_step'],'seed':c['scratch_sweep']['seed'],'map_sha256':c['scratch_sweep']['map_checkpoint_sha256']}),flush=True)
os._exit(0)
"""
    result=subprocess.run([core.shared.PYTHON,'-c',code,str(path)],cwd=ROOT,capture_output=True,text=True)
    if result.returncode:return None
    return __import__('json').loads(result.stdout.strip().splitlines()[-1])


def execute():
    def halt(*_):core.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    plan=read(ART/'protocol.json');rows=read(ART/'observations.json');latest={};starts={}
    for stage in plan['stages']:
        if core.STOP:break
        arm=stage['arm'];cp=latest.get(arm,stage['initial_resume'][arm]);start=starts.get(arm,stage['initial_resume_step'])
        origin=start;segments=[];attempts=[];values=None
        if core.DEADLINE-time.time()<(stage['steps']-start)*.45+180:
            core.status('NO_TIME_FOR_STAGE',next=stage);break
        for attempt in range(1,6):
            spec=copy.deepcopy(stage);spec.pop('initial_resume');spec.pop('initial_resume_step')
            spec.update(resume={arm:cp},resume_step=start,label=stage['label']+f'_attempt{attempt}')
            values=core.run_stage(spec,plan);folder=RUNS/spec['label']/arm
            if values is not None:
                trace=read(folder/'train_trace.json');segments+=trace
                attempts.append(dict(attempt=attempt,status='COMPLETE',from_step=start,to_step=stage['steps'],checkpoint=values[0]['checkpoint'],trace_sha256=sha(folder/'train_trace.json')))
                break
            if core.STOP:return
            candidates=sorted(folder.glob('step_*.ckpt'),key=lambda p:int(p.stem.split('_')[-1]),reverse=True)
            recovered=None
            for path in candidates:
                meta=valid_checkpoint(path)
                if meta and start<meta['step']<stage['steps'] and meta['seed']==stage['seed']:
                    recovered=(path,meta);break
            if recovered is None:
                attempts.append(dict(attempt=attempt,status='FAILED_NO_RETAINED_UPDATES',from_step=start))
                save(ART/(stage['label']+'.attempts.json'),attempts)
                continue
            path,meta=recovered;trace=read(folder/'recovery_trace.json')
            trace=[r for r in trace if start<=r['step']<meta['step']]
            assert [r['step'] for r in trace]==list(range(start,meta['step']))
            segments+=trace
            attempts.append(dict(attempt=attempt,status='FAILED_RECOVERED_PREFIX',from_step=start,to_step=meta['step'],checkpoint=str(path),checkpoint_sha256=sha(path),trace_sha256=sha(folder/'recovery_trace.json')))
            cp=str(path);start=meta['step'];save(ART/(stage['label']+'.attempts.json'),attempts)
        if values is None:
            core.status('FAILED_RECOVERY',stage=stage,attempts=attempts);return
        assert [r['step'] for r in segments]==list(range(origin,stage['steps']))
        save(ART/(stage['label']+'.stitched_trace.json'),segments)
        save(ART/(stage['label']+'.attempts.json'),dict(status='COMPLETE',attempts=attempts,retained_updates=stage['steps']-origin,discarded_partial_updates_excluded=True,numerical_drift_disclosed=True))
        latest[arm]=values[0]['checkpoint'];starts[arm]=stage['steps']
        rows+=values;save(ART/'observations.json',rows);core.comparisons(rows)
    else:
        cp={n:latest[n] for n in ('fresh657','freshK8','lane_only')}
        for n,manifest in [(128,ART/'sorted128.json'),(4096,ART/'sorted4096.json'),
                            (16384,ROOT/'tmp/control657_paired_20261012/eval16384_files.json')]:
            if core.DEADLINE-time.time()<n/4096*150+120:break
            spec=dict(label=f'endpoint_scope{n}',evaluation=cp,scenes=n,manifest=str(manifest),seed=4817,steps=16384,checkpoints=[])
            values=core.run_stage(spec,plan)
            if values is None:return
            rows+=values;save(ART/'observations.json',rows)
    core.status('COMPLETE',observations=len(rows),full128epoch_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    execute() if a.execute else prepare()
