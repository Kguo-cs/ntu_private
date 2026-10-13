"""Resume after the user's default-experiment change, preserving completed work."""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_init_diffusion_final_long_20261013 as prior
from tools import run_init_diffusion_long_audit_recovery_20261013 as recovery
from tools import run_init_diffusion_long_audit_20261013 as core
read,save,sha,now=core.read,core.save,core.sha,core.now
OLD=prior.ART;ART=OLD/'rebased';RUNS=prior.RUNS/'rebased'
original_sources=prior.sources


def sources():
    return {**original_sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}


def prepare():
    import yaml
    assert not (ART/'protocol.json').exists()
    plan=read(OLD/'protocol.json');current=original_sources()
    changed=[p for p in current if current[p]!=plan['source_hashes'].get(p)]
    assert changed==['configs/run.yaml'],changed
    folder=prior.RUNS/'long_fresh657_16384_attempt1'
    ex=read(folder/'execution.json');assert ex['returncode']==0
    assert 'RUNTIME_LOADER_SUCCESS' in (folder/'console.log').read_text()
    values=read(folder/'summary.json')
    core.RUNS=folder.parent
    try:core.verify(values,ex['spec'])
    finally:core.RUNS=RUNS
    for name,expected in plan['source_hashes'].items():
        if name!='configs/run.yaml':assert sha(ROOT/name)==expected,name
    # The selected experiment is explicitly overridden in the native harness.
    # Its resumed model config is copied from the prior checkpoint, not from
    # the changed default group; compare the actual resolved model dictionaries.
    base=Path(values[0]['resumed_from']).parent/'config.yaml'
    actual=folder/'fresh657/config.yaml'
    assert yaml.safe_load(base.read_text())['model']==yaml.safe_load(actual.read_text())['model']
    trace=read(folder/'fresh657/train_trace.json')
    assert [r['step'] for r in trace]==list(range(12288,16384))
    rows=read(OLD/'observations.json')+values
    save(folder/'verified_completion.json',dict(status='VERIFIED_COMPLETE',summary_sha256=sha(folder/'summary.json'),finished_utc=now(),
         source_exception='Only configs/run.yaml default experiment changed; explicit experiment override and checkpoint-resolved model config unchanged. All other registered files identical.'))
    save(ART/'source_change_audit.json',dict(changed=changed,old_sha256=plan['source_hashes']['configs/run.yaml'],new_sha256=current['configs/run.yaml'],actual_model_config_identical=True,retained_updates=len(trace)))
    for n in ('recipe.json','variants.json','sorted128.json','sorted256.json','sorted4096.json'):save(ART/n,read(OLD/n))
    plan=copy.deepcopy(plan)
    completed={(r['variant'],r['budget']) for r in rows if not r['evaluation_only']}
    plan['stages']=[s for s in plan['stages'] if (s['arm'],s['steps']) not in completed]
    plan.update(registered_utc=now(),source_hashes=sources(),previous_protocol_sha256=sha(OLD/'protocol.json'),
                local_input_hashes={n:sha(ART/n) for n in ('recipe.json','variants.json','sorted128.json','sorted256.json','sorted4096.json')})
    plan['imported_sources'][str((folder/'summary.json').relative_to(ROOT))]=sha(folder/'summary.json')
    save(ART/'observations.json',rows);save(ART/'protocol.json',plan);core.comparisons(rows);core.status('REGISTERED')


def execute():
    plan=read(ART/'protocol.json');rows=read(ART/'observations.json')
    # Finish the registered training intervals. A separate final-evaluation
    # phase uses all imported and completed endpoints, including the baseline.
    import signal,time
    def halt(*_):core.STOP=True
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    starts={};prefixes={}
    for stage in plan['stages']:
        if core.STOP:return
        arm=stage['arm'];cp=prefixes.get(arm,stage['initial_resume'][arm]);start=starts.get(arm,stage['initial_resume_step'])
        origin=start;segments=[];attempts=[];values=None
        if core.DEADLINE-time.time()<(stage['steps']-start)*.45+180:core.status('NO_TIME_FOR_STAGE',next=stage);return
        for attempt in range(1,6):
            spec=copy.deepcopy(stage);spec.pop('initial_resume');spec.pop('initial_resume_step')
            spec.update(resume={arm:cp},resume_step=start,label=stage['label']+f'_attempt{attempt}')
            values=core.run_stage(spec,plan);folder=RUNS/spec['label']/arm
            if values is not None:
                segments+=read(folder/'train_trace.json');attempts.append(dict(attempt=attempt,status='COMPLETE',from_step=start,to_step=stage['steps']));break
            if core.STOP:return
            found=None
            for path in sorted(folder.glob('step_*.ckpt'),key=lambda p:int(p.stem.split('_')[-1]),reverse=True):
                meta=recovery.valid_checkpoint(path)
                if meta and start<meta['step']<stage['steps'] and meta['seed']==stage['seed']:found=(path,meta);break
            if found is None:attempts.append(dict(attempt=attempt,status='FAILED_NO_RETAINED_UPDATES',from_step=start));continue
            path,meta=found;trace=[r for r in read(folder/'recovery_trace.json') if start<=r['step']<meta['step']]
            assert [r['step'] for r in trace]==list(range(start,meta['step']))
            segments+=trace;attempts.append(dict(attempt=attempt,status='FAILED_RECOVERED_PREFIX',from_step=start,to_step=meta['step'],checkpoint_sha256=sha(path)))
            cp=str(path);start=meta['step'];save(ART/(stage['label']+'.attempts.json'),attempts)
        if values is None:core.status('FAILED_RECOVERY',stage=stage,attempts=attempts);return
        assert [r['step'] for r in segments]==list(range(origin,stage['steps']))
        save(ART/(stage['label']+'.stitched_trace.json'),segments);save(ART/(stage['label']+'.attempts.json'),attempts)
        prefixes[arm]=values[0]['checkpoint'];starts[arm]=stage['steps'];rows+=values
        save(ART/'observations.json',rows);core.comparisons(rows)
    core.status('COMPLETE_LONG_TRAIN',observations=len(rows),full128epoch_verified=False)


recovery.ART,recovery.RUNS=ART,RUNS
core.ART,core.RUNS=ART,RUNS;core.sources=sources
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    execute() if a.execute else prepare()
