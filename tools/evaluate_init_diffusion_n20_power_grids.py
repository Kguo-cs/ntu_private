"""Standalone future native TEST registration for the CPU-reviewed N20 grids.

No execution occurs without --execute. Registration hashes and snapshots only
verified terminal main/compact endpoints; it never loads checkpoint payloads.
The active overnight queues, production, and their frozen runners are untouched.
All sampler R values refer to a fresh p1 run on the same frozen checkpoint.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.n20_power_grid_sampler import (sample_with_power_grid, validate_reviewed_flow,
                                         REVIEWED_RECIPE, REVIEWED_FLOW_SHA256)
from tools.evaluate_init_diffusion_sampler import (runtime_seed, runtime_seed_value,
    runtime_loader_sources, runtime_loader_stats, defer_runtime_gc, GC_ENV, RUNTIME_POLICY)

METRICS = ('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')
GEN_SEEDS = (2817,3817)
ARMS = (
    {'arm':'p1_legacy_floor','power':1.,'denominator_policy':'legacy_floor'},
    {'arm':'p05_legacy_floor','power':.5,'denominator_policy':'legacy_floor'},
    {'arm':'p2_legacy_floor','power':2.,'denominator_policy':'legacy_floor'},
    {'arm':'p2_actual_time','power':2.,'denominator_policy':'actual_time'},
)
SOURCES = {
    'main':(ROOT/'output/reports/init_diffusion_657_edges_20261013',ROOT/'logs/init_diffusion_657_edges_20261013'),
    'compact':(ROOT/'output/reports/init_diffusion_657_compact_20261013',ROOT/'logs/init_diffusion_657_compact_20261013'),
}
CALIBRATION = ROOT/'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json'
ACTIVE_HOOKS = []


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def names_sha(files):return hashlib.sha256(('\n'.join(files)+'\n').encode()).hexdigest()


def save(path,value,*,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w') as handle:
        handle.write(json.dumps(value,indent=2,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())


def read_config(path):
    import yaml
    value=yaml.safe_load(Path(path).read_text())
    require(isinstance(value,dict),'Native config must be a plain mapping')
    return value


def collect_sources():
    from tools.report_init_diffusion_657_compact import collect
    return {name:collect(name,art,runs) for name,(art,runs) in SOURCES.items()}


def code_sources():
    paths=(Path(__file__),ROOT/'tools/n20_power_grid_sampler.py',ROOT/'tools/evaluate_init_diffusion_sampler.py',
           ROOT/'tools/report_init_diffusion_657_compact.py',ROOT/'tools/report_init_diffusion_657_edges.py',
           ROOT/'src/run.py',ROOT/'src/smart/modules/smart_decoder.py',
           ROOT/'src/smart/datamodules/scalable_datamodule.py',ROOT/'src/smart/datasets/scalable_dataset.py')
    result={str(path.resolve()):sha(path) for path in paths}
    result.update(runtime_loader_sources())
    return result


def parse_deadline(value):
    deadline=datetime.fromisoformat(value)
    require(deadline.tzinfo is not None,'UTC deadline must include an explicit timezone')
    return deadline.timestamp()


def validate_source_config(cfg):
    model=cfg['model']['model_config'];decoder=model['decoder'];init=decoder['init_diffusion']
    require(decoder['init_decoder']=='flow' and decoder['initial_scene_only'] is True,
            'Require native initial-scene InitDiffusion only')
    require(init['sampling_steps']==20 and init['use_ema'] is True,'Require saved N20+EMA recipe')
    for name,expected in REVIEWED_RECIPE.items():
        if name in ('joint_type','feature_type'):continue
        require(init.get(name, 'gaussian' if name.endswith('_source') else None)==expected,
                'Unreviewed saved representation: '+name)
    require(cfg['data']['test_batch_size']==128,'Require native N128 TEST split')
    require(cfg['trainer']['precision']=='32-true','Require native float32 recipe')
    require(model['sd_reference_mode']=='matched','Require matched physical metric reference')
    require(model.get('sd_r_reference') is None,'Clear static/historical R references before registration')
    require(model.get('sd_require_full_set') is False,'This registration is exactly4096, not full50k')
    return cfg


def build_registration(*,source,variant,budget,train_seeds,manifest,output,deadline_utc,min_cell_seconds=120):
    """Read-only CPU registration builder. No Torch/checkpoint payload import."""
    require(source in SOURCES,'Source must be current main or compact; historical runs forbidden')
    require(type(budget) is int and budget in (4096,10240),'Register completed4096/10240 terminal endpoints only')
    require(len(train_seeds) in (1,2) and len(set(train_seeds))==len(train_seeds)
            and set(train_seeds)<={4817,5817},'Require explicit unique registered TRAIN seeds')
    require(type(min_cell_seconds) is int and min_cell_seconds>=30,'Require a realistic registered cell bound')
    parse_deadline(deadline_utc)
    output=Path(output).resolve();manifest=Path(manifest).resolve()
    require(not output.exists(),'Registration/output must be a new directory')
    raw=read(manifest);files=raw.get('files') if isinstance(raw,dict) else raw
    require(isinstance(files,list) and len(files)==len(set(files))==4096
            and all(isinstance(name,str) and Path(name).name==name for name in files),
            'Require exact unique4096 ordered cache filenames')
    collection=collect_sources()[source]
    require(collection.get('protocol_sha256') and collection.get('protocol'),'Missing frozen source protocol')
    require(collection['protocol_sha256']==sha(SOURCES[source][0]/'protocol.json'),
            'Collected protocol digest is stale')
    sources=[]
    for train_seed in train_seeds:
        rows=[r for r in collection['rows'] if r['variant']==variant and r['budget']==budget
              and r['seed']==train_seed and r['num_samples']==4096]
        # Duplicate aliases never become extra replicate checkpoints.
        require(len(rows)==2 and {r['eval_seed'] for r in rows}==set(GEN_SEEDS),
                'Missing or duplicate verified endpoint GEN pair for TRAIN '+str(train_seed))
        require(len({r['stage'] for r in rows})==1,'GEN pair comes from different stages')
        stage_name=rows[0]['stage'];stages=[s for s in collection['verified_stages'] if s['stage']==stage_name]
        require(len(stages)==1 and stages[0]['spec']['steps']==budget
                and not stages[0]['spec'].get('evaluation'),'Checkpoint must belong to a completed terminal TRAIN stage')
        require(not any(q['stage']==stage_name for q in collection.get('quarantined',[])),
                'Selected stage is quarantined')
        stage=(SOURCES[source][1]/stage_name).resolve()
        for key,filename in (('summary_sha256','summary.json'),('contract_sha256','edge_contract.json'),('marker_sha256','edge_verified_completion.json')):
            require(stages[0][key]==sha(stage/filename),'Collected terminal stage digest is stale: '+key)
        require(all(r['metrics_sha256']==sha(r['metrics_path']) and r['summary_sha256']==sha(stage/'summary.json') for r in rows),
                'Collected native metric/summary digest is stale')
        require(stage.parent==SOURCES[source][1].resolve(),'Selected stage escapes current source directory')
        checkpoint=Path(rows[0]['checkpoint']).resolve();config=checkpoint.parent/'config.yaml'
        require(checkpoint.name=='last.ckpt' and checkpoint.parent.parent==stage,
                'Require exact verified terminal checkpoint, never current rolling checkpoint')
        require(all(r['checkpoint']==str(checkpoint) and r['evaluation_weights']=='ema'
                    and r['eval_manifest_sha256']==names_sha(files) and r['actual_train_offset']==0
                    and r['resumed_from'] is None and r['map_frozen'] is True
                    and r['generator_num_parameters']<=8100000 for r in rows),
                'Endpoint identity/exposure/manifest/parameter scope differs')
        cfg=validate_source_config(read_config(config))
        require(sha(CALIBRATION)==rows[0]['calibration_sha256'],'TRAIN calibration SHA differs')
        require(sha(rows[0]['map_checkpoint'])==rows[0]['map_checkpoint_sha256'],'Fixed map source SHA differs')
        require(all(r['native_metric_identity']==rows[0]['native_metric_identity']
                    and r['native_membership']==rows[0]['native_membership'] for r in rows),
                'Native metric/reference membership identity differs within GEN pair')
        require(rows[0]['native_membership']['reference_mode']=='matched'
                and rows[0]['native_metric_identity']['map_source']=='reference','Require agents evaluated on reference lanes')
        identities={str(path):sha(path) for path in (stage/'edge_verified_completion.json',stage/'edge_contract.json',
                    stage/'verified_completion.json',stage/'summary.json',config,checkpoint,CALIBRATION,
                    Path(rows[0]['map_checkpoint']),SOURCES[source][0]/'protocol.json')}
        for r in rows:identities[r['metrics_path']]=sha(r['metrics_path'])
        sources.append({'TRAIN_seed':train_seed,'trained_updates':budget,'source_stage':str(stage),
            'variant':variant,'source_checkpoint':str(checkpoint),'source_checkpoint_sha256':sha(checkpoint),
            'source_config':str(config),'source_config_sha256':sha(config),
            'checkpoint_hash_provenance':'Freshly pinned verified-terminal bytes; original stage did not record checkpoint byte SHA at historical evaluation',
            'checkpoint':str(output/'snapshots'/f'train{train_seed}_step{budget}.ckpt'),
            'config':str(output/'snapshots'/f'train{train_seed}_config.yaml'),
            'generator_parameter_count':rows[0]['generator_num_parameters'],
            'calibration_sha256':rows[0]['calibration_sha256'],'map_checkpoint_sha256':rows[0]['map_checkpoint_sha256'],
            'native_metric_identity':rows[0]['native_metric_identity'],'native_membership':rows[0]['native_membership'],
            'terminal_rows':rows,'evidence_hashes':identities,'test_raw_dir':str(Path(cfg['data']['test_raw_dir']).resolve())})
    require(len({s['test_raw_dir'] for s in sources})==1,'TRAIN seeds use different evaluation data roots')
    raw_dir=Path(sources[0]['test_raw_dir'])
    data_hashes={str(raw_dir/name):sha(raw_dir/name) for name in files}
    source_hashes=code_sources()
    source_hashes.update({str((ROOT/name).resolve()):digest for name,digest in collection['protocol']['sources'].items()})
    require(all(sha(path)==digest for path,digest in source_hashes.items()),'Current model/runtime source changed')
    cells=[]
    for arm in ARMS:
        for endpoint in sources:
            for gen_seed in GEN_SEEDS:
                cells.append({**arm,'TRAIN_seed':endpoint['TRAIN_seed'],'GEN_seed':gen_seed,
                    'trained_updates':budget,'scenes':4096,'evaluation_batch_size':128,'sampling_steps':20,
                    'checkpoint':endpoint['checkpoint'],'checkpoint_sha256':endpoint['source_checkpoint_sha256'],
                    'source_config':endpoint['config'],'source_config_sha256':endpoint['source_config_sha256'],
                    'generator_parameter_count':endpoint['generator_parameter_count'],
                    'eval_files_sha256':names_sha(files),'native_metric_identity':endpoint['native_metric_identity'],
                    'native_membership':endpoint['native_membership']})
    return {'schema_version':1,'status':'FROZEN_FUTURE_N20_REGISTRATION','evaluation_only':True,
        'source':source,'variant':variant,'budget':budget,'TRAIN_seeds':list(train_seeds),'GEN_seeds':list(GEN_SEEDS),
        'arms':list(ARMS),'cells':cells,'endpoint_sources':sources,'files':files,'data_hashes':data_hashes,
        'source_hashes':source_hashes,'manifest_source':str(manifest),'manifest_source_sha256':sha(manifest),
        'eval_files_sha256':names_sha(files),'output':str(output),'deadline_utc':deadline_utc,
        'min_cell_seconds':min_cell_seconds,'runtime_policy':RUNTIME_POLICY,
        'precision':'32-true','float32_matmul_precision':'medium',
        'created_utc':datetime.now(timezone.utc).isoformat(),'main_model_selection_unchanged':True,
        'score_reference':'Fresh p1 on same checkpoint/TRAIN/GEN/cohort; no historical control or model-vs657 claim',
        'claim_limits':'Auxiliary inference-only matrix; not128-epoch convergence or model architecture search evidence'}


def freeze_registration(plan,output):
    output=Path(output).resolve()
    require(str(output)==plan['output'] and not output.exists(),'New registration directory required')
    output.mkdir(parents=True,exist_ok=False);(output/'snapshots').mkdir()
    for endpoint in plan['endpoint_sources']:
        for source_key,dest_key,hash_key in (('source_checkpoint','checkpoint','source_checkpoint_sha256'),
                                            ('source_config','config','source_config_sha256')):
            require(sha(endpoint[source_key])==endpoint[hash_key],'Terminal source changed before snapshot')
            with Path(endpoint[source_key]).open('rb') as src,Path(endpoint[dest_key]).open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
            require(sha(endpoint[dest_key])==endpoint[hash_key],'Snapshot bytes differ from terminal source')
    verify_terminal_lineage(plan)
    path=output/'registration.json';save(path,plan,exclusive=True)
    save(output/'registration_frozen.json',{'status':'FROZEN','registration_sha256':sha(path),
        'created_before_any_execution':True,'registered_cells':len(plan['cells'])},exclusive=True)
    return path


def verify_terminal_lineage(plan):
    collection=collect_sources()[plan['source']]
    for endpoint in plan['endpoint_sources']:
        stage_name=Path(endpoint['source_stage']).name
        stages=[s for s in collection['verified_stages'] if s['stage']==stage_name]
        require(len(stages)==1 and stages[0]['spec']['steps']==plan['budget']
                and not stages[0]['spec'].get('evaluation'),'Terminal source stage is no longer verified')
        rows=[r for r in collection['rows'] if r['stage']==stage_name and r['variant']==plan['variant']
              and r['budget']==plan['budget'] and r['seed']==endpoint['TRAIN_seed']]
        require(sorted(rows,key=lambda r:r['eval_seed'])==sorted(endpoint['terminal_rows'],key=lambda r:r['eval_seed']),
                'Frozen terminal GEN pair differs from current verified source')
        require(all(sha(path)==digest for path,digest in endpoint['evidence_hashes'].items()),
                'Terminal evidence/source changed during registration or verification')
    return collection


def verify_registration(path):
    path=Path(path).resolve();plan=read(path);marker=read(path.parent/'registration_frozen.json')
    require(marker['status']=='FROZEN' and marker['registration_sha256']==sha(path)
            and marker['created_before_any_execution'] is True,'Registration snapshot changed or absent')
    require(plan['schema_version']==1 and plan['evaluation_only'] is True and plan['arms']==list(ARMS)
            and plan['GEN_seeds']==list(GEN_SEEDS),'Registered arms/GEN policy changed')
    require(plan['output']==str(path.parent) and len(plan['files'])==len(set(plan['files']))==4096
            and names_sha(plan['files'])==plan['eval_files_sha256'],'Registered output/scene scope changed')
    require(sha(plan['manifest_source'])==plan['manifest_source_sha256'],'Original manifest file changed')
    require(all(sha(p)==h for p,h in plan['source_hashes'].items()),'Registered model/runtime source changed')
    require(all(sha(p)==h for p,h in plan['data_hashes'].items()),'Registered dataset bytes changed')
    for endpoint in plan['endpoint_sources']:
        require(all(sha(p)==h for p,h in endpoint['evidence_hashes'].items()),'Terminal evidence/checkpoint/config changed')
        require(sha(endpoint['checkpoint'])==endpoint['source_checkpoint_sha256']
                and sha(endpoint['config'])==endpoint['source_config_sha256'],'Pinned checkpoint/config changed')
    expected=[{**arm,'TRAIN_seed':e['TRAIN_seed'],'GEN_seed':g,'trained_updates':plan['budget'],
        'scenes':4096,'evaluation_batch_size':128,'sampling_steps':20,'checkpoint':e['checkpoint'],
        'checkpoint_sha256':e['source_checkpoint_sha256'],'source_config':e['config'],
        'source_config_sha256':e['source_config_sha256'],'generator_parameter_count':e['generator_parameter_count'],
        'eval_files_sha256':plan['eval_files_sha256'],'native_metric_identity':e['native_metric_identity'],
        'native_membership':e['native_membership']} for arm in ARMS for e in plan['endpoint_sources'] for g in GEN_SEEDS]
    require(plan['cells']==expected and marker['registered_cells']==len(expected),'Registered Cartesian cell membership changed')
    require(set(plan['TRAIN_seeds'])=={e['TRAIN_seed'] for e in plan['endpoint_sources']}
            and len(plan['TRAIN_seeds'])==len(plan['endpoint_sources']),'Registered TRAIN membership changed')
    verify_terminal_lineage(plan)
    parse_deadline(plan['deadline_utc'])
    return plan


def live_search_processes():
    patterns=('run_init_diffusion_657_edges_overnight.py','run_init_diffusion_657_compact_overnight.py',
              'runtime_cached_torch_loader_sweep.py')
    found=[]
    for path in Path('/proc').glob('[0-9]*'):
        try:
            command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            state=(path/'stat').read_text().split(') ',1)[1].split()[0]
        except (OSError,IndexError):continue
        if state!='Z' and any(pattern in command for pattern in patterns):found.append({'pid':int(path.name),'state':state,'command':command})
    return found


def gpu_compute_pids():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,timeout=10,check=True)
    values=[line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(all(value.isdecimal() for value in values),'Unknown GPU process query result; refuse execution')
    return [int(value) for value in values]


def execution_guard(plan,own_pid=None):
    require(not live_search_processes(),'Main/compact or native training process is live; future TEST may not interfere')
    gpu_pids=gpu_compute_pids()
    require(not [pid for pid in gpu_pids if pid!=own_pid],'GPU is busy; future TEST must run serially')
    remaining=parse_deadline(plan['deadline_utc'])-time.time()
    require(remaining>=plan['min_cell_seconds'],'Insufficient registered deadline budget for a complete cell')
    return remaining


def tensor_digest(module):
    import torch
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):continue
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class PowerGridHookCore:
    """TEST-only binding with explicit success/exception restoration."""
    def __init__(self,*,cell,audit_path,defer_gc=False):
        self.cell=copy.deepcopy(cell);self.audit_path=Path(audit_path);self.defer_gc=defer_gc
        require({k:cell[k] for k in ('arm','power','denominator_policy')} in ARMS,'Unregistered sampler arm')
        require(cell['sampling_steps']==20 and cell['GEN_seed'] in GEN_SEEDS,'Invalid N20/GEN cell')
        self.record={'cell':self.cell,'evaluation_only':True,'training_executed':False,'finished':False,
            'sample_calls':0,'denoiser_forward_calls':0,'completed_test_batches':0,'generated_scene_count':0,
            'per_batch':[],'runtime_policy':RUNTIME_POLICY if defer_gc else 'CPU_fixture'}
        self.flow=None;self.counter_handle=None;self.installed=False;self.batch_start_calls=None

    def on_fit_start(self,*args):raise ValueError('Power-grid override is TEST-only; fit forbidden')
    def on_train_start(self,*args):raise ValueError('Power-grid override is TEST-only; training forbidden')
    def on_validation_start(self,*args):raise ValueError('Power-grid override requires explicit TEST')

    def on_test_start(self,trainer,pl_module):
        require(str(trainer.state.fn) in ('test','TrainerFn.TESTING'),'Require explicit TEST')
        require(not pl_module.training and trainer.world_size==1 and trainer.precision=='32-true','Require eval/single-device/32-true')
        require(trainer.datamodule.test_batch_size==self.cell['evaluation_batch_size'],'Native batch split differs')
        decoder=pl_module.encoder.init_decoder;flow=decoder.G1
        require(decoder.use_ema and decoder.ema is not None
                and decoder.ema.num_updates==self.cell['trained_updates'],'Restored EMA budget differs')
        require(decoder.sampling_steps==20 and not flow.use_sde and not flow.use_refiner,'Require supervised N20 ODE')
        validate_reviewed_flow(flow)
        require(getattr(pl_module,'strict_loading',True) is True,'Require strict checkpoint model loading')
        require(sum(p.numel() for p in flow.parameters())==self.cell['generator_parameter_count'],'Actual parameter count changed')
        self.flow=flow;self.decoder=decoder;self.loss_before=copy.deepcopy(decoder.component_loss)
        self.weights_before=tensor_digest(flow);self.eps_before=flow.t_eps
        self.sample_was_instance_attribute='sample' in flow.__dict__;self.old_sample_attribute=flow.__dict__.get('sample')
        self.original_sample=flow.sample;self.expected_ema=decoder.ema.num_updates
        if self.defer_gc:
            import gc
            require(not gc.isenabled() and os.environ.get(GC_ENV)=='1','Child GC isolation was not active')
            self.record['runtime_loader_source_hashes']=runtime_loader_sources()
        runtime_seed(self.cell['GEN_seed'])
        require(runtime_seed_value()==self.cell['GEN_seed'],'GEN reset failed after strict checkpoint load')
        self.record.update(actual_seed_after_reset=runtime_seed_value(),EMA_num_updates=self.expected_ema,
            weights_before_sha256=self.weights_before,loaded_t_eps=self.eps_before,
            actual_evaluation_batch_size=trainer.datamodule.test_batch_size)
        try:
            self.counter_handle=flow.model.register_forward_pre_hook(self._count_forward)
            def sampled(agent,map_feature,steps=20,branch_steps=None):
                require(not flow.training and steps==20 and branch_steps in (None,0),'Unexpected training/sampler mode')
                require(self.batch_start_calls is not None,'Generation occurred outside native TEST batch')
                detail={}
                result=sample_with_power_grid(flow,agent,map_feature,power=self.cell['power'],steps=20,
                                             denominator_policy=self.cell['denominator_policy'],audit=detail)
                require(detail['actual_denoiser_forward_calls']==20 and detail['all_output_values_finite'],'Nonfinite or non-N20 generation')
                self.record['sample_calls']+=1
                self.last_sample_detail=detail
                return result
            flow.sample=sampled;self.installed=True
            ACTIVE_HOOKS.append(self)
            save(self.audit_path,self.record)
        except BaseException:
            self.restore();raise

    def _count_forward(self,module,args):self.record['denoiser_forward_calls']+=1

    def on_test_batch_start(self,trainer,pl_module,batch,batch_idx,dataloader_idx=0):
        require(self.installed and batch_idx==self.record['completed_test_batches'],'Native TEST batch order changed')
        require(self.batch_start_calls is None,'Overlapping TEST batches')
        self.batch_start_calls=(self.record['sample_calls'],self.record['denoiser_forward_calls'])
        self.current_scene_count=int(batch.num_graphs)

    def on_test_batch_end(self,trainer,pl_module,outputs,batch,batch_idx,dataloader_idx=0):
        require(self.batch_start_calls is not None and batch_idx==self.record['completed_test_batches'],'Unexpected TEST batch end')
        calls,nfe=self.batch_start_calls
        require(self.record['sample_calls']-calls==1 and self.record['denoiser_forward_calls']-nfe==20,'Actual per-batch NFE/sample count differs')
        expected=min(self.cell['evaluation_batch_size'],self.cell['scenes']-self.record['generated_scene_count'])
        require(self.current_scene_count==expected,'Actual TEST scene count/batch split changed')
        self.record['per_batch'].append({'batch_index':batch_idx,'scenes':self.current_scene_count,
            'sample_calls':1,'denoiser_forward_calls':20,'sampler':self.last_sample_detail})
        self.record['completed_test_batches']+=1;self.record['generated_scene_count']+=self.current_scene_count
        self.batch_start_calls=None

    def on_test_end(self,trainer,pl_module):
        try:
            expected=math.ceil(self.cell['scenes']/self.cell['evaluation_batch_size'])
            require(self.record['completed_test_batches']==expected and self.record['generated_scene_count']==self.cell['scenes']
                and self.record['denoiser_forward_calls']==20*expected,'Native TEST matrix cell incomplete')
            require(self.decoder.component_loss==self.loss_before and self.decoder.ema.num_updates==self.expected_ema,
                    'Training objective or EMA budget changed')
            require(tensor_digest(self.flow)==self.weights_before and self.flow.t_eps==self.eps_before,'Original generator weights/floor changed')
            self.record.update(finished=True,training_loss_config_unchanged=True,original_generator_weights_unchanged=True,
                original_t_eps_unchanged=True,EMA_updates_unchanged=True)
            if self.defer_gc:self.record['loader_cache_at_test_end']=runtime_loader_stats()
        finally:
            self.restore();save(self.audit_path,self.record)

    def on_exception(self,trainer,pl_module,exception):
        self.record.update(finished=False,error=repr(exception));self.restore();save(self.audit_path,self.record)

    def restore(self):
        if self.counter_handle is not None:self.counter_handle.remove();self.counter_handle=None
        if self.installed:
            if self.sample_was_instance_attribute:self.flow.sample=self.old_sample_attribute
            else:del self.flow.sample
            self.installed=False
        if self.flow is not None and hasattr(self,'eps_before'):
            require(self.flow.t_eps==self.eps_before,'Original scalar floor unexpectedly mutated')
        self.record['sample_binding_restored']=self.flow is not None and hasattr(self,'original_sample') and self.flow.sample==self.original_sample
        if self in ACTIVE_HOOKS:ACTIVE_HOOKS.remove(self)


def restore_active_hooks():
    for hook in list(ACTIVE_HOOKS):hook.restore()


def make_power_grid_hook(**kwargs):
    import lightning as L
    class Hook(PowerGridHookCore,L.Callback):pass
    return Hook(**kwargs)


def build_eval_config(source,cell,folder,manifest_path):
    cfg=copy.deepcopy(source);cfg.update(action='test',ckpt_path=cell['checkpoint'],seed=cell['TRAIN_seed'],logger=False)
    cfg['float32_matmul_precision']='medium';cfg['paths'].update(output_dir=str(folder),log_dir=str(folder))
    model=cfg['model']['model_config'];model['initial_map_checkpoint']=None;model['sd_r_reference']=None
    model.update(sd_reference_mode='matched',sd_require_full_set=False,sd_expected_scenes=50000)
    model['decoder']['init_diffusion'].update(normalizer_path=None,sampling_steps=20)
    cfg['data'].update(test_batch_size=128,num_workers=0,pin_memory=False,persistent_workers=False,
                       scenario_dreamer_eval_set=str(manifest_path))
    cfg['trainer'].update(accelerator='gpu',devices=1,precision='32-true',limit_test_batches=1.,logger=False,
                          enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False)
    cfg['callbacks']={'n20_power_audit':{'_target_':'tools.evaluate_init_diffusion_n20_power_grids.make_power_grid_hook',
                                       'cell':cell,'audit_path':str(folder/'sampler_audit.json'),'defer_gc':True}}
    return cfg


def validate_native_report(report,audit,cell):
    require(audit.get('finished') is True and audit.get('cell')==cell
        and audit.get('sample_binding_restored') is True and audit.get('training_executed') is False
        and audit.get('original_generator_weights_unchanged') is True
        and audit.get('original_t_eps_unchanged') is True and audit.get('EMA_updates_unchanged') is True
        and audit.get('actual_seed_after_reset')==cell['GEN_seed'],'Native adapter execution was not certified')
    batches=math.ceil(cell['scenes']/cell['evaluation_batch_size'])
    require(audit['sample_calls']==batches and audit['denoiser_forward_calls']==20*batches
        and audit['completed_test_batches']==batches and audit['generated_scene_count']==cell['scenes']
        and len(audit['per_batch'])==batches,'Native NFE/scene membership incomplete')
    require(all(r['denoiser_forward_calls']==20 and r['sample_calls']==1
        and r['sampler']['denominator_policy']==cell['denominator_policy']
        and r['sampler']['power']==cell['power'] for r in audit['per_batch']),'Per-batch effective grid/policy differs')
    require(report['reference_mode']=='matched' and report['map_source']=='reference'
        and report['num_samples']==report['num_gt_samples']==cell['scenes']
        and report['unverified_generation_timestep_count']==0 and report['require_full_set'] is False,
        'Native physical metric/input scope differs')
    for key,value in cell['native_metric_identity'].items():require(report.get(key)==value,'Native reference identity changed: '+key)
    require(report['decoder']['sampling_steps']==20 and report['decoder']['use_ema'] is True
        and report['decoder']['ema_num_updates']==cell['trained_updates']
        and report['runtime']['torch_initial_seed']==cell['GEN_seed']
        and report['runtime']['trainer_precision']=='32-true'
        and report['runtime']['float32_matmul_precision']=='medium','Native EMA/seed/precision differs')
    require(all(math.isfinite(float(report['agent_metrics'][k])) and report['agent_metrics'][k]>=0 for k in METRICS)
        and report['num_generated_vehicles']>0 and report['num_gt_vehicles']>0,'Invalid native metric populations')


def cell_folder(plan,index):
    cell=plan['cells'][index]
    return Path(plan['output'])/'cells'/f"train{cell['TRAIN_seed']}_gen{cell['GEN_seed']}_{cell['arm']}"


def validate_checkpoint_payload(payload,cell):
    """Actual loaded optimizer endpoint and EMA metadata, never just filenames."""
    import torch
    require(payload.get('global_step')==cell['trained_updates'],'Actual checkpoint optimizer step differs')
    state=payload.get('state_dict',{});extra=state.get('encoder.init_decoder._extra_state',{})
    require(extra.get('ema',{}).get('num_updates')==cell['trained_updates'],'Saved actual EMA exposure differs')
    metadata=payload.get('scratch_sweep',{})
    require(metadata.get('seed')==cell['TRAIN_seed'] and metadata.get('saved_global_step')==cell['trained_updates']
            and metadata.get('actual_train_offset')==0 and metadata.get('resumed_from') is None,
            'Actual fresh TRAIN endpoint metadata differs')
    calibration=read(CALIBRATION)
    for name in ('normal_mean','normal_scale'):
        actual=state.get('encoder.init_decoder.G1.model.'+name)
        require(torch.is_tensor(actual),'Checkpoint lacks physical TRAIN normalizer')
        expected=torch.as_tensor(calibration[name],dtype=actual.dtype).reshape(1,-1)
        require(torch.equal(actual.cpu(),expected),'Checkpoint TRAIN normalizer differs: '+name)
    return {'global_step':payload['global_step'],'EMA_num_updates':extra['ema']['num_updates'],
            'TRAIN_seed':metadata['seed'],'normalizer_verified_without_recalibration':True,
            'checkpoint_hash_provenance':'Pinned at future registration, not historically recorded by original stage'}


def execute_cell(registration,index):
    plan=verify_registration(registration);execution_guard(plan,own_pid=os.getpid())
    require(type(index) is int and 0<=index<len(plan['cells']),'Invalid registered cell')
    defer_runtime_gc()
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    from omegaconf import OmegaConf
    from src.run import run
    OmegaConf.register_new_resolver('sim_root',lambda:str(ROOT),replace=True)
    cell=plan['cells'][index];folder=cell_folder(plan,index)
    payload=torch.load(cell['checkpoint'],map_location='cpu',weights_only=False)
    checkpoint_metadata=validate_checkpoint_payload(payload,cell)
    del payload
    require(not folder.exists(),'Cell evidence already exists; never overwrite or consume partial output')
    folder.mkdir(parents=True,exist_ok=False)
    manifest=folder/'eval_files.pkl';manifest.write_bytes(pickle.dumps({'files':plan['files']}))
    save(folder/'checkpoint_metadata.json',checkpoint_metadata,exclusive=True)
    cfg=OmegaConf.create(build_eval_config(read_config(cell['source_config']),cell,folder,manifest))
    OmegaConf.save(cfg,folder/'config.yaml');started=time.time()
    try:run(cfg)
    finally:
        from tools import evaluate_init_diffusion_n20_power_grids as canonical
        canonical.restore_active_hooks()
    report_path=folder/'sd_agent_metrics.json';audit_path=folder/'sampler_audit.json'
    report,audit=read(report_path),read(audit_path);validate_native_report(report,audit,cell)
    verify_registration(registration)
    row={**cell,**report['agent_metrics'],'evaluation_only':True,'evaluation_weights':'ema',
        'registration_sha256':sha(registration),'native_report_path':str(report_path),
        'native_report_sha256':sha(report_path),'sampler_audit_sha256':sha(audit_path),
        'generated_onroad_fraction':report['feature_counts_generated'][1]/report['num_generated_vehicles'],
        'num_generated_vehicles':report['num_generated_vehicles'],'num_gt_vehicles':report['num_gt_vehicles'],
        'eval_seconds':time.time()-started,'reference':'same-checkpoint fresh p1; no historical denominator'}
    save(folder/'cell_summary.json',row,exclusive=True)
    print('DONE_N20_POWER_GRID '+str(folder),flush=True)
    return row


def verify_completed_cell(plan,index):
    folder=cell_folder(plan,index);row=read(folder/'cell_summary.json');execution=read(folder/'child_execution.json')
    require(execution['status']=='success' and execution['returncode']==0,'No successful independent child completion')
    require(all(sha(path)==digest for path,digest in execution['artifact_hashes'].items()),'Completed child artifacts changed')
    require(sha(folder/'console.log')==execution['log_sha256']
        and ('DONE_N20_POWER_GRID '+str(folder)) in (folder/'console.log').read_text().splitlines(),
        'Missing exact independently certified native marker')
    cell=plan['cells'][index]
    require(all(row[k]==v for k,v in cell.items()),'Cell summary identity differs')
    validate_native_report(read(folder/'sd_agent_metrics.json'),read(folder/'sampler_audit.json'),cell)
    require(all(row[k]==read(folder/'sd_agent_metrics.json')['agent_metrics'][k] for k in METRICS),'Summary metrics differ')
    return row


def summarize(plan,rows):
    expected={(c['TRAIN_seed'],c['GEN_seed'],c['arm']) for c in plan['cells']}
    actual={(r['TRAIN_seed'],r['GEN_seed'],r['arm']) for r in rows}
    require(len(actual)==len(rows) and actual<=expected,'Duplicate or unregistered comparison cells')
    summary={'complete':actual==expected,'rows':rows,'registration':str(Path(plan['output'])/'registration.json'),
             'main_model_selection_unchanged':True,'score_reference':plan['score_reference'],'scores':[]}
    require(all(all(math.isfinite(float(r[k])) and float(r[k])>=0 for k in METRICS)
        and r['num_generated_vehicles']>0 and r['num_gt_vehicles']>0
        and math.isfinite(r['generated_onroad_fraction']) and 0<=r['generated_onroad_fraction']<=1 for r in rows),
        'Invalid/nonfinite metric or population in summary')
    if not summary['complete']:return summary
    for arm in ARMS:
        ratios=[];per_train={}
        for train in plan['TRAIN_seeds']:
            values=[r for r in rows if r['TRAIN_seed']==train and r['arm']==arm['arm']]
            controls=[r for r in rows if r['TRAIN_seed']==train and r['arm']=='p1_legacy_floor']
            means={k:sum(r[k] for r in values)/2 for k in METRICS}
            baseline={k:sum(r[k] for r in controls)/2 for k in METRICS}
            require(all(baseline[k]>0 for k in METRICS),'Zero p1 denominator cannot support ratio score')
            relative={k:means[k]/baseline[k] for k in METRICS}
            onroad_drop=sum(r['generated_onroad_fraction'] for r in controls)/2-sum(r['generated_onroad_fraction'] for r in values)/2
            gates=all(relative[k]<=1.15 for k in ('length_jsd','width_jsd','speed_jsd')) and onroad_drop<=.02
            gates=gates and all(.9<=r['num_generated_vehicles']/r['num_gt_vehicles']<=1.1 for r in values)
            per_train[str(train)]={'R':sum(relative.values())/7,'R_attr':sum(relative[k] for k in ('length_jsd','width_jsd','speed_jsd'))/3,
                                  'ratios':relative,'gates_pass':gates,'onroad_drop':onroad_drop}
            ratios.append((means,baseline))
        pooled={k:sum(m[k] for m,b in ratios)/sum(b[k] for m,b in ratios) for k in METRICS}
        pooled_R=sum(pooled.values())/7
        all_train=all(s['R']<1 and s['gates_pass'] for s in per_train.values())
        summary['scores'].append({**arm,'sampler_R_vs_same_checkpoint_p1':pooled_R,
            'sampler_R_attr':sum(pooled[k] for k in ('length_jsd','width_jsd','speed_jsd'))/3,
            'per_TRAIN':per_train,'all_TRAIN_improve_and_pass':all_train,
            'robust_sampler_improvement':pooled_R<1. and all_train,
            'TRAIN_replicates':len(plan['TRAIN_seeds']),'model_vs_657_R_claimed':False})
    return summary


def execute_registration(registration):
    plan=verify_registration(registration);execution_guard(plan)
    lock_path=ROOT/'tmp/init_diffusion_n20_power_grid_gpu.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,cell in enumerate(plan['cells']):
            plan=verify_registration(registration);execution_guard(plan)
            folder=cell_folder(plan,index)
            if folder.exists():rows.append(verify_completed_cell(plan,index));continue
            # Parent owns logs in a separate sibling until child creates cell folder.
            pending=Path(plan['output'])/'execution'/f'cell_{index:02d}';pending.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,str(Path(__file__).resolve()),'--registration',str(registration),'--execute','--cell',str(index)]
            execution={'status':'running','cell':cell,'command':command,'started_unix':time.time(),
                       'registration_sha256':sha(registration)}
            log_path=pending/'console.log';process=None
            try:
                with log_path.open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,**{GC_ENV:'1'}),start_new_session=True)
                    execution['child_pid']=process.pid;save(pending/'child_execution.json',execution)
                    timeout=max(1.,parse_deadline(plan['deadline_utc'])-time.time())
                    code=process.wait(timeout=timeout)
                require(code==0,'Isolated native TEST failed '+str(code))
                require(folder.is_dir(),'Child returned without evidence directory')
                shutil.copyfile(log_path,folder/'console.log')
                paths=(folder/'cell_summary.json',folder/'sd_agent_metrics.json',folder/'sampler_audit.json',folder/'config.yaml',folder/'eval_files.pkl',folder/'checkpoint_metadata.json')
                execution.update(status='success',returncode=code,log_sha256=sha(folder/'console.log'),
                                 artifact_hashes={str(p):sha(p) for p in paths},ended_unix=time.time())
                save(pending/'child_execution.json',execution);save(folder/'child_execution.json',execution,exclusive=True)
                rows.append(verify_completed_cell(plan,index))
                save(Path(plan['output'])/'summary.json',summarize(plan,rows))
            except BaseException as error:
                if process is not None and process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGINT);process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    except ProcessLookupError:process.wait(timeout=10)
                execution.update(status='failed',returncode=process.returncode if process else None,error=repr(error),ended_unix=time.time())
                save(pending/'child_execution.json',execution);raise
    return summarize(plan,rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=tuple(SOURCES),default='main')
    parser.add_argument('--variant');parser.add_argument('--budget',type=int,choices=(4096,10240),default=10240)
    parser.add_argument('--train-seeds',type=int,nargs='+',default=[4817,5817])
    parser.add_argument('--manifest',type=Path,default=ROOT/'tmp/control657_paired_20261012/eval4096_files.json')
    parser.add_argument('--output',type=Path);parser.add_argument('--deadline-utc')
    parser.add_argument('--min-cell-seconds',type=int,default=120)
    parser.add_argument('--registration',type=Path);parser.add_argument('--register',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--cell',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.registration:
        require(not args.register,'Use either a new registration or frozen execution')
        if args.cell is not None:
            require(args.execute,'Child execution needs explicit --execute');defer_runtime_gc()
            execute_cell(args.registration,args.cell);sys.stdout.flush();sys.stderr.flush();os._exit(0)
        if args.execute:return execute_registration(args.registration)
        plan=verify_registration(args.registration);print('VERIFIED_FROZEN_ONLY '+str(len(plan['cells'])));return plan
    require(not args.execute and args.cell is None,'GPU execution requires a previously frozen registration')
    require(args.variant and args.output and args.deadline_utc,'Variant/output/deadline are required for planning')
    plan=build_registration(source=args.source,variant=args.variant,budget=args.budget,train_seeds=args.train_seeds,
        manifest=args.manifest,output=args.output,deadline_utc=args.deadline_utc,min_cell_seconds=args.min_cell_seconds)
    if args.register:
        path=freeze_registration(plan,args.output);print('REGISTERED_CPU_ONLY '+str(path));return path
    print('DRY_RUN_CPU_ONLY '+str(len(plan['cells']))+' cells; no output directory/GPU/checkpoint payload load')
    return plan


if __name__=='__main__':main()
