"""CPU-only audit of saved checkpoint RNG states for verified N20 stages.

The parent never imports torch. One isolated sim worker hides CUDA, disables GC,
uses the cached torch loader, maps every tensor to CPU and hard-exits on success.
Saved endpoint states are not evidence of identical per-step noise trajectories.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_init_diffusion_657_iteration import read,sha
PYTHON='/home/ke/miniconda3/envs/sim/bin/python'
CHANNELS=('python','numpy','torch_cpu','torch_cuda')


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as handle:
        temporary=Path(handle.name);json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def canonical(value,torch,np):
    """Hash state contents, avoiding storage addresses in Tensor pickle bytes."""
    if torch.is_tensor(value):
        if value.device.type!='cpu':raise RuntimeError('Checkpoint RNG tensor was not mapped to CPU')
        tensor=value.detach().contiguous()
        raw=tensor.numpy().tobytes()
        return {'type':'torch.Tensor','dtype':str(tensor.dtype),'shape':list(tensor.shape),'bytes_sha256':hashlib.sha256(raw).hexdigest()}
    if isinstance(value,np.ndarray):
        return {'type':'numpy.ndarray','dtype':value.dtype.str,'shape':list(value.shape),
            'bytes_sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}
    if isinstance(value,np.generic):return canonical(value.item(),torch,np)
    if isinstance(value,(list,tuple)):
        return {'type':type(value).__name__,'items':[canonical(item,torch,np) for item in value]}
    if isinstance(value,dict):return {'type':'dict','items':[[str(key),canonical(item,torch,np)] for key,item in sorted(value.items(),key=lambda pair:str(pair[0]))]}
    if value is None or isinstance(value,(str,int,bool)):return value
    if isinstance(value,float):return {'type':'float','hex':value.hex()}
    raise TypeError('Unsupported saved RNG state type: '+type(value).__name__)


def fingerprint(value,torch,np):
    structure=canonical(value,torch,np)
    digest=hashlib.sha256(json.dumps(structure,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'sha256':digest,'structure':structure}


def state_details(channel,value,torch,np):
    present=value is not None and not (channel=='torch_cuda' and len(value)==0)
    result={'present':present}
    if not present:return result
    result.update(fingerprint(value,torch,np))
    if channel=='python':result['format']={'version':value[0],'state_words':len(value[1]),'cached_gaussian_present':value[2] is not None}
    elif channel=='numpy':result['format']={'algorithm':value[0],'position':int(value[2]),'cached_gaussian_present':bool(value[3])}
    elif channel=='torch_cpu':result['format']={'bytes':int(value.numel())}
    elif channel=='torch_cuda':
        devices=[]
        for index,tensor in enumerate(value):
            raw=tensor.detach().contiguous().numpy().tobytes()
            info={'saved_device_index':index,'bytes':len(raw),'bytes_sha256':hashlib.sha256(raw).hexdigest()}
            if len(raw)==16:
                info.update(saved_seed_uint64_le=int.from_bytes(raw[:8],'little'),
                    saved_offset_uint64_le=int.from_bytes(raw[8:16],'little'),
                    format_note='Decoded16-byte saved CUDA generator state; no CUDA API called')
            devices.append(info)
        result['format']={'saved_device_count':len(devices),'devices':devices}
    return result


def validate_saved_metadata(metadata,entry):
    if metadata.get('seed')!=entry['seed']:raise RuntimeError('Saved checkpoint TRAIN seed differs')
    if metadata.get('saved_global_step')!=entry['budget']:raise RuntimeError('Saved checkpoint metadata exposure differs')
    for key,value in entry.get('expected_scratch_metadata',{}).items():
        if metadata.get(key)!=value:raise RuntimeError('Saved checkpoint scope differs from verified native row: '+key)


def worker(manifest_path,output):
    # Must be set before Torch/NumPy/Lightning imports. No model is constructed.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SIM_SWEEP_DEFER_CYCLIC_GC='1')
    import gc
    gc.disable();gc.collect=lambda *unused,**kwargs:0
    import torch,numpy as np
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader,runtime_policy
    install_cached_loader(torch)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized in CPU audit worker')
    snapshots=[];manifest=read(manifest_path)
    for entry in manifest['checkpoints']:
        path=Path(entry['path']);before=path.stat()
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized during checkpoint deserialization')
        if int(checkpoint['global_step'])!=entry['budget']:raise RuntimeError('Saved checkpoint exposure differs')
        rng=checkpoint.get('scratch_sweep_rng')
        if not isinstance(rng,dict):raise RuntimeError('Checkpoint has no saved scratch RNG states')
        metadata=checkpoint.get('scratch_sweep',{})
        validate_saved_metadata(metadata,entry)
        snapshot={'path':str(path),'checkpoint_sha256':file_sha(path),'bytes':before.st_size,
            'global_step':int(checkpoint['global_step']),'scratch_metadata':metadata,
            'states':{channel:state_details(channel,rng.get(channel),torch,np) for channel in CHANNELS},
            'cuda_initialized':torch.cuda.is_initialized()}
        after=path.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
            raise RuntimeError('Checkpoint changed during CPU audit')
        snapshots.append(snapshot)
        del checkpoint,rng
        print('CPU_CHECKPOINT_RNG '+json.dumps({'completed':len(snapshots),'expected':len(manifest['checkpoints']),
            'path':str(path),'cuda_initialized':torch.cuda.is_initialized()}),flush=True)
    result={'status':'CPU_WORKER_COMPLETE','generated_utc':datetime.now(timezone.utc).isoformat(),
        'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'cuda_initialized':torch.cuda.is_initialized(),
        'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),
        'runtime_policy':runtime_policy(),'snapshots':snapshots}
    atomic_json(output,result)
    print('CHECKPOINT_RNG_AUDIT_WORKER_SUCCESS',flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


def same_states(candidate,control):
    return {channel:(candidate['states'][channel]['sha256']==control['states'][channel]['sha256']
            if candidate['states'][channel]['present'] and control['states'][channel]['present'] else None)
            for channel in CHANNELS}


def plan_pairs(source,tier,budgets):
    from tools import report_init_diffusion_657_compact as report
    main=report.collect('main',report.MAIN_ART,report.MAIN_RUNS)
    compact=report.collect('compact',report.ART,report.RUNS)
    main_protocol=main['protocol'];compact_protocol=compact['protocol']
    aligned=bool(main_protocol and compact_protocol and all(compact_protocol['sources'].get(k)==v for k,v in main_protocol['sources'].items()))
    all_rows=main['rows']+compact['rows']
    for row in all_rows:row['cross_source_alignment']=aligned
    pairs=[];seen=set();missing=[]
    for row in all_rows:
        if row['variant']=='control657' or row['tier']!=tier or row['budget'] not in budgets:
            continue
        if source!='both' and row['source']!=source:continue
        key=(row['source'],row['variant'],row['seed'],row['budget'],row['checkpoint'])
        if key in seen:continue
        seen.add(key);controls=report.primary_controls([row],all_rows)
        if controls is None:
            missing.append({'source':row['source'],'variant':row['variant'],'seed':row['seed'],
                'budget':row['budget'],'reason':'No exact verified fresh657 checkpoint scope'});continue
        control=controls[0]
        fields=('source','variant','stage','seed','eval_seed','budget','seen_training_examples','num_samples',
            'checkpoint','initial_generator_sha256','initial_map_sha256','train_order_sha256','calibration_sha256','batch_size','map_checkpoint_sha256',
            'metrics_path','metrics_sha256','summary_sha256')
        pairs.append({'candidate':{key:row[key] for key in fields},'control':{key:control[key] for key in fields},
            'same_initial_G1_hash':row['initial_generator_sha256']==control['initial_generator_sha256'],
            'same_initial_map_hash':row['initial_map_sha256']==control['initial_map_sha256']})
    checkpoints={}
    for pair in pairs:
        for role in ('candidate','control'):
            row=pair[role];checkpoints[row['checkpoint']]={'path':row['checkpoint'],'budget':row['budget'],'seed':row['seed'],
                'expected_scratch_metadata':{key:row[key] for key in ('seed','batch_size','train_order_sha256',
                    'calibration_sha256','map_checkpoint_sha256','initial_generator_sha256','initial_map_sha256')}}
    return {'source':source,'tier':tier,'budgets':budgets,'pairs':pairs,'checkpoints':list(checkpoints.values()),
        'missing_pairs':missing,'shared_source_alignment':aligned,
        'source_protocol_sha256':{'main':main['protocol_sha256'],'compact':compact['protocol_sha256']}}


def finish_audit(manifest,worker_result):
    snapshots={row['path']:row for row in worker_result['snapshots']}
    comparisons=[]
    for pair in manifest['pairs']:
        candidate=snapshots[pair['candidate']['checkpoint']];control=snapshots[pair['control']['checkpoint']]
        saved=same_states(candidate,control)
        comparisons.append({**pair,'same_saved_endpoint_rng':saved,
            'candidate_checkpoint_sha256':candidate['checkpoint_sha256'],
            'control_checkpoint_sha256':control['checkpoint_sha256'],
            'saved_CUDA_format':{'candidate':candidate['states']['torch_cuda'].get('format'),
                                 'control':control['states']['torch_cuda'].get('format')},
            'initial_training_rng_recorded':False,'per_step_training_noise_recorded':False,
            'interpretation':'Matching saved endpoint RNG states show matching generator state at this exposure, not identical initial RNG state or per-step draws/assignment.'})
    return {'status':'CPU_READ_ONLY_COMPLETE','generated_utc':datetime.now(timezone.utc).isoformat(),
        'CUDA_VISIBLE_DEVICES':worker_result['CUDA_VISIBLE_DEVICES'],'cuda_initialized':worker_result['cuda_initialized'],
        'scope':{'source':manifest['source'],'tier':manifest['tier'],'budgets':manifest['budgets']},
        'comparisons':comparisons,'missing_pairs':manifest['missing_pairs'],
        'source_protocol_sha256':manifest['source_protocol_sha256'],'snapshots':worker_result['snapshots'],
        'runtime_policy':worker_result['runtime_policy'],'torch_version':worker_result['torch_version'],
        'numpy_version':worker_result['numpy_version'],
        'summary':{'paired_checkpoints':len(comparisons),
            'same_saved_endpoint_rng_counts':{channel:sum(r['same_saved_endpoint_rng'][channel] is True for r in comparisons) for channel in CHANNELS},
            'different_saved_endpoint_rng_counts':{channel:sum(r['same_saved_endpoint_rng'][channel] is False for r in comparisons) for channel in CHANNELS},
            'missing_saved_rng_counts':{channel:sum(r['same_saved_endpoint_rng'][channel] is None for r in comparisons) for channel in CHANNELS}},
        'evidence_limits':{'initial_model_hash_equality':'Recorded by native initialization metadata; model weights equality does not prove initial RNG state equality.',
            'saved_endpoint_state':'Exact content hashes of Python/NumPy/TorchCPU/saved TorchCUDA RNG states; these are cumulative checkpoint endpoints.',
            'initial_training_rng':'Not recorded at fresh on_train_start, so not established by this tool.',
            'per_step_trajectory':'Not recorded. Equal endpoints do not prove identical per-step RNG draws, consumer ordering or noise-to-agent assignment. Different CPU RNG endpoints do not by themselves prove different CUDA noise.',
            'CUDA_safety':'Saved CUDA state is a CPU byte tensor; no CUDA generator API, model forward, training or evaluation is used.'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=('main','compact','both'),default='main')
    parser.add_argument('--tier',choices=('screen','medium','long4096','full50k'),default='screen')
    parser.add_argument('--budgets',default='384,768')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    output=args.output or ROOT/f'output/reports/init_diffusion_657_{"edges" if args.source=="main" else "compact"}_20261013/graph_checkpoint_rng_audit.json'
    if args.worker:
        try:worker(args.worker,output)
        except BaseException:
            import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    manifest=plan_pairs(args.source,args.tier,[int(value) for value in args.budgets.split(',')])
    manifest_path=output.with_name(output.stem+'_manifest.json');atomic_json(manifest_path,manifest)
    if args.plan_only:
        print(json.dumps({'plan':str(manifest_path),'pairs':len(manifest['pairs']),'CPU_workers_started':False}));return
    if not manifest['pairs']:
        atomic_json(output,{'status':'WAITING_NO_VERIFIED_PAIRED_CHECKPOINTS','manifest':manifest,'cuda_initialized':False,
            'note':'Parent imports standard library/report metadata only; no Torch worker launched.'});return
    worker_output=output.with_name(output.stem+'_cpu_snapshots.json');console=output.with_name(output.stem+'_cpu_console.log')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    command=[PYTHON,str(Path(__file__).resolve()),'--worker',str(manifest_path),'--output',str(worker_output)]
    with console.open('w') as handle:
        child=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    if child.returncode!=0 or 'CHECKPOINT_RNG_AUDIT_WORKER_SUCCESS' not in console.read_text():
        atomic_json(output,{'status':'CPU_WORKER_FAILED','returncode':child.returncode,'console':str(console),'partial_snapshots_not_evidence':True})
        raise RuntimeError('CPU checkpoint RNG worker failed; inspect '+str(console))
    result=read(worker_output)
    if result['status']!='CPU_WORKER_COMPLETE' or result['cuda_initialized'] or len(result['snapshots'])!=len(manifest['checkpoints']):
        raise RuntimeError('CPU-only complete-state audit verification failed')
    audit=finish_audit(manifest,result);audit['CPU_worker_returncode']=child.returncode
    audit['tool_sha256']=sha(__file__);audit['manifest_sha256']=sha(manifest_path);audit['cpu_snapshots_sha256']=sha(worker_output)
    atomic_json(output,audit)
    print(json.dumps({'report':str(output),'cuda_initialized':False,**audit['summary']},indent=2))

if __name__=='__main__':main()
