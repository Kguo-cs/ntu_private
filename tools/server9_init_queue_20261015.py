"""Finite two-GPU queue: preserve completed predecessors, smoke, then train.

No process is stopped and no failed stage is retried automatically. Each GPU is
claimed only after its registered predecessor completes its final full50K val.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,pickle,resource,subprocess,time


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def save(p,value):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,p)
def alive(pid):
    try:return Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def gpu_apps():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return [(r[0].strip(),int(r[1])) for r in csv.reader(result.splitlines()) if len(r)==2]

def gpu_occupied(job):
    """Allow only explicitly recorded, unchanged user preprocessing clients."""
    allowed={int(v['pid']):v['cmdline_sha256'] for v in job.get('allowed_gpu_clients',[])}
    for uuid,pid in gpu_apps():
        if uuid!=job['gpu_uuid']:continue
        path=Path('/proc',str(pid),'cmdline')
        try:unchanged=pid in allowed and hashlib.sha256(path.read_bytes()).hexdigest()==allowed[pid]
        except FileNotFoundError:continue
        if not unchanged:return True
    if job.get('minimum_free_mib') is not None:
        free=int(subprocess.check_output(['nvidia-smi','--id='+job['gpu_uuid'],
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<int(job['minimum_free_mib']):return True
    return False

def check_source(root,protocol):
    raw=(root/'SOURCE_MANIFEST.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==protocol['source_manifest_sha256']
    for name,item in json.loads(raw)['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==item['sha256'],name

def verify_predecessor(job,target):
    if job.get('reviewed_early_stop'):
        review=read(job['reviewed_early_stop'])
        assert review['status']=='STOPPED_AFTER_VERIFIED_SNAPSHOT'
        assert review['predecessor_pid']==job['predecessor_pid']
        saved=Path(review['report_checkpoint']['path']);digest=hashlib.sha256()
        with saved.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==review['report_checkpoint']['sha256']
        assert review['report_checkpoint']['global_step']==job['predecessor_epochs']*964
        report=read(review['native_report_snapshot'])
    else:
        text=Path(job['predecessor_log']).read_bytes()[-2000000:].decode(errors='replace')
        assert '`max_epochs='+str(job['predecessor_epochs'])+'` reached' in text,'Predecessor did not finish normally'
        report=read(job['predecessor_report'])
    for name in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256',
                 'official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):
        assert report[name]==target[name],name
    assert report['decoder']['ema_num_updates']==job['predecessor_epochs']*964,'Predecessor final validation missing'
    return report

def limit_files():
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(max(soft,16384),hard),hard))

def launch(root,art,protocol,job,phase):
    check_source(root,protocol)
    assert not gpu_occupied(job),'GPU acquired by another process or insufficient memory'
    name=job['candidate'];python=protocol['python']
    out=(art/(name+'_smoke') if phase=='smoke' else Path(protocol['repo'])/'src/logs'/name/
         (protocol.get('host_label','server9')+'_gpu'+str(job['gpu_index'])+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True,exist_ok=False)
    command=[python,'-u','src/run0.py','experiment='+name,'optimizer=default',
             'data.scenario_dreamer_train_sample_list='+str(art/'training_map2_sd.pkl'),
             'data.scenario_dreamer_eval_set='+str(art/'scenario_dreamer_val.pkl'),'hydra.run.dir='+str(out)]
    if phase=='smoke':
        command += ['~logger','~callbacks','trainer.max_steps=2','trainer.limit_train_batches=2',
                    'trainer.limit_val_batches=0','+trainer.enable_model_summary=false','data.num_workers=0']
    else:
        command += ['logger.wandb.name='+name+'_'+protocol.get('host_label','server9')+'_gpu'+str(job['gpu_index']),
                    'logger.wandb.group=continuous_20261015',
                    'logger.wandb.tags=[midpoint24,full50k,source_'+protocol['source_manifest_sha256'][:12]+']']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=job['gpu_uuid'],WANDB_MODE='disabled' if phase=='smoke' else 'online',
             MPLCONFIGDIR='/tmp/sim_mpl_continuous',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    log=out/'console.log'
    with log.open('x') as f:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit_files)
    record=dict(status='RUNNING_'+phase.upper(),phase=phase,pid=child.pid,output=str(out),log=str(log),
                candidate=name,gpu_uuid=job['gpu_uuid'],command=command,started_utc=now(),exploratory=True,
                source_manifest_sha256=protocol['source_manifest_sha256'])
    save(art/(job['label']+'_'+phase+'.json'),record)
    return child,record

def verify_smoke(protocol,job,record):
    text=Path(record['log']).read_text()
    assert '`max_steps=2` reached' in text and 'Traceback (most recent call last)' not in text
    if job['label']=='omega025replica':assert 'Seed set to 6817' in text
    script='''import sys,json,torch
from pathlib import Path
folder=Path(sys.argv[1]);paths=list(folder.rglob('*.ckpt'));assert len(paths)==1,paths
c=torch.load(paths[0],map_location='cpu',weights_only=False);state=c['state_dict'];extra=state['encoder.init_decoder._extra_state']
assert c['global_step']==2 and extra['ema']['num_updates']==2
assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
if sys.argv[2]=='omega10':assert extra['denoiser_options']['heading_velocity_readout']=='shared_state_row3'
if sys.argv[2]=='omega05':
 assert extra['denoiser_options']['heading_velocity_readout']=='shared_state_row3'
 assert extra['denoiser_options']['denoiser_num_layers']==3
 assert extra['component_loss']['components']['heading']==dict(kind='mse',time_power=0,weight=.05,scale='none')
if sys.argv[2]=='omega025replica':
 assert extra['denoiser_options']['heading_velocity_readout']=='shared_state_row3'
 assert extra['denoiser_options']['denoiser_num_layers']==3
 assert extra['denoiser_options']['attribute_source'] is None
 assert extra['component_loss']['components']['heading']==dict(kind='mse',time_power=0,weight=.025,scale='none')
if sys.argv[2] in ('deepomega025','deepomega10'):
 assert extra['denoiser_options']['heading_velocity_readout']=='shared_state_row3'
 assert extra['denoiser_options']['denoiser_num_layers']==5
 assert extra['component_loss']['components']['heading']==dict(kind='mse',time_power=0,weight=.025 if sys.argv[2]=='deepomega025' else .1,scale='none')
if sys.argv[2]=='trainmap_lanes':assert state['encoder._extra_state']['initial_map_ema']['num_updates']==2
print(json.dumps(dict(status='PASS',global_step=2,ema_updates=2,checkpoint=str(paths[0]),finite_state=True)))
'''
    result=subprocess.run([protocol['python'],'-c',script,record['output'],job['label']],capture_output=True,text=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=90)
    assert result.returncode==0,result.stderr[-1600:]
    return json.loads(result.stdout)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    protocol=read(args.protocol);root=Path(protocol['source_root']);art=args.protocol.parent
    check_source(root,protocol)
    target=read(root/'configs/metrics/sd_vw_lane_conditioned_targets.json')
    for split,expected in protocol['data_membership'].items():
        names=sorted(n for n in os.listdir(Path(protocol['repo'])/'src/waymo_data/full'/split) if n.endswith(('.pt','.pkl')))
        assert len(names)==expected['count']
        assert hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()==expected['sorted_membership_sha256']
        path=art/(split+'.pkl')
        if not path.exists():path.write_bytes(pickle.dumps({'files':names}))
    states={j['label']:dict(status='WAITING_FOR_PREDECESSOR',predecessor_pid=j['predecessor_pid'],
            candidate=j['candidate'],gpu_uuid=j['gpu_uuid']) for j in protocol['jobs']}
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN_SOURCE_DATA_PASS',states=states)));return
    children={};save(art/'queue_status.json',dict(updated_utc=now(),states=states))
    while any(s['status'] not in ('STARTED_FULL','FAILED_REQUIRES_REVIEW') for s in states.values()):
        for job in protocol['jobs']:
            label=job['label'];state=states[label]
            try:
                if state['status'] in ('STARTED_FULL','FAILED_REQUIRES_REVIEW'):continue
                if state['status']=='SMOKE_RUNNING':
                    child,record=children[label]
                    result=child.poll()
                    if result is None:continue
                    assert result==0,'Smoke process failed: '+str(result)
                    verification=verify_smoke(protocol,job,record)
                    save(art/(label+'_smoke_verified.json'),verification)
                    state['status']='WAITING_FOR_FULL_GPU'
                if state['status']=='WAITING_FOR_PREDECESSOR':
                    if alive(job['predecessor_pid']):continue
                    final=verify_predecessor(job,target)
                    save(art/(label+'_predecessor_final50k.json'),final)
                    state['status']='WAITING_FOR_SMOKE_GPU'
                if gpu_occupied(job):continue
                phase='smoke' if state['status']=='WAITING_FOR_SMOKE_GPU' else 'full'
                child,record=launch(root,art,protocol,job,phase)
                children[label]=(child,record)
                state.update(status='SMOKE_RUNNING' if phase=='smoke' else 'STARTED_FULL',pid=child.pid,log=record['log'])
                print(json.dumps(dict(time=now(),label=label,**state)),flush=True)
            except Exception as exc:
                state.update(status='FAILED_REQUIRES_REVIEW',error=type(exc).__name__+': '+str(exc))
                print(json.dumps(dict(time=now(),label=label,**state)),flush=True)
        save(art/'queue_status.json',dict(updated_utc=now(),states=states))
        if any(s['status'] not in ('STARTED_FULL','FAILED_REQUIRES_REVIEW') for s in states.values()):time.sleep(20)
    print('FINITE_QUEUE_FINISHED',flush=True)

if __name__=='__main__':main()
