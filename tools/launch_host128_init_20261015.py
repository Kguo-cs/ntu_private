"""Prepare and launch an owned native InitDiffusion job on the authorized 5090."""
from pathlib import Path
import argparse,hashlib,json,pickle,shlex,subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
SOCKET='/tmp/sim_host128_continuous_24h.sock'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['smoke','full','status'])
    args=parser.parse_args()
    stage=json.loads((BASE/'shared_omega_remote_staging_host128.json').read_text())
    expected=json.loads((BASE/'host128_data_membership_expected.json').read_text())['expected']
    sample_hashes={}
    for split in expected:
        if split=='training_map2_sd':
            with (ROOT/'src/waymo_data/full/training_map2_sd_files.pkl').open('rb') as f:names=pickle.load(f)['files']
        else:names=json.loads((BASE/'full_sampler_50k_v2/full50000_files.json').read_text())
        for name in sorted(names)[::max(1,len(names)//16)][:16]:
            sample_hashes[split+'/'+name]=hashlib.sha256((ROOT/'src/waymo_data/full'/split/name).read_bytes()).hexdigest()
    remote=r'''from pathlib import Path
import os,json,subprocess,datetime,hashlib,pickle
root=Path(STAGE)
phase=PHASE
expected=EXPECTED
sample_hashes=SAMPLES
python='/home/ke/miniconda3/envs/sim/bin/python'
repo=Path('/home/ke/keguo/sim')
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for name,row in manifest['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==row['sha256'],name
art=root/'deployment';art.mkdir(exist_ok=True)
if phase=='status':
 records={p.name:json.loads(p.read_text()) for p in art.glob('*.json')}
 for row in records.values():
  if isinstance(row,dict) and 'pid' in row:
   p=Path('/proc')/str(row['pid']);row['process_exists']=p.exists()
   try:row['process_state']=(p/'stat').read_text().split(') ',1)[1].split()[0]
   except FileNotFoundError:pass
 print(json.dumps(records));raise SystemExit
members={}
for split,ref in expected.items():
 names=sorted(n for n in os.listdir(repo/'src/waymo_data/full'/split) if n.endswith(('.pt','.pkl')))
 digest=hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()
 assert len(names)==ref['count'] and digest==ref['sorted_membership_sha256'],split
 path=art/(split+'.pkl')
 if not path.exists():path.write_bytes(pickle.dumps({'files':names}))
 members[split]=dict(count=len(names),sorted_membership_sha256=digest)
for name,digest in sample_hashes.items():
 assert hashlib.sha256((repo/'src/waymo_data/full'/name).read_bytes()).hexdigest()==digest,name
map_path=repo/'src/waymo_data/AIRL32_interdis_mean_203_disnodiff_nonego_epoch=9-step=152190_0.79036.ckpt'
h=hashlib.sha256()
with map_path.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='610c1f94a0f573fd078e251af37c38c73255b969e52154e66546588996ef99fb'
(art/'data_verification.json').write_text(json.dumps(dict(members=members,sample_file_sha256=sample_hashes,map_sha256=h.hexdigest(),limitation='Complete filename membership;32 sampled file contents checked, not all scene bytes.'),indent=2)+'\n')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not apps,'GPU occupied: preserve jobs and review before launch'
experiment='init_diffusion_657_n24_road_k8_omega025_full'
command=[python,'-u','src/run0.py','experiment='+experiment,'optimizer=default',
 'data.scenario_dreamer_train_sample_list='+str(art/'training_map2_sd.pkl'),
 'data.scenario_dreamer_eval_set='+str(art/'scenario_dreamer_val.pkl')]
if phase=='smoke':
 out=art/'native_smoke_v3'
 command += ['~logger','~callbacks','trainer.max_steps=2','trainer.limit_train_batches=2',
 'trainer.limit_val_batches=0','+trainer.enable_model_summary=false',
 'data.num_workers=0','hydra.run.dir='+str(out)]
else:
 status=json.loads((art/'smoke_v3_status.json').read_text())
 p=Path('/proc')/str(status['pid']);assert not p.exists() or (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z','Smoke still running'
 log=Path(status['log']).read_text();assert '`max_steps=2` reached' in log and 'Traceback (most recent call last)' not in log,'Review smoke before full run'
 out=repo/'src/logs'/experiment/('host128_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
 command += ['logger.wandb.name='+experiment+'_host128','logger.wandb.group=continuous_20261015',
 'logger.wandb.tags=[shared_omega025,midpoint24,full50k,source_'+SOURCE_SHA[:12]+']',
 'hydra.run.dir='+str(out)]
record_path=art/('smoke_v3_status.json' if phase=='smoke' else phase+'_status.json')
assert not record_path.exists(),'Inspect existing deployment before retry'
out.mkdir(parents=True,exist_ok=False)
log_path=out/'console.log'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-8eb64d88-b334-0531-2915-510cbd1a8c5d',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/sim_mpl_continuous',
 WANDB_MODE='disabled' if phase=='smoke' else 'online')
with log_path.open('x') as f:
 def limits():
  import resource
  soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE,(min(max(soft,16384),hard),hard))
 child=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
record=dict(status='STARTED_'+phase.upper(),pid=child.pid,phase=phase,command=command,log=str(log_path),output=str(out),source_root=str(root),source_manifest_sha256=SOURCE_SHA,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),experiment=experiment,exploratory=True,full_quality_unproven=True)
record_path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''.replace('STAGE',repr(stage['root'])).replace('PHASE',repr(args.phase)).replace('EXPECTED',repr(expected)).replace('SAMPLES',repr(sample_hashes)).replace('SOURCE_SHA',repr(stage['source_manifest_sha256']))
    result=subprocess.run(['ssh','-S',SOCKET,'-o','BatchMode=yes','-o','ConnectTimeout=10','ke@10.87.114.128',
        'python3 -c '+shlex.quote(remote)],text=True,capture_output=True,timeout=90)
    if result.returncode:
        print(result.stderr[-2000:]);raise SystemExit(result.returncode)
    record=json.loads(result.stdout)
    (BASE/('host128_'+args.phase+'_deployment.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
