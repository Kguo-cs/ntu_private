"""Snapshot the genuine completed bq7z12zy final checkpoint, without GPU work."""
from pathlib import Path
import hashlib,json,shlex,subprocess
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
DEST=BASE/'full_checkpoint_snapshots/deep_final128_20261015'
SSH=['ssh','-S','/tmp/sim_server9_continuous_32884.sock','-p','32884','-o','BatchMode=yes','-o','ConnectTimeout=15','guoke@sprl-server9.dynip.ntu.edu.sg']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def fetch():
    DEST.mkdir(parents=True,exist_ok=True)
    final=DEST/'provenance.json'
    if final.exists():
        record=json.loads(final.read_text());assert sha(DEST/'model.ckpt')==record['sha256'];return record
    script=r'''from pathlib import Path
import json,hashlib,subprocess,os
out=Path('/home/guoke/sim/src/logs/cc_road_k8_match_s24_v2h2_d5/2026-10-14_10-19-39')
p=Path('/proc/150825/stat')
if p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z':
 print(json.dumps(dict(status='WAITING_FOR_REGISTERED_FINAL_TRAINING_PROCESS',pid=150825)));raise SystemExit
report=json.loads((out/'sd_agent_metrics.json').read_text())
assert report['num_samples']==report['num_gt_samples']==50000 and report['full_membership']
assert report['decoder']['ema_num_updates']==123392
paths=[p for p in out.rglob('*.ckpt') if p.name=='last.ckpt' or '123392' in p.name]
paths.sort(key=lambda p:('123392' in p.name,p.stat().st_mtime_ns),reverse=True)
assert paths,'No final checkpoint'
path=paths[0];config=out/'.hydra/config.yaml'
probe="import torch,json,sys; c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); e=c['state_dict']['encoder.init_decoder._extra_state']; print(json.dumps(dict(epoch=int(c['epoch']),global_step=int(c['global_step']),ema_updates=int(e['ema']['num_updates']))))"
r=subprocess.run(['/home/guoke/miniconda3/envs/sim/bin/python','-c',probe,str(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=90,check=True)
metadata=json.loads(r.stdout);assert metadata['epoch']==127 and metadata['global_step']==metadata['ema_updates']==123392
h=hashlib.sha256()
with path.open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
print(json.dumps(dict(status='GENUINE_FINAL_CHECKPOINT_READY',path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest(),source_config=str(config),source_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),final_report=str(out/'sd_agent_metrics.json'),**metadata)))
'''
    result=subprocess.run(SSH+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=120,check=True)
    metadata=json.loads(result.stdout)
    if metadata['status'].startswith('WAITING'):return metadata
    for remote,name,digest in [(metadata['path'],'model.ckpt',metadata['sha256']),
                               (metadata['source_config'],'original_config.yaml',metadata['source_config_sha256'])]:
        target=DEST/name
        if target.exists():assert sha(target)==digest;continue
        part=target.with_suffix(target.suffix+'.part')
        with part.open('wb') as f:
            subprocess.run(SSH+['cat '+shlex.quote(remote)],stdout=f,stderr=subprocess.PIPE,timeout=180,check=True)
        assert sha(part)==digest,'Transferred file digest differs; preserve partial artifact'
        part.replace(target)
    metadata.update(origin='Completed128epoch training, actual epoch127/globalstep123392/EMA123392',original_remote_path=metadata['path'])
    (DEST/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

if __name__=='__main__':print(json.dumps(fetch()))
