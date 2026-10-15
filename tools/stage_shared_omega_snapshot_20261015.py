"""Stage a verified source snapshot inside the user's remote sim; no training."""
from pathlib import Path
import argparse,json,shlex,subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',choices=['server9','host128'],default='server9')
    parser.add_argument('--kind',choices=['shared_omega','shared_omega05','shared_omega025_replica','time_scaled_heading','deep_shared_omega','deep_shared_omega10'],default='shared_omega')
    args=parser.parse_args()
    host,repo,python,socket,port={
        'server9':('guoke@sprl-server9.dynip.ntu.edu.sg','/home/guoke/sim',
                   '/home/guoke/miniconda3/envs/sim/bin/python','/tmp/sim_server9_continuous_32884.sock','32884'),
        'host128':('ke@10.87.114.128','/home/ke/keguo/sim',
                   '/home/ke/miniconda3/envs/sim/bin/python','/tmp/sim_host128_continuous_24h.sock','22'),
    }[args.server]
    package=json.loads((BASE/('latest_'+args.kind+'_package.json')).read_text())
    destination=repo+'/.codex_runs/'+args.kind+'_'+package['prepared_utc']
    remote=r'''import sys,io,json,hashlib,tarfile,os,subprocess,datetime
from pathlib import Path
root=Path(DEST)
payload=sys.stdin.buffer.read()
assert hashlib.sha256(payload).hexdigest()==ARCHIVE_SHA
assert not root.exists(), 'Preserve existing snapshots'
root.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
 for item in archive.getmembers():
  name=Path(item.name)
  assert item.isfile() and not name.is_absolute() and '..' not in name.parts
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(archive.extractfile(item).read())
manifest_bytes=(root/'SOURCE_MANIFEST.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest()==MANIFEST_SHA
manifest=json.loads(manifest_bytes)
for name,expected in manifest['files'].items():
 target=root/name
 assert target.stat().st_size==expected['bytes']
 assert hashlib.sha256(target.read_bytes()).hexdigest()==expected['sha256'],name
shared=Path(SHARED_DATA);assert shared.is_dir()
(root/'src/waymo_data').symlink_to(shared,target_is_directory=True)
python=REMOTE_PYTHON
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled',MPLCONFIGDIR='/tmp/sim_mpl_continuous',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for candidate in manifest['candidates']:
 result=subprocess.run([python,'src/run.py','experiment='+candidate,'optimizer=default','--cfg','job','--resolve'],cwd=root,env=env,text=True,capture_output=True,timeout=90)
 (root/(candidate+'.resolved.yaml')).write_text(result.stdout)
 (root/(candidate+'.config_preflight.log')).write_text(result.stderr)
 assert result.returncode==0,'Config preflight failed; preserve logs'
record=dict(status='STAGED_CONFIG_PREFLIGHT_PASS_NOT_TRAINING',staged_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),source_manifest_sha256=MANIFEST_SHA,archive_sha256=ARCHIVE_SHA,verified_source_files=len(manifest['files']),shared_data=str(shared),python=python,candidates=manifest['candidates'],training_started=False,limitation='LocalCPU/native16step GPU checks are separate. Review currentpaired evidence and actualGPU availability before deployment.')
(root/'STAGING_STATUS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''.replace('DEST',repr(destination)).replace('ARCHIVE_SHA',repr(package['archive_sha256'])).replace('MANIFEST_SHA',repr(package['source_manifest_sha256'])).replace('SHARED_DATA',repr(repo+'/src/waymo_data')).replace('REMOTE_PYTHON',repr(python))
    result=subprocess.run(['ssh','-S',socket,'-p',port,
        '-o','BatchMode=yes','-o','ConnectTimeout=15',host,
        'python3 -c '+shlex.quote(remote)],input=Path(package['archive']).read_bytes(),capture_output=True,timeout=210)
    if result.returncode:
        print(result.stderr.decode(errors='replace')[-1600:]);raise SystemExit(result.returncode)
    record=json.loads(result.stdout)
    record['server']=args.server
    name=args.kind+'_remote_staging'+('' if args.server=='server9' else '_host128')+'.json'
    (BASE/name).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
