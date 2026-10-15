"""Preserve a reviewed weak map-training run, then queue a controlled full run."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'


def main():
    stage = json.loads((BASE / 'deep_shared_omega10_remote_staging.json').read_text())
    assert stage['status'] == 'STAGED_CONFIG_PREFLIGHT_PASS_NOT_TRAINING'
    proof = json.loads((BASE / 'deep_shared_omega10_gpu512_verified_20261015.json').read_text())
    assert proof['status'] == 'REAL_NATIVE_DEEP_GPU_FULL512_TWO_STEP_PASS' and proof['heading_weight'] == .1
    old = json.loads((BASE / 'server9_finite_queue_deployment_20261015.json').read_text())
    code = (ROOT / 'tools/server9_init_queue_20261015.py').read_text()
    packet = dict(stage=stage, old_job_record=str(Path(old['protocol']).parent / 'trainmap_lanes_full.json'),
        old_source_root=old['source_root'], code=code,
        queue_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        data_membership=json.loads((BASE / 'host128_data_membership_expected.json').read_text())['expected'],
        registered_utc=datetime.now(timezone.utc).isoformat())
    remote = r'''
from pathlib import Path
import csv,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time
q=json.load(sys.stdin);stage=q['stage'];root=Path(stage['root']);art=root/'queued_deployment'
assert not art.exists(),'Preserve existing transaction; inspect its records before another action'
old=json.loads(Path(q['old_job_record']).read_text());pid=old['pid']
assert old['candidate']=='init_diffusion_657_n24_road_k8_trainmap_lanes_midpoint_full' and pid==176366
proc=Path('/proc')/str(pid);argv=proc.joinpath('cmdline').read_bytes().decode().split('\0');argv=[x for x in argv if x]
assert 'experiment='+old['candidate'] in argv
assert proc.stat().st_uid==os.getuid()
cwd=os.readlink(proc/'cwd');assert Path(cwd)==Path(q['old_source_root'])
uuid='GPU-c73719d8-82b0-9d62-132f-b8abb22e755f'
def clients():
 rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
 return [int(r[1]) for r in rows if r[0].strip()==uuid]
assert clients()==[pid],'Review unexpected clients before replacing GPU1'
out=Path(old['output']);report=json.loads((out/'sd_agent_metrics.json').read_text())
target=json.loads((root/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
scope=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt')
assert all(report[k]==target[k] for k in scope)
assert report['decoder']['ema_num_updates']==23136,'Re-review the newer checkpoint before stopping'
m=report['agent_metrics'];assert m['ang_dev_jsd']>.09 and m['length_jsd']>.20 and m['speed_jsd']>.04
primary=list(out.glob('bc/*/checkpoints/*23136*.ckpt'));latest=list(out.glob('bc/*/checkpoints/last.ckpt'))
assert len(primary)==len(latest)==1
art.mkdir();snapshot=art/'predecessor_trainmap24_snapshot';snapshot.mkdir()
review_path=art/'EARLY_STOP_REVIEW.json'
job=dict(label='deepomega10',gpu_index=1,gpu_uuid=uuid,predecessor_pid=pid,predecessor_epochs=24,
 predecessor_experiment=old['candidate'],predecessor_log=old['log'],predecessor_report=str(out/'sd_agent_metrics.json'),
 reviewed_early_stop=str(review_path),candidate='init_diffusion_657_n24_road_k8_match2_d5_omega10_full')
protocol=dict(registered_utc=q['registered_utc'],source_root=str(root),source_manifest_sha256=stage['source_manifest_sha256'],
 queue_code_sha256=q['queue_code_sha256'],repo='/home/guoke/sim',python=stage['python'],jobs=[job],data_membership=q['data_membership'],
 authorization='User explicitly authorizes all GPUs and autonomous stopping/replacement; preserve checkpoint,optimizer,EMA,config,log and original frozen source.',
 reason='At24epochs map-training recipe has angle.1147,length.2295,speed.0510 with no heading recovery from16epochs. Replace this expensive lower-priority recipe, retaining shape4 until its matched-sampler review. New GPU1/GPU3 five-layer omega pair differs only in heading weight.1 versus.025.',
 no_automatic_failure_retries=True)
assert hashlib.sha256(q['code'].encode()).hexdigest()==q['queue_code_sha256']
script=art/'finite_queue.py';script.write_text(q['code']);config=art/'QUEUE_PROTOCOL.json';config.write_text(json.dumps(protocol,indent=2)+'\n')
dry=subprocess.run(['python3',str(script),'--protocol',str(config),'--dry-run'],cwd=root,text=True,capture_output=True,timeout=120)
(art/'dry_run.stdout').write_text(dry.stdout);(art/'dry_run.stderr').write_text(dry.stderr);assert dry.returncode==0,'No process stopped: queue preflight failed'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
saved=[]
for source,name in [(primary[0],'report_epoch24.ckpt'),(latest[0],'latest_recoverable.ckpt')]:
 before=source.stat();dest=snapshot/name;shutil.copyfile(source,dest);after=source.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'No process stopped: checkpoint changed while copying'
 digest=sha(dest);assert sha(source)==digest
 saved.append(dict(path=str(dest),original_path=str(source),bytes=dest.stat().st_size,sha256=digest))
probe="""import torch,json,sys
torch.set_num_threads(1);rows=[]
for path in sys.argv[1:]:
 c=torch.load(path,map_location='cpu',weights_only=False);s=c['state_dict'];g=s['encoder.init_decoder._extra_state']['ema'];m=s['encoder._extra_state']['initial_map_ema'];step=int(c['global_step']);assert g['num_updates']==m['num_updates']==step
 assert all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))
 rows.append(dict(epoch=int(c['epoch']),global_step=step,EMA=int(g['num_updates']),map_EMA=int(m['num_updates']),optimizer_present=bool(c['optimizer_states'])))
print(json.dumps(rows))"""
p=subprocess.run([stage['python'],'-c',probe,*[r['path'] for r in saved]],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),text=True,capture_output=True,check=True,timeout=90)
metadata=json.loads(p.stdout);assert metadata[0]['global_step']==23136 and metadata[1]['global_step']>=23136
for item,meta in zip(saved,metadata):item.update(meta)
native=snapshot/'native_report.json';native.write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(out/'.hydra/config.yaml',snapshot/'original_config.yaml')
review=dict(status='SNAPSHOT_VERIFIED_SIGNAL_PENDING',predecessor_pid=pid,report_checkpoint=saved[0],latest_recoverable_checkpoint=saved[1],native_report_snapshot=str(native),original_config=str(snapshot/'original_config.yaml'),original_log=old['log'],original_source_root=cwd,original_source_manifest_sha256=sha(Path(cwd)/'SOURCE_MANIFEST.json'),original_command=argv,original_wandb_id='iczokmcx',reason=protocol['reason'],resume_note='Original frozen source/config retained; restore latest_recoverable.ckpt on a free GPU using original command plus ckpt_path and W&B resume. Conventional epoch-boundary resume is supported; exact uninterrupted RNG trajectory is not claimed.')
review_path.write_text(json.dumps(review,indent=2)+'\n')
assert proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]==argv and clients()==[pid]
os.kill(pid,signal.SIGINT);review['stop_signal']='SIGINT';review['signal_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();review_path.write_text(json.dumps(review,indent=2)+'\n')
def alive():
 try:return proc.joinpath('stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
deadline=time.monotonic()+45
while alive() and time.monotonic()<deadline:time.sleep(1)
if alive():
 review['status']='SIGINT_SENT_WAITING_FOR_PROCESS_EXIT';review_path.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(dict(status=review['status'],review=str(review_path),protocol=str(config),training_started=False)));raise SystemExit
review['status']='STOPPED_AFTER_VERIFIED_SNAPSHOT';review['stopped_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();review_path.write_text(json.dumps(review,indent=2)+'\n')
with (art/'queue.log').open('x') as f:child=subprocess.Popen(['python3','-u',str(script),'--protocol',str(config)],cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='REVIEWED_EARLY_STOP_AND_FINITE_QUEUE_REGISTERED',pid=child.pid,protocol=str(config),review=str(review_path),source_root=str(root),source_manifest_sha256=stage['source_manifest_sha256'],jobs=[job],report_checkpoint=saved[0],latest_recoverable_checkpoint=saved[1],log=str(art/'queue.log'),dry_run=json.loads(dry.stdout))
(art/'DRIVER.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''
    result = subprocess.run(['ssh', '-S', '/tmp/sim_server9_continuous_32884.sock', '-p', '32884',
        '-o', 'BatchMode=yes', 'guoke@sprl-server9.dynip.ntu.edu.sg', 'python3 -c ' + shlex.quote(remote)],
        input=json.dumps(packet), text=True, capture_output=True, timeout=240)
    if result.returncode:
        print(result.stderr[-2200:])
        raise SystemExit(result.returncode)
    record = json.loads(result.stdout)
    (BASE / 'server9_deep_omega10_replacement_20261015.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
