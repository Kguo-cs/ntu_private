"""Stop a reviewed weak96epoch run only after preserving recovery artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'

REMOTE=r'''
from pathlib import Path
import csv,datetime,hashlib,json,os,shutil,signal,subprocess,sys,time
q=json.load(sys.stdin);root=Path(q['stage']['root']);art=root/'queued_deployment'
assert not art.exists(),'Preserve existing transaction; inspect instead of retrying'
pid=169361;proc=Path('/proc')/str(pid)
argv=[x for x in proc.joinpath('cmdline').read_bytes().decode().split('\0') if x]
old_name='cc_road_k8_match_s24_d5_v2h2_shape4'
assert 'run1.py' in argv and 'experiment='+old_name in argv
assert proc.stat().st_uid==os.getuid() and os.readlink(proc/'cwd')=='/home/guoke/sim/src'
uuid='GPU-99ecb4b0-f53e-f13d-486b-b56208be7faa'
def clients():
 rows=csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines())
 return [int(r[1]) for r in rows if r[0].strip()==uuid]
assert clients()==[pid]
out=Path(q['old']['path']).parent;report=json.loads((out/'sd_agent_metrics.json').read_text())
assert report==q['old']['report'],'Re-review any newer validation before stopping'
assert report['decoder']['ema_num_updates']==92544
target=json.loads((root/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
scope=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt')
assert all(report[k]==target[k] for k in scope)
assert report['agent_metrics']['ang_dev_jsd']>.035 and report['agent_metrics']['length_jsd']>.19
primary=list(out.glob('bc/*/checkpoints/*92544*.ckpt'));latest=list(out.glob('bc/*/checkpoints/last.ckpt'))
assert len(primary)==len(latest)==1
art.mkdir();snapshot=art/'predecessor_shape4_96_snapshot';snapshot.mkdir();review_path=art/'EARLY_STOP_REVIEW.json'
job=dict(label='omega025replica',gpu_index=2,gpu_uuid=uuid,predecessor_pid=pid,predecessor_epochs=96,
 predecessor_experiment=old_name,predecessor_log=os.readlink(proc/'fd/1'),predecessor_report=str(out/'sd_agent_metrics.json'),
 reviewed_early_stop=str(review_path),candidate='init_diffusion_657_n24_road_k8_omega025_seed6817_full')
reason='Shape4 still misses angle/length after96epochs andits64epoch matchedMP24 review missesangle. Replace lower-priority continuation with independent TRAINseed ofverified3layer recipe; preserve weights/optimizer/EMA/source/config/log. No quality guarantee from changingseed.'
protocol=dict(registered_utc=q['registered_utc'],source_root=str(root),source_manifest_sha256=q['stage']['source_manifest_sha256'],
 queue_code_sha256=q['queue_code_sha256'],repo='/home/guoke/sim',python=q['stage']['python'],jobs=[job],data_membership=q['data_membership'],
 authorization='User explicitly authorizes allGPU allocation andautonomous stopping/replacement, preserving recovery artifacts.',reason=reason,
 no_automatic_failure_retries=True,TRAIN_seed=6817,predeclared_evaluation_epochs=[96,104],predeclared_GEN=[2817,3817],full50K_required=True,
 limitation='Independent TRAINseed butdifferenthardware fromoriginal5090; do notattribute allchanges solelytoseed.')
script=art/'finite_queue.py';assert hashlib.sha256(q['code'].encode()).hexdigest()==q['queue_code_sha256'];script.write_text(q['code'])
config=art/'QUEUE_PROTOCOL.json';config.write_text(json.dumps(protocol,indent=2)+'\n')
dry=subprocess.run(['python3',str(script),'--protocol',str(config),'--dry-run'],cwd=root,text=True,capture_output=True,timeout=120)
(art/'dry_run.stdout').write_text(dry.stdout);(art/'dry_run.stderr').write_text(dry.stderr)
assert dry.returncode==0,'No process stopped: source/data preflight failed'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
saved=[]
for source,name in [(primary[0],'report_epoch96.ckpt'),(latest[0],'latest_recoverable.ckpt')]:
 before=source.stat();dest=snapshot/name;shutil.copyfile(source,dest);after=source.stat()
 assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'No process stopped: checkpoint changed'
 digest=sha(dest);assert sha(source)==digest
 saved.append(dict(path=str(dest),original_path=str(source),bytes=dest.stat().st_size,sha256=digest))
probe="""import gc;gc.disable()
import torch,json,sys
torch.set_num_threads(1);rows=[]
for path in sys.argv[1:]:
 c=torch.load(path,map_location='cpu',weights_only=False);s=c['state_dict'];g=s['encoder.init_decoder._extra_state']['ema'];step=int(c['global_step']);assert g['num_updates']==step
 assert c['optimizer_states'] and all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))
 rows.append(dict(epoch=int(c['epoch']),global_step=step,EMA=int(g['num_updates']),optimizer_present=True))
print(json.dumps(rows),flush=True)
"""
p=subprocess.run([q['stage']['python'],'-c',probe,*[r['path'] for r in saved]],capture_output=True,text=True,timeout=90,
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
(art/'checkpoint_probe.stderr').write_text(p.stderr);assert p.returncode==0,'No process stopped: checkpoint validation failed'
metadata=json.loads(p.stdout);assert metadata[0]['global_step']==92544 and metadata[1]['global_step']>=92544
for item,meta in zip(saved,metadata):item.update(meta)
native=snapshot/'native_report.json';native.write_text(json.dumps(report,indent=2)+'\n')
shutil.copyfile(out/'.hydra/config.yaml',snapshot/'original_config.yaml')
source_archive=snapshot/'source_at_review'
for name in ('src/smart','src/utils','configs'):
 shutil.copytree(Path('/home/guoke/sim')/name,source_archive/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile('/home/guoke/sim/src/run1.py',source_archive/'src/run1.py')
source_manifest={str(p.relative_to(source_archive)):sha(p) for p in sorted(source_archive.rglob('*')) if p.is_file()}
(source_archive/'SOURCE_MANIFEST.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
shutil.copyfile(job['predecessor_log'],snapshot/'log_prefix_at_review.log')
review=dict(status='SNAPSHOT_VERIFIED_SIGNAL_PENDING',predecessor_pid=pid,report_checkpoint=saved[0],latest_recoverable_checkpoint=saved[1],
 native_report_snapshot=str(native),original_config=str(snapshot/'original_config.yaml'),original_log=job['predecessor_log'],
 original_source_root='/home/guoke/sim/src',source_at_review=str(source_archive),source_at_review_manifest_sha256=sha(source_archive/'SOURCE_MANIFEST.json'),
 source_caveat='Original process had no immutable launchmanifest; this preserves source observed atreview, notproof oflaunch-timebytes.',
 original_command=argv,reason=reason,resume_note='Savedcheckpoint includesoptimizer/EMA; resume onfreeGPU withoriginalconfig. Exactuninterrupted RNGcontinuation isnotclaimed.')
review_path.write_text(json.dumps(review,indent=2)+'\n')
assert [x for x in proc.joinpath('cmdline').read_bytes().decode().split('\0') if x]==argv and clients()==[pid]
os.kill(pid,signal.SIGINT);review.update(signal_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stop_signal='SIGINT')
review_path.write_text(json.dumps(review,indent=2)+'\n')
def alive():
 try:return proc.joinpath('stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
deadline=time.monotonic()+45
while alive() and time.monotonic()<deadline:time.sleep(1)
if alive():
 review['status']='SIGINT_SENT_WAITING_FOR_PROCESS_EXIT';review_path.write_text(json.dumps(review,indent=2)+'\n')
 print(json.dumps(dict(status=review['status'],review=str(review_path),protocol=str(config),training_started=False)));raise SystemExit
review.update(status='STOPPED_AFTER_VERIFIED_SNAPSHOT',stopped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
review_path.write_text(json.dumps(review,indent=2)+'\n')
with (art/'queue.log').open('x') as f:child=subprocess.Popen(['python3','-u',str(script),'--protocol',str(config)],cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='REVIEWED_SHAPE4_STOP_AND_REPLICA_QUEUE_REGISTERED',pid=child.pid,protocol=str(config),review=str(review_path),source_root=str(root),
 source_manifest_sha256=q['stage']['source_manifest_sha256'],jobs=[job],report_checkpoint=saved[0],latest_recoverable_checkpoint=saved[1],log=str(art/'queue.log'),dry_run=json.loads(dry.stdout))
(art/'DRIVER.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''

def main():
 stage=json.loads((BASE/'shared_omega025_replica_remote_staging.json').read_text())
 assert stage['status']=='STAGED_CONFIG_PREFLIGHT_PASS_NOT_TRAINING'
 proof=json.loads((BASE/'omega025_seed_replication_cpu_proof_20261015.json').read_text())
 assert proof['status']=='OMEGA025_INDEPENDENT_TRAIN_SEED_CPU_PREFLIGHT_PASS'
 for name,digest in proof['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
 fleet=json.loads((BASE/'latest_fleet_full_validation_20261015.json').read_text())
 old=next(r for r in fleet['reports'] if r['experiment']=='cc_road_k8_match_s24_d5_v2h2_shape4')
 assert old['report']['decoder']['ema_num_updates']==92544
 m=old['report']['agent_metrics'];assert m['ang_dev_jsd']>.035 and m['length_jsd']>.19
 code=(ROOT/'tools/server9_init_queue_20261015.py').read_text()
 packet=dict(stage=stage,proof=proof,old=old,code=code,queue_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
  data_membership=json.loads((BASE/'host128_data_membership_expected.json').read_text())['expected'],
  registered_utc=datetime.now(timezone.utc).isoformat())
 result=subprocess.run(['ssh','-S','/tmp/sim_server9_continuous_32884.sock','-p','32884','-o','BatchMode=yes',
  'guoke@sprl-server9.dynip.ntu.edu.sg','python3 -c '+shlex.quote(REMOTE)],input=json.dumps(packet),text=True,capture_output=True,timeout=240)
 if result.returncode:print(result.stderr[-2500:]);raise SystemExit(result.returncode)
 record=json.loads(result.stdout)
 (BASE/'server9_omega025_replica_replacement_20261015.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))

if __name__=='__main__':main()
