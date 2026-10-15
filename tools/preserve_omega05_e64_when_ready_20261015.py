"""Preserve preregistered epoch64 weights; CPU/SSH only, no GPU job launch."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,shlex,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
ART=BASE/'omega05_e64_preservation_v1'
SSH=['ssh','-S','/tmp/sim_host128_continuous_24h.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','ke@10.87.114.128']
FETCH=ROOT/'tools/fetch_shape4_e64_checkpoint_20261015.py'
REMOTE=r'''from pathlib import Path
import json,os
out=Path('/home/ke/keguo/sim/src/logs/init_diffusion_657_n24_road_k8_omega05_full/host128_gpu0_20261015_163222')
p=Path('/proc/790922');live=False
try:
 state=(p/'stat').read_text().split(') ',1)[1].split()[0];args=(p/'cmdline').read_bytes().decode().split('\0')
 live=state!='Z' and p.stat().st_uid==os.getuid() and 'experiment=init_diffusion_657_n24_road_k8_omega05_full' in args
except (FileNotFoundError,PermissionError,ProcessLookupError):pass
try:
 report=json.loads((out/'sd_agent_metrics.json').read_text());step=report['decoder']['ema_num_updates']
except (FileNotFoundError,ValueError,KeyError):step=None
paths=list(out.glob('bc/*/checkpoints/*61696*.ckpt'));files=[]
for f in paths:
 try:s=f.stat();files.append(dict(path=str(f),size=s.st_size,mtime_ns=s.st_mtime_ns))
 except FileNotFoundError:pass
print(json.dumps(dict(training_pid=790922,training_process_live=live,native_report_EMA=step,checkpoint_files=files)))
'''
def save(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def main():
 ART.mkdir(exist_ok=False)
 registration=json.loads((BASE/'omega05_e64_registration_20261015.json').read_text())
 assert registration['epoch']==63 and registration['EMA']==61696
 assert sha(FETCH)==registration['fetch_source_sha256']
 save(ART/'protocol.json',dict(registered_utc=now(),prior_evaluation_registration=str(BASE/'omega05_e64_registration_20261015.json'),prior_registration_sha256=sha(BASE/'omega05_e64_registration_20261015.json'),fetch_source_sha256=sha(FETCH),tool_sha256=sha(Path(__file__)),checkpoint_EMA=61696,training_pid=790922,no_training_actions=True,no_GPU_launches=True,require_stable_checkpoint_two_observations=True,no_automatic_fetch_failure_retries=True))
 previous_files=None
 while True:
  assert sha(FETCH)==registration['fetch_source_sha256'],'Downloader changed; preserve and review'
  try:
   result=subprocess.run(SSH+['python3','-c',shlex.quote(REMOTE)],text=True,capture_output=True,check=True,timeout=30);state=json.loads(result.stdout)
  except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
   if isinstance(exc,subprocess.CalledProcessError) and exc.returncode!=255:raise
   save(ART/'status.json',dict(status='WAITING_FOR_SAME_REMOTE_JOB_OBSERVATION',error_type=type(exc).__name__,updated_utc=now()));time.sleep(60);continue
  files=state['checkpoint_files'];report_step=state['native_report_EMA']
  # Capture the semantic native64 report promptly, independently of checkpoint save timing.
  archive=BASE/'native_full50K_history/host128/init_diffusion_657_n24_road_k8_omega05_full'
  archived=list(archive.glob('ema61696_*.json'))
  if report_step==61696 and not archived:
   refreshed=subprocess.run([sys.executable,str(ROOT/'tools/refresh_init_fleet_validation_20261015.py')],cwd=ROOT,capture_output=True,text=True,timeout=60)
   if refreshed.returncode:raise RuntimeError('Native64 report collection failed; review beforefetch')
   archived=list(archive.glob('ema61696_*.json'))
  if len(files)==1 and files==previous_files and len(archived)==1:
   save(ART/'status.json',dict(status='PRESERVING_GENUINE64_CHECKPOINT',updated_utc=now(),remote=state))
   with (ART/'fetch.stdout').open('x') as out,(ART/'fetch.stderr').open('x') as err:
    result=subprocess.run([sys.executable,str(FETCH),'--profile','omega05_e64'],cwd=ROOT,stdout=out,stderr=err,timeout=600)
   assert result.returncode==0,'Fetch failed; no automatic retry'
   dest=BASE/'full_checkpoint_snapshots/omega05_e64_20261015';provenance=json.loads((dest/'provenance.json').read_text())
   assert provenance['epoch']==63 and provenance['global_step']==provenance['ema_updates']==61696 and sha(dest/'model.ckpt')==provenance['sha256']
   save(ART/'status.json',dict(status='OMEGA05_E64_GENUINE_CHECKPOINT_PRESERVED',completed_utc=now(),checkpoint=str(dest/'model.ckpt'),checkpoint_sha256=provenance['sha256'],CPU_GPU_evaluation_not_started=True));print('OMEGA05_E64_GENUINE_CHECKPOINT_PRESERVED',flush=True);return
  if not state['training_process_live'] and not files:
   raise RuntimeError('Verifiedtrainingprocessabsentandregistered64checkpointmissing; review actual state, do not restart')
  if report_step is not None and report_step>61696 and not archived:
   raise RuntimeError('Native64reportnotpreservedbeforeoverwrite; review, no fabricated native report')
  save(ART/'status.json',dict(status='WAITING_FOR_REGISTERED64_CHECKPOINT',updated_utc=now(),remote=state,exact64_native_archives=len(archived),checkpoint_stable_two_observations=bool(files and files==previous_files)))
  print(json.dumps(dict(updated_utc=now(),remote=state)),flush=True);previous_files=files;time.sleep(60)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  if ART.exists():save(ART/'failure.json',dict(status='STOPPED_REQUIRES_REVIEW',error=str(exc),updated_utc=now()))
  raise
