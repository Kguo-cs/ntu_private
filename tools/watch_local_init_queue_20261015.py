"""Read-only local GPU queue health; never restart or stop experiments."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,os,re,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'

def save(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def process(pid,expected=None):
 if not isinstance(pid,int):return dict(pid=pid,live=False)
 p=Path('/proc')/str(pid)
 try:
  state=(p/'stat').read_text().split(') ',1)[1].split()[0]
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  owned=p.stat().st_uid==os.getuid()
  return dict(pid=pid,live=state!='Z' and owned and bool(args[0]) and (expected is None or any(expected in arg for arg in args)),state=state,expected_command_matches=expected is None or any(expected in arg for arg in args))
 except (FileNotFoundError,ProcessLookupError,PermissionError):return dict(pid=pid,live=False)

def snapshot():
 recovery=json.loads((BASE/'reviewed_queue_recovery_20261015.json').read_text())
 specs=[('omega10_e80_confirmation_v2',recovery['omega10_driver_pid'],'run_omega10_e80_confirmation_20261015.py'),('omega025_e104_grid_screen_v2',recovery['grid']['pid'],'run_omega025_e104_grid_screen_20261015.py'),(Path(recovery['rectified']['artifact']).name,recovery['rectified']['pid'],'run_native_floor02_prefix_20261015.py')]
 extra=BASE/'local_queue_extra_jobs.json'
 if extra.exists():
  specs.extend((job['family'],job['pid'],job['script']) for job in json.loads(extra.read_text()))
 rows=[];anomalies=[]
 for family,driver,script in specs:
  folder=BASE/family
  try:status=json.loads((folder/'status.json').read_text())
  except (FileNotFoundError,ValueError):status=dict(status='STATUS_NOT_READABLE')
  phase=status['status'];complete=phase.endswith('_COMPLETE');driver_state=process(driver,script);child=None
  if phase.startswith('RUNNING'):
   child=process(status.get('pid'))
   if not child['live']:anomalies.append(family+':GPU_STAGE_PROCESS_MISSING')
  if not complete and not driver_state['live']:anomalies.append(family+':QUEUE_DRIVER_EXITED_BEFORE_COMPLETE')
  if (folder/'failure.json').exists():anomalies.append(family+':FAILURE_RECORD_PRESENT')
  row=dict(family=family,phase=phase,driver=driver_state,child=child)
  log=Path(status['log']) if status.get('log') else None
  if log is not None and log.exists():
   with log.open('rb') as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-8192));tail=f.read().decode(errors='replace')
   progress=re.findall(r'(?:Epoch \d+|Testing DataLoader \d+|Validation DataLoader \d+):[^\r\n]*',tail)
   row.update(log=str(log),log_modified_unix=log.stat().st_mtime,last_progress=progress[-1] if progress else None)
  rows.append(row)
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10,check=True).stdout.strip()
 previous_path=BASE/'local_queue_health.json';previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
 first=previous.get('anomaly_first_seen_unix',{})
 current={name:first.get(name,time.time()) for name in anomalies};confirmed=[name for name,t in current.items() if time.time()-t>=120 or name.endswith('FAILURE_RECORD_PRESENT')]
 result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),monitor_pid=os.getpid(),read_only_process_actions=True,no_automatic_restarts=True,compute_processes_csv=gpu,queues=rows,anomalies=anomalies,anomaly_first_seen_unix=current,confirmed_alerts=confirmed,status='REVIEW_REQUIRED' if confirmed else 'OBSERVING_TRANSITION' if anomalies else 'HEALTHY')
 save(previous_path,result)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
 with (BASE/'local_queue_health.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   result=snapshot();print(json.dumps(dict(status=result['status'],observed_utc=result['observed_utc'],alerts=result['confirmed_alerts'],phases=[r['phase'] for r in result['queues']])),flush=True)
   if args.once:return
   time.sleep(60)
if __name__=='__main__':main()
