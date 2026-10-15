"""Read-only W&B/server snapshots for the user-authorized ongoing search."""
from __future__ import annotations
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import refresh_wandb_cc_readonly as api

ART = ROOT / 'output/reports/init_diffusion_continuous_20261014'
SOCKET = '/tmp/sim_server9_continuous_32884.sock'
STOP = False
REMOTE_SCRIPT = r'''
import json,os,re,subprocess
from pathlib import Path
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'],text=True)
jobs=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  stat=(p/'stat').read_text().split()
  if stat[4]!=p.name or stat[2]=='Z':continue
  args=(p/'cmdline').read_bytes().decode().split('\0')
  scripts=[x for x in args if re.fullmatch(r'(?:.*/)?run[0-3]?\.py',x)]
  if not scripts:continue
  log=Path(os.readlink(p/'fd/1'))
  with log.open('rb') as f:
   f.seek(0,2);size=f.tell();f.seek(max(0,size-262144));text=f.read().decode(errors='replace')
  text=re.sub(r'\x1b\[[0-9;]*[a-zA-Z]','',text)
  progress=re.findall(r'Epoch \d+:[^\r\n]*',text)
  evaluation_progress=re.findall(r'(?:Validation|Testing) DataLoader \d+:[^\r\n]*',text)
  phases=re.findall(r'(Epoch \d+|Validation DataLoader \d+|Testing DataLoader \d+):[^\r\n]*',text)
  suffix=re.findall(r'v_num=([a-z0-9]+)',text)
  jobs.append(dict(pid=int(p.name),entrypoint=scripts[0],experiment=next((x.split('=',1)[1] for x in args if x.startswith('experiment=')),None),log=str(log),progress=progress[-1] if progress else None,evaluation_progress=evaluation_progress[-1] if evaluation_progress else None,current_phase=phases[-1] if phases else None,process_state=stat[2],log_modified_unix=log.stat().st_mtime,wandb_suffix=suffix[-1] if suffix else None,error_in_tail=bool(re.search(r'Traceback \(most recent call last\)|CUDA out of memory|Error executing job',text))))
 except (FileNotFoundError,PermissionError,ProcessLookupError,OSError):pass
print(json.dumps(dict(gpu_csv=gpu,compute_csv=apps,jobs=jobs)))
'''


def halt(*unused):
    global STOP
    STOP = True


def snapshot():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ART / 'wandb' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    # Archive native full50K reports independently of W&B availability. A new
    # epoch may overwrite the remote filename while the old semantic report
    # remains preserved locally by checkpoint/EMA step and content hash.
    try:
        result = subprocess.run([sys.executable, str(ROOT / 'tools/refresh_init_fleet_validation_20261015.py')],
            cwd=ROOT, text=True, capture_output=True, timeout=60)
        (folder / 'native_fleet_refresh.stdout').write_text(result.stdout)
        (folder / 'native_fleet_refresh.stderr').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError('Native fleet collection failed')
        native = json.loads((ART / 'latest_fleet_full_validation_20261015.json').read_text())
        api.write(folder / 'native_fleet_full50K.json', native)
        print('NATIVE_FULL50K_ARCHIVED ' + str(len(native['verified_native_report_archives'])), flush=True)
    except Exception as exc:
        api.write(folder / 'native_fleet_refresh_error.json', dict(
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            error_type=type(exc).__name__, read_only=True))
    api.inventory(folder)
    inventory = json.loads((folder / 'inventory.json').read_text())
    previous_path = ART / 'wandb_signatures.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    selected, signatures = [], {}
    watched = {'0vftb3cc', 'nnsjk5vl', 'iu3fa4gx', 'cchgfuhr', 'bq7z12zy', 'efzhwmu3'}
    watched.update(r['id'] for r in inventory['runs'][:16]
                   if r['name'].startswith(('cc_', 'init_diffusion_')))
    for record in inventory['runs']:
        if record['id'] not in watched:
            continue
        sig = hashlib.sha256(json.dumps(record['summary'], sort_keys=True).encode()).hexdigest()
        signatures[record['id']] = sig
        if previous.get(record['id']) != sig:
            selected.append(record['id'])
    if selected:
        api.histories(folder, selected)
        failed = json.loads((folder / 'history_fetch_status.json').read_text())['errors']
        for error in failed:
            signatures.pop(error['run_id'], None)
    api.write(previous_path, signatures)
    result = subprocess.run(
        ['ssh', '-S', SOCKET, '-p', '32884', '-o', 'BatchMode=yes', '-o',
         'ConnectTimeout=15', 'guoke@sprl-server9.dynip.ntu.edu.sg', 'python3', '-'],
        input=REMOTE_SCRIPT, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Read-only SSH snapshot failed')
    server = json.loads(result.stdout)
    extra = subprocess.run(
        ['ssh', '-S', '/tmp/sim_host128_continuous_24h.sock', '-o', 'BatchMode=yes',
         '-o', 'ConnectTimeout=10', 'ke@10.87.114.128', 'python3', '-'],
        input=REMOTE_SCRIPT, text=True, capture_output=True, timeout=60)
    if extra.returncode == 0:
        host128 = json.loads(extra.stdout)
        host128.update(fetched_at_utc=datetime.now(timezone.utc).isoformat(), read_only=True,
                       ssh_host='ke@10.87.114.128')
        api.write(folder / 'host128.json', host128)
        api.write(ART / 'latest_host128.json', host128)
    else:
        api.write(ART / 'host128_monitor_error.json', dict(
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            status='SSH_READ_ONLY_REFRESH_FAILED', returncode=extra.returncode))
    server.update(fetched_at_utc=datetime.now(timezone.utc).isoformat(), read_only=True)
    api.write(folder / 'server.json', server)
    api.write(ART / 'latest_server.json', server)
    api.write(ART / 'latest_snapshot.json', dict(folder=str(folder),
              inventory_fetched_at_utc=inventory['fetched_at_utc'], requested_histories=selected))
    print('SNAPSHOT_COMPLETE ' + stamp, flush=True)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    with (ART / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not STOP:
            try:
                snapshot()
                api.write(ART / 'watcher_status.json', dict(status='READ_ONLY_MONITOR_ACTIVE', pid=__import__('os').getpid()))
            except Exception as exc:
                api.write(ART / 'watcher_status.json', dict(status='SNAPSHOT_FAILED', error_type=type(exc).__name__))
                print('SNAPSHOT_FAILED ' + type(exc).__name__, flush=True)
            next_refresh = time.monotonic() + 900
            while not STOP and time.monotonic() < next_refresh:
                time.sleep(min(10, next_refresh - time.monotonic()))


if __name__ == '__main__':
    main()
