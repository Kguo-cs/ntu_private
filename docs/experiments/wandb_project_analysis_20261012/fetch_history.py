"""Read-only W&B history snapshot; no run initialization or remote writes."""
import os
os.environ.setdefault('WANDB_CACHE_DIR', '/tmp/sim_wandb_analysis_cache_20261012')
os.environ.setdefault('WANDB_SILENT', 'true')
import concurrent.futures
import datetime
import json
import sys
from pathlib import Path
import wandb
ROOT = Path(__file__).resolve().parent
inventory = json.loads((ROOT / 'inventory.json').read_text())
selected = set(sys.argv[1:])
runs = [r for r in inventory['runs'] if not selected or r['id'] in selected]
(ROOT / 'histories').mkdir(exist_ok=True)
def fetch(record):
    rid = record['id']
    path = ROOT / 'histories' / (rid + '.json')
    if path.exists():
        return rid, 'cached', len(json.loads(path.read_text())['rows'])
    api = wandb.Api(timeout=30)
    run = api.run('u3006612/bc/' + rid)
    ceiling = int(record['summary'].get('_step', 0)) + 1
    rows = list(run.scan_history(page_size=1000, max_step=ceiling))
    payload = {'run_id':rid,'name':record['name'],
        'fetched_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'method':'unsampled scan_history(keys=None)',
        'snapshot_max_step_exclusive':ceiling,'row_count':len(rows),'rows':rows}
    path.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False))
    return rid, 'downloaded', len(rows)
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    pending={pool.submit(fetch,r):r for r in runs}
    for future in concurrent.futures.as_completed(pending):
        record=pending[future]
        try:
            print(*future.result(),flush=True)
        except Exception as exc:
            errors.append({'run_id':record['id'],'error_type':type(exc).__name__})
            print(record['id'],'ERROR',type(exc).__name__,flush=True)
(ROOT/'history_fetch_status.json').write_text(json.dumps({'requested':len(runs),
    'errors':errors,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print('COMPLETE requested',len(runs),'errors',len(errors),flush=True)
