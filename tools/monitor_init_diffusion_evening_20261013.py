"""Compact, once-per-minute live progress; never counts incomplete evaluations."""
import json
from pathlib import Path
import re
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'output/reports/init_diffusion_evening_20261013'
RUNS=ROOT/'logs/init_diffusion_evening_20261013'
if (ART/'final_long/rebased/sampler_merge/protocol.json').exists():ART=ART/'final_long/rebased/sampler_merge';RUNS=RUNS/'final_long/rebased/sampler_merge'
elif (ART/'final_long/rebased/protocol.json').exists():ART=ART/'final_long/rebased';RUNS=RUNS/'final_long/rebased'
elif (ART/'final_long/protocol.json').exists():ART=ART/'final_long';RUNS=RUNS/'final_long'
elif (ART/'long_audit/recovery/protocol.json').exists():ART=ART/'long_audit/recovery';RUNS=RUNS/'long_audit/recovery'
elif (ART/'long_audit/protocol.json').exists():ART=ART/'long_audit';RUNS=RUNS/'long_audit'
elif (ART/'recovery/protocol.json').exists():ART=ART/'recovery';RUNS=RUNS/'recovery'
BASE_ART=ROOT/'output/reports/init_diffusion_evening_20261013'
END=datetime.fromisoformat('2026-10-13T22:30:00+08:00').timestamp()
while time.time()<END:
    s=json.loads((ART/'status.json').read_text())
    stage=s.get('stage',{});label=stage.get('label',s.get('active_stage'))
    message={'local_time':datetime.now().strftime('%H:%M:%S'),'status':s.get('status'),'stage':label}
    if label:
        p=RUNS/label/'console.log'
        if p.exists():
            with p.open('rb') as stream:
                stream.seek(max(0,p.stat().st_size-12000));tail=stream.read().decode(errors='replace')
            rr=re.findall(r'STEP (\d+): loss=([\d.]+), grad=([\d.]+), elapsed=([\d.]+)s',tail)
            if rr:message.update(step=int(rr[-1][0]),target=stage.get('steps'),loss=float(rr[-1][1]),elapsed_seconds=float(rr[-1][3]))
    if (ART/'comparisons.json').exists():
        cc=json.loads((ART/'comparisons.json').read_text())['comparisons']
        message['completed_comparisons']=len(cc)
    if (BASE_ART/'wandb/watcher_status.json').exists():message['wandb_utc']=json.loads((BASE_ART/'wandb/watcher_status.json').read_text()).get('inventory_utc')
    for phase in ('n20_sampler','matching_followup'):
        path=BASE_ART/phase/'status.json'
        if path.exists():message[phase]=json.loads(path.read_text())
    print(json.dumps(message),flush=True)
    if s.get('status') in ('FAILED_STAGE','PARTIAL_EXCLUDED','GPU_OCCUPIED'):
        print('NEEDS_REVIEW',flush=True);break
    time.sleep(min(50,max(0,END-time.time())))
