"""Read-only full50K report collection across both authorized training hosts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,json,math,subprocess
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'
REMOTE=r'''from pathlib import Path
import json,sys,time
request=json.load(sys.stdin);items=list(request['paths'])
for path in request['job_records']:
 p=Path(path)
 if p.exists():
  r=json.loads(p.read_text());items.append(dict(experiment=r['candidate'],path=str(Path(r['output'])/'sd_agent_metrics.json')))
rows=[]
for item in items:
 p=Path(item['path'])
 if not p.exists():rows.append(dict(**item,status='NO_COMPLETE_REPORT_YET'));continue
 for attempt in range(3):
  try:
   before=p.stat().st_mtime_ns;r=json.loads(p.read_text());assert p.stat().st_mtime_ns==before
   rows.append(dict(**item,status='READ',modified_unix=p.stat().st_mtime,report=r));break
  except (ValueError,FileNotFoundError,AssertionError):
   if attempt==2:rows.append(dict(**item,status='REPORT_CHANGED_WHILE_READING'))
   time.sleep(.2)
print(json.dumps(rows))
'''

def main():
    old=json.loads((BASE/'latest_full_validation_review_20261015.json').read_text())
    queue=json.loads((BASE/'server9_finite_queue_deployment_20261015.json').read_text())
    host128=json.loads((BASE/'host128_full_deployment.json').read_text())
    requests=[('server9',['ssh','-S','/tmp/sim_server9_continuous_32884.sock','-p','32884','-o','BatchMode=yes','guoke@sprl-server9.dynip.ntu.edu.sg'],
       dict(paths=[dict(experiment=r['experiment'],path=r['path']) for r in old['reports']],
            job_records=[str(Path(queue['protocol']).parent/(name+'_full.json')) for name in ('omega10','trainmap_lanes')])),
       ('host128',['ssh','-S','/tmp/sim_host128_continuous_24h.sock','-o','BatchMode=yes','ke@10.87.114.128'],
       dict(paths=[dict(experiment=host128['experiment'],path=str(Path(host128['output'])/'sd_agent_metrics.json'))],job_records=[]))]
    def collect(item):
        name,ssh,request=item
        result=subprocess.run(ssh+['python3','-c',__import__('shlex').quote(REMOTE)],input=json.dumps(request),capture_output=True,text=True,timeout=60,check=True)
        return [dict(host=name,**row) for row in json.loads(result.stdout)]
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[r for batch in pool.map(collect,requests) for r in batch]
    target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    scope=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt')
    summaries=[]
    for row in rows:
        if row['status']!='READ':summaries.append({k:row[k] for k in ('host','experiment','status')});continue
        report=row['report'];m=report['agent_metrics'];mismatches=[k for k in scope if report.get(k)!=target[k]]
        assert all(math.isfinite(float(m[k])) and m[k]>=0 for k in target['targets'])
        ratios={k:m[k]/v for k,v in target['targets'].items()}
        summaries.append(dict(host=row['host'],experiment=row['experiment'],status='VERIFIED_FULL50K' if not mismatches else 'SCOPE_MISMATCH',
            scope_mismatches=mismatches,ema_updates=report['decoder']['ema_num_updates'],metrics=m,
            passed_targets=[k for k,v in ratios.items() if v<1],target_ratios=ratios))
    record=dict(fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,reports=rows,summaries=summaries)
    (BASE/'latest_fleet_full_validation_20261015.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(fetched_utc=record['fetched_utc'],summaries=summaries)))

if __name__=='__main__':main()
