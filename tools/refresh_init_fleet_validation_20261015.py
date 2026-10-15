"""Read-only full50K report collection across both authorized training hosts."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,math,os,subprocess,tempfile
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'

def save_json(path,value):
    """Readers must see a complete report even during overlapping refreshes."""
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as handle:
        json.dump(value,handle,indent=2);handle.write('\n');temporary=handle.name
    os.replace(temporary,path)

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
    deep_queue=BASE/'server9_deep_omega_queue_deployment_20261015.json'
    if deep_queue.exists():
        deployment=json.loads(deep_queue.read_text())
        requests[0][2]['job_records'].extend(str(Path(deployment['protocol']).parent/(job['label']+'_full.json'))
            for job in deployment['jobs'])
    stronger_queue=BASE/'server9_deep_omega10_replacement_20261015.json'
    if stronger_queue.exists():
        deployment=json.loads(stronger_queue.read_text())
        if 'jobs' in deployment:
            requests[0][2]['job_records'].extend(str(Path(deployment['protocol']).parent/(job['label']+'_full.json'))
                for job in deployment['jobs'])
    omega05_queue=BASE/'host128_omega05_queue_deployment_20261015.json'
    if omega05_queue.exists():
        deployment=json.loads(omega05_queue.read_text())
        requests[1][2]['job_records'].extend(str(Path(deployment['protocol']).parent/(job['label']+'_full.json'))
            for job in deployment['jobs'])
    replica_queue=BASE/'server9_omega025_replica_replacement_20261015.json'
    if replica_queue.exists():
        deployment=json.loads(replica_queue.read_text())
        if 'jobs' in deployment:
            requests[0][2]['job_records'].extend(str(Path(deployment['protocol']).parent/(job['label']+'_full.json'))
                for job in deployment['jobs'])
    def collect(item):
        name,ssh,request=item
        result=subprocess.run(ssh+['python3','-c',__import__('shlex').quote(REMOTE)],input=json.dumps(request),capture_output=True,text=True,timeout=60,check=True)
        return [dict(host=name,**row) for row in json.loads(result.stdout)]
    with ThreadPoolExecutor(max_workers=2) as pool:rows=[r for batch in pool.map(collect,requests) for r in batch]
    target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    scope=('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256','official_source_sha256','reference_source_manifest_sha256','feature_counts_gt')
    summaries=[]
    archive_paths=[]
    for row in rows:
        if row['status']!='READ':summaries.append({k:row[k] for k in ('host','experiment','status')});continue
        report=row['report'];m=report['agent_metrics'];mismatches=[k for k in scope if report.get(k)!=target[k]]
        if not mismatches:
            # Preserve each native checkpoint's exact semantic report before
            # the next validation overwrites sd_agent_metrics.json remotely.
            canonical=json.dumps(report,sort_keys=True,separators=(',',':')).encode()
            digest=hashlib.sha256(canonical).hexdigest()
            folder=BASE/'native_full50K_history'/row['host']/row['experiment']
            folder.mkdir(parents=True,exist_ok=True)
            archived=folder/('ema'+str(report['decoder']['ema_num_updates'])+'_'+digest[:16]+'.json')
            if not archived.exists():
                envelope=dict(fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    host=row['host'],remote_path=row['path'],remote_modified_unix=row['modified_unix'],
                    semantic_report_sha256=digest,report=report)
                save_json(archived,envelope)
            archive_paths.append(str(archived))
        assert all(math.isfinite(float(m[k])) and m[k]>=0 for k in target['targets'])
        ratios={k:m[k]/v for k,v in target['targets'].items()}
        summaries.append(dict(host=row['host'],experiment=row['experiment'],status='VERIFIED_FULL50K' if not mismatches else 'SCOPE_MISMATCH',
            scope_mismatches=mismatches,ema_updates=report['decoder']['ema_num_updates'],metrics=m,
            passed_targets=[k for k,v in ratios.items() if v<1],target_ratios=ratios))
    record=dict(fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,reports=rows,summaries=summaries,verified_native_report_archives=archive_paths)
    save_json(BASE/'latest_fleet_full_validation_20261015.json',record)
    print(json.dumps(dict(fetched_utc=record['fetched_utc'],summaries=summaries)))

if __name__=='__main__':main()
