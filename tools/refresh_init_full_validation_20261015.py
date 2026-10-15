"""Read-only SSH refresh and strict full50K scope verification."""
from pathlib import Path
import datetime
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'


def main():
    previous = json.loads((BASE / 'latest_full_validation_review_20261015.json').read_text())
    paths = [{'experiment': r['experiment'], 'path': r['path']} for r in previous['reports']]
    script = '''import json,time
from pathlib import Path
paths=PATHS
reports=[]
for item in paths:
 p=Path(item['path'])
 for attempt in range(3):
  try:
   before=p.stat().st_mtime_ns
   report=json.loads(p.read_text())
   after=p.stat().st_mtime_ns
   if before!=after:raise ValueError('Report changed while reading')
   reports.append(dict(**item,modified_unix=p.stat().st_mtime,report=report))
   break
  except (ValueError,FileNotFoundError):
   if attempt==2:raise
   time.sleep(.2)
print(json.dumps(reports))
'''.replace('PATHS', repr(paths))
    result = subprocess.run(['ssh', '-S', '/tmp/sim_server9_continuous_32884.sock',
        '-p', '32884', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        'guoke@sprl-server9.dynip.ntu.edu.sg', 'python3', '-'],
        input=script, text=True, capture_output=True, timeout=60, check=True)
    reports = json.loads(result.stdout)
    target = json.loads((ROOT / 'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
    scope = ['num_samples', 'num_gt_samples', 'full_membership', 'map_source',
             'eval_set_sha256', 'official_source_sha256',
             'reference_source_manifest_sha256', 'feature_counts_gt']
    summaries = []
    for row in reports:
        report = row['report']
        mismatches = [k for k in scope if report.get(k) != target[k]]
        assert not mismatches, (row['experiment'], mismatches)
        metrics = report['agent_metrics']
        passed = [k for k, limit in target['targets'].items() if metrics[k] < limit]
        decoder = report['decoder']
        summaries.append(dict(experiment=row['experiment'], modified_unix=row['modified_unix'],
            ema_num_updates=decoder.get('ema_num_updates'),
            sampling_steps=decoder.get('sampling_steps'),
            strict_scope_verified=True, passed_targets=passed, metrics=metrics))
    now = datetime.datetime.now(datetime.timezone.utc)
    record = dict(fetched_utc=now.isoformat(), read_only=True, reports=reports, summaries=summaries)
    archive = BASE / ('full_validation_review_' + now.strftime('%Y%m%dT%H%M%SZ') + '.json')
    archive.write_text(json.dumps(record, indent=2) + '\n')
    (BASE / 'latest_full_validation_review_20261015.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(archive=str(archive), summaries=summaries), ensure_ascii=False))


if __name__ == '__main__':
    main()
