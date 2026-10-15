"""Register a finite source-verified queue on two authorized server9 GPUs."""
from pathlib import Path
import hashlib,json,shlex,subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'

def main():
    stage=json.loads((BASE/'shared_omega_remote_staging.json').read_text())
    assert stage['source_manifest_sha256']=='be9295dbc802d2eda7671048df9cdd4e9b8fda4f7dc9ecf6a8ad497b172a6392'
    proofs=['shared_heading_full_config_preflight_20261015.json','trainmap_lanes_midpoint_full_config_preflight.json']
    for name in proofs: assert 'PASS' in json.loads((BASE/name).read_text())['status']
    server=json.loads((BASE/'latest_server.json').read_text());reports=json.loads((BASE/'latest_full_validation_review_20261015.json').read_text())
    definitions=[dict(label='omega10',gpu_index=0,gpu_uuid='GPU-5e8b1b5b-2414-c9e1-ca16-888cc1d74fbc',
        predecessor_pid=159714,predecessor_epochs=80,predecessor_experiment='cc_road_k8_match_s24_v2h2_d5_e80',
        candidate='init_diffusion_657_n24_road_k8_omega10_full'),
        dict(label='trainmap_lanes',gpu_index=1,gpu_uuid='GPU-c73719d8-82b0-9d62-132f-b8abb22e755f',
        predecessor_pid=150825,predecessor_epochs=128,predecessor_experiment='cc_road_k8_match_s24_v2h2_d5',
        candidate='init_diffusion_657_n24_road_k8_trainmap_lanes_midpoint_full')]
    for job in definitions:
        old=next(j for j in server['jobs'] if j['pid']==job['predecessor_pid'])
        assert old['experiment']==job['predecessor_experiment']
        job['predecessor_log']=old['log']
        job['predecessor_report']=next(r['path'] for r in reports['reports'] if r['experiment']==job['predecessor_experiment'])
    code=(ROOT/'tools/server9_init_queue_20261015.py').read_text()
    protocol=dict(registered_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        source_root=stage['root'],source_manifest_sha256=stage['source_manifest_sha256'],
        queue_code_sha256=hashlib.sha256(code.encode()).hexdigest(),repo='/home/guoke/sim',
        python='/home/guoke/miniconda3/envs/sim/bin/python',jobs=definitions,
        data_membership=json.loads((BASE/'host128_data_membership_expected.json').read_text())['expected'],
        authorization='User authorizes all GPUs on these machines and autonomous task placement/replacement; existing near-complete tasks finish normally.',
        gates=['Registered predecessor exits normally with final full50K scope/EMA step count verified',
               'Selected GPU UUID has no compute clients','All frozen source manifest entries verified',
               'Candidate native2step GPU smoke and checkpoint EMA verified'],
        no_automatic_failure_retries=True,limitation='Exploratory full runs, not proven improvements. Omega weights run on different training hardware; not a strict hardware-paired ablation.')
    packet=dict(protocol=protocol,code=code)
    remote=r'''from pathlib import Path
import sys,json,subprocess,os,hashlib
payload=json.load(sys.stdin);protocol=payload['protocol'];root=Path(protocol['source_root']);art=root/'queued_deployment'
assert not art.exists(),'Existing queue must be reviewed'
for job in protocol['jobs']:
 p=Path('/proc')/str(job['predecessor_pid'])/'cmdline'
 if p.exists():assert 'experiment='+job['predecessor_experiment'] in p.read_bytes().replace(b'\0',b' ').decode()
path=root/'tools/server9_init_queue_20261015.py';assert not path.exists();assert hashlib.sha256(payload['code'].encode()).hexdigest()==protocol['queue_code_sha256']
path.write_text(payload['code']);art.mkdir();config=art/'QUEUE_PROTOCOL.json';config.write_text(json.dumps(protocol,indent=2)+'\n')
result=subprocess.run(['python3',str(path),'--protocol',str(config),'--dry-run'],cwd=root,text=True,capture_output=True,timeout=120)
(art/'dry_run.stdout').write_text(result.stdout);(art/'dry_run.stderr').write_text(result.stderr)
assert result.returncode==0,'Queue dry run failed; preserve logs'
with (art/'queue.log').open('x') as f:
 child=subprocess.Popen(['python3','-u',str(path),'--protocol',str(config)],cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='FINITE_QUEUE_REGISTERED',pid=child.pid,protocol=str(config),log=str(art/'queue.log'),source_root=str(root),source_manifest_sha256=protocol['source_manifest_sha256'],jobs=protocol['jobs'],dry_run=json.loads(result.stdout))
(art/'DRIVER.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''
    result=subprocess.run(['ssh','-S','/tmp/sim_server9_continuous_32884.sock','-p','32884','-o','BatchMode=yes',
        '-o','ConnectTimeout=15','guoke@sprl-server9.dynip.ntu.edu.sg','python3 -c '+shlex.quote(remote)],
        input=json.dumps(packet),text=True,capture_output=True,timeout=180)
    if result.returncode:print(result.stderr[-2000:]);raise SystemExit(result.returncode)
    record=json.loads(result.stdout);(BASE/'server9_finite_queue_deployment_20261015.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
