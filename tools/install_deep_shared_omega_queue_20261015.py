"""Queue one proved five-layer candidate after GPU3's genuine final validation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'


def build_deep_protocol():
    stage = json.loads((BASE / 'deep_shared_omega_remote_staging.json').read_text())
    assert stage['server'] == 'server9' and stage['status'] == 'STAGED_CONFIG_PREFLIGHT_PASS_NOT_TRAINING'
    proof = json.loads((BASE / 'deep_shared_omega_gpu512_verified_20261015.json').read_text())
    assert proof['status'] == 'REAL_NATIVE_DEEP_GPU_FULL512_TWO_STEP_PASS'
    assert proof['parameters'] == 12562954 and proof['batch_size'] == 512
    server = json.loads((BASE / 'latest_server.json').read_text())
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(server['fetched_at_utc'])).total_seconds()
    assert 0 <= age < 1800, 'Refresh actual server process inventory first'
    previous = next(j for j in server['jobs'] if j['pid'] == 163727)
    experiment = 'init_diffusion_657_n24_road_k8_match4_guidance_full'
    assert previous['experiment'] == experiment and not previous['error_in_tail']
    fleet = json.loads((BASE / 'latest_fleet_full_validation_20261015.json').read_text())
    report = next(r for r in fleet['reports'] if r['experiment'] == experiment)
    code = (ROOT / 'tools/server9_init_queue_20261015.py').read_text()
    protocol = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(), source_root=stage['root'],
        source_manifest_sha256=stage['source_manifest_sha256'],
        queue_code_sha256=hashlib.sha256(code.encode()).hexdigest(), repo='/home/guoke/sim',
        python=stage['python'],
        jobs=[dict(label='deepomega025', gpu_index=3,
            gpu_uuid='GPU-7a7221a5-df40-e22a-d8f4-4b5aca3bda00',
            predecessor_pid=163727, predecessor_epochs=128, predecessor_experiment=experiment,
            predecessor_log=previous['log'], predecessor_report=report['path'],
            candidate='init_diffusion_657_n24_road_k8_match2_d5_omega025_full')],
        data_membership=json.loads((BASE / 'host128_data_membership_expected.json').read_text())['expected'],
        authorization='User authorizes all GPU placement and task replacement; this queue stops no existing task.',
        rationale='Raw025 has genuine native72epoch full50K evidence with six targets passed; five-layer x0 has completed128epoch evidence. Test their combination from scratch, keeping the five-layer recipe otherwise fixed. Native8 time-scaling study does not authorize automatic promotion.',
        gates=['Predecessor normal128epoch completion and final full50K/EMA123392',
               'No compute clients on registered GPU UUID', 'Frozen source manifest unchanged',
               'Real native batch512 two-step smoke; finite state, EMA2, five layers and shared angular readout verified'],
        no_automatic_failure_retries=True,
        limitation='Hypothesis under test, not a demonstrated improvement. Different hardware and training schedules prevent treating unrelated existing runs as a strict paired ablation.')
    return protocol


def build_host128_omega05_protocol():
    stage = json.loads((BASE / 'shared_omega05_remote_staging_host128.json').read_text())
    assert stage['server'] == 'host128' and stage['status'] == 'STAGED_CONFIG_PREFLIGHT_PASS_NOT_TRAINING'
    proof = json.loads((BASE / 'omega05_cpu_preflight_20261015.json').read_text())
    assert proof['status'] == 'OMEGA05_NATIVE_FULL_CONFIG_CPU_PASS_NOT_DEPLOYED'
    assert proof['parameters'] == 8083978 and proof['initial_CPU_RNG_identical']
    for name, digest in proof['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    previous = json.loads((BASE / 'host128_full_deployment.json').read_text())
    assert previous['pid'] == 707688 and previous['experiment'] == 'init_diffusion_657_n24_road_k8_omega025_full'
    code = (ROOT / 'tools/server9_init_queue_20261015.py').read_text()
    return dict(registered_utc=datetime.now(timezone.utc).isoformat(), source_root=stage['root'],
        source_manifest_sha256=stage['source_manifest_sha256'], queue_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        repo='/home/ke/keguo/sim', python=stage['python'], host_label='host128',
        jobs=[dict(label='omega05', gpu_index=0, gpu_uuid='GPU-8eb64d88-b334-0531-2915-510cbd1a8c5d',
            predecessor_pid=previous['pid'], predecessor_epochs=128, predecessor_experiment=previous['experiment'],
            predecessor_log=previous['log'], predecessor_report=str(Path(previous['output']) / 'sd_agent_metrics.json'),
            candidate='init_diffusion_657_n24_road_k8_omega05_full', minimum_free_mib=22000,
            allowed_gpu_clients=[dict(pid=736500, cmdline_sha256='711bd5ec76cacd8d44479bec1fc677167dfa620cbe61fe54e216bc6974aa6b93',
                reason='User-owned data2initmap preprocessing; leave running. Allow only unchanged registered command identity.')])],
        data_membership=json.loads((BASE / 'host128_data_membership_expected.json').read_text())['expected'],
        authorization='User authorizes all GPU allocation and new full training. Stop no current task; preserve user preprocessing.',
        rationale='Same3layer recipe and seed, changing only heading loss weight.025 to.05, between ongoing.025 and.1. CPU proof confirms identical initialG1/map andRNG; no final-quality claim.',
        gates=['Original raw025 normal128epoch completion, full50K scope andEMA123392',
               'Frozen source and exactTRAIN/VAL filename membership',
               'GPU empty except unchanged registered user preprocessing, with at least22GiB free',
               'Same-card native batch512 two-step smoke, finite state, EMA2,3layers, shared angular readout andweight.05'],
        no_automatic_failure_retries=True,
        limitation='A hypothesis, not an established improvement. SameGPU/seed/config controls hardware but not an exact uninterrupted training RNG trajectory. Existing final025 outputs remain immutable in their own run directory.')


def deploy(protocol, host128=False):
    code = (ROOT / 'tools/server9_init_queue_20261015.py').read_text()
    remote = r'''
from pathlib import Path
import hashlib,json,os,subprocess,sys
payload=json.load(sys.stdin);p=payload['protocol'];root=Path(p['source_root']);art=root/'queued_deployment'
assert not art.exists(),'Preserve existing deployment; review its records instead'
proc=Path('/proc')/str(p['jobs'][0]['predecessor_pid'])/'cmdline'
if proc.exists():assert 'experiment='+p['jobs'][0]['predecessor_experiment'] in proc.read_bytes().replace(b'\0',b' ').decode()
assert hashlib.sha256(payload['code'].encode()).hexdigest()==p['queue_code_sha256']
art.mkdir();script=art/'finite_queue.py';script.write_text(payload['code'])
config=art/'QUEUE_PROTOCOL.json';config.write_text(json.dumps(p,indent=2)+'\n')
result=subprocess.run(['python3',str(script),'--protocol',str(config),'--dry-run'],cwd=root,capture_output=True,text=True,timeout=120)
(art/'dry_run.stdout').write_text(result.stdout);(art/'dry_run.stderr').write_text(result.stderr)
assert result.returncode==0,'Preflight failed; preserve logs without retry'
with (art/'queue.log').open('x') as f:
 child=subprocess.Popen(['python3','-u',str(script),'--protocol',str(config)],cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='FINITE_QUEUE_REGISTERED',pid=child.pid,protocol=str(config),log=str(art/'queue.log'),source_root=str(root),source_manifest_sha256=p['source_manifest_sha256'],jobs=p['jobs'],dry_run=json.loads(result.stdout),training_started=False)
(art/'DRIVER.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''
    ssh = (['ssh','-S','/tmp/sim_host128_continuous_24h.sock','-o','BatchMode=yes','ke@10.87.114.128'] if host128 else
           ['ssh','-S','/tmp/sim_server9_continuous_32884.sock','-p','32884','-o','BatchMode=yes','guoke@sprl-server9.dynip.ntu.edu.sg'])
    result = subprocess.run(ssh + ['python3 -c ' + shlex.quote(remote)],
        input=json.dumps(dict(protocol=protocol, code=code)), capture_output=True, text=True, timeout=180)
    if result.returncode:
        print(result.stderr[-1800:])
        raise SystemExit(result.returncode)
    record = json.loads(result.stdout)
    filename = 'host128_omega05_queue_deployment_20261015.json' if host128 else 'server9_deep_omega_queue_deployment_20261015.json'
    (BASE / filename).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('deep_server9','omega05_host128'),default='deep_server9')
    args = parser.parse_args()
    is_host128 = args.profile == 'omega05_host128'
    deploy(build_host128_omega05_protocol() if is_host128 else build_deep_protocol(), host128=is_host128)
