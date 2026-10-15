"""Preserve an actual intermediate shape4 checkpoint without stopping training."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/reports/init_diffusion_continuous_20261014'
DEST = BASE / 'full_checkpoint_snapshots/shape4_e64_20261015'
SSH = ['ssh', '-S', '/tmp/sim_server9_continuous_32884.sock', '-p', '32884',
       '-o', 'BatchMode=yes', 'guoke@sprl-server9.dynip.ntu.edu.sg']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['shape4_e64', 'omega025_e80', 'omega025_e96', 'omega025_e104', 'omega10_e80'], default='shape4_e64')
    args = parser.parse_args()
    global DEST, SSH
    profiles = dict(
        shape4_e64=dict(repo='/home/guoke/sim', python='/home/guoke/miniconda3/envs/sim/bin/python',
            run_output='/home/guoke/sim/src/logs/cc_road_k8_match_s24_d5_v2h2_shape4/2026-10-14_21-39-52',
            epoch=63, step=61696),
        omega025_e80=dict(repo='/home/ke/keguo/sim', python='/home/ke/miniconda3/envs/sim/bin/python',
            run_output='/home/ke/keguo/sim/src/logs/init_diffusion_657_n24_road_k8_omega025_full/host128_20261015_101728',
            epoch=79, step=77120),
        omega10_e80=dict(repo='/home/guoke/sim', python='/home/guoke/miniconda3/envs/sim/bin/python',
            run_output='/home/guoke/sim/src/logs/init_diffusion_657_n24_road_k8_omega10_full/server9_gpu0_20261015_023313',
            epoch=79, step=77120),
        omega025_e104=dict(repo='/home/ke/keguo/sim', python='/home/ke/miniconda3/envs/sim/bin/python',
            run_output='/home/ke/keguo/sim/src/logs/init_diffusion_657_n24_road_k8_omega025_full/host128_20261015_101728',
            epoch=103, step=100256),
        omega025_e96=dict(repo='/home/ke/keguo/sim', python='/home/ke/miniconda3/envs/sim/bin/python',
            run_output='/home/ke/keguo/sim/src/logs/init_diffusion_657_n24_road_k8_omega025_full/host128_20261015_101728',
            epoch=95, step=92544))
    request = dict(profile=args.profile, **profiles[args.profile])
    DEST = BASE / ('full_checkpoint_snapshots/' + args.profile + '_20261015')
    if args.profile.startswith('omega025_'):
        SSH = ['ssh', '-S', '/tmp/sim_host128_continuous_24h.sock', '-o', 'BatchMode=yes', 'ke@10.87.114.128']
    if args.profile in ('omega025_e104','omega025_e96'):
        # A later validation already replaced the remote filename. Use its
        # previously verified immutable report, never the final128 report.
        archives=list((BASE/'native_full50K_history/host128/init_diffusion_657_n24_road_k8_omega025_full').glob('ema'+str(request['step'])+'_*.json'))
        assert len(archives)==1, 'Require one genuine preserved native104 report'
        envelope=json.loads(archives[0].read_text());report=envelope['report']
        assert envelope['remote_path']==request['run_output']+'/sd_agent_metrics.json'
        target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
        for key in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256',
                    'official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):
            assert report[key]==target[key]
        assert report['decoder']['ema_num_updates']==request['step']
        request.update(preserved_report=report,preserved_report_archive=str(archives[0]),
                       preserved_report_archive_sha256=sha(archives[0]))
    if args.profile=='omega10_e80':
        archives=list((BASE/'native_full50K_history/server9/init_diffusion_657_n24_road_k8_omega10_full').glob('ema77120_*.json'))
        assert len(archives)==1, 'Require one verified nativeomega10 epoch80 archive'
        envelope=json.loads(archives[0].read_text());report=envelope['report']
        assert envelope['remote_path']==request['run_output']+'/sd_agent_metrics.json'
        target=json.loads((ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json').read_text())
        for key in ('num_samples','num_gt_samples','full_membership','map_source','eval_set_sha256',
                    'official_source_sha256','reference_source_manifest_sha256','feature_counts_gt'):
            assert report[key]==target[key]
        assert report['decoder']['ema_num_updates']==77120
        request.update(preserved_report=report,preserved_report_archive=str(archives[0]),
                       preserved_report_archive_sha256=sha(archives[0]))
    DEST.mkdir(parents=True, exist_ok=True)
    final = DEST / 'provenance.json'
    if final.exists():
        record = json.loads(final.read_text())
        assert sha(DEST / 'model.ckpt') == record['sha256']
        print(json.dumps(record))
        return
    remote = r'''
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys
q=json.load(sys.stdin);out=Path(q['run_output']);step=q['step']
report=q.get('preserved_report') or json.loads((out/'sd_agent_metrics.json').read_text())
assert report['num_samples']==report['num_gt_samples']==50000 and report['full_membership']
assert report['decoder']['ema_num_updates']==step,'Preserve exact intermediate native report before later validation'
paths=list(out.glob(f'bc/*/checkpoints/*{step}*.ckpt'))
assert len(paths)==1,paths
source=paths[0];before=source.stat()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
snapshot=Path(q['repo'])/'.codex_runs'/(q['profile']+'_snapshot_'+stamp)
snapshot.mkdir(exist_ok=False);target=snapshot/'model.ckpt';shutil.copyfile(source,target)
after=source.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):d.update(block)
 return d.hexdigest()
digest=sha(target);assert sha(source)==digest
probe="import torch,json,sys; torch.set_num_threads(1); c=torch.load(sys.argv[1],map_location='cpu',weights_only=False); e=c['state_dict']['encoder.init_decoder._extra_state']; print(json.dumps(dict(epoch=int(c['epoch']),global_step=int(c['global_step']),ema_updates=int(e['ema']['num_updates']))))"
p=subprocess.run([q['python'],'-c',probe,str(target)],env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),text=True,capture_output=True,check=True,timeout=90)
metadata=json.loads(p.stdout);assert metadata==dict(epoch=q['epoch'],global_step=step,ema_updates=step)
config=snapshot/'original_config.yaml';shutil.copyfile(out/'.hydra/config.yaml',config)
native=snapshot/'native_report.json';native.write_text(json.dumps(report,indent=2)+'\n')
record=dict(status='GENUINE_INTERMEDIATE_CHECKPOINT_READY',path=str(target),original_remote_path=str(source),sha256=digest,bytes=target.stat().st_size,source_config=str(config),source_config_sha256=sha(config),native_report=str(native),native_report_sha256=sha(native),remote_snapshot=str(snapshot),origin=f"Actual{q['epoch']+1}epoch checkpoint saved during128epoch training; not final128epoch weights",preserved_report_archive=q.get('preserved_report_archive'),preserved_report_archive_sha256=q.get('preserved_report_archive_sha256'),**metadata)
(snapshot/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
'''
    result = subprocess.run(SSH + ['python3 -c ' + shlex.quote(remote)],
                            input=json.dumps(request), text=True, capture_output=True, check=True, timeout=120)
    record = json.loads(result.stdout)
    for source, name, digest in [(record['path'], 'model.ckpt', record['sha256']),
            (record['source_config'], 'original_config.yaml', record['source_config_sha256']),
            (record['native_report'], 'native_report.json', record['native_report_sha256'])]:
        target = DEST / name
        if target.exists():
            assert sha(target) == digest
            continue
        partial = target.with_suffix(target.suffix + '.part')
        with partial.open('wb') as handle:
            subprocess.run(SSH + ['cat ' + shlex.quote(source)], stdout=handle,
                           stderr=subprocess.PIPE, check=True, timeout=180)
        assert sha(partial) == digest
        partial.replace(target)
    final.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
