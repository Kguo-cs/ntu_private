"""Prepare immutable, small sim snapshots for authorized remote experiments."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,subprocess,tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/reports/init_diffusion_continuous_20261014'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['shared_omega','shared_omega05','shared_omega025_replica','empirical_speed_replica','time_scaled_heading','deep_shared_omega','deep_shared_omega10'],required=True)
    args=parser.parse_args()
    proof=json.loads((BASE/'shared_heading_full_config_preflight_20261015.json').read_text())
    assert proof['status']=='CPU_NATIVE_FULL_CONFIGS_PASS_NOT_DEPLOYED'
    candidates=['init_diffusion_657_n24_road_k8_omega025_full','init_diffusion_657_n24_road_k8_omega10_full',
                'init_diffusion_657_n24_road_k8_lines_midpoint_full',
                'init_diffusion_657_n24_road_k8_trainmap_lanes_midpoint_full']
    if args.kind=='shared_omega05':
        proof=json.loads((BASE/'omega05_cpu_preflight_20261015.json').read_text())
        assert proof['status']=='OMEGA05_NATIVE_FULL_CONFIG_CPU_PASS_NOT_DEPLOYED'
        for name,digest in proof['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        candidates=['init_diffusion_657_n24_road_k8_omega05_full']
    if args.kind=='shared_omega025_replica':
        proof=json.loads((BASE/'omega025_seed_replication_cpu_proof_20261015.json').read_text())
        assert proof['status']=='OMEGA025_INDEPENDENT_TRAIN_SEED_CPU_PREFLIGHT_PASS'
        assert proof['seed']==6817 and proof['frozen_map_exact'] and proof['initial_G1_differs']
        for name,digest in proof['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        candidates=['init_diffusion_657_n24_road_k8_omega025_seed6817_full']
    if args.kind=='empirical_speed_replica':
        proof=json.loads((BASE/'empirical_speed_seed6817_cpu_proof_20261015.json').read_text())
        assert proof['status']=='EMPIRICAL_SPEED_SEED6817_SOURCE_ONLY_PAIR_CPU_PASS_NOT_DEPLOYED'
        assert proof['seed']==6817 and proof['initial_parameters_buffers_CPU_RNG_exact']
        for name,digest in proof['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        candidates=['init_diffusion_657_n24_road_k8_omega025_empirical_speed_seed6817_full']
    if args.kind=='time_scaled_heading':
        proof=json.loads((BASE/'time_scaled_heading_cpu_proof_20261015.json').read_text())
        assert proof['status']=='PASS' and proof['native_full_config_factory']
        for name,digest in proof['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        candidates=['init_diffusion_657_n24_road_k8_omega025_timescaled_full']
    if args.kind in ('deep_shared_omega','deep_shared_omega10'):
        proof=json.loads((BASE/(args.kind+'_cpu_proof_20261015.json')).read_text())
        assert proof['status']=='NATIVE_DEEP_SHARED_OMEGA_CPU_PROOF_PASS'
        for name,digest in proof['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        gpu=json.loads((BASE/(args.kind+'_gpu512_verified_20261015.json')).read_text())
        assert gpu['status']=='REAL_NATIVE_DEEP_GPU_FULL512_TWO_STEP_PASS'
        candidates=['init_diffusion_657_n24_road_k8_match2_d5_omega'+('10' if args.kind.endswith('10') else '025')+'_full']
        proof=dict(CPU=proof,GPU512=gpu)
    paths=[]
    for directory in ['src/smart','src/utils','src/my_data_process','configs']:
        names=subprocess.check_output(['rg','--files',directory],cwd=ROOT,text=True).splitlines()
        paths.extend(Path(n) for n in names if Path(n).suffix in ('.py','.yaml','.yml','.json'))
    paths.extend(p.relative_to(ROOT) for p in (ROOT/'src').glob('*.py'))
    resources=['src/run.py','src/__init__.py','src/smart/tokens/map_traj_token5.pkl',
        'src/smart/tokens/agent_vocab_555_s2.pkl','src/smart/metrics/SOURCES.md',
        'tools/shared_heading_velocity_20261015.py','tools/shared_heading_full/__init__.py',
        'tools/shared_heading_full/model.py',
        'docs/experiments/init_diffusion_scratch_loss_search_20261011/train_normalizer.json','.project-root']
    if args.kind=='time_scaled_heading':
        resources += ['tools/time_scaled_heading_full/__init__.py','tools/time_scaled_heading_full/model.py']
    if args.kind in ('deep_shared_omega','deep_shared_omega10'):
        resources += ['tools/run_native_prefix_training_20261015.py',
                      'tools/runtime_cached_torch_loader_sweep.py','tools/torch_pickle_class_cache.py']
    if args.kind=='empirical_speed_replica':
        resources += ['tools/run_native_prefix_training_20261015.py',
                      'tools/runtime_cached_torch_loader_sweep.py','tools/torch_pickle_class_cache.py',
                      'docs/experiments/init_diffusion_train_attribute_source_full_20261015/quantiles.pt']
    paths=sorted(set(paths+[Path(n) for n in resources if (ROOT/n).is_file()]))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=BASE/'remote_packages'/(args.kind+'_'+stamp);folder.mkdir(parents=True,exist_ok=False)
    files={str(n):dict(sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),bytes=(ROOT/n).stat().st_size) for n in paths}
    manifest=dict(prepared_utc=stamp,state='PREPARED_NOT_DEPLOYED',kind=args.kind,files=files,
        candidates=candidates,
        runtime='Ownsim environment; source snapshot inside remote sim, sharedwaymo_data symlink.',
        caveat='FreshG1/fixedAIRLmap, GTego pose/shape/type; SD/VW conditioning differs. No fullquality claim.')
    payload=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    (folder/'SOURCE_MANIFEST.json').write_bytes(payload)
    archive=folder/'source_snapshot.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        contents=[(str(n),(ROOT/n).read_bytes()) for n in paths]
        contents += [('SOURCE_MANIFEST.json',payload),('LOCAL_PREFLIGHT.json',(json.dumps(proof,indent=2)+'\n').encode())]
        for name,data in contents:
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644;item.mtime=0
            tar.addfile(item,io.BytesIO(data))
    record=dict(prepared_utc=stamp,folder=str(folder),archive=str(archive),kind=args.kind,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(payload).hexdigest(),files=len(paths),bytes=archive.stat().st_size,
        remote_started=False)
    (folder/'package.json').write_text(json.dumps(record,indent=2)+'\n')
    (BASE/('latest_'+args.kind+'_package.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
