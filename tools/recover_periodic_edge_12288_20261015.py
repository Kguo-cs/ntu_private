"""Preserve the failed run and finish equal budgets in bounded1024-step jobs."""
from pathlib import Path
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_periodic_edge_training_20261015 as engine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='periodic_edge_12288_recovery_v2')
    parser.add_argument('--runner',default='tools/runtime_tensor_cache_recovery_20261015.py')
    parser.add_argument('--failed-stage',default='periodic_edge_12288_v1')
    args=parser.parse_args()
    failed=engine.BASE/args.failed_stage
    assert engine.read(failed/'status.json')['returncode']==-11
    assert engine.read(engine.BASE/'safe_tensor_cache_parity_20261015.json')['status']=='PASS'
    art=engine.BASE/args.name
    runs=ROOT/'logs/init_diffusion_continuous_20261014'/args.name
    art.mkdir(parents=True,exist_ok=True)
    runs.mkdir(parents=True,exist_ok=True)
    assert not (art/'protocol.json').exists()
    engine.RUNNER=ROOT/args.runner
    isolated='isolated_cache' in engine.RUNNER.name
    if isolated:
        assert engine.read(engine.BASE/'isolated_cache_reader_parity_20261015.json')['status']=='PASS'
    engine.ARMS={'control':0,'harmonic8':8}
    source=engine.BASE/'periodic_edge_long_v1'
    for name in ('recipe.json','variants.json','eval8192.json'):
        engine.save(art/name,engine.read(source/name))
    frozen=engine.source_hashes()
    frozen[str(Path(__file__).relative_to(ROOT))]=engine.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ('tools/runtime_tensor_cache_recovery_20261015.py',
                 'tools/runtime_periodic_edge_sweep_20261015.py'):
        frozen[name]=engine.hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if isolated:
        name='tools/tensor_cache_numpy_worker_20261015.py'
        frozen[name]=engine.hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    engine.save(art/'protocol.json',dict(registered_utc=engine.now(),source_hashes=frozen,
        preserved_failed_run=str(failed),start_steps=6144,end_steps=12288,process_span=1024,
        save_only_guard_interval=512,TRAIN=4817,GEN=[2817,3817],scenes=8192,NFE=24,
        loader=('SpawnCPU reader and NumPy pipe; no TRAIN Torchdeserialization in modelprocess.' if isolated
                else 'Nativeweights_only=True for tensor-onlyTRAIN caches; checkpoint/VAL loader retained.'),
        parity='512 files /5632 tensor fields: values,dtypes,shapes,strides identical.',
        isolated_reader_parity=(engine.read(engine.BASE/'isolated_cache_reader_parity_20261015.json')
                               if isolated else None),
        reason='Python SIGSEGV atweakref handling; rootcause remains unproved. Addfaulthandler, '
               'boundedprocess lifespan and checkpoints to preserve progress if another failure occurs.',
        limitation='I/O/lifecycle recovery only. No objective/model/data-value changes; no automatic retry '
                   'after another native failure, and no fulltraining success claim.'))
    def verify():
        for name,digest in frozen.items():
            assert engine.hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    while engine.occupied():
        engine.save(art/'status.json',dict(status='WAITING_FOR_GPU'))
        time.sleep(10)
    originals=engine.read(source/'observations.json')
    endpoints=[]
    all_rows=[]
    for arm in ('control','harmonic8'):
        checkpoints={r['checkpoint'] for r in originals if r['variant']==arm and r['budget']==6144}
        assert len(checkpoints)==1
        checkpoint=Path(next(iter(checkpoints)))
        for target in range(7168,12289,1024):
            verify()
            label=f'{arm}_to{target}'
            engine.ART=art/'stages'/label
            engine.RUNS=runs/'stages'/label
            engine.ART.mkdir(parents=True,exist_ok=True)
            engine.RUNS.mkdir(parents=True,exist_ok=True)
            for name in ('recipe.json','variants.json','eval8192.json'):
                engine.save(engine.ART/name,engine.read(art/name))
            engine.save(art/'status.json',dict(status='RUNNING_BOUNDED_RECOVERY',arm=arm,
                target_steps=target,stage=str(engine.ART),resume=str(checkpoint)))
            try:
                rows,checkpoint=engine.stage(arm,target,resume=checkpoint)
            except Exception:
                engine.save(art/'status.json',dict(status='FAILED_REQUIRES_STACK_REVIEW',arm=arm,
                    target_steps=target,stage=str(engine.ART),last_verified_checkpoint=str(checkpoint)))
                raise
            verify()
            policy=checkpoint.parents[1].parent/(checkpoint.parents[1].name+'.tensor_cache_policy.json')
            assert engine.read(policy)['safe_loads']>0
            if isolated:
                policy=checkpoint.parents[1].parent/(checkpoint.parents[1].name+'.isolated_cache_io.json')
                assert engine.read(policy)['loads']>0
            all_rows.extend(rows)
            engine.save(art/'observations.json',all_rows)
            engine.save(art/'last_verified_checkpoint.json',dict(arm=arm,step=target,checkpoint=str(checkpoint)))
            if target==12288:
                endpoints.extend(rows)
    comparison=engine.compare(endpoints,'harmonic8',12288)
    engine.save(art/'comparison.json',comparison)
    engine.save(art/'status.json',dict(status='PERIODIC_12288_RECOVERY_COMPLETE',completed_utc=engine.now(),
        next='Review pairedlater curves. Keep nativefailure evidence and require additionalTRAIN/full50K beforepromotion.'))


if __name__=='__main__':
    main()
