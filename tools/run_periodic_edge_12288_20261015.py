"""Longer equal-exposure check after the strong-angle / worse-speed K8 result."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_periodic_edge_training_20261015 as engine


def main():
    parent = engine.BASE / 'periodic_edge_long_v1'
    assert engine.read(parent/'status.json')['status'] == 'PERIODIC_6144_PAIRED_CHECK_COMPLETE'
    evidence = engine.read(parent/'comparison.json')
    assert evidence['ratios']['ang_dev_jsd'] < .85
    assert all(r['ratios']['ang_dev_jsd'] < .85 for r in evidence['paired'])
    engine.ART = engine.BASE / 'periodic_edge_12288_v1'
    engine.RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/periodic_edge_12288_v1'
    engine.ARMS = {'control':0,'harmonic8':8}
    engine.ART.mkdir(parents=True,exist_ok=True)
    engine.RUNS.mkdir(parents=True,exist_ok=True)
    assert not (engine.ART/'protocol.json').exists()
    for name in ('recipe.json','variants.json','eval8192.json'):
        engine.save(engine.ART/name,engine.read(parent/name))
    predecessor = engine.read(engine.BASE/'deep_e103_midpoint_driver_pid.json')['pid']
    frozen = engine.source_hashes()
    frozen[str(Path(__file__).relative_to(ROOT))] = engine.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine.save(engine.ART/'protocol.json',dict(registered_utc=engine.now(),predecessor_pid=predecessor,
        source_hashes=frozen,TRAIN=4817,GEN=[2817,3817],scenes=8192,NFE=24,
        start_steps=6144,end_steps=12288,paired_evidence=evidence,
        decision='Explicit longer check: K8 angle improves21.5% in bothGEN, speed worsens32.3%. '
                 'The user targets require every metric; existing full-trained roadK8 has speed/shape '
                 'headroom, so a worse averaged proxy does not alone settle final target potential.',
        continuation='Restore own6144 optimizer/scheduler/EMA/RNG and next unconsumed fullTRAIN scene.',
        limitation='One TRAIN with repeated fixed full-data order. No all7/full128epoch inference '
                   'from short ratios, and no automatic remote replacement.'))
    while engine.alive(predecessor) or engine.occupied():
        engine.save(engine.ART/'status.json',dict(status='WAITING_FOR_OWNED_E103_STUDY_AND_GPU'))
        time.sleep(10)
    def verify():
        for name,digest in frozen.items():
            assert engine.hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    originals=engine.read(parent/'observations.json')
    rows=[]
    for arm in ('control','harmonic8'):
        verify()
        checkpoints={r['checkpoint'] for r in originals if r['variant']==arm and r['budget']==6144}
        assert len(checkpoints)==1
        result,_=engine.stage(arm,12288,resume=Path(next(iter(checkpoints))))
        verify()
        rows.extend(result)
        engine.save(engine.ART/'observations.json',rows)
    comparison=engine.compare(rows,'harmonic8',12288)
    engine.save(engine.ART/'comparison.json',comparison)
    engine.save(engine.ART/'status.json',dict(status='PERIODIC_12288_PAIRED_CHECK_COMPLETE',
        completed_utc=engine.now(),next='Review later angle/speed/shape tradeoffs; '
        'require anotherTRAIN and full50K before a confident full-training recommendation.'))


if __name__=='__main__':
    main()
