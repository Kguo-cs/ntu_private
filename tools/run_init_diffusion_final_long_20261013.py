"""Continue matched map audit using an existing identical-config K8 lineage."""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import run_init_diffusion_long_audit_recovery_20261013 as recovery
from tools import run_init_diffusion_long_audit_20261013 as core
read,save,sha,now=core.read,core.save,core.sha,core.now
OLD=recovery.ART
ART=ROOT/'output/reports/init_diffusion_evening_20261013/final_long'
RUNS=ROOT/'logs/init_diffusion_evening_20261013/final_long'
original_sources=recovery.sources


def sources():
    return {**original_sources(),str(Path(__file__).relative_to(ROOT)):sha(__file__)}


def prepare():
    import yaml
    assert not (ART/'protocol.json').exists()
    prior=read(OLD/'protocol.json');rows=read(OLD/'observations.json')
    for n in ('recipe.json','variants.json','sorted128.json','sorted256.json','sorted4096.json'):
        save(ART/n,read(OLD/n))
    baseline=next(r['checkpoint'] for r in rows if r['variant']=='fresh657' and r['budget']==12288)
    lane=ROOT/'logs/init_diffusion_evening_20261013/recovery/train4817_lane_only_6826/lane_only/last.ckpt'
    k8=ROOT/'logs/init_diffusion_657_edges_20261013/long4817/a2a_k8/step_6826.ckpt'
    reference=ROOT/'logs/wandb_cc_jsd_priority_20261013/train4817_freshK8/freshK8'
    old=read(k8.parent/'initialization.json');other=read(reference/'initialization.json')
    for k in ('seed','batch_size','train_order_sha256','calibration_sha256','map_checkpoint_sha256','initial_map_sha256','initial_generator_sha256','gradient_clip_val'):
        assert old[k]==other[k],k
    assert yaml.safe_load((k8.parent/'config.yaml').read_text())['model']==yaml.safe_load((reference/'config.yaml').read_text())['model']
    proof=read(k8.parent.parent/'verified_completion.json');assert proof['status']=='VERIFIED_COMPLETE'
    assert proof['summary_sha256']==sha(k8.parent.parent/'summary.json')
    meta=recovery.valid_checkpoint(k8);assert meta['step']==6826 and meta['seed']==4817
    stages=[]
    for arm,path,start in [('fresh657',Path(baseline),12288),('freshK8',k8,6826),('lane_only',lane,6826)]:
        for target in (8192,12288,16384):
            if target<=start:continue
            stages.append(dict(arm=arm,seed=4817,steps=target,initial_resume={arm:str(path)},initial_resume_step=start,
                               scenes=4096,manifest=str(ROOT/'tmp/control657_paired_20261012/eval4096_files.json'),
                               checkpoints=[],label=f'long_{arm}_{target}'))
    plan=copy.deepcopy(prior)
    plan.update(registered_utc=now(),source_hashes=sources(),stages=stages,
                previous_protocol_sha256=sha(OLD/'protocol.json'),
                local_input_hashes={n:sha(ART/n) for n in ('recipe.json','variants.json','sorted128.json','sorted256.json','sorted4096.json')},
                prefix_reuse='Existing completed fresh a2a_k8 run, identical model configuration, initial seeded generator/map hash, TRAIN order/calibration/optimizer recipe. Relabeled freshK8; no cross-arm warm start. Backend drift, not bitwise equivalence.')
    plan['imported_sources'][str((OLD/'observations.json').relative_to(ROOT))]=sha(OLD/'observations.json')
    for p in [k8,k8.parent/'config.yaml',k8.parent/'initialization.json',k8.parent.parent/'verified_completion.json',Path(baseline),Path(baseline).parent/'config.yaml',Path(baseline).parent/'initialization.json']:
        plan['imported_sources'][str(p.relative_to(ROOT))]=sha(p)
    save(ART/'prefix_reuse_verification.json',dict(status='PASS',configuration_identical=True,initial_hashes_identical=True,checkpoint_sha256=sha(k8),checkpoint_metadata=meta,trace_first_loss_difference=0.00032806396484375,bitwise_equivalence=False))
    save(ART/'observations.json',rows);save(ART/'protocol.json',plan);core.comparisons(rows);core.status('REGISTERED')


recovery.ART,recovery.RUNS=ART,RUNS
core.ART,core.RUNS=ART,RUNS
core.sources=sources
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    recovery.execute() if args.execute else prepare()
