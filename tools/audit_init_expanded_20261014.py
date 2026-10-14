"""Recompute metrics for completed expanded-search observations independently."""
from __future__ import annotations
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_init_overnight_20261014 import divergence, METRICS
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014'


def audit():
    entries=[('training',r) for r in read(ART/'observations.json')]
    for path in sorted((ROOT/'logs/init_diffusion_expanded_20261014/samplers').glob('*/verified.json')):
        entries += [(path.parent.name,r) for r in read(path.parent/'summary.json')]
    checks=0;seen=set();truth={};cells=[]
    for mode,row in entries:
        key=(mode,row['variant'],row['seed'],row['budget'],row['eval_seed'],row['num_samples'])
        assert key not in seen;seen.add(key);checks+=1
        path=Path(row['metrics_path']);report=read(path);hp=path.parent/'feature_histograms.json';hist=read(hp)
        assert report['num_samples']==report['num_gt_samples']==row['num_samples'];checks+=1
        assert report['reference_mode']=='matched' and report['map_source']=='reference';checks+=1
        assert report['unverified_generation_timestep_count']==0;checks+=1
        assert report['decoder']['sampling_steps'] in (20,24);checks+=1
        assert report['decoder']['ema_num_updates']==row['budget'] and report['decoder']['use_ema'];checks+=1
        assert row['map_frozen'] and row['generator_num_parameters']==row['generator_trainable_parameters'];checks+=1
        assert report['runtime']['trainer_precision']=='32-true';checks+=1
        assert report['runtime']['float32_matmul_precision']=='medium';checks+=1
        if row['num_samples']==50000:assert report['full_membership'];checks+=1
        metrics={}
        for i,name in enumerate(METRICS):
            actual=divergence(hist[name],10 if i<2 else 100)['value']
            assert math.isclose(actual,row[name],rel_tol=1e-9,abs_tol=1e-10);checks+=1
            assert math.isclose(actual,report['agent_metrics'][name],rel_tol=1e-9,abs_tol=1e-10);checks+=1
            cohort=(row['eval_manifest_sha256'],name)
            if cohort in truth:assert truth[cohort]==hist[name]['matched_gt'];checks+=1
            else:truth[cohort]=hist[name]['matched_gt']
            metrics[name]=actual
        col=100*report['num_colliding_vehicles']/report['num_generated_vehicles']
        assert math.isclose(col,row['collision_rate'],rel_tol=1e-9,abs_tol=1e-10);checks+=1
        cells.append(dict(key=key,metrics=metrics,collision_rate=col,report_sha256=sha(path),histogram_sha256=sha(hp)))
    result=dict(status='PASS' if cells else 'NO_COMPLETED_CELLS',audited_utc=now(),
        unique_actual_cells=len(cells),checks=checks,cells=cells,
        auditor_sha256=sha(__file__),observations_sha256=sha(ART/'observations.json'),
        scope='Independent histogram/count checks. Training/model correctness remains separately verified by native loader and checkpoints.')
    save(ART/'native_metric_audit.json',result)
    print({k:result[k] for k in ('status','unique_actual_cells','checks')})


if __name__=='__main__':audit()
