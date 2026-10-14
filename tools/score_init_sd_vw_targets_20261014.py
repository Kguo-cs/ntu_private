"""Report per-metric SD/VW target gaps without averaging away failed metrics."""
from __future__ import annotations
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now
ART=ROOT/'output/reports/init_diffusion_expanded_20261014'
TARGET_FILE=ROOT/'configs/metrics/sd_vw_lane_conditioned_targets.json'
TARGETS=read(TARGET_FILE)['targets']


def score(metrics):
    if not all(isinstance(metrics.get(k),(float,int)) and math.isfinite(metrics[k]) and metrics[k]>=0 for k in TARGETS):
        raise ValueError('All seven finite nonnegative metrics are required')
    ratios={k:metrics[k]/limit for k,limit in TARGETS.items()}
    failed=[k for k,r in ratios.items() if r>=1.0]
    return dict(metrics={k:metrics[k] for k in TARGETS},target_ratios=ratios,
        passed_metrics=7-len(failed),failed_metrics=failed,all_seven_below=not failed,
        worst_target_ratio=max(ratios.values()),
        required_reduction_fraction={k:max(0.,1.-TARGETS[k]/metrics[k]) if metrics[k]>0 else 0. for k in TARGETS})


def report_scope(report):
    target=read(TARGET_FILE)
    keys=('eval_set_sha256','official_source_sha256','reference_source_manifest_sha256')
    return dict(full50000=report.get('num_samples')==report.get('num_gt_samples')==50000 and report.get('full_membership') is True,
        same_cohort=report.get('eval_set_sha256')==target['eval_set_sha256'],
        same_metric_sources=report.get('official_source_sha256')==target['official_source_sha256'],
        same_reference_source=report.get('reference_source_manifest_sha256')==target['reference_source_manifest_sha256'],
        reference_maps=report.get('map_source')=='reference',
        matching_gt_feature_counts=report.get('feature_counts_gt')==target['feature_counts_gt'],
        generation_verified=report.get('unverified_generation_timestep_count')==0,
        numerical_comparison_verifiable=all(report.get(k)==target[k] for k in keys)
            and report.get('num_samples')==report.get('num_gt_samples')==50000
            and report.get('full_membership') is True and report.get('map_source')=='reference'
            and report.get('unverified_generation_timestep_count')==0
            and report.get('feature_counts_gt')==target['feature_counts_gt'])


def refresh():
    rows=read(ART/'observations.json');groups={};cells=[]
    for row in rows:
        report=read(row['metrics_path']);scope=report_scope(report);s=score(row)
        cell=dict(variant=row['variant'],TRAIN=row['seed'],GEN=row['eval_seed'],
            updates=row['budget'],scenes=row['num_samples'],checkpoint=row['checkpoint'],
            report=row['metrics_path'],report_sha256=sha(row['metrics_path']),
            scope=scope,full128epoch_verified=False,**s)
        cell['verified_full50k_numerical_pass']=s['all_seven_below'] and scope['numerical_comparison_verifiable']
        cells.append(cell)
        key=(row['variant'],row['seed'],row['budget'],row['num_samples'],row['checkpoint'])
        groups.setdefault(key,[]).append(cell)
    candidates=[]
    for key,values in groups.items():
        mean=score({k:statistics.fmean(r['metrics'][k] for r in values) for k in TARGETS})
        candidates.append(dict(variant=key[0],TRAIN=key[1],updates=key[2],scenes=key[3],
            checkpoint=key[4],GEN=[r['GEN'] for r in values],mean_metrics=mean,
            all_prespecified_GEN_pass={r['GEN'] for r in values}=={2817,3817}
                and all(r['verified_full50k_numerical_pass'] for r in values),
            warning='Mean of per-GEN metrics is descriptive, not pooled-histogram JSD or an all-GEN success claim.'))
    remote=[]
    path=ART/'wandb/latest_analysis.json'
    if path.exists():
        for run in read(path)['manifest']:
            values=run.get('last_validation_metrics') or {}
            if not all(isinstance(values.get(k),(int,float)) for k in TARGETS):continue
            remote.append(dict(run=run['id'],name=run['name'],epoch=run.get('last_validation_epoch'),
                validation_scope=run['conditions'].get('validation_scope'),
                full_membership_independently_verified=False,**score(values)))
    local=[]
    for name,path in [('657','src/logs/init_diffusion_lane_conditioned_loss_search_best/2026-10-11_13-33-53/sd_agent_metrics.json')]:
        p=ROOT/path
        if p.exists():
            r=read(p);local.append(dict(name=name,source=path,sha256=sha(p),scope=report_scope(r),**score(r['agent_metrics'])))
    result=dict(generated_utc=now(),targets_sha256=sha(TARGET_FILE),scorer_sha256=sha(__file__),
        observations_sha256=sha(ART/'observations.json'),targets=TARGETS,cells=cells,candidates=candidates,
        remote_observations=remote,verified_local_references=local,
        interpretation=read(TARGET_FILE)['criterion'],conditioning_caveat=read(TARGET_FILE)['comparison_caveat'])
    save(ART/'all_seven_target_progress.json',result)
    return result


if __name__=='__main__':
    result=refresh()
    print({'cells':len(result['cells']),'full50k_all_GEN_successes':sum(r['all_prespecified_GEN_pass'] for r in result['candidates']),
           'remote_latest':[{k:r[k] for k in ('run','epoch','passed_metrics','failed_metrics')} for r in result['remote_observations']]})
