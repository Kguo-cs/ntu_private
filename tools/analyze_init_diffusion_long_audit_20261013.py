"""Independent native-histogram QA, cohort audit and long-run evidence report."""
import argparse
import html
import json
import math
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scipy.spatial.distance import jensenshannon
from tools.run_init_diffusion_657_iteration import read, save, now, sha
from tools.analyze_init_diffusion_evening_20261013 import verify_values, independent_score
BASE=ROOT/'output/reports/init_diffusion_evening_20261013'
ART=BASE/'long_audit'
FOCUS=('speed_jsd','ang_dev_jsd','length_jsd')
METRICS=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')
END=datetime.fromisoformat('2026-10-13T22:30:00+08:00').timestamp()


def main():
    rows=read(ART/'observations.json')
    comparisons=read(ART/'comparisons.json')['comparisons'] if (ART/'comparisons.json').exists() else []
    checks, files=verify_values(rows)
    for c in comparisons:
        a=[r for r in rows if r['variant']==c['arm'] and r['budget']==c['budget'] and r['num_samples']==4096 and not r['evaluation_only']]
        b=[r for r in rows if r['variant']=='fresh657' and r['budget']==c['budget'] and r['num_samples']==4096 and not r['evaluation_only']]
        independent=independent_score(a,b)
        for k in ('R_focus','R_jsd','R_attr'):
            assert math.isclose(c[k], independent[k], abs_tol=1e-12);checks+=1
        for r,control in zip(sorted(a,key=lambda v:v['eval_seed']),sorted(b,key=lambda v:v['eval_seed'])):
            for k in ('eval_manifest_sha256','seen_training_examples','train_order_sha256','calibration_sha256','initial_map_sha256'):
                assert r[k]==control[k];checks+=1
            assert r['initial_shared_generator_sha256']==control['initial_shared_generator_sha256'];checks+=1
    old=read(BASE/'recovery/observations.json');scope=[]
    for budget in (4096,16384):
        for arm in sorted({r['variant'] for r in rows if r['budget']==budget}):
            for n in (128,256,4096,16384):
                rr=[r for r in rows if r['variant']==arm and r['budget']==budget and r['num_samples']==n and r['evaluation_only']]
                if not rr:continue
                means={k:statistics.fmean(r[k] for r in rr) for k in METRICS}
                hist=read(Path(rr[0]['metrics_path']).parent/'feature_histograms.json')
                reference_shift={k:float(jensenshannon(hist[k]['matched_gt'],hist[k]['full_gt'])**2*(10 if i<2 else 100)) for i,k in enumerate(METRICS[:6])}
                scope.append(dict(arm=arm,budget=budget,scenes=n,cohort='random16384' if n==16384 else 'sorted prefix',GEN=[r['eval_seed'] for r in rr],metrics=means,
                                  metric_seed_sd={k:statistics.stdev(r[k] for r in rr) if len(rr)>1 else None for k in METRICS},
                                  GT_vs_full_JSD=reference_shift,
                                  samples_gt=rr[0]['num_gt_vehicles'],samples_generated_mean=statistics.fmean(r['num_generated_vehicles'] for r in rr)))
            rr=[r for r in (old if budget==4096 else rows) if r['variant']==arm and r['budget']==budget and r['seed']==4817 and r['num_samples']==4096 and (budget==4096 or not r['evaluation_only'])]
            if len(rr)==2:scope.append(dict(arm=arm,budget=budget,scenes=4096,cohort='random4096',GEN=[r['eval_seed'] for r in rr],metrics={k:statistics.fmean(r[k] for r in rr) for k in METRICS}))
    expanded=[]
    end=[r for r in rows if r['budget']==16384 and r['num_samples']==16384]
    for arm in ('freshK8','lane_only'):
        a=[r for r in end if r['variant']==arm];b=[r for r in end if r['variant']=='fresh657']
        if len(a)==len(b)==2:expanded.append(dict(arm=arm,scenes=16384,**independent_score(a,b)))
    wb=read(BASE/'wandb/latest_analysis.json')
    records={r['id']:r for r in wb['manifest']}
    online=[]
    for rid in ('iu3fa4gx','myqhg1rq','8bblh5do','zww9tc6r'):
        r=records[rid];p=read(r['history_file'])
        val=[v for v in p['rows'] if v.get('epoch') is not None and all(isinstance(v.get(k),(int,float)) for k in FOCUS)]
        online.append(dict(id=rid,name=r['name'],conditions=r['conditions'],history_sha256=sha(r['history_file']),
                           points=[{k:v[k] for k in ('epoch',)+FOCUS+('lat_dev_jsd',)} for v in val]))
    state=read(ART/'status.json')
    report=dict(generated_utc=now(),status=state,observations=len(rows),comparisons=comparisons,scope=scope,expanded=expanded,online=online,
                QA=dict(status='PASS',checks=checks,file_hashes=files),
                full_training_recommendation='Withdraw prior lane recommendation until matched longer/full evidence. A quick ratio below1 is not a128epoch claim.',
                caveats=['One TRAIN4817, two GEN; repeated fixed training order differs from reshuffled full training.',
                         'Same-arm optimizer/EMA/scheduler/RNG restoration has small backend numerical drift.',
                         'W&B numeric1 is ambiguous: int1 batch vs float1.0 full validation. Earlier128/256-scene inference is withdrawn; actual report membership is required. TRAIN seeds/backend also differ.'])
    save(ART/'analysis.json',report)
    esc=lambda x:html.escape(str(x))
    def table(headers,values):
        return '<table><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in values)+'</table>'
    ctable=table(['arm','step','R_focus','R_jsd','speed ratio','ang ratio','length ratio','gates'],
                 [[c['arm'],c['budget'],f"{c['R_focus']:.4f}",f"{c['R_jsd']:.4f}",*[f"{c['ratios'][k]:.4f}" for k in FOCUS],c['gates_pass']] for c in comparisons])
    stable=table(['arm','step','cohort','scenes','speed','ang','length'],[[s['arm'],s['budget'],s['cohort'],s['scenes'],*[f"{s['metrics'][k]:.5f}" for k in FOCUS]] for s in scope])
    wtable=table(['run','validation scope','epoch','speed','ang','length'],[[v['name'],v['conditions'].get('validation_scope',{}).get('scenes') or 'numeric1: unverified',int(p['epoch']),*[f'{p[k]:.5f}' for k in FOCUS]] for v in online for p in v['points']])
    page='''<!doctype html><meta charset="utf-8"><title>Long-run audit</title><style>body{font:15px system-ui;max-width:1250px;margin:32px auto;color:#15304a}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #dce3ea;padding:7px}th{background:#eaf1f8}.warning{background:#fff1d2;padding:16px}img{max-width:100%}</style>'''
    page+=f'<h1>InitDiffusion 长期结果复核</h1><p>UTC {esc(now())}；{esc(state.get("status"))}；数值QA {checks} checks PASS。</p><p class="warning">撤回 lane 配置作为完整训练首选的建议。固定地图、N20、约808万参数；三组同预算对照，不用 collision 排序。当前不是128epoch结论。</p><h2>更长训练的配对结果</h2>'+ctable+'<h2>同一权重的评价范围检查</h2>'+stable+'<h2>W&amp;B 实际曲线</h2><p>不同验证范围不能直接排行。de相对dbf没有新增训练算法，实际实验配置不同。</p>'+wtable
    page+='<h2>限制</h2><ul>'+''.join('<li>'+esc(v)+'</li>' for v in report['caveats'])+'</ul>'
    (ART/'analysis.html').write_text(page)
    print(json.dumps(dict(updated_utc=now(),observations=len(rows),comparisons=len(comparisons),checks=checks,status=state.get('status'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--active-dir',type=Path);args=p.parse_args()
    if args.active_dir:ART=args.active_dir.resolve()
    previous=None
    while True:
        fs=[ART/'observations.json',ART/'comparisons.json',BASE/'wandb/latest_analysis.json']
        signature=tuple(sha(f) if f.exists() else None for f in fs)
        if signature!=previous:
            main();previous=signature
        if not args.watch or time.time()>=END:break
        time.sleep(10)
