"""Independent histogram/score QA and live HTML for the evening search."""
from __future__ import annotations
import argparse
import html
import json
import math
import time
from pathlib import Path
import statistics
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scipy.spatial.distance import jensenshannon
from tools.run_init_diffusion_657_iteration import read,save,now,sha
ART=ROOT/'output/reports/init_diffusion_evening_20261013'
BASE_ART=ART
METRICS=('nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate')
FOCUS=('speed_jsd','ang_dev_jsd','length_jsd')
DEADLINE=datetime.fromisoformat('2026-10-13T22:30:00+08:00').timestamp()

def verify_values(values):
    count=0;paths={}
    for r in values:
        p=Path(r['metrics_path']);report=read(p);hf=p.parent/'feature_histograms.json';hist=read(hf)
        paths[str(p)]=sha(p);paths[str(hf)]=sha(hf)
        for index,k in enumerate(METRICS[:6]):
            computed=float(jensenshannon(hist[k]['generated'],hist[k]['matched_gt'])**2*(10 if index<2 else 100))
            assert math.isclose(computed,r[k],abs_tol=1e-10,rel_tol=0),(p,k,computed,r[k]);count+=1
        assert report['num_samples']==r['num_samples'] and report['decoder']['sampling_steps']==20
        assert report['decoder']['ema_num_updates']==r['budget'];count+=3
    return count,paths

def independent_score(a,b):
    seeds=sorted({r['seed'] for r in a})
    mean=lambda rows,k:statistics.fmean(statistics.fmean(r[k] for r in rows if r['seed']==seed) for seed in seeds)
    ratios={k:mean(a,k)/mean(b,k) for k in METRICS}
    return dict(R_focus=statistics.fmean(ratios[k] for k in FOCUS),R_jsd=statistics.fmean(ratios[k] for k in METRICS[:6]),R_attr=statistics.fmean(ratios[k] for k in METRICS[3:6]),ratios=ratios)

def main():
    rows=read(ART/'observations.json');comparisons=read(ART/'comparisons.json')['comparisons']
    checks,files=verify_values(rows)
    for c in comparisons:
        a=[r for r in rows if r['variant']==c['arm'] and r['budget']==c['budget'] and r['seed'] in c['TRAIN_seeds']]
        b=[next(r for r in rows if r['variant']==c['reference'] and (r['seed'],r['budget'],r['eval_seed'])==(q['seed'],q['budget'],q['eval_seed'])) for q in a]
        result=independent_score(a,b)
        for k in ('R_focus','R_jsd','R_attr'):
            assert math.isclose(c[k],result[k],abs_tol=1e-12,rel_tol=0);checks+=1
        for q,base in zip(a,b):
            ah=read(Path(q['metrics_path']).parent/'feature_histograms.json');bh=read(Path(base['metrics_path']).parent/'feature_histograms.json')
            assert all(ah[k]['matched_gt']==bh[k]['matched_gt'] for k in METRICS[:6]);checks+=6
            for k in ('train_order_sha256','eval_manifest_sha256','calibration_sha256','initial_map_sha256','map_checkpoint_sha256','seen_training_examples'):
                assert q[k]==base[k],k;checks+=1
            assert q.get('initial_shared_generator_sha256',q['initial_generator_sha256'])==base.get('initial_shared_generator_sha256',base['initial_generator_sha256']);checks+=1
    expanded={}
    for n in (16384,50000):
        p=ART/f'expanded{n}_result.json'
        if not p.exists():continue
        value=read(p);rr=read(ART/f'expanded{n}_observations.json');count,ff=verify_values(rr);checks+=count;files.update(ff)
        for ref,name in [('fresh657','against657'),('lane_only','against_lane_only')]:
            result=independent_score([r for r in rr if r['variant']==value['arm']],[r for r in rr if r['variant']==ref])
            for k in ('R_focus','R_jsd','R_attr'):assert math.isclose(result[k],value[name][k],abs_tol=1e-12,rel_tol=0);checks+=1
        expanded[str(n)]=value
    latest=read(BASE_ART/'wandb/latest_analysis.json') if (BASE_ART/'wandb/latest_analysis.json').exists() else None
    state=read(ART/'status.json') if (ART/'status.json').exists() else {}
    report=dict(generated_utc=now(),status=state,observations=len(rows),completed_comparisons=comparisons,expanded=expanded,numeric_qa=dict(status='PASS',checks=checks,file_hashes=files),wandb=latest,
        focus_definition='R_focus = mean(speed ratio, angle ratio, length ratio). Means over GEN first, then equally over TRAIN. Lower is better; different budgets/cohorts are not cross-ranked.',
        no_final128epoch_claim=True)
    save(ART/'analysis.json',report)
    esc=lambda s:html.escape(str(s))
    table=[]
    for c in sorted(comparisons,key=lambda v:(v['reference'],v['budget'],v['TRAIN_seeds'],v['R_focus'])):
        rr=c['ratios'];table.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [c['arm'],c['reference'],c['TRAIN_seeds'],c['budget'],f"{c['R_focus']:.4f}",f"{c['R_jsd']:.4f}",f"{rr['speed_jsd']:.4f}",f"{rr['ang_dev_jsd']:.4f}",f"{rr['length_jsd']:.4f}",f"{rr['width_jsd']:.4f}",c['gates_pass']])+'</tr>')
    online=[]
    if latest:
        for r in latest['manifest']:
            if r['id'] not in ('myqhg1rq','8bblh5do','zww9tc6r','4e89o3nz','iege6y14','iu3fa4gx','4bo59qfn'):continue
            m=r['last_validation_metrics'];co=r['conditions'];online.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['name'],r['state'],r['last_validation_epoch'],f"{co['val_batch_size']} × {co['limit_val_batches']}",*[f'{m[k]:.5f}' if m else 'pending' for k in FOCUS]])+'</tr>')
    page=f'''<!doctype html><meta charset="utf-8"><title>InitDiffusion evening search</title><style>body{{font:15px system-ui;max-width:1500px;margin:36px auto;color:#17324d}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #dde4ea;padding:8px;text-align:left}}th{{background:#e9f0f7}}.note{{padding:16px;background:#fff3d5}}pre{{white-space:pre-wrap}}</style><h1>InitDiffusion：speed / ang / length</h1><p>更新 UTC {esc(report['generated_utc'])}；截止北京时间 22:30；当前状态 {esc(state.get('status'))}；独立数值 QA：{checks} checks PASS。</p><div class="note">N20，参数上限810万，固定AIRL map encoder，fresh G1/optimizer/EMA。R_focus 是三项配对相对 JSD 的平均。collision 仅记录。短训练不证明128 epoch效果；跨评价场景数量不能直接排序。EMA窗口与完整训练不同。</div><h2>本地完成的配对结果</h2><table><tr><th>arm</th><th>reference</th><th>TRAIN</th><th>steps</th><th>R_focus</th><th>R_jsd</th><th>speed ratio</th><th>ang ratio</th><th>length ratio</th><th>width ratio</th><th>gates</th></tr>{''.join(table)}</table><h2>最新 W&B 观察</h2><p>不同验证范围、训练种子及继续训练起点需单独考虑；线上R的固定参考不等于本轮配对R。</p><table><tr><th>run</th><th>state</th><th>last val epoch</th><th>configured scenes</th><th>speed</th><th>ang</th><th>length</th></tr>{''.join(online)}</table><h2>扩大评价</h2><pre>{esc(json.dumps(expanded,ensure_ascii=False,indent=2))}</pre>'''
    (ART/'analysis.html').write_text(page)
    print(json.dumps(dict(updated_utc=report['generated_utc'],status=state.get('status'),observations=len(rows),comparisons=len(comparisons),checks=checks)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--active-dir',type=Path);args=p.parse_args()
    if args.active_dir:ART=args.active_dir.resolve()
    previous=None
    while True:
        if (ART/'observations.json').stat().st_mtime_ns>(ART/'comparisons.json').stat().st_mtime_ns:
            time.sleep(.5);continue
        signature=tuple(sha(f) if f.exists() else None for f in [ART/'observations.json',ART/'comparisons.json',BASE_ART/'wandb/latest_analysis.json',ART/'expanded16384_result.json',ART/'expanded50000_result.json'])
        if signature!=previous:
            main();previous=signature
        if not args.watch or time.time()>=DEADLINE:break
        time.sleep(10)
