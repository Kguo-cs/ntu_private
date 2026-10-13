"""Independent numeric QA and live report for the JSD-first second round."""
from __future__ import annotations
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scipy.spatial.distance import jensenshannon
from tools.run_wandb_cc_scratch_20261013 import read,save,sha,now,METRICS
ART=ROOT/"output/reports/init_diffusion_jsd_round2_20261013"


def independent_score(cells,base):
    seeds=sorted({r['seed'] for r in cells})
    def mean(rows,key):
        return statistics.fmean(statistics.fmean(r[key] for r in rows if r['seed']==seed) for seed in seeds)
    ratios={k:mean(cells,k)/mean(base,k) for k in METRICS}
    return {"R_jsd":statistics.fmean(ratios[k] for k in METRICS[:6]),
        "R_attr":statistics.fmean(ratios[k] for k in METRICS[3:6]),
        "R_geometry":statistics.fmean(ratios[k] for k in METRICS[:3]),"ratios":ratios}


def main():
    active=ART/'speed_isolation' if (ART/'speed_isolation/protocol.json').exists() else ART
    rows=read(active/"observations.json")
    comparisons=read(active/"comparisons.json")['comparisons']
    checks=[];files={}
    scopes=[(rows,comparisons)]
    expanded_path=active/"expanded_result.json"
    expanded=None
    if expanded_path.exists():
        expanded=read(expanded_path)
        scopes.append((read(active/"expanded_observations.json"),[expanded]))
    for values,comps in scopes:
        base={(r['seed'],r['budget'],r['eval_seed']):r for r in values if r['variant']=='fresh657'}
        for r in values:
            path=Path(r['metrics_path']);histfile=path.parent/'feature_histograms.json'
            report=read(path);hist=read(histfile)
            files[str(path)]=sha(path);files[str(histfile)]=sha(histfile)
            for index,key in enumerate(METRICS[:6]):
                value=jensenshannon(hist[key]['generated'],hist[key]['matched_gt'])**2*(10 if index<2 else 100)
                assert math.isclose(value,r[key],abs_tol=1e-10,rel_tol=0)
                checks.append('native_histogram_JSD')
            assert report['decoder']['sampling_steps']==20
            assert report['decoder']['ema_num_updates']==r['budget']
            control=base.get((r['seed'],r['budget'],r['eval_seed']))
            if control is not None:
                other=read(Path(control['metrics_path']).parent/'feature_histograms.json')
                for key in METRICS[:6]:assert hist[key]['matched_gt']==other[key]['matched_gt']
                for key in ('train_order_sha256','eval_manifest_sha256','calibration_sha256',
                    'initial_generator_sha256','initial_map_sha256','map_checkpoint_sha256','seen_training_examples'):
                    assert r[key]==control[key],key
                checks.append('pair_identity')
        for c in comps:
            candidates=[r for r in values if r['variant']==c['arm'] and r['budget']==c['budget'] and r['seed'] in c['TRAIN_seeds']]
            controls=[base[(r['seed'],r['budget'],r['eval_seed'])] for r in candidates]
            independently=independent_score(candidates,controls)
            for key in ('R_jsd','R_attr','R_geometry'):
                assert math.isclose(c[key],independently[key],abs_tol=1e-12,rel_tol=0)
                checks.append('relative_score')
    direct=[]
    seed_diagnostics=[]
    controls={(r['seed'],r['budget'],r['eval_seed']):r for r in rows if r['variant']=='fresh657'}
    for c in comparisons:
        for r in rows:
            if r['variant']==c['arm'] and r['budget']==c['budget'] and r['seed'] in c['TRAIN_seeds']:
                seed_diagnostics.append({'arm':r['variant'],'budget':r['budget'],
                    'TRAIN':r['seed'],'GEN':r['eval_seed'],
                    **independent_score([r],[controls[(r['seed'],r['budget'],r['eval_seed'])]])})
    for arm in ('lane_only','lane_attr_ot','speed_only','speed_pooled'):
        for budget in (2048,4096):
            cells=[r for r in rows if (r['variant'],r['seed'],r['budget'])==(arm,5817,budget)]
            base=[r for r in rows if (r['variant'],r['seed'],r['budget'])==('lane_agentmax05',5817,budget)]
            if len(cells)==len(base)==2:
                direct.append({'arm':arm,'denominator':'lane_agentmax05','TRAIN':5817,'budget':budget,
                    **independent_score(cells,base)})
    probe=ART/'cpu_speed_probe/results.json'
    result={'updated_utc':now(),'status':read(active/'status.json') if (active/'status.json').exists() else {'status':'REGISTERED'},
        'protocol_sha256':sha(ART/'protocol.json'),'active_artifact_directory':str(active),
        'active_protocol_sha256':sha(active/'protocol.json'),
        'adaptive_amendment':read(ART/'adaptive_amendment.json') if (ART/'adaptive_amendment.json').exists() else None,
        'primary':'R_jsd; collision auxiliary','comparisons':comparisons,'direct_ablations':direct,
        'per_GEN_diagnostics':seed_diagnostics,
        'expanded':expanded,'CPU_diagnostic':read(probe) if probe.exists() else None,
        'numeric_QA':{'all_checks_pass':True,'checks':len(checks),
            'counts':{k:checks.count(k) for k in sorted(set(checks))}},'native_artifact_hashes':files,
        'limits':['Imported control sources differ in disabled guidance/default and score-logging code; initialized states and data/statistics/GT histograms match.',
                  'New short training repeats a fixed shuffled full order; full training reshuffles each epoch.',
                  'No128epoch or full50000 final-training improvement established.',
                  'Expanded16384 keeps4096 search scenes; not independent held-out evaluation.',
                  'CPU clamp probe is not pooled into GPU quality measurements.']}
    if result['adaptive_amendment']:
        result['limits'].append('Planned6826-stage was stopped before completion to isolate the speed effect; its partial training output is excluded.')
    save(ART/'analysis.json',result)
    parts=["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>JSD round2</title>",
        "<style>body{max-width:1350px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#20344b}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td,th{border:1px solid #ccd6e3;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf3fa}</style>",
        "<h1>InitDiffusion：JSD 第二轮探索</h1>",
        f"<p>更新UTC {result['updated_utc']}；{html.escape(result['status']['status'])}；N20；G1 8,083,978参数。</p>",
        "<p>R_jsd是六项JSD相对配对657的比值平均，R_attr是length/width/speed三个比值平均；越低越好。每个TRAIN内先平均GEN指标。不同预算/场景/TRAIN的行按各自范围解释，collision不参与排序。</p>"]
    def table(values,title,direct=False):
        parts.append('<h2>'+title+'</h2><table><tr><th>候选 / TRAIN / budget</th><th>R_jsd</th><th>R_attr</th>'+''.join('<th>'+k+'</th>' for k in METRICS)+'<th>检查</th></tr>')
        for r in values:
            seed=r.get('TRAIN_seeds',r.get('TRAIN'))
            parts.append('<tr><td>'+html.escape(f"{r['arm']} / {seed} / {r['budget']}")+'</td>'+''.join(f"<td>{r[k]:.4f}</td>" for k in ('R_jsd','R_attr'))+''.join(f"<td>{r['ratios'][k]:.4f}</td>" for k in METRICS)+f"<td>{r.get('gates_pass','—')}</td></tr>")
        parts.append('</table>')
    table(comparisons,'配对657比较（包含已完成的前轮观测，未完成阶段不计分）')
    table(direct,'新单项实验相对同种子lane_agentmax05（分母不同）',True)
    if expanded:table([expanded],'16384场景扩展验证')
    if result['CPU_diagnostic']:
        parts.append('<h2>CPU小样本speed截断诊断</h2><pre>'+html.escape(json.dumps(result['CPU_diagnostic'],indent=2,ensure_ascii=False))+'</pre>')
    parts.append('<p>独立数值校验：'+str(result['numeric_QA'])+'</p><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in result['limits'])+'</ul><p><a href="analysis.json">完整结果与哈希</a> · <a href="protocol.json">实验注册</a></p></html>')
    (ART/'analysis.html').write_text('\n'.join(parts))
    print(json.dumps({'comparisons':len(comparisons),'checks':len(checks),'status':result['status']['status']}))


if __name__=='__main__':main()
