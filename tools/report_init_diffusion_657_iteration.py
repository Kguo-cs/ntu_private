"""Render verified optimization stages as a local HTML report and JSON table."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_init_diffusion_657_iteration import ART,RUNS,METRICS,read,save,score


def build():
    controls=[]
    for label in ('train4817','train5817','screen4817','full50000_4817'):
        controls+=read(ROOT/f'logs/control657_paired_20261012/{label}/summary.json')
    extra_control=RUNS/'full_control5817'/'summary.json'
    if (extra_control.parent/'verified_completion.json').is_file():controls+=read(extra_control)
    rows=[];stages=[];seen_observations=set();aliases=[]
    for stage in sorted(RUNS.glob('*')):
        if not (stage/'verified_completion.json').is_file():continue
        completion=read(stage/'verified_completion.json')
        stages.append({'stage':stage.name,**completion})
        current=read(stage/'summary.json')
        unique=[r for r in current if r['metrics_path'] not in seen_observations]
        duplicate=len(current)-len(unique)
        if duplicate:aliases.append({'stage':stage.name,'shared_observations':duplicate,'independent_replicates':False})
        seen_observations.update(r['metrics_path'] for r in unique)
        current=unique
        for variant in sorted({r['variant'] for r in current}):
            for budget in sorted({r['budget'] for r in current if r['variant']==variant}):
                selected=[r for r in current if r['variant']==variant and r['budget']==budget]
                try:s=score(selected,controls)
                except RuntimeError as e:s={'R':None,'reason':str(e)}
                rows.append({'stage':stage.name,'variant':variant,'budget':budget,'equivalent_epochs':selected[0]['seen_training_examples']/493542,
                    'num_samples':selected[0]['num_samples'],'seed':selected[0]['seed'],'GEN_N':len(selected),**s})
    provisional=[]
    for stage in sorted(RUNS.glob('*')):
        if (stage/'verified_completion.json').is_file() or not (stage/'summary.json').is_file():continue
        current=read(stage/'summary.json')
        for variant in sorted({r['variant'] for r in current}):
            for budget in sorted({r['budget'] for r in current if r['variant']==variant}):
                selected=[r for r in current if r['variant']==variant and r['budget']==budget]
                if {r['eval_seed'] for r in selected}!={2817,3817}:continue
                try: paired=score(selected,controls)
                except RuntimeError:continue
                provisional.append({'stage':stage.name,'variant':variant,'budget':budget,
                    'stage_execution_verified':False,**paired})
    latest=read(ART/'status.json') if (ART/'status.json').exists() else {'status':'NOT_STARTED'}
    followup=read(ART/'round2_status.json') if (ART/'round2_status.json').is_file() else None
    if followup and followup['status'].startswith('COMPLETE'):latest=followup
    sampler=read(ART/'sampler_status.json') if (ART/'sampler_status.json').is_file() else None
    decisions={name:read(ART/name) for name in ('medium.json','round2_medium.json','round2_long.json','round2_full.json','sampler_comparison.json','sampler_full_comparison.json') if (ART/name).is_file()}
    output={'sampler_status':sampler,'decisions':decisions,'aliases_not_independent_replicates':aliases,'round2_status':followup,'generated_utc':datetime.now(timezone.utc).isoformat(),'status':latest,'verified_stages':stages,'comparisons':rows,'provisional_completed_budget_pairs':provisional}
    save(ART/'progress_report.json',output)
    headers=['阶段','候选','等价 epoch','场景','TRAIN','GEN数','R ↓','R_attr ↓','门槛','Near比','Lat比','Ang比','Length比','Width比','Speed比','Collision比']
    body=[]
    for r in rows:
        ratios=r.get('ratios',{})
        fmt=lambda n:'—' if n is None else f'{n:.4f}'
        cells=[r['stage'],r['variant'],f"{r['equivalent_epochs']:.3f}",str(r['num_samples']),str(r['seed']),str(r['GEN_N']),fmt(r.get('R')),fmt(r.get('R_attr')),'PASS' if r.get('gates_pass') else 'FAIL']+[fmt(ratios.get(k)) for k in METRICS]
        body.append('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>')
    recommendations=read(ART/'recommended_full_configs_20261012.json') if (ART/'recommended_full_configs_20261012.json').is_file() else None
    recommendation_html=''
    if recommendations:
        recommendation_html='<p><strong>目前建议完整训练的候选（768step筛选，128epoch未验证）</strong>：'+', '.join(html.escape(r['variant'])+' R='+str(round(r['R'],4)) for r in recommendations['rows'])+'</p>'
    decision_html=''.join('<p><strong>'+html.escape(name)+'</strong><br>'+html.escape(str(value))+'</p>' for name,value in decisions.items() if name.endswith('full.json') or name=='sampler_full_comparison.json')
    text=f'''<!DOCTYPE html><html lang="zh"><meta charset="utf-8"><title>657 baseline 优化实验</title>
<style>body{{font-family:system-ui,sans-serif;margin:32px;color:#182432;background:#f8fafc}}h1{{font-size:24px}}table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}th,td{{padding:9px;border:1px solid #d6e0e9;text-align:right}}th{{background:#e8eef6}}td:nth-child(2),td:first-child{{text-align:left}}p{{max-width:1100px;line-height:1.65}}.scroll{{overflow:auto}}</style>
<h1>以 657 为 baseline 的从头训练优化实验</h1><p>更新时间 UTC：{html.escape(output['generated_utc'])}；状态：{html.escape(str(latest.get('status')))}。只显示完整核验的 stage；运行中或失败尝试不计为评价证据。</p>
<p>固定 AIRL map encoder，fresh G1，TRAIN-only calibration，batch 512，EMA，32-true，N20/t_eps=.05。R = 七项候选/对应训练预算的657指标比取平均；GEN先在各TRAIN内取平均，再对TRAIN等权。单seed表中R是该TRAIN的结果。不能用不同场景数、预算或GEN的分母。</p>
<p>门槛：Length/Width/Speed各不超过657的1.15倍；onroad下降不超过2个百分点；车辆数/GT在[0.9,1.1]。改善需要R&lt;1且所有TRAIN通过门槛。短预算排名不是128epoch收敛结论；学习曲线反转必须保留。</p>
<p>原始657的128epoch日志只评价256个场景，不能作为这里4096或50k评价的分母。快速实验每epoch重复同一份seed打乱的TRAIN顺序；原生完整训练的逐epoch shuffle不同；原生normalizer_path=None，而配对快速实验共用8192个TRAIN场景的统计量。这些曲线用于预测，不能证明128epoch最优。</p>
<p>采样研究状态：{html.escape(str(sampler))}。采样只在固定657权重上比较；4096选择冻结后再用50k复查，包含新的N20配对参考。</p>
{recommendation_html}
{decision_html}
<p>已完成预算的学习曲线与原生 shape 直方图（待第二轮完成后更新）：</p>
<img src="paired_learning_curves.svg" style="max-width:100%;background:white" alt="Paired learning curves">
<img src="shape_histograms.svg" style="max-width:100%;background:white" alt="Native shape histograms">
<img src="long_confirmation_curves.svg" style="max-width:100%;background:white" alt="Longer paired confirmation curves">
<div class="scroll"><table><thead><tr>{''.join('<th>'+x+'</th>' for x in headers)}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
<p>恢复执行所共享的观察不重复计数，也不算独立重复实验：{html.escape(str(aliases))}</p>
<p>运行中 stage 已产生的成对预算结果（暂未核验整个stage退出，不能当作最终入选证据）：{html.escape(str([(r["variant"],r["budget"],round(r["R"],4)) for r in provisional]))}</p>
<p>计划、变体、源码快照、配对评分及checkpoint位置保存在同目录JSON；完整训练轨迹、config和原生sd_agent_metrics.json位于 logs/init_diffusion_657_iteration_20261012。</p></html>'''
    (ART/'report.html').write_text(text,encoding='utf-8')
    print(f'verified stages={len(stages)}, scored budgets={len(rows)}, status={latest.get("status")}')


if __name__=='__main__':build()
