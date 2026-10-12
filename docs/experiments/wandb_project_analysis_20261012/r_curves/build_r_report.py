"""Plot all saved W&B runs and audit short/long R rank agreement."""
import sys
sys.path.insert(0, '/tmp/sim_pdf_report_deps')
import collections
import datetime
import hashlib
import html
import json
import math
import statistics
import textwrap
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
SIM=BASE.parents[2]
OUT=SIM/'output/pdf/wandb_short_long_R_20261012.pdf'
M=['nearest_dist_jsd','lat_dev_jsd','ang_dev_jsd','length_jsd','width_jsd','speed_jsd','collision_rate']
ATTR=M[3:6]
inventory=json.loads((BASE/'inventory.json').read_text())
manifest=json.loads((BASE/'run_manifest.json').read_text())
raw=json.loads((BASE/'validation_history.json').read_text())
records={r['id']:r for r in inventory['runs']}; configs={r['id']:r for r in manifest}
H={}
for rid,rows in raw.items():
    buckets=collections.defaultdict(list)
    for a in rows:
        if a.get('epoch') is not None:
            buckets[int(a['epoch'])].append(a)
    H[rid]={e:{k:statistics.mean(a[k] for a in rr) for k in M} for e,rr in buckets.items()}

MATCHED=[r['id'] for r in manifest if r['init_decoder']=='flow' and r['reference_mode']=='matched']
LONG=['iu3fa4gx','vd23g1bq','0yhfg9ue','pd7zg04g','4bedivn1']
NAMES={'iu3fa4gx':'657 / B512','vd23g1bq':'frozen shared / B512','0yhfg9ue':'frozen shared / B256','pz8g135r':'overnight separate / B256','pd7zg04g':'scratch / B512 run3','4bedivn1':'scratch / B512 run0','v62ymyjj':'LR2e-4 / min .01','fs0kq6fc':'LR2e-4 / min .05','c486qs4q':'LR5e-4 / min .01'}
COLORS={'iu3fa4gx':'#276eb5','vd23g1bq':'#ba621a','0yhfg9ue':'#328b56','pz8g135r':'#b24785','pd7zg04g':'#839898','4bedivn1':'#7760a5','v62ymyjj':'#78442c','fs0kq6fc':'#beab22','c486qs4q':'#464647'}
ANCHORS={('flow','matched'):('vd23g1bq',99),('flow','full'):('yj3x4z1n',123),('scenario_dreamer','full'):('xd6ibd0p',63)}
GROUP_LABEL={('flow','matched'):'InitDiffusion / matched reference',('flow','full'):'InitDiffusion / full reference (historical)',('scenario_dreamer','full'):'Scenario Dreamer / full reference (historical)'}

def score(values,baseline,keys=M):
    if any(baseline[k]<=0 for k in keys):
        raise ValueError('Reference metric is zero; ratio undefined')
    return float(np.mean([values[k]/baseline[k] for k in keys]))

def mean_vector(rid,epochs):
    assert all(e in H[rid] for e in epochs),(rid,epochs)
    return {k:statistics.mean(H[rid][e][k] for e in epochs) for k in M}

def ranks(values):
    from scipy.stats import rankdata
    return rankdata(values,method='average')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
figures=[]
def save(fig,name,title,caption,ids=None):
    for suffix in ['png','svg']:
        fig.savefig(ROOT/f'{name}.{suffix}',dpi=160)
    plt.close(fig)
    figures.append({'name':name,'title':title,'caption':caption,'run_ids':ids or []})

for anchor,tag in [('vd23g1bq','frozen512'),('iu3fa4gx','657')]:
    fixed=H[anchor][99]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for col,keys in enumerate([M,ATTR]):
        for rid in MATCHED:
            es=sorted(H[rid]); vals=[score(H[rid][e],fixed,keys) for e in es]
            axes[0,col].plot(es,vals,label=NAMES[rid],color=COLORS[rid],lw=1.8,ls='--' if rid in ['v62ymyjj','fs0kq6fc','c486qs4q'] else '-')
            es=[e for e in es if e in H[anchor]]
            vals=[score(H[rid][e],H[anchor][e],keys) for e in es]
            axes[1,col].plot(es,vals,label=NAMES[rid],color=COLORS[rid],lw=1.8,ls='--' if rid in ['v62ymyjj','fs0kq6fc','c486qs4q'] else '-')
        for row in range(2):
            ax=axes[row,col];ax.axhline(1,color='#666',ls=':',lw=1);ax.grid(alpha=.2)
            ax.set(xlabel='Validation epoch (zero based)',ylabel='R (lower is better)')
            if anchor=='iu3fa4gx' and col==0:ax.set_yscale('log')
        metric='R7' if col==0 else 'R_attr'
        axes[0,col].set_title(f'{metric}: fixed reference = {NAMES[anchor]}, epoch99')
        axes[1,col].set_title(f'{metric}: same-epoch reference = {NAMES[anchor]}')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    fig.suptitle('Observed single-run ratios; 9 matched runs, not a paired multi-seed pooled score',fontsize=13)
    save(fig,f'matched_R_{tag}','Matched runs - '+NAMES[anchor],
         'Top: one fixed epoch99 reference, so numerical scale is stable. Bottom: equal-epoch reference; R=1 is the control at that epoch. Same-epoch curves stop where the control stops. Seeds, hardware and recipes differ.',MATCHED)

fixed=H['vd23g1bq'][99]
point_rows=[];window_rows=[]
for rid in LONG:
    point_rows.append({'id':rid,'name':NAMES[rid],**{f'{key}_{e}':score(H[rid][e],fixed,keys) for key,keys in [('R7',M),('R_attr',ATTR)] for e in [11,55,99]}})
    window_rows.append({'id':rid,'name':NAMES[rid],**{f'{key}_{label}':score(mean_vector(rid,epochs),fixed,keys) for key,keys in [('R7',M),('R_attr',ATTR)] for label,epochs in [('short',[3,7,11]),('long',[91,95,99])]}})

correlations=[]
for anchor in ['vd23g1bq','iu3fa4gx']:
    b=H[anchor][99]
    for ids,label in [(LONG,'five_runs_including_controls'),([r for r in LONG if r!=anchor],'four_runs_excluding_this_control')]:
        for keys,metric in [(M,'R7'),(ATTR,'R_attr')]:
            for early in [3,7,11,15,23,31,39,55,63,75,83,91,95]:
                x=[score(H[r][early],b,keys) for r in ids];y=[score(H[r][99],b,keys) for r in ids]
                correlations.append({'anchor':anchor,'anchor_epoch':99,'cohort':label,'n':len(ids),'metric':metric,'early_epoch':early,'late_epoch':99,'spearman':float(spearmanr(x,y).statistic),'kendall':float(kendalltau(x,y).statistic)})
            x=[score(mean_vector(r,[3,7,11]),b,keys) for r in ids];y=[score(mean_vector(r,[91,95,99]),b,keys) for r in ids]
            correlations.append({'anchor':anchor,'anchor_epoch':99,'cohort':label,'n':len(ids),'metric':metric,'early_epochs':[3,7,11],'late_epochs':[91,95,99],'spearman':float(spearmanr(x,y).statistic),'kendall':float(kendalltau(x,y).statistic)})

fig,axes=plt.subplots(1,2,figsize=(13,5.3),layout='constrained')
for ax,keys,metric in zip(axes,[M,ATTR],['R7','R_attr']):
    xs=[11,55,99];scores=[[score(H[r][e],fixed,keys) for r in LONG] for e in xs]
    rank=np.array([ranks(v) for v in scores])
    rho=float(spearmanr(scores[0],scores[-1]).statistic)
    for j,rid in enumerate(LONG):
        ax.plot(range(3),rank[:,j],marker='o',lw=2,color=COLORS[rid],label=NAMES[rid])
        ax.text(2.06,rank[-1,j],NAMES[rid],va='center',fontsize=8,color=COLORS[rid])
    ax.set_xlim(-.1,3.25);ax.set_ylim(5.4,.6);ax.set_yticks(range(1,6));ax.set_xticks(range(3),['epoch11','epoch55','epoch99']);ax.set_ylabel('Rank (1 is best)');ax.grid(alpha=.15)
    ax.set_title(f'{metric} rank agreement: early11 vs late99, Spearman={rho:.2f}')
fig.suptitle('Same 5 runs, same fixed reference (frozen B512 epoch99); ranks cross',fontsize=13)
save(fig,'short_long_rank','Short vs long rank agreement',
     'Only five matched-reference runs reached epoch99. Scores at all epochs use the same fixed baseline vector. Each line is one run; low sample size and differing seed/hardware prevent a universal prediction claim.',LONG)

fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),layout='constrained')
for ax,metric in zip(axes,['R7','R_attr']):
    for anchor,style in [('vd23g1bq','-'),('iu3fa4gx','--')]:
        rr=[a for a in correlations if a.get('cohort')=='five_runs_including_controls' and a['anchor']==anchor and a['metric']==metric and 'early_epoch' in a]
        ax.plot([r['early_epoch'] for r in rr],[r['spearman'] for r in rr],marker='o',ls=style,label=NAMES[anchor]+' epoch99')
    ax.axhline(0,color='#777',ls=':',lw=1);ax.set_ylim(-1.05,1.05);ax.grid(alpha=.2);ax.set(xlabel='Early evaluation epoch',ylabel='Spearman rank correlation with epoch99',title=metric);ax.legend(fontsize=8,frameon=False)
fig.suptitle('Ranking depends on reference; five-run descriptive sensitivity, not a significance test',fontsize=13)
save(fig,'rank_correlation_sensitivity','Reference sensitivity of short/long rank correlation',
     'The 657 reference has very small lateral JSD; ratios against it give lateral differences a large influence on R7. R_attr early ordering is weak under both reference choices. Correlations are descriptive (n=5).',LONG)

observations=[];atlas=[];coverage=[]
for group,(anchor,epoch) in ANCHORS.items():
    rr=[r for r in manifest if (r['init_decoder'],r['reference_mode'])==group]
    b=H[anchor][epoch]
    for r in rr:
        rid=r['id']
        for e,a in sorted(H[rid].items()):
            observations.append({'id':rid,'name':r['name'],'initializer':group[0],'reference_mode':group[1],'epoch':e,'baseline_id':anchor,'baseline_epoch':epoch,'R7_fixed':score(a,b),'R_attr_fixed':score(a,b,ATTR),**{f'ratio_{k}':a[k]/b[k] for k in M},**a})
        coverage.append(rid)
    for start in range(0,len(rr),6):
        chunk=rr[start:start+6]
        fig,axes=plt.subplots(3,2,figsize=(12.5,7.5))
        fig.subplots_adjust(left=.065,right=.975,bottom=.075,top=.865,hspace=.86,wspace=.24)
        for ax,r in zip(axes.flat,chunk):
            rid=r['id'];es=sorted(H[rid]);r7=[score(H[rid][e],b) for e in es];ra=[score(H[rid][e],b,ATTR) for e in es]
            ax.plot(es,r7,lw=1.6,color='#276eb5',label='R7');ax.plot(es,ra,lw=1.6,color='#c76b21',label='R_attr');ax.axhline(1,color='#777',ls=':',lw=.7);ax.set_yscale('log');ax.grid(alpha=.16)
            title=rid+' | '+r['state']+' | seed '+str(r['seed'])+' | B'+str(r['train_batch_size'])+'\n'+ '\n'.join(textwrap.wrap(r['name'],65))
            ax.set_title(title,fontsize=8,pad=5);ax.tick_params(labelsize=8);ax.set_xlabel('Validation epoch',fontsize=8);ax.set_ylabel('Ratio (log)',fontsize=8);ax.legend(loc='upper right',ncol=2,fontsize=7,frameon=True,framealpha=.85,borderpad=.3)
        for ax in list(axes.flat)[len(chunk):]:ax.axis('off')
        fig.suptitle(GROUP_LABEL[group]+f' | runs {start+1}-{start+len(chunk)} / {len(rr)}',fontsize=14,y=.982)
        fig.text(.5,.927,f'Fixed control: {anchor} epoch{epoch}; R7 = mean of 7 ratios; R_attr = mean of length/width/speed ratios.',ha='center',fontsize=9)
        fig.text(.5,.01,'Different seeds / ego constraints / map crop / architecture / data may apply. Historical pages are descriptive, not a fair leaderboard.',ha='center',fontsize=8,color='#444')
        name=f"atlas_{group[0]}_{group[1]}_{start//6+1:02d}"
        save(fig,name,GROUP_LABEL[group]+f' - chart page {start//6+1}',
             f'Each panel contains one run. Baseline {anchor}, epoch{epoch} is fixed. Six plots per page. Original history is not sampled; repeated evaluations at one epoch are averaged at metric level.',[r['id'] for r in chunk])
        atlas.append(figures[-1]);print('atlas',name,'runs',len(chunk),flush=True)
assert len(coverage)==135 and len(set(coverage))==135

result={'source_inventory':str(BASE/'inventory.json'),'source_snapshot_fetched_at_utc':inventory.get('fetched_at_utc'),
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'definitions':{'R7':'Arithmetic mean of seven candidate/control metric ratios.','R_attr':'Arithmetic mean of length/width/speed candidate/control metric ratios.','fixed':'Same control metric vector at all epochs.','same_epoch':'Control vector at the same validation epoch; only exact overlapping epochs used.','pooling':'These are single-run observed ratios. They are not the original paired TRAIN/GEN multi-seed pooled R.'},
 'anchors':[{'initializer':g[0],'reference_mode':g[1],'run_id':r,'epoch':e} for g,(r,e) in ANCHORS.items()],
 'point_comparison':point_rows,'window_comparison':window_rows,'correlations':correlations,'run_coverage':coverage,'atlas_pages':atlas,
 'limitations':['Frozen snapshot from prior project analysis; actively running trials have no demonstrated long-term result.','Input conditions and version/hardware differ; historical full-reference charts are not a fair leaderboard.','Primary correlation uses five matched-reference runs reaching epoch99; only three reach127 and overnight stops at55.','No pairing of generation seeds, no independent training replicates per recipe, no remote dataset hashes.','Per-epoch metric averaging for duplicate logged evaluations is not pooled-histogram JSD.']}
(ROOT/'R_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
(ROOT/'R_history.json').write_text(json.dumps(observations,ensure_ascii=False,allow_nan=False))

# ReportLab document: text pages plus one landscape figure per page.
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
C=canvas.Canvas(str(OUT),pagesize=A4);C.setTitle('W&B short and long training R analysis - 135 runs');C.setAuthor('sim project analysis')
PAGE=0
CN=ParagraphStyle('cn',fontName='STSong-Light',fontSize=10.5,leading=17,textColor=colors.HexColor('#253448'))
HEAD=ParagraphStyle('head',fontName='STSong-Light',fontSize=21,leading=29,textColor=colors.HexColor('#163354'))

def footer(w,h,run_ids=None):
    C.setStrokeColor(colors.HexColor('#d7dfe7'));C.line(30,29,w-30,29)
    C.setFont('Helvetica',8);C.setFillColor(colors.HexColor('#526176'));C.drawString(30,16,'W&B u3006612/bc | saved snapshot 2026-10-12 | single-run observed ratios');C.drawRightString(w-30,16,str(PAGE))
    if run_ids:
        C.setFont('Helvetica',8);x=30;y=36
        for rid in run_ids:
            label=rid+'   ';C.drawString(x,y,label);wid=pdfmetrics.stringWidth(label,'Helvetica',8);C.linkURL(records[rid]['url'],(x,y-2,x+wid,y+9),relative=0);x+=wid

def paragraph(text,x,y,width,style=CN):
    p=Paragraph(text,style);w,h=p.wrap(width,900);p.drawOn(C,x,y-h);return y-h-12

def textpage(title,paragraphs,table=None):
    global PAGE
    C.setPageSize(A4);w,h=A4;PAGE+=1;y=h-42
    y=paragraph(title,40,y,w-80,HEAD)
    for text in paragraphs:y=paragraph(text,40,y,w-80)
    if table:
        t=Table(table,colWidths=[150]+[(w-230)/(len(table[0])-1)]*(len(table[0])-1))
        t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf5')),('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor('#253448')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')]),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),('ALIGN',(1,0),(-1,-1),'RIGHT')]))
        tw,th=t.wrap(w-80,h);assert y-th>45,(title,y,th);t.drawOn(C,40,y-th)
    footer(w,h);C.showPage()

textpage('短期 R 能否预测长期 R？',[
 '已覆盖此前快照中的全部135个run。共有4558次完整七指标评价，按相同epoch合并后得到'+str(len(observations))+'个曲线点。图册包含9个matched InitDiffusion、116个full InitDiffusion和10个Scenario Dreamer run。',
 '主要结论：在5个达到epoch99的近期matched run中，短期排序并不稳定。固定frozen B512 epoch99基准后，epoch11与99的Spearman相关为：R7=0.40，R_attr=-0.10。短期三轮均值(3/7/11)与长期三轮均值(91/95/99)为R7=0.40、R_attr=0.00。',
 '最明确的反转是frozen shared B256：早期属性排名第1，到epoch99变为第5。同epoch对照中，其R7从epoch11的0.9653变为99的1.0850，R_attr从0.9604变为1.2538。',
 'R7的排序还对control敏感：改用657 epoch99作固定基准，早期11与99相关变为0.90，而R_attr仍为-0.10。657的lateral JSD非常小，使这一项在R7中占据很大影响。',
 '以上是小样本描述，不是显著性或普遍预测能力证明。不同run的seed、硬件及配方并不完全一致；原短期search的paired pooled R=0.8975不能与这里单run的新基准R直接等同。',
 '后续页面先给出两种control下的固定/同epoch曲线、排序交叉与敏感性，再逐run展示全部曲线。所有曲线均使用完整历史，未从W&B抽样。'
])

point_table=[['Run (fixed frozen B512 e99)','R11','R99','Attr11','Attr99']]
for row in point_rows:point_table.append([row['id'],f"{row['R7_11']:.4f}",f"{row['R7_99']:.4f}",f"{row['R_attr_11']:.4f}",f"{row['R_attr_99']:.4f}"])
textpage('数值对照与解释',[
 'R7(t)=七个metric(t)/control_metric的比值平均；R_attr只平均length、width、speed三项。均越小越好。固定基准与同epoch基准都沿用该定义，区别只在分母。',
 '固定基准：始终使用同一个control epoch99向量，适合检查同一个run的真实收敛变化及短长期排序。同epoch基准：使用control在相同epoch的向量，适合判断相同训练进度下的相对优势。R=1表示与相应control相等。',
 '同epoch R上升不一定代表原始指标恶化，也可能是control改善更快。不能把各指标各自最好的epoch拼成一个不存在的checkpoint。',
 '主要排名分析的5个run都达到epoch99，按相同训练epoch比较；B256更新次数约为B512两倍。历史其他组的输入约束、数据、crop和模型可能不同，仅作描述。原始评价多数只有前256场景。',
 '下表全部使用固定frozen B512 epoch99向量，因而control自己的epoch11 R大于1，epoch99恰为1。'
],point_table)

# Primary plots then atlas; one landscape page per standalone figure.
for info in figures:
    C.setPageSize(landscape(A4));w,h=landscape(A4);PAGE+=1
    C.setFont('Helvetica-Bold',13);C.setFillColor(colors.HexColor('#163354'));C.drawString(30,h-27,info['title'])
    img=ImageReader(str(ROOT/(info['name']+'.png')));iw,ih=img.getSize();scale=min((w-60)/iw,(h-115)/ih);dw,dh=iw*scale,ih*scale
    C.drawImage(img,(w-dw)/2,h-48-dh,dw,dh)
    style=ParagraphStyle('caption',fontName='Helvetica',fontSize=8,leading=10,textColor=colors.HexColor('#47576a'))
    paragraph(html.escape(info['caption']),30,65,w-60,style)
    footer(w,h,info['run_ids']);C.showPage()
textpage('下一步如何使用这些图',[
 '目前不适合只用前10余epoch的R选出唯一配置。它能筛掉明显失败的run，但属性排序需要更长的验证窗口；现有5个run在epoch23以后属性相关性约0.5-0.6，仍不是可靠保证。',
 '将所有候选在固定场景清单、固定TRAIN normalizer和固定GEN seed下评价。建议在约12、24、56、100个epoch保存统一预算结果，至少保留多个候选到中后期；单次R较差也不应自动淘汰。',
 '同时观察R7、R_attr与七个分量比值，补充onroad率、生成vehicle数量及属性直方图。采用多seed复核与最终未参与选参的完整50k审计。',
 '当前快照中三个新LR run只到早期，overnight最后验证为55，没有长期99/127结果；不以缺失结果补齐、插值或外推它们的排名。',
 '数据与算法均保存于docs/experiments/wandb_project_analysis_20261012/r_curves/：R_history.json、R_analysis.json、build_r_report.py、每张PNG/SVG图。固定/同epochR之间不做跨定义拼接。'
])
C.save()

# Navigable local index plus a hash/coverage manifest.
parts=['<!doctype html><html><meta charset="utf-8"><title>W&B R curves</title><style>body{max-width:1250px;margin:30px auto;font:16px sans-serif;color:#243448}img{width:100%}section{margin:35px 0;border-top:1px solid #ddd}a{color:#276eb5}</style><body><h1>W&B short / long R - 135 runs</h1><p>R7: mean of seven candidate/control ratios. R_attr: length/width/speed. Fixed and same-epoch references differ. Historical cohorts are descriptive, not a fair leaderboard.</p>']
for info in figures:
    parts.append('<section><h2>'+html.escape(info['title'])+'</h2><p>'+html.escape(info['caption'])+'</p><img src="'+info['name']+'.png"><p>'+ ' | '.join('<a href="'+records[rid]['url']+'">'+rid+'</a>' for rid in info['run_ids'])+'</p></section>')
parts.append('</body></html>');(ROOT/'index.html').write_text(''.join(parts))
qa={'run_count':len(coverage),'unique_runs':len(set(coverage)),'curve_points':len(observations),'plot_pages':len(figures),'pdf_pages':PAGE,'pdf_path':str(OUT),'pdf_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'point_table':point_rows,'window_table':window_rows,'correlation11_vs99':[a for a in correlations if a.get('early_epoch')==11],'pdf_visual_qa':'pending'}
(ROOT/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print(json.dumps(qa,ensure_ascii=False,indent=2),flush=True)
