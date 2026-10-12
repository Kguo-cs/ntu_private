"""Create the final paired-control comparison PDF from verified observations."""
import argparse,json,math,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if Path('/tmp/sim_pdf_report_deps').is_dir():sys.path.insert(0,'/tmp/sim_pdf_report_deps')
from tools.build_control657_report import PDFReport,KEYS,SHORT,ATTR
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt
import numpy as np


def get_group(data,phase,exposure=None):
    groups=[g for g in data['groups'] if g['phase']==phase]
    if exposure is None:return max(groups,key=lambda g:g['exposure'])
    return next(g for g in groups if g['exposure']==exposure)


def draw(data,out):
    figs={}
    for phase,title in [('long','Matched4096 - two TRAIN models'),('medium','Matched4096 - adaptive TRAIN4817')]:
        gs=sorted([g for g in data['groups'] if g['phase']==phase],key=lambda g:g['exposure'])
        fig,axs=plt.subplots(2,2,figsize=(10.6,7.3))
        for ax,key in zip(axs.flat,('R','R_attr','lat_dev_jsd','speed_jsd')):
            for arm,label,color in [('control','Old control','#c37b20'),('separate_speed','Separate speed','#437ba4'),('lane_center_context','Lane context','#668f52')]:
                points=[(g['equivalent_epochs'],a[key] if key in ('R','R_attr') else a['ratios'][key]) for g in gs for a in g['arms'] if a['variant']==arm]
                if points:
                    x,y=zip(*points);ax.plot(x,y,'o-',label=label,color=color)
            ax.axhline(1,color='#263b4d',lw=1,label='Fresh657=1')
            if key in ('R_attr','speed_jsd'):ax.axhline(1.15,color='#b65a56',ls=':',lw=.8,label='15% attr threshold')
            ax.set_title(key);ax.set_xlabel('Actual exposures / 493,542');ax.set_ylabel('Ratio to matched657');ax.grid(alpha=.2)
            vals=[v for line in ax.lines for v in line.get_ydata()]
            if vals and min(vals)>0 and max(vals)/min(vals)>15:ax.set_yscale('log')
        handles,labels=axs.flat[0].get_legend_handles_labels()
        fig.suptitle(title,y=.99,fontsize=14);fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False,fontsize=8)
        fig.subplots_adjust(top=.81,bottom=.13,hspace=.45,wspace=.32)
        fig.text(.5,.015,'Lower is better. GEN averaged within TRAIN; TRAIN means weighted equally. No interpolation.',ha='center',fontsize=8)
        p=out/(phase+'_ratios.png');fig.savefig(p,dpi=180);plt.close(fig);figs[phase]=p
    return figs


def make(data,out,pdf):
    assert data['complete'],'Final report requires complete matched cells'
    figures=draw(data,out);r=PDFReport(data,figures,pdf);w=r.width
    final=get_group(data,'long');old=next(a for a in final['arms'] if a['variant']=='control');candidate=next(a for a in final['arms'] if a['variant']=='separate_speed')
    wins=[a for a in final['arms'] if a['status']=='BEATS_657']
    conclusion='separate_speed 在本次配对短预算比较中满足替换657的条件。' if candidate['status']=='BEATS_657' else 'separate_speed 在本次配对短预算比较中未满足替换657的条件，保留657。'
    r.page('InitDiffusion 配对复查 | control：657f1bb8')
    r.p('本次已补训 TRAIN4817、5817 的两个657 control，使用原search的固定AIRL地图、TRAIN统计、曝光预算和评价场景。旧候选直接复用原始实测记录，不重新训练。')
    r.p('**结论：'+conclusion+'**')
    r.table(['最终4096 / 10.6224 epoch','R','R_attr','门槛 / 决策'],[['657 control','1.0000','1.0000','CONTROL']]+[[a['label'],f"{a['R']:.4f}",f"{a['R_attr']:.4f}",a['status']] for a in final['arms']],[170,65,65,w-300])
    r.p('R为七项指标candidate/control比值的平均，R_attr为Length、Width、Speed三个比值的平均；越低越好。先在每个TRAIN内平均GEN2817/3817，再等权平均两个TRAIN。R不是每场景JSD均值，也不是逐种子R的均值。')
    r.p('旧报告 separate_speed 的R=0.8975、R_attr=0.8269只表示相对旧control。更换分母后必须读取上表；本报告逐预算重新配对，不将新分数与旧分数或原128epoch分数相乘。')
    r.p(f"证据：{data['original_observations']}条历史观测；{data['matched_new_control_observations']}条新control配对观测，另有{data['auxiliary_observations']}条不同scope的辅助观测未纳入排名。两个模型各训练5,242,620次曝光；**这不是128epoch最终收敛比较。**")
    r.p('报告生成 UTC：'+data['generated_utc'],small=True)

    r.page('最终预算 | 逐项指标、覆盖与TRAIN差异')
    base=candidate['control_values_raw']
    r.table(['指标 raw JSD / Col%','657','旧control','separate_speed','sep变化%'],[[s,f'{base[k]:.6f}',f"{old['candidate_values_raw'][k]:.6f}",f"{candidate['candidate_values_raw'][k]:.6f}",f"{candidate['changed_percent'][k]:+.2f}%"] for s,k in zip(SHORT,KEYS)],[100,80,80,90,w-350])
    r.p('六项JSD为无量纲raw值；Col%为vehicle-weighted百分比。helper原生值Near/Lat乘10，Ang/Length/Width/Speed乘100；固定缩放在比值中抵消。负变化表示改善。')
    r.table(['TRAIN / sep','R','R_attr','onroad%','V/GT','gate'],[[seed,f"{s['R']:.4f}",f"{s['R_attr']:.4f}",f"{s['metrics']['generated_onroad_fraction']*100:.2f}",f"{s['metrics']['num_generated_vehicles']/s['metrics']['num_gt_vehicles']:.4f}",'PASS' if s['eligible'] else 'FAIL'] for seed,s in candidate['per_training_seed'].items()],[72,60,65,65,65,w-327])
    r.p('门槛：Length、Width、Speed各自<=657的1.15倍；onroad下降不超过2个百分点；生成vehicle/匹配GT vehicle在[0.9,1.1]。pooled和每个TRAIN均需通过，且pooled R<1。不能用R_attr平均值替代逐项门槛。')
    failures=[k for k,v in candidate['gate_checks'].items() if not v]
    r.p('separate_speed pooled失败项：'+(', '.join(failures) if failures else '无')+'；全部TRAIN门槛：'+('PASS' if candidate['each_training_seed_gates_pass'] else 'FAIL')+'。')

    r.page('相同曝光学习曲线 | 两TRAIN / 两GEN')
    r.image('long')
    r.table(['等价epoch','方法','R','R_attr','结果'],[[f"{g['equivalent_epochs']:.4f}",a['label'],f"{a['R']:.4f}",f"{a['R_attr']:.4f}",a['status']] for g in sorted([x for x in data['groups'] if x['phase']=='long'],key=lambda x:x['exposure']) for a in g['arms']],[70,140,65,65,w-340])
    r.p('所有点都是实际checkpoint，不插值。每个点均与同一曝光、TRAIN/GEN和4096 manifest的657比较。早期checkpoint来自同一次从头训练的前缀，不增加独立TRAIN数量。',small=True)

    r.page('Screen重排名 | 0.7967 epoch / 1024场景')
    screen=get_group(data,'screen')
    r.p('以下18个候选标签使用TRAIN4817、GEN2817；它们已参与过adaptive筛选。该表为更换control后的回顾性排序，不是独立确认或最终收敛排序。')
    r.table(['候选','R','R_attr','onroad%','gate / 决策'],[[a['label'],f"{a['R']:.4f}",f"{a['R_attr']:.4f}",f"{a['metrics']['generated_onroad_fraction']*100:.2f}",a['status']] for a in screen['arms']],[154,64,64,68,w-350])
    r.p('完整0.3984和0.7967epoch数字、逐TRAIN门槛和车辆计数保存在comparison.json与rankings.csv。screen通过并不代表最终通过。',small=True)

    r.page('Medium重比较 | 4096场景 / TRAIN4817')
    r.image('medium')
    r.table(['等价epoch','方法','R','R_attr','结果'],[[f"{g['equivalent_epochs']:.4f}",a['label'],f"{a['R']:.4f}",f"{a['R_attr']:.4f}",a['status']] for g in sorted([x for x in data['groups'] if x['phase']=='medium'],key=lambda x:x['exposure']) for a in g['arms']],[70,140,65,65,w-340])
    r.p('medium只有一个TRAIN和两个GEN；GEN重复不能作为独立训练样本。lane_center_context没有TRAIN5817的原始最终预算观测，本次不会用medium成绩冒充两TRAIN确认。',small=True)

    r.page('Full50k scope audit | 已有权重，不再择优')
    full=get_group(data,'full50000');sep=next(a for a in full['arms'] if a['variant']=='separate_speed');b=sep['control_values_raw']
    r.table(['方法 / TRAIN4817','R','R_attr','结果'],[[a['label'],f"{a['R']:.4f}",f"{a['R_attr']:.4f}",a['status']] for a in full['arms']],[170,70,70,w-310])
    r.table(['指标','657 raw','sep raw','变化%'],[[s,f'{b[k]:.6f}',f"{sep['candidate_values_raw'][k]:.6f}",f"{sep['changed_percent'][k]:+.2f}%"] for s,k in zip(SHORT,KEYS)],[100,110,110,w-320])
    r.p('full50k复用TRAIN4817的相同最终权重，并平均GEN2817/3817。50,000是cache entries，对应43,601个unique原场景key；这不是50,000个独立原场景，也没有新增独立TRAIN。full与4096 GT population不同，单独计算比值，不能合并排名。')
    r.p('50k中包含此前被评价的条目，因此不能称为新的unseen test。本次保持20步、t_eps=.05；hybrid40属于另一推断研究，没有混入这些训练recipe指标。')

    r.page('训练配方与验证 | 恢复657整套设置')
    history=json.loads((ROOT/'docs/experiments/init_diffusion_overnight_search_20261011/analysis/actual_observations.json').read_text())
    recipes=[data['protocol']['recipe_sha256'],next(a['component_loss'] for a in history if a['variant']=='control'),next(a['component_loss'] for a in history if a['variant']=='separate_speed')]
    new=json.loads((ROOT/'tmp/control657_paired_20261012/recipe.json').read_text())
    def text(recipe,k):
        v=recipe['components'][k];return f"{v['kind']} / p{v['time_power']} / w{v['weight']:g}"
    r.table(['loss component','657','旧control','separate_speed'],[[k,text(new,k),text(recipes[1],k),text(recipes[2],k)] for k in ('position','heading','length','width','velocity','type','collision')],[90,140,140,w-370])
    r.p('657与旧control均B512 / angle ego context / shared speed head。separate_speed为B256 / sincos context / 独立speed head，并同时调整完整loss recipe。这些组合比较不能单独证明speed head、sin/cos或某一loss分量是原因。')
    r.p('兼容验证：隔离CPU真实两场景批次，当前默认兼容路径与657原始denoiser/flow代码在相同权重和随机种子下，六项loss及全部252组G1梯度逐值一致。此检查仅证明已检验路径，不等于两次GPU训练逐bit重放。')
    r.p('新control与历史旧control的G1初始权重、固定map权重、TRAIN次序和TRAIN normalizer哈希一致。语义cohort还校验scene manifest、metric backend、reference source、precision、条件及20步采样。')
    r.p('训练使用sim环境与原scratch runner；map固定AIRL32，G1/AdamW/scheduler/EMA从头；LR=.0005、clip=.5、32-true、matmul medium；每个epoch重复seeded完整TRAIN次序，最后不足batch保留。')

    r.page('结论、适用范围与可复查证据')
    r.p('**'+conclusion+'** 新control使原候选得到新的绝对分母；旧search报告相对旧control的实测改善仍是历史事实，不能自动解释为优于657。')
    r.p('这轮回答的是：在原固定地图、TRAIN归一化和10.62epoch预算下，原候选是否优于657配方。原W&B的657 native128epoch使用另一训练协议；两项结果必须分开，不能依据本次短预算声称128epoch排名已经确定。')
    r.p('TRAIN4817曾参与原adaptive选择；5817仅确认原先冻结的separate_speed配方。若利用本次新排名挑选另一screen候选，应先冻结配置，再补充相同最终预算的5817或新TRAIN种子。两个TRAIN的差异仅作描述，不能证明稳健最优。')
    r.p('后续优化应保留明确的657 control，单项改变loss或模型，并保持map、normalizer、batch、EMA、sampler和evaluation membership一致。需要比较128epoch时，双方必须按同一完整训练协议从头训练。')
    for title,path in [('严格配对数据','output/reports/control657_paired_20261012/comparison.json'),('逐阶段数值','output/reports/control657_paired_20261012/rankings.csv'),('新control实际观测','output/reports/control657_paired_20261012/new_control_observations.json'),('原始候选实测','docs/experiments/init_diffusion_overnight_search_20261011/analysis/actual_observations.json'),('执行协议','tmp/control657_paired_20261012/paired_protocol.json'),('源码兼容检查','tmp/control657_paired_20261012/legacy_compatibility.json'),('训练 / 评价日志','logs/control657_paired_20261012')]:r.p(title+'：`'+path+'`',small=True)
    r.p('原始观测SHA256：'+data['protocol']['history_sha256'],small=True)
    r.p('新control loss SHA256：'+data['protocol']['recipe_sha256'],small=True)
    r.p('完整checkpoint与每条metric JSON哈希均列在new_control_observations.json；driver要求完整cell和成功exit后才将stage计入报告。',small=True)
    def footer(canvas,doc):
        canvas.setFont('Latin',7);canvas.setFillColorRGB(.35,.42,.48)
        canvas.drawString(42,24,'InitDiffusion | Paired657 control | 2026-10-12')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    pdf.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=42,title='InitDiffusion paired657 comparison',author='sim experiments').build(r.story,onFirstPage=footer,onLaterPages=footer)
    print('PDF_CREATED',pdf)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'output/reports/control657_paired_20261012/comparison.json');p.add_argument('--pdf',type=Path,default=ROOT/'output/pdf/init_diffusion_control657_paired_20261012.pdf');a=p.parse_args()
    make(json.loads(a.data.read_text()),a.data.parent,a.pdf)
