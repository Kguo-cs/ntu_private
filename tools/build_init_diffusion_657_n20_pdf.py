"""Build an eight-page Chinese N20 search report from the combined evidence JSON.

The combined JSON is the only source of scores, metric values, measured resources,
and rankings. Authorized control and graph audits provide nonmetric RNG and
configuration context. This tool performs no training and never imports torch.

Follow the PDF skill marker contract before the first PDF-authoring execution.
The report is a DRAFT by default and may be refreshed using the same input path.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPS = Path('/tmp/sim_pdf_report_deps')
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String, Polygon
from tools.build_scratch_loss_search_pdf import register_fonts, markup
from tools.report_init_diffusion_657_compact import same_fresh657_initializer

METRICS = ['nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd', 'collision_rate']
SHORT = ['Near', 'Lat', 'Ang', 'Length', 'Width', 'Speed', 'Col%']
ATTR = METRICS[3:6]
TIERS = {'screen': 0, 'medium': 1, 'long4096': 2, 'long': 2, 'full50k': 3, 'full': 3}
NAMES = {'control657': '657 control', 'norm_time_every_layer': 'norm_time', 'a2a_r20': 'A radius20', 'a2a_r40': 'A radius40', 'a2a_k8': 'A K8', 'a2a_k16': 'A K16', 'map_r20': 'Map radius20', 'map_r40': 'Map radius40', 'map_k16': 'Map K16', 'map_k64': 'Map K64'}
BLUE = colors.HexColor('#155574')
INK = colors.HexColor('#263b4d')
AUDITS = [ROOT / 'output/reports/init_diffusion_657_edges_20261013/control_config_runtime_audit.json', ROOT / 'output/reports/init_diffusion_657_edges_20261013/control_checkpoint_rng_audit.json', ROOT / 'output/reports/init_diffusion_657_edges_20261013/graph_checkpoint_rng_audit.json', ROOT / 'output/reports/init_diffusion_657_edges_20261013/medium_checkpoint_rng_audit.json']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fmt(v, n=4):
    return '-' if v is None else f'{float(v):.{n}f}'


def time_pair(s):
    if not s:
        return '未知'
    t = datetime.fromisoformat(s.replace('Z', '+00:00'))
    return f"UTC {t.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S} / 北京 {t.astimezone(timezone(timedelta(hours=8))):%Y-%m-%d %H:%M:%S}"


def label(c):
    name = NAMES.get(c['variant'], c['variant'])
    return name if c.get('source') == 'main' else f"compact/{name}"


def score(c, field='primary_fresh657'):
    s = c.get(field) or {}
    return s if s.get('available') else {}


def gate(s):
    if not s or s.get('available') is False:
        return '未配对'
    return 'PASS' if s.get('gates_pass') and s.get('all_TRAIN_gates_pass', True) else 'FAIL'


def gate_detail(s):
    if not s or s.get('available') is False:
        return '缺少对应分母'
    fails = [k.replace('_jsd', '') for k, v in s.get('gates', {}).items() if not v]
    if not s.get('all_TRAIN_gates_pass', True) and not fails:
        fails = ['有 TRAIN 失败']
    return ', '.join(fails) or 'PASS'


def qualified(c):
    s = score(c)
    return bool(s and c.get('each_observed_TRAIN_R_lt1_and_gates') and s.get('R', 1) < 1 and gate(s) == 'PASS')


def scopes(comparisons, terminal=False):
    groups = defaultdict(list)
    for c in comparisons:
        if terminal and not c.get('evidence', {}).get('complete_terminal_evidence'):
            continue
        key = (c['tier'], c['budget'], c['num_samples'], tuple(c.get('TRAIN', [])), tuple(c.get('GEN', [])))
        groups[key].append(c)
    return sorted(groups.items(), key=lambda kv: (TIERS.get(kv[0][0], -1), kv[0][1], kv[0][2]))


def primary_scope(groups):
    """Prioritize terminal full/long dual-TRAIN, then medium/screen evidence.

    High-budget single-TRAIN records cannot supersede a complete registered
    dual-TRAIN scope. Scores remain comparable only within the selected scope.
    """
    eligible = []
    for key, rows in groups:
        level = TIERS.get(key[0], -1)
        if level >= 2 and len(key[3]) < 2:
            continue
        if not any(c.get('evidence', {}).get('complete_terminal_evidence') for c in rows):
            continue
        eligible.append((key, rows))
    return max(eligible, key=lambda g: (TIERS.get(g[0][0], -1), g[0][1], g[0][2], len(g[0][3]), len(g[0][4])), default=None)


def primary_winner(group):
    if not group:
        return None
    candidates = [c for c in group[1] if c['variant'] != 'control657'
                  and c.get('evidence', {}).get('complete_terminal_evidence') and qualified(c)]
    return min(candidates, key=lambda c: score(c)['R'], default=None)


def parameter_text(c):
    values = c.get('resources', {}).get('actual_generator_parameters', [])
    return ', '.join(f'{int(v):,}' for v in values) if values else '未记录'


def best_order(rows):
    return sorted(rows, key=lambda c: (0 if score(c) else 1, score(c).get('R', float('inf')), label(c)))


def scope_text(k):
    return f"{k[0]} | step {k[1]} | N={k[2]} | TRAIN {','.join(map(str,k[3]))} | GEN {','.join(map(str,k[4]))}"


def get_stage(data, c):
    stages = data.get('sources', {}).get(c['source'], {}).get('verified_stages', [])
    eligible = [s for s in stages if s['stage'] in c.get('complete_stages', [])]
    return eligible[0] if eligible else None


def training_resource_stages(data, c):
    # full50k is a re-evaluation: match exact checkpoint paths to completed fit
    # stages in the same combined source instead of reporting zero as fit time.
    checkpoints={r['checkpoint'] for r in c.get('native_observations',[])}
    matches=[]
    for stage in data.get('sources',{}).get(c['source'],{}).get('verified_stages',[]):
        if (stage.get('runtime') or {}).get('train_seconds',0)<=0:continue
        observations=[r for row in data['comparisons'] if row['source']==c['source'] for r in row.get('native_observations',[]) if r['stage']==stage['stage']]
        if checkpoints & {r['checkpoint'] for r in observations}:
            matches.append(stage)
    return sorted(matches,key=lambda x:x['stage'])


def plot_curves(data, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    for ax, n in zip(axes, (1024, 4096)):
        groups = defaultdict(list)
        for c in data['comparisons']:
            if c['num_samples'] == n and score(c):
                # Different TRAIN/GEN scopes are not joined into one curve.
                groups[(c['source'], c['variant'], tuple(c['TRAIN']), tuple(c['GEN']), c.get('eval_manifest_sha256'))].append(c)
        for idx, (_, rows) in enumerate(sorted(groups.items())):
            rows.sort(key=lambda c: c['budget'])
            ax.plot([c['budget'] for c in rows], [score(c)['R'] for c in rows], marker='o', markersize=3, linewidth=1.2, label=label(rows[0])+((' T'+ '+'.join(map(str,rows[0]['TRAIN']))) if n==4096 else ''))
        ax.axhline(1, color='#38434d', ls='--', lw=.8)
        ax.set(title=f'Matched cohort: N={n}', xlabel='Training steps', ylabel='R / fresh same-scope 657')
        ax.grid(alpha=.2)
        if not groups:
            ax.text(.5,.5,'No verified paired cells yet',ha='center',va='center',transform=ax.transAxes,color='#637785')
        else:
            ax.legend(fontsize=5.6, loc='best', ncol=2, frameon=False)
    fig.tight_layout()
    path = out / 'n20_learning_curves.png'
    fig.savefig(path, dpi=190)
    fig.savefig(out / 'n20_learning_curves.svg')
    plt.close(fig)
    return path


class Report:
    def __init__(self, data, source, pdf, out, status, audits, recipe):
        register_fonts()
        self.data, self.source, self.pdf, self.out = data, source, pdf, out
        self.status, self.audits, self.recipe = status, audits, recipe
        self.width = A4[0] - 76
        self.story = []
        self.body = ParagraphStyle('n20body', fontName='Latin', fontSize=9, leading=14.2, wordWrap='CJK', textColor=INK, spaceAfter=8)
        self.small = ParagraphStyle('n20small', parent=self.body, fontSize=7.4, leading=11.2, spaceAfter=5)
        self.cell = ParagraphStyle('n20cell', parent=self.body, fontSize=7.25, leading=10.1, spaceAfter=0)
        self.heading = ParagraphStyle('n20heading', parent=self.body, fontSize=18, leading=25, textColor=BLUE, spaceAfter=13)
        self.sub = ParagraphStyle('n20sub', parent=self.body, fontSize=11.1, leading=16, textColor=BLUE, spaceAfter=7)
        self.generated = datetime.now(timezone.utc).isoformat()
        self.primary_scopes = scopes(data['comparisons'], terminal=True)
        self.screen = max((g for g in self.primary_scopes if g[0][0] == 'screen'), key=lambda g:g[0][1], default=None)
        self.latest = primary_scope(self.primary_scopes)

    def p(self, text, small=False):
        self.story.append(Paragraph(markup(str(text)), self.small if small else self.body))

    def subheading(self, text):
        self.story.append(Paragraph(markup(text), self.sub))

    def page(self, title):
        if self.story:
            self.story.append(PageBreak())
        self.story.append(Paragraph(markup(title), self.heading))

    def table(self, heads, rows, widths=None, dense=False):
        if not rows:
            self.p('暂无完成且可配对的实测记录。', True)
            return
        style = ParagraphStyle('dense', parent=self.cell, fontSize=6.8, leading=9.6) if dense else self.cell
        cells = [[Paragraph(markup(str(v)), style) for v in row] for row in [heads, *rows]]
        t = Table(cells, colWidths=widths or [self.width / len(heads)] * len(heads), repeatRows=1, hAlign='LEFT')
        pad = 4.4 if dense else 5.8
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8da9ba')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d3e0e7')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad)]))
        self.story.extend([t,Spacer(1,9)])

    def image(self, path, height):
        self.story.extend([Image(str(path),width=self.width,height=height),Spacer(1,9)])

    def rank_table(self, rows, limit=10):
        rows = best_order(rows)
        self.table(['方案','R(C/B)','R_attr/B','B gates','R(A/B)','直接 R(C/A)','直接 A gates'], [[label(c),fmt(score(c).get('R')),fmt(score(c).get('R_attr')),gate(score(c)),fmt((c.get('same_fresh657_initializer') or same_fresh657_initializer(score(c),score(c,'matched_initializer'))).get('initializer_R')),fmt(score(c,'matched_initializer').get('R')),gate(score(c,'matched_initializer'))] for c in rows[:limit]], [121,62,53,62,72,77,self.width-447], dense=True)
        if len(rows)>limit:
            self.p(f'正文列出前 {limit} 个方案；完整 {len(rows)} 个方案见输入 JSON。',True)

    def build(self):
        d=self.data
        completed=sum(len(s.get('verified_stages',[])) for s in d['sources'].values())
        # 1: Executive scope, rather than overclaiming a convergence prediction.
        self.page('InitDiffusion N20 低参数搜索 | '+self.status.upper())
        self.p(('**这是运行中的证据快照。** ' if self.status=='draft' else '**这是本轮完成证据的汇总快照。** ')+'所有排名、指标、资源数值均来自 combined live_report.json 中的完成验证记录；未完成训练不计入结果。')
        self.p('本轮目标是在采样 20 步和受控网络规模下，比较 657 baseline、训练特征归一化 + 逐层时间条件，以及图半径和邻居上限的消融。较短训练用于筛选，不能保证 128 epoch 的最终优劣。')
        if self.latest:
            winner=primary_winner(self.latest)
            self.p('**首页结论以最高已完成证据为准：** '+scope_text(self.latest[0])+'。')
            if winner:
                self.p(f"本范围内满足每个 TRAIN 的 R<1 与全部门槛、且 R 最低的是 **{label(winner)}**：R={fmt(score(winner)['R'])}，R_attr={fmt(score(winner)['R_attr'])}，实测 G1 参数 {parameter_text(winner)}。该结论只适用于此训练曝光与评价范围；完整 128 epoch 仍未证实。")
                worsened=[f'{SHORT[METRICS.index(k)]} ratio={fmt(v)}（+{100*(v-1):.2f}%）' for k,v in score(winner).get('ratios',{}).items() if v>1]
                if worsened:self.p('**逐项取舍：** '+ '；'.join(worsened)+'。R 改善不代表七项都改善；gates 也没有约束 Lat / Ang 必须下降。',True)
            else:
                done=[c for c in self.latest[1] if c['variant']!='control657']
                if done:
                    self.p('**本层没有已完成候选同时满足每个 TRAIN 的 R<1 和全部门槛，当前保留 657 baseline。** 不用较短 screen 的改善覆盖本层未获胜结果。')
                else:
                    self.p('**本层尚无完成候选可证明优于 657，当前保留 657 baseline。** 已完成 control 不等于候选失败；正在训练或未配对的候选仍不判胜负。')
        else:
            self.p('**暂无 complete_terminal_evidence，当前保留 657 baseline。** 部分阶段或缺失 seed 不用于主结论。')
        crop=self.source.get('map_crop_context') or {}
        crop_text=(f"{crop.get('init_map_crop','未知')}；half_extent={crop.get('init_map_half_extent','未知')} m（原生实验 base_config）" if crop else 'base_config 未记录；不从默认配置推定')
        self.table(['约束','本轮口径'],[['初始化','固定 AIRL map encoder；G1 从头初始化；TRAIN-only calibration'],['生成输入','GT 地图；固定 ego position / heading / shape / type；ego speed 生成'],['地图裁剪',crop_text],['预算与公平性','同 TRAIN / GEN、样本曝光、场景清单、校准、EMA；fresh657 同阶段分母'],['采样 / 训练精度','N20；32-true；matmul medium；EMA 评价'],['主队列参数上限','8,100,000；实测参数见资源页'],['截止时间','2026-10-13 10:00:00 北京 / 2026-10-13 02:00:00 UTC']],[110,self.width-110])
        self.p(f'已验证 stage：{completed}；当前 compact 状态：{d.get("compact_state","未知")}。')
        if self.screen:
            ss=self.screen[1]; good=[c for c in ss if c['variant']!='control657' and qualified(c)]
            anchorwins=[c for c in ss if c['variant'] not in ('control657',c.get('initializer_name')) and score(c,'matched_initializer').get('R',1)<1 and gate(score(c,'matched_initializer'))=='PASS']
            self.p(f'辅助 screen 信号：终点有 {len(ss)} 个完成方案；其中 {len(good)} 个观察到 R<1 且通过 fresh657 门槛。直接相对 anchor 同时 R<1 与 gates PASS 的额外候选有 {len(anchorwins)} 个（诊断项，非晋级条件）。screen 只作辅助筛选，不替代上述较高预算结论。')
        self.p('数据快照：'+time_pair(d.get('generated_utc')),True)
        self.p('PDF 生成：'+time_pair(self.generated),True)

        # 2: Exposure- and cohort-stratified ranking.
        self.page('分层排名 | 不混合评价规模与训练预算')
        self.table(['层级','step / N','TRAIN / GEN','完成方案'],[[k[0],f'{k[1]} / {k[2]}',f"{','.join(map(str,k[3]))} / {','.join(map(str,k[4]))}",len(rows)] for k,rows in self.primary_scopes[-6:]],[70,75,280,self.width-425],dense=True)
        if self.screen:
            self.subheading(scope_text(self.screen[0]))
            self.rank_table(self.screen[1],10)
        if self.latest and self.latest!=self.screen:
            self.subheading('最新较高预算：'+scope_text(self.latest[0]))
            self.rank_table(self.latest[1],4)
        self.p('C=候选，B=该候选配对 fresh657，A=记录中同范围 initializer（本快照为 A K8）。A 与自身的直接 R=1 不算增量改善。R 是七项指标比的均值；R_attr 是 Length/Width/Speed 比的均值。GEN 先在 TRAIN 内平均，再对 TRAIN 等权平均指标后计算比值。',True)
        self.p('完整 A 可用时，compact medium/long 的 R 条件为 R(C/B)<min(1,R(A/B))，gates 针对 B，另需阶段、曲线、参数与逐 TRAIN/GEN 要求。直接 R(C/A) 和 A gates 仅作诊断，并非晋级规则；两种判断不等价。A 缺失不补值；long 可按已冻结规则用阈值 1，full50k 仅比较 fresh657。',True)
        self.p('如果对应 fresh657 尚未完成，表格标为“未配对”；不借用历史 control，不把 control 自身的数值漂移描述成算法提升。',True)

        # 3: All seven actual metrics.
        self.page('实测七项指标 | 原生尺度，越低越好')
        rows=best_order(self.screen[1]) if self.screen else []
        if self.screen:self.p(scope_text(self.screen[0]),True)
        self.table(['方案',*SHORT],[[label(c),*[fmt(c.get('native_mean_metrics',{}).get(k)) for k in METRICS]] for c in rows[:12]],[115]+[(self.width-115)/7]*7,dense=True)
        self.p('Near / Lat = JSD×10；Ang / Length / Width / Speed = JSD×100；Col% = 生成车辆中发生碰撞的比例。各项尺度不同，不能直接平均这些原始数值。',True)
        if self.latest and self.latest!=self.screen:
            self.subheading('最新预算的已完成指标')
            self.p(scope_text(self.latest[0]),True)
            self.table(['方案',*SHORT],[[label(c),*[fmt(c.get('native_mean_metrics',{}).get(k)) for k in METRICS]] for c in best_order(self.latest[1])[:5]],[115]+[(self.width-115)/7]*7,dense=True)
        self.subheading('如何判断属性退化')
        self.p('Length、Width、Speed 分别要求 ratio≤1.15；另外要求 on-road 相对 control 下降不超过 0.02、生成车辆数 / GT 处于 [0.9,1.1]。R 降低但某一项超过门槛，仍记为 FAIL。')
        self.p('这里使用 matched reference 和官方参考地图计算 lane-conditioned agent metrics。子集结果只能说明该子集；不能直接替代全体场景评价。',True)

        # 4: Per-training-seed losses are never hidden behind pooling.
        self.page('逐 TRAIN 结果 | pooled 改善不等于稳定改善')
        chosen=self.latest[1] if self.latest else []
        if self.latest:self.p('最新预算：'+scope_text(self.latest[0]),True)
        train_scopes=[self.latest] if self.latest else []
        if self.screen and self.latest!=self.screen:
            train_scopes.append(self.screen)
            self.p('同时保留 screen 终点各 TRAIN 的独立胜负。表中 scope 分开，不跨范围平均。',True)
        per=[]
        for scope_key, scope_rows in train_scopes:
          for c in best_order(scope_rows):
            seedrows=c.get('per_TRAIN',[])
            for r in seedrows:
                p=r.get('primary_fresh657') or {}; a=r.get('matched_initializer') or {}
                per.append([label(c)+' / '+str(scope_key[1]),r['TRAIN'],','.join(map(str,r.get('GEN',[]))),fmt(p.get('R')),fmt(p.get('R_attr')),gate_detail(p),fmt(a.get('R')),gate(a)])
        self.table(['方案','TRAIN','GEN','R(C/B)','R_attr/B','B gates / 失败项','直接R(C/A)','直接A gates'],per[:14],[104,39,58,48,47,99,57,self.width-452],dense=True)
        if len(per)>14:self.p(f'正文显示前14行；完整逐 TRAIN 共{len(per)}行，见输入 JSON。',True)
        self.p('每个 TRAIN 都必须 R<1 且通过门槛，才能描述为两个 TRAIN 一致改善。只有一个 TRAIN 的 screen / medium 不自动升级为双 TRAIN 确证。',True)
        self.table(['条件','本快照'],[['完整 128 epoch + 全集评价','已证实' if d.get('native_128epoch_full_evaluation_verified') else '未证实'],['双 TRAIN 全50k获胜方案',str(sum(c.get('both_TRAIN_terminal_improves_fresh657') is True and c['tier']=='full50k' for c in d['comparisons']))],['缺失 seed 或阶段','不补值；不插值；不通过其他阶段的 control 配对']],[145,self.width-145])
        if self.latest and any(len(c.get('TRAIN',[]))>1 for c in chosen):
            failing=[label(c) for c in chosen if score(c).get('R',1)<1 and not c.get('each_observed_TRAIN_R_lt1_and_gates')]
            if failing:self.p('pooled R<1 但存在 TRAIN 失败：'+', '.join(failing)+'。',True)

        # 5: Actual resource records, no projected parameter counts.
        self.page('资源开销 | 实测参数、训练时间与内存')
        resource_rows=[]
        for c in rows:
            st=get_stage(d,c) or {};rt=st.get('runtime') or {};res=c.get('resources') or {}
            params=res.get('actual_generator_parameters') or st.get('actual_generator_parameters') or []
            trainable=res.get('actual_generator_trainable_parameters') or st.get('actual_generator_trainable_parameters') or []
            resource_rows.append([label(c),','.join(map(str,params)) or '-',','.join(map(str,trainable)) or '-',fmt(rt.get('train_seconds_per_step'),3),fmt(rt.get('train_seconds'),1),fmt(rt.get('cuda_peak_allocated_mib'),0),fmt(rt.get('cuda_peak_reserved_mib'),0)])
        self.table(['方案','G1 params','trainable','s/step','train s','alloc MiB','reserve MiB'],resource_rows[:12],[113,72,72,52,60,73,self.width-442],dense=True)
        self.p(f'此表按 screen 同预算展示前 {min(12,len(resource_rows))} / {len(resource_rows)} 项。完整各阶段实测参数、时间和内存见 combined JSON 的 comparisons.resources 与 sources.verified_stages.runtime；较高预算表也仅展示前5项，不代表其余方案不存在。',True)
        frozen=sorted({n for c in d['comparisons'] for n in c.get('resources',{}).get('actual_frozen_map_parameters',[])})
        self.p('G1 参数上限 8,100,000 只约束生成器。冻结 map encoder 实测参数：'+(', '.join(f'{int(v):,}' for v in frozen) if frozen else '本快照未记录')+'，单独统计，不计入 G1 cap。上述 train s 只统计 fit；原生 checkpoint 评价耗时另计。',True)
        base=next((c for c in d['comparisons'] if c.get('source')=='main' and c['variant']=='control657'),None)
        if base:
            res=base.get('resources',{}); g=res.get('actual_generator_parameters',[]); m=res.get('actual_frozen_map_parameters',[])
            if len(g)==len(m)==1:
                self.p(f'657 实测 G1 {int(g[0]):,} + 固定 map {int(m[0]):,} = 网络主体 {int(g[0])+int(m[0]):,}；这个总数不与 G1 cap 比较。',True)
        self.p('CUDA 峰值来自训练回调采样记录中的进程累计峰值；fit 后才执行原生 checkpoint 评价。这是阶段级实测，不是单层精确内存。allocated 和 reserved 分开列出，避免把缓存保留当成网络实际占用。',True)
        if self.latest and self.latest!=self.screen:
            self.subheading('较高预算资源')
            hi=[]
            for c in best_order(chosen)[:5]:
                training=training_resource_stages(d,c)
                for st in training:
                    rt=st.get('runtime') or {}
                    train_seeds=[st.get('spec',{}).get('seed')]
                    hi.append([label(c)+' / T'+','.join(map(str,train_seeds)),parameter_text(c),fmt(rt.get('train_seconds_per_step'),3),fmt(rt.get('train_seconds'),1),fmt(rt.get('cuda_peak_allocated_mib'),0)])
            self.table(['方案 / TRAIN','G1 params','s/step','train s','alloc MiB'],hi,[142,95,75,110,self.width-422],dense=True)
            if self.latest[0][0]=='full50k':
                evaluation=[(c,get_stage(d,c) or {}) for c in best_order(chosen)]
                self.p('全50k阶段只复用上述10240 step checkpoint评价；fit时间按相同 checkpoint 的已验证long训练记录列出。全集原生评价另计：'+ '；'.join(label(c)+' 首个TRAIN '+fmt((st.get('runtime') or {}).get('eval_seconds'),1)+' s' for c,st in evaluation)+'。完整各TRAIN评价耗时见JSON。',True)
        self.p('半径与 K 的修改不新增可训练参数，但改变注意力边数、运算量和条件信息。只有同一预算下的实测时间和峰值才适合做资源比较。')

        # 6: Code-generated vector diagram and metric tradeoffs.
        self.page('图设计 | 条件信息与计算量的取舍')
        self.p('当前构图在同一 batch 场景内按距离选择最近 K 个候选，再按半径过滤。A2A 不连自身；map2agent 使用地图到 agent 的距离。坐标单位为物理米。')
        self.story.extend([self.graph_diagram(), Spacer(1, 8)])
        self.p('半径变小会删除较远的信息；K 变小会减少密集区域的邻居。高噪声时 agent 位置偏离 GT，较小地图半径可能削弱地图条件；这是一项机制解释，不是已确定的性能原因。',True)
        singles=[c for c in rows if c['variant'] not in ('control657',c.get('initializer_name'))]
        trade=[]
        for c in singles:
            a=score(c,'matched_initializer');ratio=a.get('ratios',{})
            trade.append([label(c),fmt(a.get('R')),fmt(ratio.get('length_jsd')),fmt(ratio.get('speed_jsd')),fmt(ratio.get('collision_rate')),gate_detail(a)])
        self.table(['方案','直接R(C/A)','Length / A','Speed / A','Col / A','直接A gates'],trade[:10],[109,67,77,77,68,self.width-398],dense=True)
        self.p('此表直接相对各记录的 initializer（本快照为 A K8），不是 compact 晋级条件。A K8 与自身的直接 R=1 不列作获胜。不能把共同的特征归一化 / 时间条件收益全部归因于图稀疏化；fresh657 比较另列。',True)
        compact_points=[c for c in d['comparisons'] if c['source']=='compact' and c['variant']=='sincos_context' and c['num_samples']==4096 and c['budget'] in (4096,10240)]
        if compact_points:
            self.subheading('短训预测被否定的例子 | 仅第一 TRAIN')
            compact_points=sorted(compact_points,key=lambda c:c['budget'])
            self.table(['方案 / step','TRAIN','R(C/B)','R(A/B)','直接R(C/A)','B / A gates'],[[label(c)+' / '+str(c['budget']),','.join(map(str,c['TRAIN'])),fmt(score(c).get('R')),fmt((c.get('same_fresh657_initializer') or {}).get('initializer_R')),fmt(score(c,'matched_initializer').get('R')),gate(score(c))+' / '+gate(score(c,'matched_initializer'))] for c in compact_points],[172,43,66,66,82,self.width-429],dense=True)
            self.p('sin/cos 在 4096 step 的短训优势没有延续到 10240 step：同 fresh657 分母下 R 高于 K8 阈值，故未晋级。第二 TRAIN / 全50k未执行，不补结果。小型网络和 MLP edge 的768 step候选均未通过 fresh657 全部门槛；不推荐作为本轮完整训练胜者。',True)
        else:
            self.p('动态半径 / time-aware K 仍是未验证机制假设；不列为已验证或推荐方案。',True)

        # 7: Observed curves and strictly scoped control/RNG audit.
        self.page('短期曲线与 657 漂移 | 证据和推断分开')
        self.image(plot_curves(d,self.out),174)
        self.p('曲线只连接 TRAIN/GEN 组合、N 和来源一致的已验证点；空白图不代表性能差，而是尚无可配对记录。384→768 的变化提示早期排名会变动，不能线性外推到 128 epoch。',True)
        config_audit=self.audits.get('control_config_runtime_audit.json',{})
        rng_audit=self.audits.get('control_checkpoint_rng_audit.json',{})
        prov=config_audit.get('row_provenance',{})
        shared=['initial_generator_sha256','initial_map_sha256','train_order_sha256','calibration_sha256','map_checkpoint_sha256']
        same=[k for k in shared if prov and len({v.get(k) for v in prov.values()})==1]
        rng_rows=rng_audit.get('comparisons',[])
        self.table(['审计事实','范围与限制'],[['初始权重、校准和训练顺序','已核对一致：'+', '.join(k.replace('_sha256','') for k in same)],['checkpoint RNG 状态',f"{len(rng_rows)} 组同曝光 checkpoint 对比；Python / NumPy / CPU / CUDA RNG 状态均相同" if rng_rows and all(all(r['same_rng'].values()) for r in rng_rows) else '审计未提供完全一致证明'],['浮点权重分化','RNG 状态相同仍出现 generator 浮点权重差异；不是完整逐 step 运算轨迹证明'],['原因推断','GPU 算术 / 归约非确定性与现象一致；尚未定位具体 kernel，不能称已证实根因']],[118,self.width-118],dense=True)
        self.p('因此本轮主排名重新训练同阶段 fresh657。历史 control 仅作背景，不出现在主分母。配置审计没有发现隐藏的活跃超参数差异；源码版本和所有环境标志未达到逐位 GPU 等价证明。',True)
        graph_rng=self.audits.get('graph_checkpoint_rng_audit.json',{}).get('summary',{})
        if graph_rng:
            pairs=graph_rng.get('paired_checkpoints',0)
            matching=graph_rng.get('same_saved_endpoint_rng_counts',{})
            if pairs and all(matching.get(k)==pairs for k in ['python','numpy','torch_cpu','torch_cuda']):
                self.p(f'图队列新增审计：{pairs} 对 checkpoint 的 Python / NumPy / CPU / CUDA 端点 RNG 状态一致。只证明该曝光终点的累计状态一致，不证明初始训练 RNG 或逐 step noise / assignment 一致。',True)
        medium_rng=self.audits.get('medium_checkpoint_rng_audit.json',{}).get('summary',{})
        if medium_rng:
            pairs=medium_rng.get('paired_checkpoints',0)
            counts=medium_rng.get('same_saved_endpoint_rng_counts',{})
            if pairs and all(counts.get(k)==pairs for k in ['python','numpy','torch_cpu','torch_cuda']):
                self.p(f'中期新增审计：norm_time / K8 / K16 在 2048 / 4096 的 {pairs} 对 checkpoint 端点 RNG 状态全部一致；审计未初始化 CUDA，不证明逐 step 抽样、noise-agent 分配或初始训练 RNG 一致。',True)
        self.p('结构候选仍有另一项限制：MLP / 宽度 / 深度改变初始化 RNG 消耗；相同 TRAIN seed 和数据顺序，不保证逐 step 使用相同训练噪声。需要独立 TRAIN 确认。',True)
        self.p('短训不是压缩版 128 epoch：LR 仍沿原 128 epoch cosine，前期接近初始 LR。EMA 默认 use_num_updates=True，实效 decay=min(配置 decay,(1+n)/(10+n))；配置 .9999 时，768 / 4096 / 10240 step 约为 .9884 / .9978 / .99912。EMA 窗口也在变化，不能据此直接归因胜负。',True)
        self.p('数据次序限制：每个 TRAIN seed 先打乱一次完整清单，后续 epoch 按相同次序重复，shuffle=False。seen examples 包含重复曝光，不代表新增独立场景；若最终训练改为逐 epoch 重新 shuffle，优化轨迹会变化。两 GEN 只是采样重复，不能替代第二个 TRAIN。',True)
        self.p('EMA 代码：torch_ema/ema.py:45,108-113；InitDiffusion 默认构造：src/smart/diffusion/initial_diffusion.py:296。',True)

        # 8: Recipe and traceable provenance, no speculative final-training scores.
        self.page('配方、来源与后续确证要求')
        dec=self.recipe.get('model',{}).get('model_config',{}).get('decoder',{}).get('init_diffusion',{})
        loss=dec.get('component_loss',{}).get('components',{})
        self.table(['657 loss component','kind','time power','weight','scale'],[[k,v.get('kind'),v.get('time_power'),v.get('weight'),v.get('scale')] for k,v in loss.items()],[162,60,90,75,self.width-387],dense=True)
        self.p('657 监督目标、固定 ego 条件和 N20 保持一致。norm_time 的增量为 state_feature_normalization=train 与 time_conditioning=every_layer。radius / K 消融只改指定一侧的图；compact 的具体已完成差异由其 protocol 与 native 记录追踪。',True)
        self.p('本轮 K8 已完成两 TRAIN × 两 GEN 的全50k membership 验证，但 checkpoint 只训练10240 step（约10.62 epoch）。推荐优先完整训练K8；K16只有单TRAIN medium证据，可作为探索项，不是第二个全50k胜者。sin/cos、MLP、小型网络与5层模型不列作本轮完整训练推荐。128 epoch最终收敛仍未证实。',True)
        input_path=self.source.get('immutable_combined_snapshot',{}).get('path',self.source['path'])
        provenance=[[input_path,self.source['sha256'][:20]+'...','不可变唯一指标 / 资源 / 排名源']]
        for name,meta in self.source['audit_files'].items():provenance.append([meta['path'],meta['sha256'][:20]+'...','非指标审计背景'])
        if self.source.get('recipe_file'):provenance.append([self.source['recipe_file']['path'],self.source['recipe_file']['sha256'][:20]+'...','仅配置配方'])
        if self.source.get('base_config_file'):provenance.append([self.source['base_config_file']['path'],self.source['base_config_file']['sha256'][:20]+'...','实际地图裁剪背景'])
        self.table(['来源路径','SHA256 摘要','用途'],provenance,[282,118,self.width-400],dense=True)
        waiting=[]
        for name,s in d['sources'].items():
            state=s.get('status') or {}
            waiting.append([name,state.get('status','未知'),state.get('active_stage') or state.get('next_task',{}).get('label') or '-',len(s.get('partials_not_evidence',[])),len(s.get('quarantined',[]))])
        self.table(['队列','当前状态','阶段','partial','quarantine'],waiting,[52,190,158,60,self.width-460],dense=True)
        self.p('失败、SIGSEGV 或部分训练输出均不作为完成指标。若新结果写入 combined JSON，可重新执行本工具刷新；PDF 保留本次输入字节 SHA 和准确快照时间，避免运行中结果混入同一报告。',True)
        self.p('复现命令：sim 环境执行 tools/build_init_diffusion_657_n20_pdf.py；默认 source 和 output 已指向本轮文件。完整来源 SHA、图表和快照见 PDF 旁的 provenance JSON。',True)
        self.finish()

    def graph_diagram(self):
        w=self.width; d=Drawing(w,158)
        boxw=(w-24)/3
        for i,title in enumerate(['Scene / physical meters','Nearest K candidates','Radius filter / attention']):
            x=i*(boxw+12)
            d.add(Rect(x,13,boxw,130,fillColor=colors.HexColor('#f0f5f8'),strokeColor=colors.HexColor('#c6d7e0')))
            d.add(String(x+9,126,title,fontName='Latin',fontSize=7.3,fillColor=BLUE))
            cx=x+boxw*.52;cy=67
            if i==0:
                d.add(Line(x+15,33,x+boxw-18,96,strokeColor=colors.HexColor('#699bb6'),strokeWidth=2))
                for px,py in [(cx,cy),(cx-30,cy+20),(cx+40,cy-20)]:d.add(Circle(px,py,5,fillColor=colors.HexColor('#da8b38'),strokeColor=None))
            else:
                d.add(Circle(cx,cy,4,fillColor=colors.HexColor('#da8b38'),strokeColor=None))
                pts=[(cx-36,cy+26),(cx+33,cy+17),(cx-29,cy-21),(cx+43,cy-19),(cx+12,cy+35)]
                if i==2:d.add(Circle(cx,cy,38,fillColor=None,strokeColor=colors.HexColor('#5f91ae'),strokeDashArray=[3,2]))
                for j,(px,py) in enumerate(pts):
                    keep=i==1 or j in (1,2,4)
                    d.add(Line(cx,cy,px,py,strokeColor=colors.HexColor('#417e9e') if keep else colors.HexColor('#cdd9df'),strokeWidth=1.2 if keep else .6))
                    d.add(Circle(px,py,3,fillColor=colors.HexColor('#719caf') if keep else colors.HexColor('#cdd9df'),strokeColor=None))
            if i<2:
                xx=x+boxw+2; yy=76
                d.add(Polygon([xx,yy-3,xx+8,yy-3,xx+8,yy-7,xx+12,yy,xx+8,yy+7,xx+8,yy+3,xx,yy+3],fillColor=BLUE,strokeColor=None))
        d.add(String(0,0,'Schematic only: no measured edge counts or performance values.',fontName='Latin',fontSize=7,fillColor=INK))
        return d

    def finish(self):
        shortsha=self.source['sha256'][:12]
        def footer(canvas,doc):
            canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#cbd8e1'));canvas.line(38,38,A4[0]-38,38)
            canvas.setFont('Latin',7);canvas.setFillColor(colors.HexColor('#5c7180'))
            canvas.drawString(38,25,f'InitDiffusion N20 | {self.status.upper()} | input SHA256 {shortsha}')
            canvas.drawRightString(A4[0]-38,25,f'{doc.page} / 8');canvas.restoreState()
        doc=SimpleDocTemplate(str(self.pdf),pagesize=A4,rightMargin=38,leftMargin=38,topMargin=38,bottomMargin=48,title=f'InitDiffusion N20 search {self.status.upper()}',author='Local experiment audit',pageCompression=1)
        doc.build(self.story,onFirstPage=footer,onLaterPages=footer)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'output/reports/init_diffusion_657_compact_20261013/live_report.json')
    p.add_argument('--output',type=Path,default=ROOT/'output/pdf/init_diffusion_657_n20_search_20261013.pdf')
    p.add_argument('--status',choices=['draft','final'],default='draft')
    p.add_argument('--snapshot-dir',type=Path,default=ROOT/'tmp/pdfs/init_diffusion_657_n20_20261013')
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve();args.snapshot_dir=args.snapshot_dir.resolve();raw=args.source.read_bytes();data=json.loads(raw)
    for k in ['comparisons','sources','generated_utc','metric_scales']:
        if k not in data:raise ValueError('Missing combined schema field '+k)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.snapshot_dir.mkdir(parents=True,exist_ok=True)
    immutable_dir=args.output.parent/'source_snapshots'
    immutable_dir.mkdir(parents=True,exist_ok=True)
    immutable_path=immutable_dir/('combined_'+digest(raw)+'.json')
    if immutable_path.is_file():
        if immutable_path.read_bytes()!=raw:raise ValueError('Content-addressed input snapshot differs from its SHA')
    else:
        with immutable_path.open('xb') as handle:handle.write(raw)
    # Compatibility pointer may refresh; immutable bytes never do.
    (args.snapshot_dir/'combined_input_snapshot.json').write_bytes(raw)
    source={'path':str(args.source.relative_to(ROOT)),'sha256':digest(raw),
        'immutable_combined_snapshot':{'path':str(immutable_path.relative_to(ROOT)),'sha256':digest(raw)},
        'audit_files':{},'recipe_file':None,'generated_utc':datetime.now(timezone.utc).isoformat(),'metric_source_policy':'immutable combined JSON snapshot only; authorized audits and recipe are nonmetric context'}
    audits={}
    for path in AUDITS:
        if path.is_file():
            b=path.read_bytes();audits[path.name]=json.loads(b);source['audit_files'][path.name]={'path':str(path.relative_to(ROOT)),'sha256':digest(b)}
    recipe={}
    control=next((c for c in data['comparisons'] if c['variant']=='control657' and c['source']=='main'),None)
    if control and control.get('native_observations'):
        cp=Path(control['native_observations'][0]['checkpoint']).parent/'config.yaml'
        if cp.is_file():
            import yaml
            b=cp.read_bytes();recipe=yaml.safe_load(b);source['recipe_file']={'path':str(cp.relative_to(ROOT)),'sha256':digest(b)}
        bp=cp.parent.parent/'base_config.yaml'
        if bp.is_file():
            import yaml
            b=bp.read_bytes();base_config=yaml.safe_load(b)
            source['base_config_file']={'path':str(bp.relative_to(ROOT)),'sha256':digest(b),'purpose':'actual native control configuration; no metric values'}
            token=base_config.get('model',{}).get('model_config',{}).get('token_processor',{})
            source['map_crop_context']={k:token.get(k) for k in ['init_map_crop','init_map_half_extent']}

    for context in [Path('/home/ke/miniconda3/envs/sim/lib/python3.11/site-packages/torch_ema/ema.py'),ROOT/'src/smart/diffusion/initial_diffusion.py']:
        if context.is_file():
            source.setdefault('schedule_context_files', []).append({'path':str(context),'sha256':digest(context.read_bytes()),'purpose':'EMA implementation background only; not a metric source'})
    report=Report(data,source,args.output,args.snapshot_dir,args.status,audits,recipe);report.build()
    source.update({'pdf_path':str(args.output.relative_to(ROOT)),'pdf_sha256':digest(args.output.read_bytes()),'pdf_generated_utc':report.generated,'combined_generated_utc':data['generated_utc'],'status':args.status.upper(),'comparison_count':len(data['comparisons']),'cuda_initialized':False,'gpu_execution':False})
    args.output.with_suffix('.provenance.json').write_text(json.dumps(source,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pdf':str(args.output),'input_sha256':source['sha256'],'pdf_sha256':source['pdf_sha256'],'status':args.status.upper()},ensure_ascii=False))


if __name__=='__main__':main()
