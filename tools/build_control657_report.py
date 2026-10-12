"""Build an evidence-scoped PDF comparison with commit657 as the control.

Consumes frozen, unsampled API snapshots; does not train or modify models.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if Path('/tmp/sim_pdf_report_deps').is_dir():
    sys.path.insert(0, '/tmp/sim_pdf_report_deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from tools.build_scratch_loss_search_pdf import register_fonts, markup

BASE = 'iu3fa4gx'
CONTROL_COMMIT = '657f1bb8c017f39e06e4e5317f734d6799eb1040'
IDS = [BASE, '4bedivn1', 'pd7zg04g', '0yhfg9ue', 'vd23g1bq', 'pz8g135r']
KEYS = ['nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd', 'collision_rate']
SHORT = ['Near', 'Lat', 'Ang', 'Length', 'Width', 'Speed', 'Col%']
ATTR = ['length_jsd', 'width_jsd', 'speed_jsd']
PARTS = ['position', 'heading', 'length', 'width', 'velocity', 'type', 'collision']
LABELS = {BASE: '657 control', '4bedivn1': 'Scratch B512', 'pd7zg04g': 'Scratch replicate',
          '0yhfg9ue': 'Frozen B256', 'vd23g1bq': 'Frozen B512', 'pz8g135r': 'Overnight + speed head'}
COLORS = {BASE: '#172433', '4bedivn1': '#d18a24', 'pd7zg04g': '#a2a8ae',
          '0yhfg9ue': '#287bb6', 'vd23g1bq': '#368e5b', 'pz8g135r': '#9358b7'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_analysis(source):
    snap = json.loads((source / 'runs_snapshot.json').read_text())
    remote = {r['id']: r for r in snap['records']}
    assert remote[BASE]['commit'] == CONTROL_COMMIT
    result = {'control_commit': CONTROL_COMMIT, 'control_run': BASE,
              'report_generated_utc': datetime.now(timezone.utc).isoformat(),
              'api_snapshot_utc': snap['read_at_utc'], 'method': 'unsampled scan_history; exact matched epoch; no interpolation',
              'score_definition': 'R_single = mean_7(candidate_native / control_native); R_attr_single = mean_Length_Width_Speed(ratio)',
              'decision_rule': 'R_single < 1 and attr <= 1.15 and observed population gates; missing population gates remain unknown',
              'runs': {}, 'pending_runs': [], 'provenance': [{'path': str((source / 'runs_snapshot.json').relative_to(ROOT)), 'sha256': sha(source / 'runs_snapshot.json')}]}
    for rid in IDS:
        obj = json.loads((source / (rid + '_history.json')).read_text())
        run = remote[rid]
        cfg = run['config']
        m = ast.literal_eval(cfg['model_config']) if isinstance(cfg['model_config'], str) else cfg['model_config']
        vals = [{**{k: float(r[k]) for k in KEYS}, 'epoch_index': int(r['epoch']),
                 'completed_epochs': int(r['epoch']) + 1, 'global_step': int(r['trainer/global_step']),
                 'timestamp': r.get('_timestamp')} for r in obj['rows'] if all(k in r for k in KEYS)]
        assert len({r['completed_epochs'] for r in vals}) == len(vals), 'Repeated validation epochs require explicit reconciliation'
        logs = [r for r in obj['rows'] if all('train/component/' + k + '_weighted_loss' in r for k in PARTS)][-100:]
        mean = {k: statistics.mean(r['train/component/' + k + '_weighted_loss'] for r in logs) for k in PARTS}
        result['runs'][rid] = {'label': LABELS[rid], 'name': run['name'], 'commit': run['commit'], 'state': run['state'],
                              'url': run['url'], 'seed': cfg['seed'], 'batch_size': cfg['data']['train_batch_size'],
                              'last_training_epoch_index': run['summary'].get('epoch'), 'history_rows': len(obj['rows']),
                              'read_at_utc': obj['read_at_utc'], 'validation': vals, 'model_config': m,
                              'data_config': cfg['data'], 'trainer_config': cfg['trainer'], 'callbacks': cfg['callbacks'],
                              'loss_window': {'count': len(logs), 'epoch_index_range': [logs[0]['epoch'], logs[-1]['epoch']],
                                              'mean': mean, 'fractions': {k: v / sum(mean.values()) for k, v in mean.items()}},
                              'n_train_models': 1, 'paired_generation_seed': False, 'population_gate': 'UNKNOWN'}
        result['provenance'].append({'path': str((source / (rid + '_history.json')).relative_to(ROOT)),
                                     'sha256': sha(source / (rid + '_history.json'))})
    anchor = {r['completed_epochs']: r for r in result['runs'][BASE]['validation']}
    for rid, run in result['runs'].items():
        for row in run['validation']:
            b = anchor[row['completed_epochs']]
            ratios = {k: row[k] / b[k] for k in KEYS}
            assert all(b[k] > 0 for k in KEYS)
            fails = [k for k in ATTR if ratios[k] > 1.15]
            row.update({'control_values': {k: b[k] for k in KEYS}, 'ratio': ratios,
                        'delta_native': {k: row[k] - b[k] for k in KEYS},
                        'change_pct': {k: (ratios[k] - 1) * 100 for k in KEYS},
                        'R_single': sum(ratios.values()) / 7,
                        'R_attr_single': sum(ratios[k] for k in ATTR) / 3,
                        'attribute_gate_failures': fails,
                        'overall_gate': 'CONTROL' if rid == BASE else 'FAIL' if fails or sum(ratios.values()) / 7 >= 1 else 'UNVERIFIED'})
        run['latest'] = run['validation'][-1]
    common = set.intersection(*[{r['completed_epochs'] for r in run['validation']} for run in result['runs'].values()])
    result['common_completed_epochs'] = max(common)
    result['common_epoch_table'] = []
    for rid, run in result['runs'].items():
        row = next(r for r in run['validation'] if r['completed_epochs'] == result['common_completed_epochs'])
        result['common_epoch_table'].append({'id': rid, 'label': run['label'], **row})
    result['common_epoch_table'].sort(key=lambda r: r['R_single'])
    result['completed128_ids'] = [rid for rid in IDS if result['runs'][rid]['latest']['completed_epochs'] == 128]
    for rid, run in remote.items():
        if run['created_at'] >= '2026-10-12' and rid not in IDS and 'lat_dev_jsd' not in run['summary']:
            result['pending_runs'].append({'id': rid, 'name': run['name'], 'state': run['state'],
                                           'commit': run['commit'], 'url': run['url'], 'score_status': 'NO_COMPLETED_VALIDATION'})
    old_path = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011/analysis/actual_observations.json'
    old = json.loads(old_path.read_text())
    old_groups = defaultdict(list)
    for row in old:
        old_groups[row['variant']].append(row)
    result['original_search'] = {
        'observations': len(old), 'control657_matched_observations': 0,
        'status': 'NOT_REBASED: no matched commit657 control at original exact exposure, seed, precision and eval cohort',
        'arms': [{'name': name, 'observations': len(rows), 'scopes': sorted({r['num_samples'] for r in rows}),
                  'seen_training_examples': sorted({r['seen_training_examples'] for r in rows}),
                  'new_control_score': None, 'status': 'MATCHED_CONTROL_MISSING'}
                 for name, rows in sorted(old_groups.items())],
        'historical_R': .8975, 'historical_R_attr': .8269,
        'scope': '10.6224 equivalent epochs, matched4096, TRAIN4817/5817 and GEN2817/3817',
        'meaning': 'Historical score versus original geo-MSE32/col20 control only; not a score versus commit657'}
    result['provenance'].append({'path': str(old_path.relative_to(ROOT)), 'sha256': sha(old_path)})
    result['limitations'] = ['One training model per run; no paired GEN seed; no confidence intervals.',
                             'Equal epochs approximate equal data exposure, not equal optimizer updates or compute.',
                             'Seeds, batch size, map training, normalizer and network changes are confounded.',
                             'Remote dataset and source-checkpoint byte hashes are unavailable.',
                             'No generated onroad counts/type populations or generated histogram exports in W&B.',
                             'R_single is dominated by very small baseline Lat values; it is not uniform degradation.',
                             'The original76-observation search has no matched657 denominator; its new ranking is unavailable.']
    result['validation_checkpoint_count'] = sum(len(r['validation']) for r in result['runs'].values())
    assert all(abs(r['R_single'] - 1) < 1e-12 for r in result['runs'][BASE]['validation'])
    return result


def make_figures(data, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figs = {}
    for name, variables, heading in [
            ('geometry_ratios', KEYS[:3] + [KEYS[-1]], 'Geometry and collision relative to657 at the same epoch'),
            ('attribute_ratios', ATTR + ['R_single'], 'Attribute metrics and composite relative to657')]:
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
        for ax, key in zip(axes.flat, variables):
            for rid in IDS[1:]:
                run = data['runs'][rid]
                rows = run['validation']
                values = [r[key] if key == 'R_single' else r['ratio'][key] for r in rows]
                ax.plot([r['completed_epochs'] for r in rows], values, color=COLORS[rid], label=LABELS[rid],
                        linestyle='--' if rid == 'pd7zg04g' else '-', lw=1.6)
            ax.axhline(1., color=COLORS[BASE], lw=1.2, label='657 = 1')
            if key in ATTR:
                ax.axhline(1.15, color='#b44c4c', lw=.8, linestyle=':', label='+15% attr gate')
            ax.set_title(key.replace('_', ' '));ax.set_xlabel('Completed data epochs');ax.set_ylabel('Ratio to same-epoch657')
            ax.set_xlim(0, 130);ax.grid(alpha=.2)
            if key in ('lat_dev_jsd', 'R_single'):ax.set_yscale('log')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, fontsize=8, frameon=False)
        fig.suptitle(heading, fontsize=14, y=.99)
        fig.text(.5, .01, 'Lower is better; runs stop at last completed validation; single-run descriptive comparisons.', ha='center', fontsize=8)
        fig.subplots_adjust(top=.79, bottom=.11, hspace=.44, wspace=.31)
        target = out / (name + '.png');fig.savefig(target, dpi=180);plt.close(fig);figs[name] = target
    return figs


def write_tables(data, out):
    columns = ['run_id', 'label', 'commit', 'completed_epochs', 'seed', 'batch_size',
               *KEYS, 'R_single', 'R_attr_single', 'attribute_gate_failures', 'overall_gate']
    # CSV is a machine-readable numeric export; the PDF is the human report.
    import csv
    for filename, ids, epoch in [('latest_same_epoch.csv', IDS, None), ('common_epoch.csv', IDS, data['common_completed_epochs']),
                                 ('completed128.csv', data['completed128_ids'], 128)]:
        with (out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns);writer.writeheader()
            for rid in ids:
                run = data['runs'][rid]
                row = run['latest'] if epoch is None else next(x for x in run['validation'] if x['completed_epochs'] == epoch)
                writer.writerow({'run_id': rid, 'label': run['label'], 'commit': run['commit'], 'completed_epochs': row['completed_epochs'],
                                 'seed': run['seed'], 'batch_size': run['batch_size'], **{k: row[k] for k in KEYS},
                                 'R_single': row['R_single'], 'R_attr_single': row['R_attr_single'],
                                 'attribute_gate_failures': ','.join(row['attribute_gate_failures']), 'overall_gate': row['overall_gate']})


class PDFReport:
    def __init__(self, data, figures, pdf):
        self.data, self.figures, self.pdf = data, figures, pdf
        register_fonts()
        self.width = A4[0] - 84
        self.story = []
        self.sections = []
        self.body = ParagraphStyle('body657', fontName='Latin', fontSize=9.1, leading=14.4, wordWrap='CJK',
                                   textColor=colors.HexColor('#263b4d'), spaceAfter=9)
        self.heading = ParagraphStyle('heading657', parent=self.body, fontSize=17, leading=24, textColor=colors.HexColor('#155574'),
                                      spaceAfter=14, keepWithNext=True)
        self.small = ParagraphStyle('small657', parent=self.body, fontSize=7.3, leading=11, spaceAfter=4)
        self.cell = ParagraphStyle('cell657', parent=self.body, fontSize=7.5, leading=10.8, spaceAfter=0)

    def p(self, text, small=False):
        self.story.append(Paragraph(markup(text), self.small if small else self.body))
        self.sections.append(text)

    def page(self, title):
        if self.story:self.story.append(PageBreak())
        self.story.append(Paragraph(markup(title), self.heading));self.sections.append(title)

    def table(self, headers, rows, widths=None):
        cells = [[Paragraph(markup(str(v)), self.cell) for v in row] for row in [headers, *rows]]
        t = Table(cells, colWidths=widths or [self.width / len(headers)] * len(headers), repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f0f5')),
                              ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#155574')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#91a5b6')),
                              ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d4dee6')),
                              ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                              ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        self.story.extend([t, Spacer(1,12)])

    def image(self, key):
        path = self.figures[key]
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,h=im.size
        self.story.extend([Image(str(path), width=self.width, height=self.width*h/w), Spacer(1,12)])

    def build(self):
        d = self.data;rr=d['runs'];e=d['common_completed_epochs']
        self.page('InitDiffusion 对比报告 | 新 control：657f1bb8')
        self.p('本报告以 commit `657f1bb8c017f39e06e4e5317f734d6799eb1040` 的实际 W&B run `iu3fa4gx` 为 control。全部候选指标除以相同完成 epoch 的 control 指标，未插值、未将不同预算混合排名。')
        self.p('**结论：目前没有已有候选满足替换657的条件。** 两个训满128 epoch的候选降低了碰撞率，但几何分布明显退化；属性退化门槛也未通过。原短预算search的得分只能保留为历史证据，不能直接改名为对657的得分。')
        self.table(['128 epoch模型','R_single','R_attr','属性gate'],
                   [[rr[r]['label'],f"{rr[r]['latest']['R_single']:.4f}",f"{rr[r]['latest']['R_attr_single']:.4f}",
                     'CONTROL' if r==BASE else ', '.join(k.replace('_jsd','') for k in rr[r]['latest']['attribute_gate_failures']) or 'PASS']
                    for r in d['completed128_ids']], [180,82,82,self.width-344])
        self.p('R_single是七个指标比值的平均，R_attr是长、宽、速度三个比值的平均，均越小越好。657自身为1。R_single接近19不代表每项都差19倍：其主要来源是Lat约122至125倍。必须同时阅读逐项指标。')
        self.p(f"证据：6个直接相关run，{d['validation_checkpoint_count']}次完整验证；统一可比较的最新公共时点是{e} epoch。两个新启动run尚无完成验证，列为待评价。每run仅一个训练模型，生成随机种子未配对，不计算置信区间。")
        self.p('本次只重算已有实测结果，没有重新训练control或候选。未对原76条search记录构造缺失的657分母。',small=True)
        self.p('API读取时间 UTC：'+d['api_snapshot_utc']+'；报告生成时间 UTC：'+d['report_generated_utc'],small=True)

        self.page('实际run、进度与比较范围')
        self.table(['Run / commit','Seed / B','最近验证','Map / context / head'],
                   [[f"{rr[r]['label']}\n{r} / {rr[r]['commit'][:8]}",f"{rr[r]['seed']} / {rr[r]['batch_size']}",
                     str(rr[r]['latest']['completed_epochs']),
                     ('AIRL frozen' if rr[r]['model_config'].get('initial_map_checkpoint') or r==BASE else 'scratch train map')+' / '+
                     rr[r]['model_config']['decoder']['init_diffusion'].get('ego_context_heading_encoding','angle')+' / '+
                     rr[r]['model_config']['decoder']['init_diffusion'].get('speed_head_kind','shared')] for r in IDS],
                   [177,69,62,self.width-308])
        self.p('以完成验证为时间单位：epoch索引127表示完成128 epoch。API state、summary中的最后训练epoch、最后上传的验证点可能不同。例如Scratch replicate已结束，但最后完整验证为100 epoch；Overnight还未完成128 epoch，不能判定其独立speed head的最终效果。')
        self.p('Native配置均使用training_map2_sd、scenario_dreamer_val，console记录493542训练entries、50000验证entries；指标是matched参考统计、官方GT地图、GT ego position/heading/shape/type条件、EMA、20步采样、square half extent50m。数据目录跨机器，未获得逐字节一致性证据。')
        self.p('相同epoch近似等样本曝光，B256的优化器更新约为B512的两倍，并不等计算量。训练seed、normalizer、map训练方式和网络结构仍混杂，因此这里是实测模型对比，不是单变量因果消融。')
        self.p('657与Frozen B256都通过run1.py使用同一RTX5090、32-true和highest matmul。run1.py硬编码highest，配置中的medium不生效；旧search使用medium。',small=True)
        if d['pending_runs']:
            self.table(['待评价run','Commit','状态'],[[x['name'],x['commit'][:8],'NO VALIDATION'] for x in d['pending_runs']], [335,78,self.width-413])

        self.page('同为128 epoch：绝对指标与替换门槛')
        base=rr[BASE]['latest'];a=rr['4bedivn1']['latest'];b=rr['0yhfg9ue']['latest']
        self.table(['Metric','657','Scratch','Frozen B256','Frozen变化%'],
                   [[s,f"{base[k]:.6f}",f"{a[k]:.6f}",f"{b[k]:.6f}",f"{b['change_pct'][k]:+.1f}%"] for k,s in zip(KEYS,SHORT)],
                   [93,88,88,103,self.width-372])
        self.p('原始日志单位：Near与Lat是raw JSD×10；Ang、Length、Width、Speed是raw JSD×100；Col%是按vehicle数加权的碰撞百分比。固定乘数在ratio中抵消，碰撞不再除100。所有指标越小越好。')
        self.p('Frozen B256：Lat=122.33倍、Ang=3.07倍、Near=2.86倍；Length差23.1%、Width差7.8%、Speed差24.0%；碰撞改善61.7%。Length与Speed超过原search允许的15%退化门槛。')
        self.p('Scratch B512：Width改善3.5%、碰撞改善70.8%，但Lat=125.06倍，Speed差32.2%，属性gate失败。R_attr=1.1286虽然低于1.15，逐项Speed仍超限；门槛不应只检查三项平均值。')
        self.p('原population gates要求onroad比例不降低超过0.02、generated vehicle/GT数量比在[0.9,1.1]。W&B未记录相应数量，所以这些gate保持UNKNOWN。已观察到R或属性gate失败，就不能替换control；不能把未记录的gate视为PASS。')

        self.page(f'公共时点：全部候选在{e} epoch的比较')
        self.table(['模型','R_single','R_attr','属性gate','判断'],
                   [[r['label'],f"{r['R_single']:.4f}",f"{r['R_attr_single']:.4f}",
                     'CONTROL' if r['id']==BASE else ', '.join(k.replace('_jsd','') for k in r['attribute_gate_failures']) or 'PASS',
                     'CONTROL' if r['id']==BASE else r['overall_gate']] for r in d['common_epoch_table']],
                   [162,76,76,115,self.width-429])
        self.p('这个表可以比较同预算下已有run的整体表现；不能把下表不同epoch的R互相排名。这里也没有控制训练seed或GEN随机数，微小改善不能当作稳健收益。')
        self.table(['最新完整验证','Epoch','R_single','R_attr','失败属性'],
                   [[rr[r]['label'],rr[r]['latest']['completed_epochs'],f"{rr[r]['latest']['R_single']:.4f}",
                     f"{rr[r]['latest']['R_attr_single']:.4f}",', '.join(k.replace('_jsd','') for k in rr[r]['latest']['attribute_gate_failures']) or '-'] for r in IDS[1:]],
                   [159,49,77,77,self.width-362])
        self.p('公共时点的Frozen B512属性平均有所改善，但R仍大于1。属性改善不足以补偿位置和车道对齐退化；当前模型不能据此替换657。')

        self.page('完整曲线：几何与碰撞的权衡')
        self.image('geometry_ratios')
        self.p('每个点都除以相同epoch的657。Lat和R使用log坐标以显示数量级。所有新配方的Lat下降比657慢，随后比值迅速上升；这不是最后一次验证的偶发波动。碰撞持续低于657，改善与几何指标呈明显权衡。')
        self.p('相关源码事实：collision处罚超出GT的圈间penetration，没有直接lane-centering/offroad项，可以通过移动位置或转动heading减少重叠。compute_penetration明确detach长宽，因此不能把尺寸退化解释为碰撞loss直接把车缩小。由于缺少onroad数量/生成直方图，不能确定具体偏移方向或offroad比例。')

        self.page('属性与checkpoint选择')
        self.image('attribute_ratios')
        self.p('128 epoch的Frozen B256 speed为0.070255；其第56 epoch为0.037617。657第64 epoch为0.025528，最后为0.056670。Late阶段反弹并非只出现在新方法中；训练loss或最后checkpoint不能代表最佳生成分布。')
        self.p('曲线只能作诊断，不能把多个epoch的不同指标最小值拼成一个最优模型，也不按各run最小R跨预算择优。本报告坚持公共epoch或同为128 epoch的比较。若需要选checkpoint，应预先规定一个多指标规则，用同一checkpoint评价所有指标并补配对GEN重复。')

        self.page('损失配方与后期有效平衡')
        self.table(['分量','657 control','Scratch','Frozen / Overnight'],
                   [['Position','L1 / p1 / w4','MSE / p1 / w32','MSE / p1 / w32'],
                    ['Heading(angle)','MSE / p2 / w1','L1 / p1 / w1','L1 / p1 / w1'],
                    ['Length','MSE / p1 / w1','MSE / p1 / w2','MSE / p2 / w2'],
                    ['Width','MSE / p2 / w1','MSE / p1 / w2','MSE / p2 / w2'],
                    ['Speed','MSE / p1 / w2','MSE / p1 / w2','MSE / p1 / w4'],
                    ['Type','CE / p0 / w1','CE / p0 / w1','CE / p0 / w1'],
                    ['Collision','L1 / p1 / w5','L1 / p1 / w20','L1 / p1 / w20']],
                   [112,130,130,self.width-372])
        self.p('p表示时间权重power，w表示分量系数。按max(t,0.05)^(-p)/C_p归一化，C1=3.995732、C2=39。Length从p1/w1改为p2/w2，高噪声t接近1的监督系数相对657仍仅约0.205倍。它更强调低噪声误差，不能保证完整生成路径更好。')
        self.table(['后期100条log','Position份额','Length+Width','Speed份额'],
                   [[rr[r]['label'],f"{rr[r]['loss_window']['fractions']['position']*100:.1f}%",
                     f"{sum(rr[r]['loss_window']['fractions'][k] for k in ['length','width'])*100:.1f}%",
                     f"{rr[r]['loss_window']['fractions']['velocity']*100:.1f}%"] for r in [BASE,'4bedivn1','0yhfg9ue']],
                   [181,110,110,self.width-401])
        self.p('上表是加权loss数值份额，不是梯度范数份额。新配方中位置约占80%，尺寸合计约2%；名义属性权重增大没有保持其相对份额。MSE本身并不是错误的Flow目标，现有日志不足以证明方差收缩、模式坍缩、EMA滞后或过拟合。主要嫌疑是整组几何目标和损失平衡改动，其单项因果贡献待消融。')

        self.page('原search：哪些结果不能直接换分母')
        old=d['original_search']
        self.p('原夜间search有76条实际observation、18个arm标签，最终selected相对于geo-MSE32/col20 control的pooled R=0.8975、R_attr=0.8269。这些数字保留历史含义。本次新R_single不是原pooled得分：每run只有一个训练模型，GEN随机种子未配对。')
        self.p('原search的matched4096、10.6224等价epoch、TRAIN4817/5817、固定normalizer和medium precision下，没有对应的657 control实验。因此全部原arm的新control得分均为NOT REBASED。不能拿657的128 epoch验证作10.62 epoch候选的分母，也不对657第8/12 epoch做插值。')
        rows=[]
        for i in range(0,len(old['arms']),2):
            pair=old['arms'][i:i+2]
            rows.append([pair[0]['name'],pair[0]['observations'],pair[1]['name'] if len(pair)>1 else '-',pair[1]['observations'] if len(pair)>1 else '-'])
        self.table(['原arm','记录数','原arm','记录数'],rows,[198,57,198,self.width-453])
        self.p('上表包括历史combo_reference，仅表示标签和记录数量，不表示每个arm都有完整最终预算结果。所有arm的scope、曝光和缺失状态已写入机器数据。原绝对指标仍有效，改变control后的R、gate、排名和最佳组合均需新的匹配分母。')
        self.p('若从657配置出发再应用旧搜索delta，会构成新的完整模型/训练配方，需要重新训练；旧候选checkpoint不能冒充这个新组合。')

        self.page('结论、后续对照与证据索引')
        self.p('**当前保留657 control，不替换为已有候选。** 新的几何损失配方没有通过同epoch或128 epoch比较；单独speed head的完整训练仍未结束。先保留657的Position L1/p1/w4、Heading angle MSE/p2/w1、Collision L1/p1/w5，再逐项尝试尺寸/速度权重与网络改动。')
        self.p('严格的新search应训练同预算、同TRAIN seed的657 control，统一map、normalizer、precision、数据次序与配对GEN种子。对完整收敛目标保留32/64/96/128 epoch；旧10.62 epoch比较需要单独补齐其原cohort的control。统一normalizer后的657-loss control不是原W&B657 run的逐项复现，应明确命名。')
        self.p('Limitations：每run N_train=1，无配对GEN或置信区间；不同seed/batch/map/normalizer仍混杂；remote文件hash未验证；population gates为UNKNOWN。当前只给已有结果的描述性比较，没有声称完成新的严格配对搜索。')
        self.p('W&B来源：',small=True)
        for rid in IDS:self.p(f"[{rr[rid]['label']} / {rid}]({rr[rid]['url']}) | commit {rr[rid]['commit'][:12]} | API {rr[rid]['read_at_utc']}",small=True)
        self.p('机器证据：output/reports/control657_20261012/comparison.json；completed128.csv；common_epoch.csv；latest_same_epoch.csv。原始API snapshot和history在tmp/control657_report_20261012/；SHA256清单随报告保存。',small=True)
        self.p('生成脚本：tools/build_control657_report.py。脚本只读取固定证据、计算ratio并生成报告，不导入模型、不训练、不更改候选选择文件。',small=True)
        def decorate(canvas,doc):
            canvas.saveState();canvas.setFont('Latin',7.5);canvas.setFillColor(colors.HexColor('#6b7f8f'))
            canvas.drawString(42,A4[1]-25,'InitDiffusion | commit657 control comparison')
            canvas.drawString(42,23,'Existing evidence only | Same-epoch ratios | 2026-10-12')
            canvas.drawRightString(A4[0]-42,23,str(doc.page));canvas.restoreState()
        doc=SimpleDocTemplate(str(self.pdf),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=48,bottomMargin=42,
                              title='InitDiffusion - commit657 control comparison',author='Sim training analysis')
        doc.build(self.story,onFirstPage=decorate,onLaterPages=decorate)
        return self.sections


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='tmp/control657_report_20261012')
    parser.add_argument('--output-dir',default='output/reports/control657_20261012')
    parser.add_argument('--pdf',default='output/pdf/init_diffusion_control657_comparison_20261012.pdf');args=parser.parse_args()
    source=(ROOT/args.source).resolve();out=(ROOT/args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    pdf=(ROOT/args.pdf).resolve();pdf.parent.mkdir(parents=True,exist_ok=True)
    data=make_analysis(source);figures=make_figures(data,out);write_tables(data,out)
    (out/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    sections=PDFReport(data,figures,pdf).build()
    body=''.join('<p>'+html.escape(s)+'</p>' for s in sections)
    (out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>657 control comparison</title><style>body{font:16px/1.8 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:20px}img{width:100%}</style><h1>InitDiffusion / 657 control comparison</h1>'+body+'<img src="geometry_ratios.png"><img src="attribute_ratios.png"><p>Full numeric tables: comparison.json / completed128.csv / common_epoch.csv</p></html>',encoding='utf-8')
    manifest={'pdf':str(pdf.relative_to(ROOT)),'pdf_sha256':sha(pdf),'data_sha256':sha(out/'comparison.json'),
              'script_sha256':sha(Path(__file__)),'source_files':data['provenance'],
              'validation_points':data['validation_checkpoint_count'],'common_epoch':data['common_completed_epochs'],
              'no_new_training':True,'original_search_rebased':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
