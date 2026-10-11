"""Render the completed local Markdown analysis as a Chinese PDF report."""
from pathlib import Path
import html
import json
import re
import sys

sys.path.insert(0, '/tmp/sim_pdf_report_deps')
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/init_diffusion_loss_search_20261011.md'
OUTPUT = ROOT / 'output/pdf/init_diffusion_loss_search_20261011.pdf'
TEMP = ROOT / 'tmp/pdfs'
TEMP.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
for name, path in [
    ('Latin', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    ('LatinBold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    ('Mono', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'),
    ('Chinese', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),
]:
    pdfmetrics.registerFont(TTFont(name, path))
pdfmetrics.registerFontFamily('Latin', normal='Latin', bold='LatinBold', italic='Latin', boldItalic='LatinBold')
pdfmetrics.registerFontFamily('Chinese', normal='Chinese', bold='Chinese', italic='Chinese', boldItalic='Chinese')


def plain(text):
    text = text.translate(str.maketrans({'–': '-', '—': '-', '−': '-', '\u2011': '-'}))
    parts = re.split(r'([\u2e80-\uffff]+)', text)
    return ''.join(f'<font name="Chinese">{html.escape(p)}</font>' if p and ord(p[0]) >= 0x2e80
                   else html.escape(p) for p in parts)


def markup(text):
    parts = re.split(r'(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))', text)
    result = []
    for p in parts:
        if p.startswith('**') and p.endswith('**'):
            result.append('<b>' + plain(p[2:-2]) + '</b>')
        elif p.startswith('`') and p.endswith('`'):
            result.append('<font name="Mono" size="8">' + plain(p[1:-1]) + '</font>')
        elif p.startswith('['):
            label = re.match(r'\[([^\]]+)\]', p).group(1)
            result.append('<font color="#156699">' + plain(label) + '</font>')
        else:
            result.append(plain(p))
    return ''.join(result)


BODY = ParagraphStyle('body', fontName='Latin', fontSize=9.3, leading=15,
                      textColor=colors.HexColor('#263342'), wordWrap='CJK', spaceAfter=8,
                      splitLongWords=True)
TITLE = ParagraphStyle('title', parent=BODY, fontSize=23, leading=31, spaceAfter=14,
                       textColor=colors.HexColor('#102f49'))
H2 = ParagraphStyle('h2', parent=BODY, fontSize=14, leading=21, spaceBefore=14,
                    spaceAfter=9, keepWithNext=True, textColor=colors.HexColor('#135a7a'))
H3 = ParagraphStyle('h3', parent=H2, fontSize=11, leading=17, spaceBefore=10)
CELL = ParagraphStyle('cell', parent=BODY, fontSize=8, leading=11.5, spaceAfter=0)
HEAD = ParagraphStyle('head', parent=CELL, textColor=colors.white)
CODE = ParagraphStyle('code', parent=BODY, fontName='Mono', fontSize=7.4, leading=11,
                      backColor=colors.HexColor('#f1f5f8'), borderPadding=8, spaceAfter=10)


def distribution_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    base = ROOT / 'logs/component_loss_quick/full50k_online'
    warm = json.loads((base / 'warm_start/feature_histograms.json').read_text())
    best = json.loads((base / 'best/feature_histograms.json').read_text())
    paths = []
    for feature, bounds, unit in zip(['speed', 'length', 'width'],
                                    [(0, 50), (1, 9), (.5, 3.5)], ['m/s', 'm', 'm']):
        fig, ax = plt.subplots(figsize=(6.8, 2.45), layout='constrained')
        key = feature + '_jsd'
        spec = warm[key]
        for label, field, data, color, ls in [
            ('GT', 'matched_gt', warm, '#111111', '-'),
            ('Warm start', 'generated', warm, '#888888', '--'),
            ('Angular MSE / 128 steps', 'generated', best, '#007cb5', '-'),
        ]:
            counts = np.asarray(data[key][field], dtype=float)
            edges = spec['minimum'] + np.arange(len(counts) + 1) * spec['bin_width']
            ax.stairs(counts / counts.sum(), edges, label=label, color=color, linestyle=ls, linewidth=1.6)
        ax.set_xlim(*bounds)
        ax.set_xlabel(f'{feature.title()} ({unit})', fontsize=11)
        ax.set_ylabel('Probability / bin', fontsize=10)
        ax.tick_params(labelsize=10)
        ax.grid(axis='y', alpha=.18)
        ax.legend(frameon=False, fontsize=8.5, loc='upper right')
        path = TEMP / f'distribution_{feature}.png'
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d9e4eb'))
    canvas.line(42, 805, 553, 805)
    canvas.setFont('Latin', 8)
    canvas.setFillColor(colors.HexColor('#617485'))
    canvas.drawString(42, 813, 'InitDiffusion | Component-loss search | 2026-10-11')
    canvas.drawString(42, 26, '94 training runs | 3 training seeds | Full 50k evaluation')
    canvas.drawRightString(553, 26, str(doc.page))
    canvas.restoreState()


lines = SOURCE.read_text().splitlines()
story = []
width = A4[0] - 84
i = 0
stacked = distribution_figure()
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line.startswith('```'):
        code = []
        i += 1
        while i < len(lines) and not lines[i].startswith('```'):
            code.append(plain(lines[i]).replace('  ', '&nbsp;&nbsp;'))
            i += 1
        story.append(Paragraph('<br/>'.join(code), CODE))
        i += 1
        continue
    if line.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            row = [x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', x) for x in row):
                rows.append(row)
            i += 1
        n = len(rows[0])
        if n == 2:
            widths = [110, width - 110]
        elif n >= 7:
            widths = [83] + [(width - 83) / (n - 1)] * (n - 1)
        elif n == 5 and rows[0][0] == '分量':
            widths = [115, 143, 67, 56, width - 381]
        else:
            widths = [width * .27] + [width * .73 / (n - 1)] * (n - 1)
        cells = [[Paragraph(markup(x), HEAD if r == 0 else CELL) for x in row]
                 for r, row in enumerate(rows)]
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#174d6b')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f7fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#d5e1e9')),
        ]))
        if len(rows) <= 6:
            story.append(KeepTogether([table, Spacer(1, 10)]))
        else:
            story.extend([table, Spacer(1, 10)])
        continue
    image = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
    if image:
        path = (SOURCE.parent / image.group(2)).resolve()
        if 'vehicle_distributions.png' in str(path):
            for panel in stacked:
                story.extend([Image(str(panel), width=width, height=width * 2.45 / 6.8), Spacer(1, 6)])
            story.extend([Paragraph(markup(image.group(1)), CELL), Spacer(1, 10)])
            i += 1
            continue
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, h = im.size
        height = width * h / w
        limit = 550
        if height > limit:
            height = limit
        story.extend([Image(str(path), width=min(width, height * w / h), height=height),
                      Paragraph(markup(image.group(1)), CELL), Spacer(1, 10)])
        i += 1
        continue
    if line.startswith('# '):
        story.append(Paragraph(markup(line[2:]), TITLE))
        story.append(Paragraph(plain('分析报告 / 2026-10-11 / sim 环境实测'), CELL))
        story.append(Spacer(1, 12))
    elif line.startswith('## '):
        story.append(Paragraph(markup(line[3:]), H2))
    elif line.startswith('### '):
        story.append(Paragraph(markup(line[4:]), H3))
    else:
        content = [line]
        if line.startswith('- '):
            content[0] = '- ' + line[2:]
        while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(('#', '|', '```', '![', '- ')):
            i += 1
            content.append(lines[i].strip())
        story.append(Paragraph(markup(' '.join(content)), BODY))
    i += 1

story.append(Paragraph(plain('源文件与结果记录'), H2))
story.append(Paragraph(plain('正文由仓库中的完整 Markdown 报告生成，保留所有结论、表格及复现命令。彩色文字表示原报告的配置或证据链接标签；具体文件位置请查阅 Markdown 版本。'), BODY))
for path in ['docs/init_diffusion_loss_search_20261011.md',
             'docs/experiments/init_diffusion_loss_search_20261011/',
             'configs/experiment/init_diffusion_lane_conditioned_loss_search_best.yaml',
             'configs/experiment/init_diffusion_lane_conditioned_loss_search_best_eval.yaml']:
    story.append(Paragraph(markup('`' + path + '`'), CELL))

doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=42, rightMargin=42,
                        topMargin=53, bottomMargin=46, title='InitDiffusion 分量 loss 搜索分析报告',
                        author='sim experiment analysis', allowSplitting=True)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
