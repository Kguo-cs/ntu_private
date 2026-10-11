"""Render a repository Markdown report to a Chinese PDF, using only its content.

Example:
  python tools/build_scratch_loss_search_pdf.py \
    --source docs/init_diffusion_scratch_loss_search_20261011.md \
    --output output/pdf/init_diffusion_scratch_loss_search_20261011.pdf \
    --footer 'InitDiffusion scratch training | Recorded local experiments'

This renderer does not create experimental figures, load metric data, or infer
run counts, evaluation scope, or findings. Images must already exist and be
referenced in the source Markdown. Follow the PDF artifact workflow before
executing this script, then render and inspect the resulting pages.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
import sys

# Reuse the installed local report dependencies; no installation or network.
OPTIONAL_DEPENDENCIES = Path("/tmp/sim_pdf_report_deps")
if OPTIONAL_DEPENDENCIES.is_dir():
    sys.path.insert(0, str(OPTIONAL_DEPENDENCIES))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 42
PAGE_WIDTH, PAGE_HEIGHT = A4
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
INK = colors.HexColor("#263342")
BLUE = colors.HexColor("#174d6b")


def register_fonts():
    for name, filename in (
        ("Latin", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("LatinBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("Mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        ("Chinese", "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ):
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Required report font not found: {filename}")
        pdfmetrics.registerFont(TTFont(name, filename))
    pdfmetrics.registerFontFamily("Latin", normal="Latin", bold="LatinBold",
                                  italic="Latin", boldItalic="LatinBold")
    pdfmetrics.registerFontFamily("Chinese", normal="Chinese", bold="Chinese",
                                  italic="Chinese", boldItalic="Chinese")


def plain(text: str) -> str:
    text = text.translate(str.maketrans({"–": "-", "—": "-", "−": "-", "\u2011": "-"}))
    parts = re.split(r"([\u2e80-\uffff]+)", text)
    return "".join(f'<font name="Chinese">{html.escape(part)}</font>'
                   if part and ord(part[0]) >= 0x2e80 else html.escape(part)
                   for part in parts)


def markup(text: str, *, code_size: float = 8.) -> str:
    tokens = re.split(r"(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|<br\s*/?>)", text)
    output = []
    for token in tokens:
        if token.startswith("**") and token.endswith("**"):
            output.append("<b>" + plain(token[2:-2]) + "</b>")
        elif token.startswith("`") and token.endswith("`"):
            output.append(f'<font name="Mono" size="{code_size:g}">' + plain(token[1:-1]) + "</font>")
        elif token.startswith("["):
            match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            label, target = match.groups()
            if target.startswith(("https://", "http://")):
                output.append(f'<link href="{html.escape(target, quote=True)}" color="#156699">'
                              + plain(label) + "</link>")
            else:
                output.append('<font color="#156699">' + plain(label) + "</font>")
        elif re.fullmatch(r"<br\s*/?>", token):
            output.append("<br/>")
        else:
            output.append(plain(token))
    return "".join(output)


def styles():
    body = ParagraphStyle("body", fontName="Latin", fontSize=9.3, leading=14.5,
                          textColor=INK, wordWrap="CJK", splitLongWords=True, spaceAfter=8)
    return {
        "body": body,
        "title": ParagraphStyle("title", parent=body, fontSize=22, leading=30,
                                textColor=colors.HexColor("#102f49"), spaceAfter=16),
        "h2": ParagraphStyle("h2", parent=body, fontSize=13.8, leading=20.5,
                             spaceBefore=13, spaceAfter=8, keepWithNext=True,
                             textColor=colors.HexColor("#135a7a")),
        "h3": ParagraphStyle("h3", parent=body, fontSize=11, leading=17,
                             spaceBefore=10, spaceAfter=6, keepWithNext=True,
                             textColor=colors.HexColor("#135a7a")),
        "caption": ParagraphStyle("caption", parent=body, fontSize=8, leading=11.5,
                                  textColor=colors.HexColor("#617485"), spaceAfter=7),
        "code": ParagraphStyle("code", parent=body, fontName="Mono", fontSize=7.3,
                               leading=10.4, backColor=colors.HexColor("#f1f5f8"),
                               borderPadding=7, spaceAfter=10),
        "list": ParagraphStyle("list", parent=body, leftIndent=12, firstLineIndent=-10),
        "quote": ParagraphStyle("quote", parent=body, leftIndent=12,
                                textColor=colors.HexColor("#617485")),
    }


def split_table_row(line: str) -> list[str]:
    """Do not split escaped pipes or pipes inside inline code."""
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    cells, cell = [], []
    escaped = in_code = False
    for char in value:
        if escaped:
            cell.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "`":
            in_code = not in_code
            cell.append(char)
        elif char == "|" and not in_code:
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(char)
    if escaped:
        cell.append("\\")
    cells.append("".join(cell).strip())
    return cells


def table_flowable(rows: list[list[str]], width: float):
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise ValueError("Markdown table rows have inconsistent column counts")
    if columns == 1:
        widths = [width]
    elif columns == 2:
        widths = [width * .26, width * .74]
    elif columns >= 7:
        # Long variant names receive a wider first column and can break at
        # any character. Numeric metric columns stay uniformly readable.
        first = width * .24
        widths = [first] + [(width - first) / (columns - 1)] * (columns - 1)
    else:
        first = width * .25
        widths = [first] + [(width - first) / (columns - 1)] * (columns - 1)
    font_size = 7.0 if columns >= 9 else 7.5 if columns >= 7 else 8.
    cell_style = ParagraphStyle("cell", fontName="Latin", fontSize=font_size,
                                leading=font_size * 1.45, textColor=INK,
                                wordWrap="CJK", splitLongWords=True, spaceAfter=0)
    head_style = ParagraphStyle("head", parent=cell_style, textColor=colors.white)
    cells = [[Paragraph(markup(cell, code_size=font_size), head_style if index == 0 else cell_style)
              for cell in row] for index, row in enumerate(rows)]
    table = LongTable(cells, colWidths=widths, repeatRows=1, hAlign="LEFT",
                      splitByRow=1, splitInRow=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f7fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#d5e1e9")),
    ]))
    return table


def is_block_start(line: str) -> bool:
    return bool(re.match(r"\s*(?:#{1,6}\s|\||```|!\[|[-*]\s|\d+[.)]\s|>)", line))


def parse_report(source: Path, text: str, report_styles: dict) -> list:
    lines = text.splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                raw = lines[index].expandtabs(4)
                leading = len(raw) - len(raw.lstrip(" "))
                code.append("&nbsp;" * leading + plain(raw[leading:]))
                index += 1
            if index >= len(lines):
                raise ValueError("Unclosed Markdown code fence")
            story.append(Paragraph("<br/>".join(code) or "&nbsp;", report_styles["code"]))
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = split_table_row(lines[index])
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row):
                    rows.append(row)
                index += 1
            if rows:
                story.extend([table_flowable(rows, CONTENT_WIDTH), Spacer(1, 10)])
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            path = Path(image.group(2))
            if not path.is_absolute():
                path = source.parent / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Report image not found: {path}")
            figure = Image(str(path))
            factor = min(CONTENT_WIDTH / figure.imageWidth, 490 / figure.imageHeight)
            figure.drawWidth = figure.imageWidth * factor
            figure.drawHeight = figure.imageHeight * factor
            figure.hAlign = "LEFT"
            story.append(KeepTogether([figure, Spacer(1, 5),
                                      Paragraph(markup(image.group(1)), report_styles["caption"]),
                                      Spacer(1, 8)]))
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2)
            style = report_styles["title" if level == 1 else "h2" if level == 2 else "h3"]
            story.append(Paragraph(markup(title), style))
        elif re.fullmatch(r"[-*_]{3,}", line):
            story.append(Spacer(1, 7))
        elif line.startswith(">"):
            story.append(Paragraph(markup(line.lstrip(">").strip()), report_styles["quote"]))
        else:
            content = [line]
            list_item = bool(re.match(r"^(?:[-*]\s|\d+[.)]\s)", line))
            while index + 1 < len(lines) and lines[index + 1].strip() and not is_block_start(lines[index + 1]):
                index += 1
                content.append(lines[index].strip())
            story.append(Paragraph(markup(" ".join(content)), report_styles["list" if list_item else "body"]))
        index += 1
    return story


def page_decoration(header_text: str, footer_text: str, report_styles: dict):
    style = report_styles["caption"]

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d9e4eb"))
        canvas.line(MARGIN, PAGE_HEIGHT - 38, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 38)
        header = Paragraph(markup(header_text, code_size=7), style)
        _, height = header.wrap(CONTENT_WIDTH, 20)
        header.drawOn(canvas, MARGIN, PAGE_HEIGHT - 33)
        footer = Paragraph(markup(footer_text, code_size=7), style)
        _, height = footer.wrap(CONTENT_WIDTH - 25, 24)
        footer.drawOn(canvas, MARGIN, max(18, 32 - height))
        canvas.setFont("Latin", 8)
        canvas.setFillColor(colors.HexColor("#617485"))
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 22, str(doc.page))
        canvas.restoreState()

    return decorate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--footer", help="Factual footer text; default shows source filename only")
    parser.add_argument("--header", default="InitDiffusion | Scratch component-loss search")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf extension")
    text = source.read_text()
    register_fonts()
    report_styles = styles()
    story = parse_report(source, text, report_styles)
    if not story:
        parser.error("Report source is empty")
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), source.stem)
    footer = args.footer or f"Source: {source.name}"
    output.parent.mkdir(parents=True, exist_ok=True)
    decoration = page_decoration(args.header, footer, report_styles)
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                                 topMargin=53, bottomMargin=49, title=title,
                                 author="sim experiment analysis", allowSplitting=True)
    document.build(story, onFirstPage=decoration, onLaterPages=decoration)
    print(output)


if __name__ == "__main__":
    main()
