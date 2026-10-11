"""Make numbered raster contact sheets for figure/PDF-page visual review.

This helper reads existing images only; it does not create or modify a PDF.
Contact sheets show page flow. Inspect each full-resolution source separately
before declaring a report visually verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont, ImageOps


def natural_key(path):
    return tuple(int(part) if part.isdigit() else part.lower()
                 for part in re.split(r"(\d+)", str(path)))


def select_images(images, input_dir=None, pattern="*.png"):
    """Keep explicit input order; naturally sort a directory's numbered pages."""
    paths = [Path(path).resolve() for path in images]
    if input_dir is not None:
        directory = Path(input_dir).resolve()
        if not directory.is_dir():
            raise ValueError(f"Input directory does not exist: {directory}")
        paths.extend(sorted((path.resolve() for path in directory.glob(pattern) if path.is_file()),
                            key=natural_key))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No source images selected")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Source image does not exist: {path}")
    return paths


def load_font(size, path=None):
    font = ImageFont.truetype(str(path) if path else "DejaVuSans.ttf", size=size)
    return font


def text_width(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def fit_label(draw, text, font, width, max_lines=2):
    """Wrap long filenames without allowing a label to cross cell boundaries."""
    lines, remaining = [], text
    while remaining and len(lines) < max_lines:
        split = len(remaining)
        while split > 0 and text_width(draw, remaining[:split], font) > width:
            split -= 1
        if split == 0:
            raise ValueError("Cell is too narrow for the requested label font")
        line, remaining = remaining[:split], remaining[split:]
        if len(lines) == max_lines - 1 and remaining:
            while line and text_width(draw, line + "…", font) > width:
                line = line[:-1]
            line += "…"
            remaining = ""
        lines.append(line)
    return lines


def inventory(paths):
    sources = []
    for ordinal, path in enumerate(paths, start=1):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            if not width or not height:
                raise ValueError(f"Source image has no pixels: {path}")
            source_format, mode = source.format, source.mode
        sources.append({"ordinal": ordinal, "path": str(path), "filename": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "width": width, "height": height, "format": source_format, "mode": mode})
    return sources


def render_montages(paths, output_dir, *, prefix="montage", columns=3, rows=3,
                    cell_width=550, cell_height=760, gap=18, font_size=20,
                    title="Raster visual overview", font_path=None, upscale=False):
    if not isinstance(prefix, str) or Path(prefix).name != prefix or prefix in ("", ".", ".."):
        raise ValueError("Prefix must be a safe basename")
    if min(columns, rows) < 1 or min(cell_width, cell_height) < 100 or gap < 1 or font_size < 10:
        raise ValueError("Positive rows/columns, cell dimensions >=100, gap >=1, font size >=10 required")
    paths = [Path(path).resolve() for path in paths]
    if not paths:
        raise ValueError("No source images selected")
    sources = inventory(paths)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(font_size, font_path)
    heading_font = load_font(font_size + 6, font_path)
    label_height = 2 * (font_size + 6) + gap
    image_height = cell_height - label_height
    if image_height < 50 or cell_width <= 2 * gap or image_height <= 2 * gap:
        raise ValueError("Cell height does not leave enough space above the label")
    header_height, footer_height = font_size + 6 + 2 * gap, font_size + 2 * gap
    per_sheet = columns * rows
    sheet_count = math.ceil(len(sources) / per_sheet)
    sheets = []
    for sheet_number in range(1, sheet_count + 1):
        cells = sources[(sheet_number - 1) * per_sheet:sheet_number * per_sheet]
        used_rows = math.ceil(len(cells) / columns)
        sheet_width = columns * cell_width + (columns + 1) * gap
        sheet_height = header_height + used_rows * cell_height + (used_rows + 1) * gap + footer_height
        sheet_path = output_dir / f"{prefix}_{sheet_number:02d}.png"
        if sheet_path.resolve() in paths:
            raise ValueError(f"Output would overwrite a source: {sheet_path}")
        sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
        draw = ImageDraw.Draw(sheet)
        heading = f"{title} — sheet {sheet_number}/{sheet_count}"
        heading = fit_label(draw, heading, heading_font, sheet_width - 2 * gap, max_lines=1)[0]
        draw.text((gap, gap), heading, font=heading_font, fill="#222222")
        placements = []
        for index, source in enumerate(cells):
            column, row = index % columns, index // columns
            x = gap + column * (cell_width + gap)
            y = header_height + gap + row * (cell_height + gap)
            draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), fill="#f6f7f8", outline="#d6d9de")
            with Image.open(source["path"]) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGBA")
                scale = min((cell_width - 2 * gap) / image.width, (image_height - 2 * gap) / image.height)
                if not upscale:
                    scale = min(scale, 1.0)
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
                image_x = x + (cell_width - image.width) // 2
                image_y = y + (image_height - image.height) // 2
                sheet.paste(image, (image_x, image_y), image)
            label = f"{source['ordinal']:03d}  {source['filename']}"
            labels = fit_label(draw, label, font, cell_width - 2 * gap)
            for line_index, line in enumerate(labels):
                draw.text((x + gap, y + image_height + line_index * (font_size + 6)),
                          line, font=font, fill="#222222")
            placements.append({"ordinal": source["ordinal"], "source_path": source["path"],
                               "cell_xyxy": [x, y, x + cell_width, y + cell_height],
                               "image_xyxy": [image_x, image_y, image_x + size[0], image_y + size[1]],
                               "rendered_labels": labels, "source_aspect_ratio": source["width"] / source["height"],
                               "source_pixels_cropped": False})
        footer = "Overview only: inspect every full-resolution source separately."
        footer = fit_label(draw, footer, font, sheet_width - 2 * gap, max_lines=1)[0]
        draw.text((gap, sheet_height - footer_height + gap // 2), footer, font=font, fill="#555555")
        sheet.save(sheet_path)
        sheets.append({"path": str(sheet_path), "sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(),
                       "width": sheet_width, "height": sheet_height, "placements": placements})
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "Raster contact sheets for visual QA; no PDF creation or completion claim",
                "full_resolution_individual_inspection_required": True,
                "sources": sources, "sheets": sheets,
                "layout": {"columns": columns, "max_rows_per_sheet": rows, "cell_width": cell_width,
                           "cell_height": cell_height, "gap": gap, "font_size": font_size,
                           "font_path": str(getattr(font, "path", font_path)), "upscale": upscale,
                           "preserve_aspect_ratio": True, "crop_sources": False}}
    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="montage")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cell-width", type=int, default=550)
    parser.add_argument("--cell-height", type=int, default=760)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--title", default="Raster visual overview")
    parser.add_argument("--upscale", action="store_true")
    args = parser.parse_args()
    try:
        paths = select_images(args.images, args.input_dir, args.pattern)
        # Prevent a refresh from treating its prior sheets as fresh source pages.
        if args.input_dir and args.input_dir.resolve() == args.output_dir.resolve():
            paths = [path for path in paths if not re.fullmatch(re.escape(args.prefix) + r"_\d+\.png", path.name)]
        result = render_montages(paths, args.output_dir, prefix=args.prefix, columns=args.columns,
                                 rows=args.rows, cell_width=args.cell_width, cell_height=args.cell_height,
                                 gap=args.gap, font_size=args.font_size, title=args.title,
                                 font_path=args.font, upscale=args.upscale)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"source_count": len(result["sources"]), "sheets": [sheet["path"] for sheet in result["sheets"]],
                      "full_resolution_inspection_still_required": True}, indent=2))


if __name__ == "__main__":
    main()
