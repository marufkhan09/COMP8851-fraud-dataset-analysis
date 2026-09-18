"""Combine a folder of screenshots into one PDF.

    python3 scripts/screenshots_to_pdf.py screenshots/ --out DELIVERABLE/screenshots.pdf

One screenshot per page, in filename order, each fitted to A4 with a caption
naming the source file. Numeric parts of filenames sort numerically, so
``shot_2.png`` comes before ``shot_10.png`` rather than after it.

Needs only Pillow.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# A4 at 150 dpi, which keeps terminal text readable without a huge file.
PAGE_W, PAGE_H = 1240, 1754
MARGIN = 60
CAPTION_BAND = 46


def natural_key(path: Path):
    """Sort so shot_2 precedes shot_10."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def find_images(folder: Path) -> List[Path]:
    images = [p for p in folder.rglob("*")
              if p.is_file() and p.suffix.lower() in EXTENSIONS]
    return sorted(images, key=natural_key)


def build_pdf(folder: Path, out_path: Path, title: str | None = None) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise SystemExit("Pillow is required:  pip install Pillow")

    folder = Path(folder)
    out_path = Path(out_path)
    images = find_images(folder)
    if not images:
        raise SystemExit(f"No images found in {folder}")

    try:
        caption_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 46)
    except OSError:
        try:
            caption_font = ImageFont.truetype("arial.ttf", 20)
            title_font = ImageFont.truetype("arial.ttf", 46)
        except OSError:
            caption_font = ImageFont.load_default()
            title_font = ImageFont.load_default()

    pages = []

    if title:
        cover = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(cover)
        draw.text((MARGIN, PAGE_H // 3), title, fill="black", font=title_font)
        draw.text((MARGIN, PAGE_H // 3 + 80),
                  f"{len(images)} screenshot{'s' if len(images) != 1 else ''}",
                  fill="#555555", font=caption_font)
        draw.line([(MARGIN, PAGE_H // 3 + 130), (PAGE_W - MARGIN, PAGE_H // 3 + 130)],
                  fill="#cccccc", width=2)
        pages.append(cover)

    for index, image_path in enumerate(images, start=1):
        try:
            with Image.open(image_path) as source:
                source.load()
                # Flatten transparency onto white; PDF has no alpha channel.
                if source.mode in ("RGBA", "LA", "P"):
                    source = source.convert("RGBA")
                    flat = Image.new("RGB", source.size, "white")
                    flat.paste(source, mask=source.split()[-1])
                    shot = flat
                else:
                    shot = source.convert("RGB")

                box_w = PAGE_W - 2 * MARGIN
                box_h = PAGE_H - 2 * MARGIN - CAPTION_BAND
                scale = min(box_w / shot.width, box_h / shot.height)
                if scale < 1:
                    shot = shot.resize(
                        (max(1, int(shot.width * scale)),
                         max(1, int(shot.height * scale))),
                        Image.LANCZOS)

                page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
                page.paste(shot, ((PAGE_W - shot.width) // 2,
                                  MARGIN + (box_h - shot.height) // 2))

                draw = ImageDraw.Draw(page)
                try:
                    label = str(image_path.relative_to(folder))
                except ValueError:
                    label = image_path.name
                draw.line([(MARGIN, PAGE_H - MARGIN - 28),
                           (PAGE_W - MARGIN, PAGE_H - MARGIN - 28)],
                          fill="#dddddd", width=1)
                draw.text((MARGIN, PAGE_H - MARGIN - 20),
                          f"{index}.  {label}", fill="#444444", font=caption_font)
                pages.append(page)
        except Exception as error:          # a corrupt file must not lose the rest
            print(f"  skipped {image_path.name}: {error}")

    if not pages:
        raise SystemExit("No readable images")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out_path, "PDF", resolution=150.0, save_all=True,
                  append_images=pages[1:])
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine screenshots into one PDF")
    parser.add_argument("folder", help="Folder containing the screenshots")
    parser.add_argument("--out", default="screenshots.pdf")
    parser.add_argument("--title", default=None,
                        help="Add a cover page with this title")
    args = parser.parse_args(argv)

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return 1

    images = find_images(folder)
    print(f"Found {len(images)} image(s) in {folder}")
    for image in images:
        print(f"   {image.name}")

    pdf = build_pdf(folder, Path(args.out), args.title)
    size = pdf.stat().st_size / 1024
    print(f"\nWrote {pdf}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
