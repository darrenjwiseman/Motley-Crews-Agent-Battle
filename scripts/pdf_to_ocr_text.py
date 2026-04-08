#!/usr/bin/env python3
"""
Render PDF pages to images and OCR with Tesseract.
Writes per-page text plus a combined file with page markers.

Usage:
  python scripts/pdf_to_ocr_text.py [--pdf PATH] [--out-dir DIR] [--dpi N] [--lang eng]

Requires: tesseract on PATH (brew install tesseract).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def normalize_whitespace(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Collapse excessive blank lines
    out: list[str] = []
    for ln in lines:
        if not ln.strip() and out and out[-1] == "":
            continue
        out.append(ln)
    return "\n".join(out).strip() + "\n"


def safe_hp_fixes(text: str) -> str:
    """Conservative OCR touch-ups; do not rewrite rule prose."""
    # Normalize spaced digits after HP when clearly numeric
    return re.sub(r"\bHP\s+([O0])(?=\b)", "HP 0", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR a PDF to per-page text files.")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Rules_Zine.pdf",
        help="Path to Rules_Zine.pdf",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "rules" / "ocr_raw",
        help="Output directory for page_N.txt and combined.txt",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Render DPI for OCR")
    parser.add_argument("--lang", default="eng", help="Tesseract language(s)")
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (6 = uniform block)",
    )
    args = parser.parse_args()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Missing PyMuPDF. Install: pip install -r requirements-rules.txt", file=sys.stderr)
        return 1

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("Missing pytesseract/Pillow. Install: pip install -r requirements-rules.txt", file=sys.stderr)
        return 1

    # Common Homebrew locations when `tesseract` is not on default PATH
    if not shutil.which("tesseract"):
        for candidate in (
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
        ):
            if Path(candidate).is_file():
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
        else:
            print(
                "tesseract not found. Install: brew install tesseract (macOS)",
                file=sys.stderr,
            )
            return 1

    if not args.pdf.is_file():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    combined_parts: list[str] = []
    zoom = args.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")

        ocr_config = f"--psm {args.psm}"
        raw = pytesseract.image_to_string(img, lang=args.lang, config=ocr_config)
        cleaned = normalize_whitespace(safe_hp_fixes(raw))

        page_no = i + 1
        marker = f"\n{'=' * 72}\n-- PAGE {page_no} of {len(doc)} --\n{'=' * 72}\n\n"
        combined_parts.append(marker + cleaned)

        page_path = args.out_dir / f"page_{page_no:02d}.txt"
        page_path.write_text(cleaned, encoding="utf-8")

    doc.close()

    combined = "".join(combined_parts)
    (args.out_dir / "combined.txt").write_text(combined, encoding="utf-8")

    print(f"Wrote {len(combined_parts)} page file(s) and combined.txt under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
