from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

from local_invoice_intelligence.config import OCR_BACKEND

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

_RAPIDOCR_ENGINE = None


def _render_page_image(doc_fitz, page_num: int, dpi: int = 300) -> Image.Image:
    page_fitz = doc_fitz.load_page(page_num)
    pix = page_fitz.get_pixmap(dpi=dpi)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    return img


def _rapidocr_engine():
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError("RapidOCR is not installed. Use `--ocr-backend tesseract`.") from exc
        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def _format_rapidocr_output(result) -> str:
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    if boxes is None or txts is None:
        return ""
    entries = []
    for idx, text in enumerate(txts):
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        box = boxes[idx]
        entries.append(
            {
                "text": cleaned,
                "left": min(point[0] for point in box),
                "top": min(point[1] for point in box),
                "height": max(max(point[1] for point in box) - min(point[1] for point in box), 1.0),
            }
        )
    entries.sort(key=lambda item: (item["top"], item["left"]))
    lines: list[list[dict]] = []
    for entry in entries:
        if not lines:
            lines.append([entry])
            continue
        current = lines[-1]
        threshold = max(current[-1]["height"], entry["height"]) * 0.65
        if abs(entry["top"] - current[-1]["top"]) <= threshold:
            current.append(entry)
        else:
            lines.append([entry])
    return "\n".join("  ".join(item["text"] for item in sorted(line, key=lambda item: item["left"])) for line in lines)


def _run_ocr(img: Image.Image, backend: str) -> str:
    normalized = backend.lower()
    if normalized == "tesseract":
        return pytesseract.image_to_string(img)
    if normalized == "rapidocr":
        return _format_rapidocr_output(_rapidocr_engine()(np.array(img)))
    raise ValueError(f"Unsupported OCR backend: {backend}")


def extract_text_hybrid(pdf_path: str | Path, ocr_backend: str | None = None) -> str:
    """Extract a layout-preserving transcript, using OCR only for weak text pages."""
    backend = ocr_backend or OCR_BACKEND
    pdf_path = str(pdf_path)
    parts = []
    with pdfplumber.open(pdf_path) as doc_plumber, fitz.open(pdf_path) as doc_fitz:
        for page_num, page_plumber in enumerate(doc_plumber.pages):
            text = (page_plumber.extract_text(layout=True) or "").strip()
            if len(text) < 50:
                text = _run_ocr(_render_page_image(doc_fitz, page_num), backend).strip()
            parts.append(f"--- START OF PAGE {page_num + 1} ---\n{text}\n--- END OF PAGE {page_num + 1} ---")
    return "\n\n".join(parts)
