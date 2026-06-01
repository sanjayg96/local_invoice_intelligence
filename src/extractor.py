import fitz
import numpy as np
import ollama
import pytesseract
import pdfplumber
from PIL import Image
from config import MODEL_PROVIDER, OCR_BACKEND, OPENAI_MODEL, TEXT_MODEL
from ollama_controls import maybe_add_thinking_directive, primary_chat_controls
from openai_provider import run_openai_extraction

# --- APPLE SILICON TESSERACT PATH ---
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract' 

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


def _run_tesseract_ocr(img: Image.Image) -> str:
    return pytesseract.image_to_string(img)


def _rapidocr_engine():
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "RapidOCR is not installed. Run `uv sync` or use `--ocr-backend tesseract`."
            ) from exc
        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def _box_left(box) -> float:
    return min(point[0] for point in box)


def _box_top(box) -> float:
    return min(point[1] for point in box)


def _box_height(box) -> float:
    ys = [point[1] for point in box]
    return max(ys) - min(ys)


def _format_rapidocr_output(result) -> str:
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or txts is None:
        return ""

    entries = []
    for idx, text in enumerate(txts):
        cleaned = str(text or "").strip()
        if not cleaned:
            continue
        box = boxes[idx]
        score = float(scores[idx]) if scores is not None else 1.0
        entries.append(
            {
                "text": cleaned,
                "left": _box_left(box),
                "top": _box_top(box),
                "height": max(_box_height(box), 1.0),
                "score": score,
            }
        )

    if not entries:
        return ""

    entries.sort(key=lambda item: (item["top"], item["left"]))
    lines = []
    current_line = []
    current_top = None
    current_height = None

    for entry in entries:
        if current_top is None:
            current_line = [entry]
            current_top = entry["top"]
            current_height = entry["height"]
            continue

        line_threshold = max(current_height or 1.0, entry["height"]) * 0.65
        if abs(entry["top"] - current_top) <= line_threshold:
            current_line.append(entry)
            current_top = (current_top + entry["top"]) / 2
            current_height = max(current_height or 1.0, entry["height"])
        else:
            lines.append(current_line)
            current_line = [entry]
            current_top = entry["top"]
            current_height = entry["height"]

    if current_line:
        lines.append(current_line)

    formatted = []
    for line in lines:
        ordered = sorted(line, key=lambda item: item["left"])
        formatted.append("  ".join(item["text"] for item in ordered))
    return "\n".join(formatted)


def _run_rapidocr(img: Image.Image) -> str:
    result = _rapidocr_engine()(np.array(img))
    return _format_rapidocr_output(result)


def _run_ocr(img: Image.Image, backend: str) -> str:
    normalized = backend.lower()
    if normalized == "tesseract":
        return _run_tesseract_ocr(img)
    if normalized == "rapidocr":
        return _run_rapidocr(img)
    raise ValueError(f"Unsupported OCR backend: {backend}")


def extract_text_hybrid(pdf_path: str, ocr_backend: str | None = None) -> str:
    """
    The 'Hybrid Eyes'. Uses pdfplumber for perfect 2D spatial text extraction.
    Falls back to PyMuPDF + OCR on a per-page basis if it's a flat scan.
    """
    backend = ocr_backend or OCR_BACKEND
    master_text = ""
    
    # Open with both libraries (pdfplumber for text layout, fitz for fast image rendering)
    doc_plumber = pdfplumber.open(pdf_path)
    doc_fitz = fitz.open(pdf_path)
    
    for page_num in range(len(doc_plumber.pages)):
        page_plumber = doc_plumber.pages[page_num]
        
        # 1. The Fast Path: Born-Digital Layout Extraction
        # This is the command that pads text with spaces to match the physical layout
        text = page_plumber.extract_text(layout=True)
        
        # Clean up in case the page is completely devoid of digital text
        if text:
            text = text.strip()
        else:
            text = ""
        
        # 2. The Validation Gate: Is this a flat scanned image?
        if len(text) < 50: 
            # print(f"  [Fallback] Page {page_num + 1} is a flat scan. Triggering OCR...")
            img = _render_page_image(doc_fitz, page_num)
            text = _run_ocr(img, backend)
        
        master_text += f"\n--- START OF PAGE {page_num + 1} ---\n"
        master_text += text
        master_text += f"\n--- END OF PAGE {page_num + 1} ---\n"
            
    doc_plumber.close()
    doc_fitz.close()
    
    return master_text


def _schema_uses_scratchpad(schema) -> bool:
    return "reasoning_process" in getattr(schema, "model_fields", {})


def _auditor_system_prompt(schema) -> str:
    base_prompt = (
        "You are an expert financial data auditor. You will be provided with a complete, "
        "multi-page text extraction of an invoice. The text has been formatted to preserve "
        "the original physical layout of the document.\n\n"
        "Return only JSON matching the requested schema. Copy exact visible values when possible. "
        "For totals, choose the final total due or gross amount, not a subtotal, tax, unit price, "
        "or line-item amount."
    )
    if not _schema_uses_scratchpad(schema):
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "CRITICAL INSTRUCTION FOR `reasoning_process`: This field is a brief scratchpad. "
        "Use it to verify the vendor, issue date, and final total before populating the JSON fields."
    )

def run_2_step_extraction(
    pdf_path: str,
    schema,
    ocr_backend: str | None = None,
    ollama_thinking: str | bool | None = None,
    model: str = TEXT_MODEL,
    provider: str = MODEL_PROVIDER,
):
    """
    Step 1 (Perception): Hybrid router extracts text via pdfplumber or Tesseract.
    Step 2 (Analysis): Llama 3.1 reads the text and enforces the JSON schema.
    """
    
    # 1. Extract the text securely, regardless of document origin
    master_transcription = extract_text_hybrid(pdf_path, ocr_backend=ocr_backend)
    # print(pdf_path)
    # print(master_transcription)
    
    # 2. The Auditor
    system_prompt = _auditor_system_prompt(schema)
    normalized_provider = provider.lower()

    if normalized_provider == "openai":
        return run_openai_extraction(
            transcription=master_transcription,
            schema=schema,
            system_prompt=system_prompt,
            model=model or OPENAI_MODEL,
        )

    if normalized_provider != "ollama":
        raise ValueError(f"Unsupported model provider: {provider}")
    
    llm_response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": maybe_add_thinking_directive(
                    f"DOCUMENT TRANSCRIPTION:\n{master_transcription}",
                    thinking=ollama_thinking,
                ),
            }
        ],
        format=schema.model_json_schema(),
        **primary_chat_controls(thinking=ollama_thinking),
    )
    
    return llm_response
