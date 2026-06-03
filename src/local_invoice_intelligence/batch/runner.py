from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from local_invoice_intelligence.config import MODEL_PROVIDER, OCR_BACKEND, TEXT_MODEL
from local_invoice_intelligence.export.excel import write_results_xlsx
from local_invoice_intelligence.extraction.fields import FieldDefinition
from local_invoice_intelligence.extraction.pipeline import extract_document


def extract_folder(
    *,
    input_dir: str | Path,
    fields: list[FieldDefinition],
    out_path: str | Path,
    artifacts_dir: str | Path | None = None,
    model: str = TEXT_MODEL,
    provider: str = MODEL_PROVIDER,
    ocr_backend: str = OCR_BACKEND,
    ollama_thinking: str | bool | None = None,
) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")
    artifacts = Path(artifacts_dir) if artifacts_dir else None
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)

    results = []
    for pdf_path in tqdm(pdfs, desc="Extracting PDFs"):
        result = extract_document(
            pdf_path=pdf_path,
            fields=fields,
            model=model,
            provider=provider,
            ocr_backend=ocr_backend,
            ollama_thinking=ollama_thinking,
        )
        results.append(result)
        if artifacts:
            artifact_path = artifacts / f"{pdf_path.stem}.json"
            artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_results_xlsx(results, fields, out_path)
    return results
