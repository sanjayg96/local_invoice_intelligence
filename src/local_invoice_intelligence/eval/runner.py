from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from local_invoice_intelligence.config import MODEL_PROVIDER, OCR_BACKEND, TEXT_MODEL
from local_invoice_intelligence.eval.adapters import ADAPTERS
from local_invoice_intelligence.eval.scoring import summarize
from local_invoice_intelligence.extraction.fields import FieldDefinition
from local_invoice_intelligence.extraction.pipeline import extract_document


def run_eval(
    *,
    dataset: str,
    dataset_root: str | Path,
    fields: list[FieldDefinition],
    model: str = TEXT_MODEL,
    provider: str = MODEL_PROVIDER,
    ocr_backend: str = OCR_BACKEND,
    ollama_thinking: str | bool | None = None,
) -> dict[str, Any]:
    adapter = ADAPTERS[dataset]
    records = []
    started = time.perf_counter()
    documents = list(adapter.iter_documents(dataset_root, fields))
    for document in tqdm(documents, desc=f"Evaluating {dataset}"):
        result = extract_document(
            pdf_path=document.pdf_path,
            fields=fields,
            model=model,
            provider=provider,
            ocr_backend=ocr_backend,
            ollama_thinking=ollama_thinking,
        )
        records.append(
            {
                "doc_id": document.doc_id,
                "source_file": str(document.pdf_path),
                "ground_truth": document.ground_truth,
                "predicted": result["fields"],
                "metadata": result["metadata"],
                "validation": result["validation"],
            }
        )
    return {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "model": model,
        "provider": provider,
        "ocr_backend": ocr_backend,
        "field_definitions": [field.__dict__ for field in fields],
        "summary": summarize(records, fields),
        "latency_s": round(time.perf_counter() - started, 4),
        "records": records,
    }
