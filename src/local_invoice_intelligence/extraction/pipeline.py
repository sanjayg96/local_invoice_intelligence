from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from local_invoice_intelligence.config import MODEL_PROVIDER, OCR_BACKEND, TEXT_MODEL
from local_invoice_intelligence.extraction.fields import FieldDefinition, field_prompt, schema_for_fields
from local_invoice_intelligence.extraction.validation import validate_values
from local_invoice_intelligence.models.ollama_provider import run_ollama_json
from local_invoice_intelligence.models.openai_provider import run_openai_json
from local_invoice_intelligence.ocr.transcript import extract_text_hybrid


SYSTEM_PROMPT = (
    "You are a careful document extraction engine. Extract only the user-requested fields "
    "from the supplied PDF transcript. The transcript preserves page boundaries and approximate "
    "layout. Return JSON only. Use null when a value is absent or not supported by the document. "
    "Do not invent values. Prefer exact visible text over normalized guesses."
)


def extract_document(
    *,
    pdf_path: str | Path,
    fields: list[FieldDefinition],
    model: str = TEXT_MODEL,
    provider: str = MODEL_PROVIDER,
    ocr_backend: str = OCR_BACKEND,
    ollama_thinking: str | bool | None = None,
    include_transcript: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    pdf_path = Path(pdf_path)
    transcript = extract_text_hybrid(pdf_path, ocr_backend=ocr_backend)
    schema = schema_for_fields(fields)
    user_prompt = (
        f"FIELDS TO EXTRACT:\n{field_prompt(fields)}\n\n"
        f"DOCUMENT TRANSCRIPT:\n{transcript}"
    )
    normalized_provider = provider.lower()
    if normalized_provider == "ollama":
        model_result = run_ollama_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=schema,
            thinking=ollama_thinking,
        )
    elif normalized_provider == "openai":
        model_result = run_openai_json(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=schema,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    values = {field.name: model_result["values"].get(field.name) for field in fields}
    validation, warnings = validate_values(values, fields)
    metadata = {
        **model_result.get("provider_metadata", {}),
        "source_file": str(pdf_path),
        "latency_s": round(time.perf_counter() - started, 4),
        "ocr_backend": ocr_backend,
        "warnings": warnings,
        "transcript_chars": len(transcript),
    }
    result = {
        "source_file": str(pdf_path),
        "fields": values,
        "validation": validation,
        "metadata": metadata,
    }
    if include_transcript:
        result["transcript"] = transcript
    return result
