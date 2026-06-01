from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

import ollama
from dateutil import parser as date_parser
from rapidfuzz import fuzz

from config import BASELINE_PLUS_PRIMARY_MODEL, BASELINE_PLUS_RESCUE_MODEL, ENABLE_THINKING
from extractor import extract_text_hybrid
from ollama_controls import (
    maybe_add_thinking_directive,
    primary_chat_controls,
    rescue_chat_controls,
    thinking_label,
)
from schema import InvoiceExtractionBase, InvoiceExtractionWithReasoning


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    description: str


DEFAULT_FIELD_SPECS: dict[str, FieldSpec] = {
    "vendor_name": FieldSpec(
        name="vendor_name",
        kind="organization",
        description="The vendor, supplier, seller, or billing entity name.",
    ),
    "vendor_address": FieldSpec(
        name="vendor_address",
        kind="address",
        description="The vendor, supplier, seller, physical, billing, or remittance address.",
    ),
    "amount_total_gross": FieldSpec(
        name="amount_total_gross",
        kind="money",
        description="The final gross invoice total, total due, amount due, balance due, or total payable.",
    ),
    "date_issue": FieldSpec(
        name="date_issue",
        kind="date",
        description="The invoice issue date, invoice date, bill date, statement date, printed date, or creation date.",
    ),
}

MONTH_RE = r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
DATE_RE = re.compile(
    rf"\b(?:\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|(?:{MONTH_RE})[a-z]*\.?\s*\d{{1,2}},?\s*\d{{2,4}}|(?:{MONTH_RE})[a-z]*\.?\s*\d{{1,2}}[/-]\d{{2,4}}|\d{{1,2}}\s*(?:{MONTH_RE})[a-z]*\.?,?\s*\d{{2,4}})\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(
    r"[$€£]?\s*\(?-?\d{1,3}(?:[,\s]\d{3})*(?:\s*\.\s*\d{2})?\)?-?|[$€£]?\s*\(?-?\d+\s*\.\s*\d{2}\)?-?"
)
ADDRESS_RE = re.compile(
    r"\b(po\s*box|p\.?\s*o\.?\s*box|street|st\.?|avenue|ave\.?|road|rd\.?|suite|ste\.?|floor|fl\.?|blvd|boulevard|\d{5}(?:-\d{4})?)\b",
    re.IGNORECASE,
)
PROSE_RE = re.compile(
    r"\b(not explicitly|not stated|unknown|cannot determine|appears to be|likely|inferred|based on|there are|the total|the date)\b",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_compact(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(value or "")).lower()


def parse_amount(value: Any) -> Optional[Decimal]:
    raw = normalize_text(value)
    if not raw:
        return None
    negative = raw.endswith("-") or (raw.startswith("(") and raw.endswith(")"))
    cleaned = raw.replace(",", "")
    cleaned = re.sub(r"(?<=\d)\s+\.(?=\d{1,2}\b)", ".", cleaned)
    cleaned = re.sub(r"(?<=\d)\s+(?=\d{2}\b)", ".", cleaned)
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned).rstrip("-")
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        parsed = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -parsed if negative and parsed > 0 else parsed


def normalize_amount(value: Any) -> str:
    parsed = parse_amount(value)
    return f"{parsed:.2f}" if parsed is not None else ""


def _expand_compact_month(value: str) -> str:
    return re.sub(
        rf"\b({MONTH_RE})(\d{{1,2}})([/-]\d{{2,4}})\b",
        lambda match: f"{match.group(1)} {match.group(2)}{match.group(3)}",
        value,
        flags=re.IGNORECASE,
    )


def parse_date(value: Any):
    raw = str(value or "")
    raw = re.sub(r"\b(?:EDT|EST|ET|CDT|CST|CT|MDT|MST|MT|PDT|PST|PT|UTC|GMT)\b", " ", raw)
    raw = re.sub(r"(?<=\d)[Il|](?=\d)", "1", raw)
    raw = re.sub(r"(?<=[A-Za-z])[Il|](?=[/-])", "1", raw)
    raw = re.sub(r"(?<=\d)[oO](?=\d)", "0", raw)
    try:
        parsed = date_parser.parse(_expand_compact_month(raw), fuzzy=True).date()
    except Exception:
        return None
    if parsed.year < 1900 or parsed.year > 2100:
        return None
    return parsed


def normalize_date(value: Any) -> str:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else ""


def normalize_for_kind(value: Any, kind: str) -> str:
    if kind == "money":
        return normalize_amount(value)
    if kind == "date":
        return normalize_date(value)
    return normalize_text(value)


def resolve_specs(fields: Optional[list[str | FieldSpec]] = None) -> dict[str, FieldSpec]:
    if fields is None:
        return dict(DEFAULT_FIELD_SPECS)
    specs = {}
    for field in fields:
        if isinstance(field, FieldSpec):
            specs[field.name] = field
        elif field in DEFAULT_FIELD_SPECS:
            specs[str(field)] = DEFAULT_FIELD_SPECS[str(field)]
        else:
            field_name = str(field)
            specs[field_name] = FieldSpec(field_name, "text", f"User-requested field named {field_name}.")
    return specs


def _json_from_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def _baseline_schema():
    return InvoiceExtractionWithReasoning if ENABLE_THINKING else InvoiceExtractionBase


def _baseline_system_prompt(schema) -> str:
    base_prompt = (
        "You are an expert financial data auditor. You will be provided with a complete, "
        "multi-page text extraction of an invoice. The text has been formatted to preserve "
        "the original physical layout of the document.\n\n"
        "Return only JSON matching the requested schema. Copy exact visible values when possible. "
        "For totals, choose the final total due or gross amount, not a subtotal, tax, unit price, "
        "or line-item amount."
    )
    if "reasoning_process" not in getattr(schema, "model_fields", {}):
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "CRITICAL INSTRUCTION FOR `reasoning_process`: This field is a brief scratchpad. "
        "Use it to verify the vendor, issue date, and final total before populating the JSON fields."
    )


def _run_baseline(
    transcription: str,
    model: str,
    ollama_thinking: str | bool | None = None,
) -> tuple[dict[str, Any], float]:
    schema = _baseline_schema()
    system_prompt = _baseline_system_prompt(schema)
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": maybe_add_thinking_directive(
                    f"DOCUMENT TRANSCRIPTION:\n{transcription}",
                    thinking=ollama_thinking,
                ),
            },
        ],
        format=schema.model_json_schema(),
        **primary_chat_controls(thinking=ollama_thinking),
        keep_alive="10m",
    )
    return _json_from_content(response["message"]["content"]), time.perf_counter() - started


def prediction_value(prediction: dict[str, Any], field_name: str) -> Any:
    if field_name == "amount_total_gross":
        return prediction.get("amount_total_gross") or prediction.get("invoice_total")
    return prediction.get(field_name)


def is_grounded(value: str, transcription: str, kind: str) -> bool:
    if not value:
        return False
    compact_value = normalize_compact(value)
    compact_doc = normalize_compact(transcription)
    if compact_value and compact_value in compact_doc:
        return True
    if kind == "money":
        normalized = normalize_amount(value)
        return bool(normalized and any(normalize_amount(match) == normalized for match in MONEY_RE.findall(transcription)))
    if kind == "date":
        normalized = normalize_date(value)
        return bool(normalized and any(normalize_date(match) == normalized for match in DATE_RE.findall(transcription)))
    return fuzz.partial_ratio(compact_value, compact_doc) >= 90


def validation_issue(value: Any, spec: FieldSpec, transcription: str) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return "blank"
    if PROSE_RE.search(text) or len(text.split()) > 24:
        return "prose_or_too_long"
    if spec.kind == "money" and parse_amount(text) is None:
        return "invalid_amount"
    if spec.kind == "date" and parse_date(text) is None:
        return "invalid_date"
    if spec.kind == "address" and not (ADDRESS_RE.search(text) or len(text.split()) >= 4):
        return "invalid_address"
    if not is_grounded(text, transcription, spec.kind):
        return "not_grounded"
    return None


def _page_from_marker(line: str, current_page: int) -> int:
    match = re.search(r"START OF PAGE\s+(\d+)", line)
    return int(match.group(1)) if match else current_page


def transcript_lines(transcription: str) -> list[tuple[int, str]]:
    lines = []
    page = 1
    for raw in transcription.splitlines():
        page = _page_from_marker(raw, page)
        text = raw.strip()
        if text:
            lines.append((page, text))
    return lines


def evidence_windows(transcription: str, spec: FieldSpec, limit: int = 8) -> list[dict[str, Any]]:
    lines = transcript_lines(transcription)
    query = normalize_text(f"{spec.name.replace('_', ' ')} {spec.kind} {spec.description}").lower()
    windows = []
    for idx, (page, _) in enumerate(lines):
        start = max(0, idx - 2)
        end = min(len(lines), idx + 3)
        text = "\n".join(line for _, line in lines[start:end])
        text_l = text.lower()
        score = fuzz.token_set_ratio(query, text_l)
        if spec.kind == "money" and MONEY_RE.search(text):
            score += 30
            if re.search(r"\b(total|amount due|balance due|payable|gross)\b", text_l):
                score += 35
            if re.search(r"\b(tax|discount|rate|unit|subtotal|paid)\b", text_l):
                score -= 20
        elif spec.kind == "date" and DATE_RE.search(text):
            score += 25
            if re.search(r"\b(inv\s*date|invoice\s*date|bill\s*date|statement\s*date|printed)\b", text_l):
                score += 35
            if re.search(r"\b(due|service|run|flight|ordered|period|payment)\b", text_l):
                score -= 18
        elif spec.kind == "address" and ADDRESS_RE.search(text):
            score += 25
            if re.search(r"\b(remit|payment|payable|mail\s+checks|pay\s+to)\b", text_l):
                score += 30
        elif spec.kind == "organization":
            if re.search(r"\b(remit|vendor|supplier|from|payable|seller)\b", text_l):
                score += 25
            if idx < 12:
                score += 12
        windows.append({"page": page, "text": normalize_text(text)[:1200], "score": round(score, 2)})
    windows.sort(key=lambda item: item["score"], reverse=True)

    deduped = []
    seen = set()
    for window in windows:
        key = normalize_compact(window["text"][:220])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)
        if len(deduped) >= limit:
            break
    return deduped


def _rescue_schema(field_names: list[str]) -> dict[str, Any]:
    result_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "evidence_quote": {"type": "string"},
            "page": {"type": "integer"},
            "reason": {"type": "string"},
        },
        "required": ["value", "evidence_quote", "page", "reason"],
    }
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": {field: result_schema for field in field_names},
                "required": field_names,
            }
        },
        "required": ["fields"],
    }


def _run_rescue(
    weak_specs: dict[str, FieldSpec],
    windows_by_field: dict[str, list[dict[str, Any]]],
    model: str,
    ollama_thinking: str | bool | None = None,
) -> tuple[dict[str, Any], float]:
    definitions = {
        name: {"kind": spec.kind, "description": spec.description}
        for name, spec in weak_specs.items()
    }
    prompt = (
        "Retry only the requested weak fields using the provided evidence windows.\n"
        "Return exact visible text only. If the evidence is insufficient, return an empty value.\n"
        "Do not compute, infer, or explain inside the value.\n\n"
        f"FIELDS:\n{json.dumps(definitions, indent=2)}\n\n"
        f"EVIDENCE WINDOWS:\n{json.dumps(windows_by_field, indent=2)}"
    )
    started = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a conservative extraction repair step. Only copy values from evidence."},
            {"role": "user", "content": maybe_add_thinking_directive(prompt, thinking=ollama_thinking)},
        ],
        format=_rescue_schema(list(weak_specs)),
        **rescue_chat_controls(thinking=ollama_thinking),
        keep_alive="10m",
    )
    return _json_from_content(response["message"]["content"]), time.perf_counter() - started


def field_record(
    value: Any,
    spec: FieldSpec,
    transcription: str,
    route: str,
    model: str,
    latency_s: float,
    reason: str = "",
    evidence_quote: str = "",
    page: Optional[int] = None,
) -> dict[str, Any]:
    issue = validation_issue(value, spec, transcription)
    text = normalize_text(value)
    return {
        "value": text if text else None,
        "normalized_value": normalize_for_kind(text, spec.kind),
        "kind": spec.kind,
        "valid": issue is None,
        "validation_issue": issue,
        "route": route,
        "model": model,
        "timings": {route: round(latency_s, 4)},
        "reason": reason,
        "evidence_quote": evidence_quote,
        "page": page,
    }


def extract_baseline_plus(
    pdf_path: str,
    fields: Optional[list[str | FieldSpec]] = None,
    primary_model: str = BASELINE_PLUS_PRIMARY_MODEL,
    rescue_model: str = BASELINE_PLUS_RESCUE_MODEL,
    ocr_backend: str | None = None,
    ollama_thinking: str | bool | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    specs = resolve_specs(fields)
    timings = {}
    warnings = []

    parse_started = time.perf_counter()
    transcription = extract_text_hybrid(pdf_path, ocr_backend=ocr_backend)
    timings["parse"] = time.perf_counter() - parse_started

    try:
        prediction, primary_latency = _run_baseline(transcription, primary_model, ollama_thinking)
        timings["primary_model"] = primary_latency
    except Exception as exc:
        warnings.append(f"primary extraction failed: {exc}")
        prediction = {}
        primary_latency = 0.0

    field_results = {}
    weak_fields = []
    for field_name, spec in specs.items():
        value = prediction_value(prediction, field_name)
        record = field_record(value, spec, transcription, "baseline_primary", primary_model, primary_latency)
        field_results[field_name] = record
        if record["validation_issue"]:
            weak_fields.append(field_name)

    windows_by_field = {}
    if weak_fields:
        weak_specs = {field_name: specs[field_name] for field_name in weak_fields}
        windows_by_field = {
            field_name: evidence_windows(transcription, specs[field_name])
            for field_name in weak_fields
        }
        try:
            rescue, rescue_latency = _run_rescue(
                weak_specs,
                windows_by_field,
                rescue_model,
                ollama_thinking,
            )
            timings["rescue_model"] = rescue_latency
            rescue_fields = rescue.get("fields", {})
            for field_name, spec in weak_specs.items():
                raw = rescue_fields.get(field_name, {})
                candidate = field_record(
                    raw.get("value"),
                    spec,
                    transcription,
                    "baseline_rescue",
                    rescue_model,
                    rescue_latency,
                    reason=normalize_text(raw.get("reason")),
                    evidence_quote=normalize_text(raw.get("evidence_quote")),
                    page=raw.get("page"),
                )
                if candidate["validation_issue"] is None:
                    field_results[field_name] = candidate
        except Exception as exc:
            warnings.append(f"rescue extraction failed: {exc}")

    return {
        "fields": field_results,
        "primary_prediction": prediction,
        "evidence_windows": windows_by_field,
        "metadata": {
            "pipeline": "baseline_plus",
            "latency_s": round(time.perf_counter() - started, 4),
            "timings": {key: round(value, 4) for key, value in timings.items()},
            "models": {"primary": primary_model, "rescue": rescue_model if weak_fields else None},
            "ocr_backend": ocr_backend,
            "ollama_thinking": thinking_label(ollama_thinking),
            "weak_fields": weak_fields,
            "warnings": warnings,
            "transcript_chars": len(transcription),
        },
    }


def flatten_baseline_plus_result(result: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for field_name, record in result.get("fields", {}).items():
        flat[field_name] = record.get("value")
    if "amount_total_gross" in flat:
        flat["invoice_total"] = flat["amount_total_gross"]
    flat["_baseline_plus"] = {
        "metadata": result.get("metadata", {}),
        "fields": result.get("fields", {}),
    }
    return flat
