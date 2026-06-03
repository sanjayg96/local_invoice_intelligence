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

from config import (
    BASELINE_PLUS_PRIMARY_MODEL,
    BASELINE_PLUS_RESCUE_MODEL,
    BLANK_ENSEMBLE_MODEL,
    ENABLE_THINKING,
)
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
HARD_VALIDATION_ISSUES = {"blank", "invalid_amount", "invalid_date", "prose_or_too_long"}
SOFT_VALIDATION_ISSUES = {"invalid_address", "not_grounded"}


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


def weakness_level(issue: Optional[str]) -> str:
    if issue is None:
        return "none"
    if issue in HARD_VALIDATION_ISSUES:
        return "hard"
    if issue in SOFT_VALIDATION_ISSUES:
        return "soft"
    return "soft"


def should_attempt_rescue(issue: Optional[str], spec: FieldSpec) -> bool:
    if issue is None:
        return False
    if weakness_level(issue) == "hard":
        return True
    return issue == "not_grounded" and spec.kind in {"money", "date"}


def should_attempt_blank_ensemble(issue: Optional[str], spec: FieldSpec) -> bool:
    return issue == "blank"


def should_shadow_rescue(issue: Optional[str], spec: FieldSpec) -> bool:
    if should_attempt_rescue(issue, spec):
        return False
    return issue in SOFT_VALIDATION_ISSUES and spec.kind in {"organization", "address"}


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


def _candidate_match_score(value: Any, text: str, kind: str) -> int:
    normalized_value = normalize_for_kind(value, kind)
    if not normalized_value:
        return 0
    if kind == "money":
        return 115 if any(normalize_amount(match) == normalized_value for match in MONEY_RE.findall(text)) else 0
    if kind == "date":
        return 115 if any(normalize_date(match) == normalized_value for match in DATE_RE.findall(text)) else 0

    compact_value = normalize_compact(value)
    compact_text = normalize_compact(text)
    if not compact_value:
        return 0
    if compact_value in compact_text:
        return 115
    return 85 if fuzz.partial_ratio(compact_value, compact_text) >= 92 else 0


def evidence_windows(
    transcription: str,
    spec: FieldSpec,
    limit: int = 8,
    candidate_values: Optional[list[Any]] = None,
) -> list[dict[str, Any]]:
    lines = transcript_lines(transcription)
    query = normalize_text(f"{spec.name.replace('_', ' ')} {spec.kind} {spec.description}").lower()
    candidate_values = candidate_values or []
    windows = []
    for idx, (page, _) in enumerate(lines):
        start = max(0, idx - 2)
        end = min(len(lines), idx + 3)
        text = "\n".join(line for _, line in lines[start:end])
        text_l = text.lower()
        score = fuzz.token_set_ratio(query, text_l)
        sources = ["keyword"]
        if spec.kind == "money" and MONEY_RE.search(text):
            score += 30
            if re.search(r"\b(total|amount due|balance due|payable|gross)\b", text_l):
                score += 35
                sources.append("money_total_label")
            if re.search(r"\b(tax|discount|rate|unit|subtotal|paid)\b", text_l):
                score -= 20
            if idx >= max(0, len(lines) - 18):
                score += 8
                sources.append("document_footer")
        elif spec.kind == "date" and DATE_RE.search(text):
            score += 25
            if re.search(r"\b(inv\s*date|invoice\s*date|bill\s*date|statement\s*date|printed)\b", text_l):
                score += 35
                sources.append("date_label")
            if re.search(r"\b(due|service|run|flight|ordered|period|payment)\b", text_l):
                score -= 18
        elif spec.kind == "address" and ADDRESS_RE.search(text):
            score += 25
            if re.search(r"\b(remit|payment|payable|mail\s+checks|pay\s+to)\b", text_l):
                score += 30
                sources.append("remit_address_label")
            if idx < 18:
                score += 8
                sources.append("document_header")
        elif spec.kind == "organization":
            if re.search(r"\b(remit|vendor|supplier|from|payable|seller)\b", text_l):
                score += 25
                sources.append("organization_label")
            if idx < 12:
                score += 12
                sources.append("document_header")

        candidate_boost = max((_candidate_match_score(value, text, spec.kind) for value in candidate_values), default=0)
        if candidate_boost:
            score += candidate_boost
            sources.append("primary_value_match")

        windows.append({
            "page": page,
            "text": normalize_text(text)[:1200],
            "score": round(score, 2),
            "sources": sorted(set(sources)),
        })
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


def evidence_support_score(value: Any, spec: FieldSpec, windows: list[dict[str, Any]]) -> float:
    if not normalize_text(value):
        return 0.0
    supported = [
        float(window.get("score") or 0.0)
        for window in windows
        if is_grounded(normalize_text(value), window.get("text", ""), spec.kind)
    ]
    return max(supported) if supported else 0.0


def rescue_quote_supports(candidate: dict[str, Any], spec: FieldSpec) -> bool:
    value = normalize_text(candidate.get("value"))
    quote = normalize_text(candidate.get("evidence_quote"))
    return bool(value and quote and is_grounded(value, quote, spec.kind))


def rescue_replacement_decision(
    primary: dict[str, Any],
    candidate: dict[str, Any],
    spec: FieldSpec,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_issue = primary.get("validation_issue")
    candidate_issue = candidate.get("validation_issue")
    primary_value = primary.get("value")
    candidate_value = candidate.get("value")
    primary_support = evidence_support_score(primary_value, spec, windows)
    rescue_support = evidence_support_score(candidate_value, spec, windows)
    quote_supported = rescue_quote_supports(candidate, spec)
    level = weakness_level(primary_issue)

    decision = {
        "accepted": False,
        "primary_issue": primary_issue,
        "weakness_level": level,
        "candidate_issue": candidate_issue,
        "primary_support_score": round(primary_support, 2),
        "rescue_support_score": round(rescue_support, 2),
        "quote_supported": quote_supported,
        "reason": "",
    }

    if not primary_issue:
        decision["reason"] = "primary_valid"
    elif candidate_issue:
        decision["reason"] = f"rescue_candidate_invalid:{candidate_issue}"
    elif not quote_supported:
        decision["reason"] = "rescue_quote_does_not_support_value"
    elif level == "hard":
        decision["accepted"] = True
        decision["reason"] = f"hard_primary_issue:{primary_issue}"
    elif rescue_support >= max(95.0, primary_support + 35.0):
        decision["accepted"] = True
        decision["reason"] = "soft_issue_with_stronger_rescue_evidence"
    else:
        decision["reason"] = "soft_issue_without_enough_rescue_evidence_gain"
    return decision


def blank_ensemble_replacement_decision(
    primary: dict[str, Any],
    candidate: dict[str, Any],
    spec: FieldSpec,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_issue = candidate.get("validation_issue")
    candidate_value = candidate.get("value")
    rescue_support = evidence_support_score(candidate_value, spec, windows)
    quote_supported = rescue_quote_supports(candidate, spec)
    decision = {
        "accepted": False,
        "primary_issue": primary.get("validation_issue"),
        "candidate_issue": candidate_issue,
        "rescue_support_score": round(rescue_support, 2),
        "quote_supported": quote_supported,
        "reason": "",
    }
    if primary.get("validation_issue") != "blank":
        decision["reason"] = "primary_not_blank"
    elif candidate_issue:
        decision["reason"] = f"ensemble_candidate_invalid:{candidate_issue}"
    elif not quote_supported:
        decision["reason"] = "ensemble_quote_does_not_support_value"
    else:
        decision["accepted"] = True
        decision["reason"] = "blank_primary_filled_by_valid_ensemble_candidate"
    return decision


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
    shadow_identity_rescue: bool = False,
    blank_only_ensemble: bool = False,
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
    rescue_fields = []
    shadow_fields = []
    for field_name, spec in specs.items():
        value = prediction_value(prediction, field_name)
        record = field_record(value, spec, transcription, "baseline_primary", primary_model, primary_latency)
        record["weakness_level"] = weakness_level(record["validation_issue"])
        field_results[field_name] = record
        if record["validation_issue"]:
            weak_fields.append(field_name)
            if blank_only_ensemble:
                if should_attempt_blank_ensemble(record["validation_issue"], spec):
                    rescue_fields.append(field_name)
                else:
                    record["rescue_skipped_reason"] = "blank_ensemble_only"
            elif should_attempt_rescue(record["validation_issue"], spec):
                rescue_fields.append(field_name)
            elif shadow_identity_rescue and should_shadow_rescue(record["validation_issue"], spec):
                shadow_fields.append(field_name)
                record["rescue_skipped_reason"] = "shadow_identity_field_preserved"
            else:
                record["rescue_skipped_reason"] = "soft_identity_field_preserved"

    windows_by_field = {}
    rescue_decisions = {}
    shadow_decisions = {}
    attempted_fields = rescue_fields + shadow_fields
    if attempted_fields:
        weak_specs = {field_name: specs[field_name] for field_name in attempted_fields}
        windows_by_field = {
            field_name: evidence_windows(
                transcription,
                specs[field_name],
                candidate_values=[field_results[field_name].get("value")],
            )
            for field_name in attempted_fields
        }
        try:
            rescue, rescue_latency = _run_rescue(
                weak_specs,
                windows_by_field,
                rescue_model,
                ollama_thinking,
            )
            timings["rescue_model"] = rescue_latency
            rescue_response_fields = rescue.get("fields", {})
            for field_name, spec in weak_specs.items():
                raw = rescue_response_fields.get(field_name, {})
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
                candidate["weakness_level"] = weakness_level(candidate["validation_issue"])
                if blank_only_ensemble:
                    decision = blank_ensemble_replacement_decision(
                        field_results[field_name],
                        candidate,
                        spec,
                        windows_by_field.get(field_name, []),
                    )
                else:
                    decision = rescue_replacement_decision(
                        field_results[field_name],
                        candidate,
                        spec,
                        windows_by_field.get(field_name, []),
                    )
                if field_name in shadow_fields:
                    decision = {**decision, "shadow_only": True}
                    shadow_decisions[field_name] = decision
                    field_results[field_name]["shadow_rescue_candidate"] = candidate
                    field_results[field_name]["shadow_rescue_decision"] = decision
                elif decision["accepted"]:
                    rescue_decisions[field_name] = decision
                    candidate["rescue_decision"] = decision
                    field_results[field_name] = candidate
                else:
                    rescue_decisions[field_name] = decision
                    field_results[field_name]["rescue_candidate"] = candidate
                    field_results[field_name]["rescue_decision"] = decision
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
            "models": {"primary": primary_model, "rescue": rescue_model if rescue_fields else None},
            "ocr_backend": ocr_backend,
            "ollama_thinking": thinking_label(ollama_thinking),
            "weak_fields": weak_fields,
            "rescue_fields": rescue_fields,
            "shadow_fields": shadow_fields,
            "rescue_decisions": rescue_decisions,
            "shadow_decisions": shadow_decisions,
            "shadow_identity_rescue": shadow_identity_rescue,
            "blank_only_ensemble": blank_only_ensemble,
            "warnings": warnings,
            "transcript_chars": len(transcription),
        },
    }


def extract_blank_ensemble(
    pdf_path: str,
    fields: Optional[list[str | FieldSpec]] = None,
    primary_model: str = BASELINE_PLUS_PRIMARY_MODEL,
    ensemble_model: str = BLANK_ENSEMBLE_MODEL,
    ocr_backend: str | None = None,
    ollama_thinking: str | bool | None = None,
) -> dict[str, Any]:
    result = extract_baseline_plus(
        pdf_path=pdf_path,
        fields=fields,
        primary_model=primary_model,
        rescue_model=ensemble_model,
        ocr_backend=ocr_backend,
        ollama_thinking=ollama_thinking,
        shadow_identity_rescue=False,
        blank_only_ensemble=True,
    )
    result["metadata"]["pipeline"] = "blank_ensemble"
    result["metadata"]["models"] = {
        "primary": primary_model,
        "blank_ensemble": ensemble_model if result["metadata"].get("rescue_fields") else None,
    }
    return result


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
