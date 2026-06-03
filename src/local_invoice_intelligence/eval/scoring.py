from __future__ import annotations

import re
from typing import Any

from dateutil import parser as date_parser
from rapidfuzz import fuzz

from local_invoice_intelligence.extraction.fields import FieldDefinition


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _number(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    cleaned = str(value).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _date(value: Any):
    if value in {None, ""}:
        return None
    try:
        return date_parser.parse(str(value), fuzzy=True).date()
    except Exception:
        return None


def score_value(expected: Any, predicted: Any, field: FieldDefinition) -> float:
    if expected in {None, ""} and predicted in {None, ""}:
        return 100.0
    if expected in {None, ""} or predicted in {None, ""}:
        return 0.0
    if field.type in {"number", "integer", "money"}:
        left = _number(expected)
        right = _number(predicted)
        return 100.0 if left is not None and right is not None and left == right else 0.0
    if field.type == "date":
        left = _date(expected)
        right = _date(predicted)
        return 100.0 if left is not None and right is not None and left == right else 0.0
    return float(fuzz.token_set_ratio(_normalize_text(predicted), _normalize_text(expected)))


def summarize(records: list[dict[str, Any]], fields: list[FieldDefinition]) -> dict[str, Any]:
    scores = {field.name: [] for field in fields}
    for record in records:
        expected = record.get("ground_truth", {})
        predicted = record.get("predicted", {})
        for field in fields:
            scores[field.name].append(score_value(expected.get(field.name), predicted.get(field.name), field))
    averages = {
        name: (sum(values) / len(values) if values else None)
        for name, values in scores.items()
    }
    populated = [value for value in averages.values() if value is not None]
    return {
        "fields": averages,
        "total_average": sum(populated) / len(populated) if populated else None,
    }
