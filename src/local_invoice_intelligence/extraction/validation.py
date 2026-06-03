from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from dateutil import parser as date_parser

from local_invoice_intelligence.extraction.fields import FieldDefinition


def _looks_like_date(value: Any) -> bool:
    if not value:
        return False
    try:
        date_parser.parse(str(value), fuzzy=True)
        return True
    except Exception:
        return False


def _looks_like_number(value: Any) -> bool:
    if not value:
        return False
    return bool(re.search(r"-?\d+(?:[,\s]\d{3})*(?:\.\d+)?", str(value)))


def validate_values(values: dict[str, Any], fields: list[FieldDefinition]) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []
    for field in fields:
        value = values.get(field.name)
        field_meta = {"definition": asdict(field), "present": value not in {None, ""}, "warnings": []}
        if field.required and not field_meta["present"]:
            field_meta["warnings"].append("missing_required")
        if field_meta["present"] and field.type in {"date"} and not _looks_like_date(value):
            field_meta["warnings"].append("invalid_date_shape")
        if field_meta["present"] and field.type in {"number", "integer", "money"} and not _looks_like_number(value):
            field_meta["warnings"].append("invalid_number_shape")
        if field_meta["warnings"]:
            warnings.extend(f"{field.name}:{warning}" for warning in field_meta["warnings"])
        metadata[field.name] = field_meta
    return metadata, warnings
