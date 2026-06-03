from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    description: str
    type: str = "string"
    required: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FieldDefinition":
        name = str(data.get("name", "")).strip()
        description = str(data.get("description", "")).strip()
        if not name:
            raise ValueError("Each field definition must include a non-empty name.")
        if not description:
            raise ValueError(f"Field `{name}` must include a description.")
        return cls(
            name=name,
            description=description,
            type=str(data.get("type") or "string").strip().lower(),
            required=bool(data.get("required", False)),
        )


def load_field_definitions(path: str | Path) -> list[FieldDefinition]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)

    if isinstance(payload, dict):
        payload = payload.get("fields", payload)
    if not isinstance(payload, list):
        raise ValueError("Field config must be a list or an object with a `fields` list.")
    return [FieldDefinition.from_mapping(item) for item in payload]


def schema_for_fields(fields: list[FieldDefinition]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required = []
    for field in fields:
        json_type = {
            "string": "string",
            "text": "string",
            "number": "string",
            "integer": "string",
            "money": "string",
            "date": "string",
            "boolean": "string",
        }.get(field.type, "string")
        properties[field.name] = {
            "type": [json_type, "null"],
            "description": field.description,
        }
        if field.required:
            required.append(field.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def field_prompt(fields: list[FieldDefinition]) -> str:
    lines = []
    for field in fields:
        required = "required" if field.required else "optional"
        lines.append(f"- {field.name} ({field.type}, {required}): {field.description}")
    return "\n".join(lines)
