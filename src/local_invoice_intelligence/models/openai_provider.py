from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from local_invoice_intelligence.config import (
    OPENAI_API_BASE_URL,
    OPENAI_MAX_OUTPUT_TOKENS,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
)
from local_invoice_intelligence.models.ollama_provider import parse_json_content


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            converted = {key: visit(value) for key, value in node.items()}
            if converted.get("type") == "object":
                converted.setdefault("additionalProperties", False)
                properties = converted.get("properties")
                if isinstance(properties, dict):
                    converted["required"] = list(properties)
            return converted
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return visit(dict(schema))


def _extract_output_text(response: dict[str, Any]) -> str:
    if response.get("output_text"):
        return str(response["output_text"])
    chunks = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    if chunks:
        return "".join(chunks)
    raise ValueError(f"OpenAI response did not contain output text: {response}")


def run_openai_json(
    *,
    model: str | None,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Local Ollama is the default provider.")
    selected_model = model or OPENAI_MODEL
    body = {
        "model": selected_model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "document_extraction",
                "schema": _strict_json_schema(json_schema),
                "strict": True,
            }
        },
        "max_output_tokens": OPENAI_MAX_OUTPUT_TOKENS,
    }
    if OPENAI_REASONING_EFFORT not in {"", "none", "off", "false"}:
        body["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
    request = urllib.request.Request(
        f"{OPENAI_API_BASE_URL.rstrip('/')}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed with HTTP {exc.code}: {detail}") from exc
    return {
        "values": parse_json_content(_extract_output_text(payload)),
        "provider_metadata": {
            "provider": "openai",
            "model": selected_model,
            "latency_s": round(time.perf_counter() - started, 4),
            "response_id": payload.get("id"),
            "usage": payload.get("usage"),
            "reasoning_effort": OPENAI_REASONING_EFFORT,
        },
    }
