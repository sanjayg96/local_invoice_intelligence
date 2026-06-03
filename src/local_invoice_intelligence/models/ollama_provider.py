from __future__ import annotations

import json
from typing import Any

import ollama

from local_invoice_intelligence.config import (
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_THINKING,
    OLLAMA_THINKING_PROMPT_DIRECTIVE,
)


def parse_thinking(value: str | bool | None = None) -> bool | str | None:
    if isinstance(value, bool) or value is None:
        return value
    normalized = str(value).strip().lower()
    if normalized in {"auto", "default", "none", ""}:
        return None
    if normalized in {"false", "off", "no", "nothink", "no_think", "disable", "disabled"}:
        return False
    if normalized in {"true", "on", "yes", "think", "enable", "enabled"}:
        return True
    if normalized in {"low", "medium", "high"}:
        return normalized
    raise ValueError("Unsupported Ollama thinking mode.")


def maybe_add_thinking_directive(content: str, thinking: str | bool | None = None) -> str:
    parsed = parse_thinking(OLLAMA_THINKING if thinking is None else thinking)
    if not OLLAMA_THINKING_PROMPT_DIRECTIVE:
        return content
    if parsed is False:
        return f"/no_think\n\n{content}"
    if parsed is True or parsed in {"low", "medium", "high"}:
        return f"/think\n\n{content}"
    return content


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def run_ollama_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict[str, Any],
    thinking: str | bool | None = None,
) -> dict[str, Any]:
    parsed_thinking = parse_thinking(OLLAMA_THINKING if thinking is None else thinking)
    controls: dict[str, Any] = {
        "options": {
            "temperature": 0.0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
        "keep_alive": "10m",
    }
    if parsed_thinking is not None:
        controls["think"] = parsed_thinking
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": maybe_add_thinking_directive(user_prompt, parsed_thinking)},
        ],
        format=json_schema,
        **controls,
    )
    content = response["message"]["content"]
    return {
        "values": parse_json_content(content),
        "provider_metadata": {"provider": "ollama", "model": model, "thinking": str(parsed_thinking)},
    }
