from __future__ import annotations

from typing import Any

from config import (
    OLLAMA_NUM_CTX,
    OLLAMA_PRIMARY_NUM_PREDICT,
    OLLAMA_RESCUE_NUM_PREDICT,
    OLLAMA_THINKING,
    OLLAMA_THINKING_PROMPT_DIRECTIVE,
)


def parse_thinking(value: str | bool | None) -> bool | str | None:
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
    raise ValueError(
        "Unsupported Ollama thinking mode. Use one of: auto, false, true, low, medium, high."
    )


def thinking_label(value: str | bool | None = None) -> str:
    parsed = parse_thinking(OLLAMA_THINKING if value is None else value)
    if parsed is None:
        return "auto"
    if parsed is False:
        return "off"
    if parsed is True:
        return "on"
    return str(parsed)


def maybe_add_thinking_directive(content: str, thinking: str | bool | None = None) -> str:
    parsed = parse_thinking(OLLAMA_THINKING if thinking is None else thinking)
    if not OLLAMA_THINKING_PROMPT_DIRECTIVE:
        return content
    if parsed is False:
        return f"/no_think\n\n{content}"
    if parsed is True or parsed in {"low", "medium", "high"}:
        return f"/think\n\n{content}"
    return content


def chat_controls(
    *,
    thinking: str | bool | None = None,
    num_predict: int = OLLAMA_PRIMARY_NUM_PREDICT,
    num_ctx: int = OLLAMA_NUM_CTX,
) -> dict[str, Any]:
    parsed_thinking = parse_thinking(OLLAMA_THINKING if thinking is None else thinking)
    controls: dict[str, Any] = {
        "options": {
            "temperature": 0.0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
    }
    if parsed_thinking is not None:
        controls["think"] = parsed_thinking
    return controls


def primary_chat_controls(*, thinking: str | bool | None = None) -> dict[str, Any]:
    return chat_controls(thinking=thinking, num_predict=OLLAMA_PRIMARY_NUM_PREDICT)


def rescue_chat_controls(*, thinking: str | bool | None = None) -> dict[str, Any]:
    return chat_controls(thinking=thinking, num_predict=OLLAMA_RESCUE_NUM_PREDICT)
