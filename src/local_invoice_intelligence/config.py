from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").lower()
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen3:14b")
OCR_BACKEND = os.getenv("OCR_BACKEND", "tesseract").lower()
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "200"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1024"))
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low").lower()

OLLAMA_THINKING = os.getenv("OLLAMA_THINKING", "false").lower()
OLLAMA_THINKING_PROMPT_DIRECTIVE = os.getenv("OLLAMA_THINKING_PROMPT_DIRECTIVE", "1") == "1"
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
