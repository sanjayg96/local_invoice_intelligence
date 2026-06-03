import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# --- USER CONFIGURATION ---
DOCILE_DATASET_PATH = "/Users/sanja/Projects/docile/data/docile"
SUBSET_JSON_PATH = DATA_DIR / "eval_subset_ids_sorted.json"
EVAL_RESULTS_PATH = RESULTS_DIR / "eval_report.json"

# --- VLM / LLM Settings ---
# The "Eyes": Pure Transcription (No Q&A, No JSON formatting)
VISION_MODEL = "llama3.2-vision:11b"

# The "Brain": Extracts JSON from Markdown
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen3:14b")
# TEXT_MODEL = "llama3.1:8b"
# TEXT_MODEL = "llama3.2:3b"

# --- Provider settings ---
# "ollama" keeps everything local. "openai" sends the extracted transcript to
# OpenAI's API for frontier-model benchmarking.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1024"))
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low").lower()

# --- Baseline-plus extraction settings ---
# pipeline mode: "baseline_plus" for original full-transcript extraction plus
# validation and targeted rescue, or "baseline" for the original extraction only.
PIPELINE_MODE = "baseline"
BASELINE_PLUS_PRIMARY_MODEL = os.getenv("BASELINE_PLUS_PRIMARY_MODEL", TEXT_MODEL)
BASELINE_PLUS_RESCUE_MODEL = os.getenv("BASELINE_PLUS_RESCUE_MODEL", TEXT_MODEL)

# OCR fallback used only for pages where pdfplumber finds very little digital text.
# Options: "tesseract" or "rapidocr".
OCR_BACKEND = os.getenv("OCR_BACKEND", "tesseract").lower()

# --- Reasoning controls ---
# Schema scratchpad is separate from Ollama/Qwen thinking. Keep it off for clean
# Qwen3 tests so --ollama-thinking is the only reasoning variable.
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() in {"1", "true", "yes", "on"}

# Ollama's built-in thinking control for hybrid reasoning models such as Qwen3.
# Use "false" for fastest bounded extraction, or "true"/"low"/"medium"/"high"
# to compare reasoning-enabled variants. "auto" leaves Ollama/model defaults alone.
OLLAMA_THINKING = os.getenv("OLLAMA_THINKING", "false").lower()
OLLAMA_THINKING_PROMPT_DIRECTIVE = os.getenv("OLLAMA_THINKING_PROMPT_DIRECTIVE", "1") == "1"
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_PRIMARY_NUM_PREDICT = int(os.getenv("OLLAMA_PRIMARY_NUM_PREDICT", "1024"))
OLLAMA_RESCUE_NUM_PREDICT = int(os.getenv("OLLAMA_RESCUE_NUM_PREDICT", "512"))

# Thermal control (seconds) to prevent heat saturation during full batch runs
THERMAL_COOLDOWN_BATCH_SIZE = 100
THERMAL_COOLDOWN_SECONDS = 50

# Misc
MODEL_TIMEOUT_SECONDS = 200
