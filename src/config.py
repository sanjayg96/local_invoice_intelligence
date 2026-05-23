import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# --- USER CONFIGURATION ---
DOCILE_DATASET_PATH = "/Users/sanja/Projects/docile/data/docile"
SUBSET_JSON_PATH = DATA_DIR / "eval_subset_ids_sorted.json"
EVAL_RESULTS_PATH = RESULTS_DIR / "eval_report.json"

# --- VLM / LLM Settings ---
# The "Eyes": Converts images to Markdown
VISION_MODEL = "glm-ocr" 

# The "Brain": Extracts JSON from Markdown
TEXT_MODEL = "llama3.1:8b"

# --- NEW FEATURES ---
# True = Higher accuracy, higher latency. Uses the Pydantic scratchpad.
# False = Max speed, zero reasoning overhead.
ENABLE_THINKING = False

# Thermal control (seconds) to prevent heat saturation during full batch runs
THERMAL_COOLDOWN_BATCH_SIZE = 10
THERMAL_COOLDOWN_SECONDS = 60

# Misc
MODEL_TIMEOUT_SECONDS = 200