"""Central configuration constants for the Arabic Complaints Classifier.

All scripts should import from here instead of redefining magic numbers/paths.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Data paths ---
DATA_TEXT_DIR = ROOT / "data" / "text"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_CANARY_DIR = ROOT / "data" / "canary"

LABELED_CSV = DATA_TEXT_DIR / "complaints_labeled.csv"
TRAIN_CSV = DATA_PROCESSED_DIR / "train.csv"
VAL_CSV = DATA_PROCESSED_DIR / "val.csv"
TEST_CSV = DATA_PROCESSED_DIR / "test.csv"
LABEL_MAP_JSON = DATA_PROCESSED_DIR / "label_map.json"

# --- Model paths ---
MODELS_DIR = ROOT / "models"
BAKEOFF_DIR = MODELS_DIR / "bakeoff"
ENSEMBLE_FINAL_DIR = MODELS_DIR / "ensemble_final"
SINGLE_FINAL_DIR = MODELS_DIR / "single_final"
ENSEMBLE_CONFIG = ENSEMBLE_FINAL_DIR / "config.json"
SINGLE_CONFIG = SINGLE_FINAL_DIR / "config.json"

# --- Schema ---
def num_labels() -> int:
    """Read NUM_LABELS dynamically from label_map.json (handles 8/9-class transitions)."""
    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        return len(json.load(f))


# --- Training defaults ---
DOMINANT_CATEGORY = "جودة الطعام"
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRAD_ACCUM = 2
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_MAX_LENGTH = 192
DEFAULT_DOMINANT_CAP = 100_000  # effectively no subsampling
DEFAULT_SEED = 42

# --- Inference defaults ---
INFERENCE_BATCH_SIZE = 64
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
DEFAULT_MIN_ARABIC_RATIO = 0.30

# --- Augmentation ---
TRAIN_ONLY_SOURCES = {"synthetic", "augmented_bt", "chatgpt_synthetic", "pseudo_labeled", "eda_augmented"}
GOOD_AUG_SOURCES = {"production", "play_store", "res1", "chatgpt_synthetic", "augmented_bt", "pseudo_labeled"}

# --- Splitting ---
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test (real only)
SPLIT_SEED = 42

# --- Tests / regression gates ---
REGRESSION_ACC_FLOOR = 0.92
REGRESSION_MACRO_F1_FLOOR = 0.85
REGRESSION_MIN_CLASS_FLOOR = 0.65

# --- HF Hub (single-model deploy) ---
DEFAULT_HF_REPO_ID_SINGLE = "FerasMad/arabic-complaints-camelbert-mix"
