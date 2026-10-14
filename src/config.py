"""Configuration constants and paths for ManSanket (मनसंकेत)."""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BENCHMARK_DATA_DIR = DATA_DIR / "benchmarks"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOCALES_DIR = BASE_DIR / "locales"

# Ensure runtime directories exist
for path in [PROCESSED_DATA_DIR, BENCHMARK_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOCALES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Emotion Taxonomy (6-class CARER benchmark standard)
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

LABEL_TO_ID = {v: k for k, v in EMOTION_LABELS.items()}

# Editorial Color Palette for Visualizations & Styling
PALETTE = {
    "bg": "#F7F5F0",
    "surface": "#FFFFFF",
    "surface_alt": "#FBF9F5",
    "border": "#E4E0D8",
    "text_primary": "#292824",
    "text_secondary": "#716F69",
    "text_muted": "#9C9990",
    "primary_accent": "#35465C",      # Deep Slate Blue
    "supporting_accent": "#A45D48",   # Terracotta
    "accent_subtle": "#E8ECEF",
    "success": "#3E6B48",
    "warning": "#C47B2B",
    "danger": "#A45D48",
}

# Emotion-specific curated editorial colors (subdued, not neon)
EMOTION_COLORS = {
    "joy": "#C98A2C",        # Warm ochre / amber
    "love": "#A85368",       # Dusky rose
    "sadness": "#4A6B82",    # Slate blue
    "anger": "#A45D48",      # Terracotta
    "fear": "#5C5470",       # Muted indigo
    "surprise": "#3E7B78",   # Deep sage teal
}

# Multilingual Interface Languages
SUPPORTED_LOCALES = ["en", "hi", "mr"]
DEFAULT_LOCALE = "en"

# Model paths
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MULTINOMIAL_NB_PATH = MODELS_DIR / "multinomial_nb.joblib"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "logistic_regression.joblib"
LINEAR_SVC_PATH = MODELS_DIR / "linear_svc.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Results paths
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
CONFUSION_MATRICES_PATH = RESULTS_DIR / "confusion_matrices.json"
MULTILINGUAL_RESULTS_PATH = RESULTS_DIR / "multilingual_benchmark_results.json"
