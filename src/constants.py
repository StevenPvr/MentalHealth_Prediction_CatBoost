"""Centralized constants for the Mental Health prediction project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODEL_DIR: Path = PROJECT_ROOT / "model_saved"
LOG_DIR: Path = RESULTS_DIR / "log"
PLOTS_DIR: Path = RESULTS_DIR / "plots"

DEFAULT_DATASET_FILENAME: str = "dataset.csv"
CLEANED_DATASET_FILENAME: str = "dataset_cleaned.parquet"
CLEANED_DATASET_CSV_FILENAME: str = "dataset_cleaned.csv"
SPLITS_PARQUET_FILENAME: str = "splits.parquet"
SPLITS_CSV_FILENAME: str = "splits.csv"
MODEL_FILENAME: str = "catboost_model.cbm"
SHAP_PLOT_FILENAME: str = "shap_summary.png"
ENCODERS_MAPPINGS_FILENAME: str = "encoders_mappings.json"

DEFAULT_RANDOM_STATE: int = 42
RARE_CATEGORY_THRESHOLD: float = 0.05
RARE_CATEGORY_LABEL: str = "others"
DEFAULT_TARGET_COLUMN: str = "treatment"
DEFAULT_EARLY_STOPPING_ROUNDS: int = 50
DEFAULT_VALIDATION_SIZE: float = 0.1
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FILE: str = "pipeline.log"

# Cross-validation and hyperparameter optimization defaults
DEFAULT_CV_N_SPLITS: int = 5
DEFAULT_OPTIMIZATION_METRIC: str = "logloss"  # one of: "auc", "logloss"
# Simple, safe grid for CatBoost tuning (kept small for runtime & tests)
HPO_PARAM_GRID: dict[str, tuple[object, ...]] = {
    "depth": (4, 6),
    "learning_rate": (0.03, 0.1),
    "l2_leaf_reg": (1.0, 3.0),
}
DEFAULT_HPO_TRIALS: int = 50

DEFAULT_MISSING_TOKENS: tuple[str, ...] = (
    "",
    " ",
    "NA",
    "N/A",
    "nan",
    "NaN",
    "null",
)


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "MODEL_DIR",
    "LOG_DIR",
    "PLOTS_DIR",
    "DEFAULT_DATASET_FILENAME",
    "CLEANED_DATASET_FILENAME",
    "CLEANED_DATASET_CSV_FILENAME",
    "SPLITS_PARQUET_FILENAME",
    "SPLITS_CSV_FILENAME",
    "MODEL_FILENAME",
    "SHAP_PLOT_FILENAME",
    "ENCODERS_MAPPINGS_FILENAME",
    "DEFAULT_RANDOM_STATE",
    "RARE_CATEGORY_THRESHOLD",
    "RARE_CATEGORY_LABEL",
    "DEFAULT_TARGET_COLUMN",
    "DEFAULT_EARLY_STOPPING_ROUNDS",
    "DEFAULT_VALIDATION_SIZE",
    "DEFAULT_CV_N_SPLITS",
    "DEFAULT_OPTIMIZATION_METRIC",
    "HPO_PARAM_GRID",
    "DEFAULT_HPO_TRIALS",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_FILE",
    "DEFAULT_MISSING_TOKENS",
]
