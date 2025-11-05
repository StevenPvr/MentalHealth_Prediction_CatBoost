"""Training pipeline module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from src.constants import RESULTS_DIR
from src.training.training import (
    filter_train,
    load_splits,
    save_model,
    separate_features_target,
    train_model,
)
from src.utils import get_logger

LOGGER = get_logger(__name__)


class HPOResult:
    """Container for hyperparameter optimization results."""

    def __init__(self, best_params: dict, best_iterations: int | None = None) -> None:
        """Initialize HPOResult with best parameters and iterations."""
        self.best_params = best_params
        self.best_iterations = best_iterations


def train_pipeline() -> None:
    """Execute the training workflow end-to-end."""
    LOGGER.info("Loading dataset splits")
    df = load_splits()
    LOGGER.info("Loaded %d rows", len(df))

    LOGGER.info("Filtering train rows")
    df_train = filter_train(df)

    train_df = cast(pd.DataFrame, df_train.drop(columns=["split"]))
    LOGGER.info("Train rows: %d", len(train_df))

    LOGGER.info("Separating features and target")
    X_train, y_train = separate_features_target(train_df)

    # Optimization is expected to have been run earlier in the workflow.
    # Load saved best params if present; otherwise proceed with empty params.
    best_params_path = RESULTS_DIR / "optimization" / "best_params.json"
    if best_params_path.exists():
        LOGGER.info("Loading saved HPO best params from %s", best_params_path)
        with best_params_path.open("r", encoding="utf-8") as fh:
            best_params = json.load(fh)
        hpo_result = HPOResult(best_params=best_params, best_iterations=None)
    else:
        LOGGER.info(
            "No saved HPO params found at %s; continuing without HPO (training with defaults).",
            best_params_path,
        )
        hpo_result = HPOResult(best_params={}, best_iterations=None)

    LOGGER.info("Training CatBoost model with best hyperparameters")
    # Use iterations suggested by HPO (mean best trees across folds) when available;
    # otherwise fall back to any 'iterations' present in best_params.
    final_iterations = getattr(hpo_result, "best_iterations", None) or hpo_result.best_params.get(
        "iterations"
    )
    model = train_model(
        X_train,
        y_train,
        hyperparams=hpo_result.best_params,
        final_iterations=final_iterations if isinstance(final_iterations, int) else None,
    )
    LOGGER.info("Model trained with %d trees", model.tree_count_)

    LOGGER.info("Saving trained model")
    save_model(model)


if __name__ == "__main__":
    train_pipeline()
