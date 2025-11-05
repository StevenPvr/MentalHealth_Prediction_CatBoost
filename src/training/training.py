"""Training utilities for the CatBoost classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore
from catboost import CatBoostClassifier  # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore

from ..constants import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_VALIDATION_SIZE,
)
from ..model import create_catboost_model
from ..utils import (
    detect_categorical_columns,
    fill_categorical_na,
    get_logger,
    model_path,
    splits_parquet_path,
)

LOGGER = get_logger(__name__)


def load_splits(base_dir: str | None = None) -> pd.DataFrame:
    """Load the prepared dataset splits from disk."""
    path = splits_parquet_path(base_dir)
    LOGGER.info("Loading dataset splits from %s", path)
    return pd.read_parquet(path)


def separate_features_target(
    df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target columns."""
    features = df.drop(columns=[target_col])
    target = cast(pd.Series, df[target_col])
    LOGGER.info("Separated features and target using column '%s'", target_col)
    return features, target


def filter_train(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the train split."""
    filtered = cast(pd.DataFrame, df[df["split"] == "train"]).copy()

    # Remove unused categorical levels introduced when the combined splits
    # artifact was created. This prevents categorical dtypes from leaking
    # categories that only exist in other splits into training-time
    # preprocessing (e.g. OneHotEncoder reading pandas categories).
    for col in filtered.select_dtypes(include=["category"]).columns:
        filtered[col] = filtered[col].cat.remove_unused_categories()

    LOGGER.info("Filtered splits: resulting dataframe has %d rows", len(filtered))
    return filtered


# Note: categorical feature detection is centralized in utils.detect_categorical_columns


def prepare_training_data(X_train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare training data by filling categorical NAs and detecting categorical features."""
    X_train_processed = fill_categorical_na(X_train)
    cat_features = detect_categorical_columns(X_train_processed)
    LOGGER.info("Prepared training data: %d categorical features detected", len(cat_features))
    return X_train_processed, cat_features


def train_with_fixed_iterations(
    model: CatBoostClassifier,
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    final_iterations: int,
) -> CatBoostClassifier:
    """Train the model with fixed iterations on full train data (no eval_set)."""
    LOGGER.info("Training final model on full train (iterations=%d, no eval_set)", final_iterations)
    model.set_params(use_best_model=False, iterations=int(final_iterations))
    model.fit(
        X_train_processed,
        y_train,
        cat_features=cat_features,
        verbose=False,
    )
    tree_count = getattr(model, "tree_count_", None)
    if tree_count is not None:
        LOGGER.info("Training finished after %d iterations", tree_count)
    return model


def train_with_early_stopping(
    model: CatBoostClassifier,
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
) -> CatBoostClassifier:
    """Train the model with internal validation and early stopping."""
    LOGGER.info(
        "Training with internal validation (early stopping, val_size=%.2f)",
        DEFAULT_VALIDATION_SIZE,
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=DEFAULT_VALIDATION_SIZE, random_state=DEFAULT_RANDOM_STATE
    )
    tr_idx, val_idx = next(splitter.split(X_train_processed, y_train))
    X_tr, y_tr = X_train_processed.iloc[tr_idx], y_train.iloc[tr_idx]
    X_val, y_val = X_train_processed.iloc[val_idx], y_train.iloc[val_idx]
    model.fit(
        X_tr,
        y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    tree_count = getattr(model, "tree_count_", None)
    if tree_count is not None:
        LOGGER.info("Training finished after %d iterations", tree_count)
    return model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparams: dict[str, Any] | None = None,
    final_iterations: int | None = None,
) -> CatBoostClassifier:
    """Train the CatBoost model on the train split.

    Classic methodology aligned with Ridge baseline:
    - HPO is done with CV + eval_set + early stopping (separately).
    - Final training uses the full train split; if `final_iterations` is
      provided (recommended), it disables early stopping, sets
      `iterations=final_iterations`, and trains without eval_set.
    - If `final_iterations` is not provided, fallback to a small internal
      validation split with early stopping.
    """
    model = create_catboost_model()
    if hasattr(model, "set_params") and hyperparams:
        model.set_params(**hyperparams)

    X_train_processed, cat_features = prepare_training_data(X_train)

    if isinstance(final_iterations, int) and final_iterations > 0:
        model = train_with_fixed_iterations(
            model, X_train_processed, y_train, cat_features, final_iterations
        )
    else:
        model = train_with_early_stopping(model, X_train_processed, y_train, cat_features)

    return model


def save_model(model: Any, path: str | Path | None = None) -> None:
    """Persist the trained model to disk."""
    target_path = Path(path) if path is not None else model_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(target_path))
    LOGGER.info("Model saved to %s", target_path)
