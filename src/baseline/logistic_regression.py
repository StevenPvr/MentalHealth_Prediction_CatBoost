"""Ridge regression baseline for categorical-only features."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import accuracy_score, log_loss, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

from ..constants import DEFAULT_CV_N_SPLITS, DEFAULT_RANDOM_STATE, DEFAULT_TARGET_COLUMN
from ..utils import get_logger, splits_parquet_path

LOGGER = get_logger(__name__)


def load_splits(base_dir: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test splits from parquet file.

    Args:
    ----
        base_dir: Optional base directory for data files.

    Returns:
    -------
        Tuple of (train_df, test_df) with split column removed.

    """
    path = splits_parquet_path(base_dir)
    LOGGER.info("Loading splits from %s", path)
    df = pd.read_parquet(path)

    train_df = cast(pd.DataFrame, df[df["split"] == "train"]).copy()
    test_df = cast(pd.DataFrame, df[df["split"] == "test"]).copy()

    if "split" in train_df.columns:
        train_df = train_df.drop(columns=["split"])
    if "split" in test_df.columns:
        test_df = test_df.drop(columns=["split"])

    # Remove any unused categorical levels that may have been introduced
    # when splits were persisted (we save a combined dataframe which can
    # cause categories to be unioned across splits). Removing unused
    # categories for the returned subsets prevents leaking category levels
    # from the test split into training-time preprocessing (e.g. OneHotEncoder
    # reading pandas' CategoricalDtype categories).
    for df_part in (train_df, test_df):
        for col in df_part.select_dtypes(include=["category"]).columns:
            # remove_unused_categories is inplace on a Series-like when
            # reassigned; ensure we keep the dtype as categorical but drop
            # levels that are not present in this subset
            df_part[col] = df_part[col].cat.remove_unused_categories()

    LOGGER.info("Loaded %d train rows and %d test rows", len(train_df), len(test_df))
    return train_df, test_df


def separate_features_target(
    df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target columns.

    Args:
    ----
        df: Input dataframe.
        target_col: Name of target column.

    Returns:
    -------
        Tuple of (features_df, target_series).

    """
    features = df.drop(columns=[target_col])
    target = cast(pd.Series, df[target_col])
    return features, target


def binarize_target(y: pd.Series) -> pd.Series:
    """Convert target to binary (yes -> 1, other -> 0).

    Args:
    ----
        y: Target series with "yes"/other values.

    Returns:
    -------
        Binary series (0/1).

    """
    return (y.astype(str) == "yes").astype(int)


def fit_onehot_encoder(X_train: pd.DataFrame) -> OneHotEncoder:
    """Fit OneHotEncoder on training data.

    Args:
    ----
        X_train: Training features (all categorical).

    Returns:
    -------
        Fitted OneHotEncoder.

    """
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        drop=None,
    )
    encoder.fit(X_train)
    LOGGER.debug("Fitted OneHotEncoder on %d features", X_train.shape[1])
    return encoder


def transform_with_onehot(encoder: OneHotEncoder, X: pd.DataFrame) -> np.ndarray:
    """Transform features using fitted OneHotEncoder.

    Args:
    ----
        encoder: Fitted OneHotEncoder.
        X: Features to transform.

    Returns:
    -------
        Transformed array (one-hot encoded).

    """
    result = encoder.transform(X)
    return np.asarray(result)


def fit_scaler(X_train_encoded: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on encoded training data.

    Args:
    ----
        X_train_encoded: One-hot encoded training features.

    Returns:
    -------
        Fitted StandardScaler.

    """
    scaler = StandardScaler()
    scaler.fit(X_train_encoded)
    LOGGER.debug("Fitted StandardScaler on %d features", X_train_encoded.shape[1])
    return scaler


def transform_with_scaler(scaler: StandardScaler, X_encoded: np.ndarray) -> np.ndarray:
    """Transform encoded features using fitted StandardScaler.

    Args:
    ----
        scaler: Fitted StandardScaler.
        X_encoded: One-hot encoded features.

    Returns:
    -------
        Scaled array.

    """
    result = scaler.transform(X_encoded)
    return np.asarray(result)


def evaluate_C_on_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    C: float,
    random_state: int,
) -> float:
    """Evaluate a C value on a single CV fold with fold-level preprocessing.

    Args:
    ----
        X_train: Full training features.
        y_train: Full training target.
        train_idx: Indices for fold training.
        val_idx: Indices for fold validation.
        C: Regularization parameter to evaluate.
        random_state: Random state for model.

    Returns:
    -------
        Logloss on the validation fold.

    """
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Fit encoder/scaler on fold train only
    encoder = fit_onehot_encoder(X_fold_train)
    X_fold_train_enc = transform_with_onehot(encoder, X_fold_train)
    X_fold_val_enc = transform_with_onehot(encoder, X_fold_val)

    scaler = fit_scaler(X_fold_train_enc)
    X_fold_train_s = transform_with_scaler(scaler, X_fold_train_enc)
    X_fold_val_s = transform_with_scaler(scaler, X_fold_val_enc)

    model = LogisticRegression(
        penalty="l2",
        C=float(C),
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_fold_train_s, y_fold_train)

    proba_positive = model.predict_proba(X_fold_val_s)[:, 1]
    fold_logloss = float(log_loss(y_fold_val, proba_positive, labels=[0, 1]))
    return fold_logloss


def evaluate_C_across_folds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float,
    kfold: StratifiedKFold,
    random_state: int,
) -> float:
    """Evaluate a C value across all CV folds and return mean logloss.

    Args:
    ----
        X_train: Training features.
        y_train: Training target.
        C: Regularization parameter.
        kfold: StratifiedKFold object.
        random_state: Random state for model.

    Returns:
    -------
        Mean logloss across folds.

    """
    fold_loglosses: list[float] = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        fold_logloss = evaluate_C_on_fold(X_train, y_train, train_idx, val_idx, C, random_state)
        fold_loglosses.append(fold_logloss)
    return float(np.mean(fold_loglosses))


def find_best_C_no_leak(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = DEFAULT_CV_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> float:
    """Find best C via CV with fold-only preprocessing to avoid leakage.

    For each fold, fits OneHotEncoder and StandardScaler on the fold's
    training partition only, then evaluates on the fold's holdout
    partition. This prevents information leakage into preprocessing when
    selecting hyperparameters.

    Args:
    ----
        X_train: Training features (categorical only).
        y_train: Binary training target.
        cv_splits: Number of CV folds.
        random_state: Random state for CV.

    Returns:
    -------
        Best C value.

    """
    Cs = np.logspace(-2, 2, 20)
    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    best_C = float(Cs[0])
    best_logloss = float("inf")

    for C in Cs:
        mean_logloss = evaluate_C_across_folds(X_train, y_train, C, kfold, random_state)
        if mean_logloss < best_logloss:
            best_logloss = mean_logloss
            best_C = float(C)

    LOGGER.info("Best C (no-leak CV): %.4f (CV logloss: %.4f)", best_C, best_logloss)
    return best_C


def train_logistic_model(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    C: float,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegression:
    """Train final Logistic Regression model on full training set.

    Args:
    ----
        X_train_scaled: Scaled training features.
        y_train: Binary training target.
        C: Inverse regularization strength.
        random_state: Random state for model.

    Returns:
    -------
        Trained LogisticRegression.

    """
    model = LogisticRegression(
        penalty="l2",
        C=float(C),
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)
    LOGGER.info("Trained LogisticRegression with C=%.4f", C)
    return model


def predict_proba_logistic(model: LogisticRegression, X_scaled: np.ndarray) -> np.ndarray:
    """Get probability predictions from Logistic Regression model.

    Args:
    ----
        model: Trained LogisticRegression.
        X_scaled: Scaled features.

    Returns:
    -------
        Probability array (n_samples, 2) with probabilities for [class_0, class_1].

    """
    return model.predict_proba(X_scaled)


def compute_metrics(y_true: pd.Series, y_pred_proba: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics matching CatBoost baseline.

    Args:
    ----
        y_true: True binary target.
        y_pred_proba: Predicted probabilities (n_samples, 2) or (n_samples,).

    Returns:
    -------
        Dictionary with logloss, auc, accuracy, f1, recall.

    """
    if y_pred_proba.ndim == 2:
        y_pred_proba_positive = y_pred_proba[:, 1]
    else:
        y_pred_proba_positive = y_pred_proba

    y_pred_bin = (y_pred_proba_positive >= 0.5).astype(int)

    # Handle AUC edge case
    auc_value: float
    if y_true.nunique() < 2:
        auc_value = float("nan")
    else:
        try:
            auc_value = float(roc_auc_score(y_true, y_pred_proba_positive))
        except ValueError:
            auc_value = float("nan")

    metrics = {
        "logloss": float(log_loss(y_true, y_pred_proba_positive, labels=[0, 1])),
        "auc": auc_value,
        "accuracy": float(accuracy_score(y_true, y_pred_bin)),
        "f1": float(f1_score(y_true, y_pred_bin)),
        "recall": float(recall_score(y_true, y_pred_bin)),
    }
    return metrics


def prepare_train_test_data(
    base_dir: str | None,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and prepare train/test data for baseline.

    Args:
    ----
        base_dir: Base directory for data.
        target_col: Target column name.

    Returns:
    -------
        Tuple of (X_train, y_train, X_test, y_test) with binarized targets.

    """
    train_df, test_df = load_splits(base_dir)
    X_train, y_train_raw = separate_features_target(train_df, target_col)
    X_test, y_test_raw = separate_features_target(test_df, target_col)
    y_train = binarize_target(y_train_raw)
    y_test = binarize_target(y_test_raw)
    return X_train, y_train, X_test, y_test


def fit_preprocessing_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[OneHotEncoder, StandardScaler, np.ndarray, np.ndarray]:
    """Fit preprocessing on train and transform train/test.

    Args:
    ----
        X_train: Training features.
        X_test: Test features.

    Returns:
    -------
        Tuple of (encoder, scaler, X_train_scaled, X_test_scaled).

    """
    onehot_encoder = fit_onehot_encoder(X_train)
    X_train_encoded = transform_with_onehot(onehot_encoder, X_train)
    X_test_encoded = transform_with_onehot(onehot_encoder, X_test)

    scaler = fit_scaler(X_train_encoded)
    X_train_scaled = transform_with_scaler(scaler, X_train_encoded)
    X_test_scaled = transform_with_scaler(scaler, X_test_encoded)
    return onehot_encoder, scaler, X_train_scaled, X_test_scaled


def train_and_evaluate_ridge_baseline(
    base_dir: str | None = None,
    target_col: str = DEFAULT_TARGET_COLUMN,
    cv_splits: int = DEFAULT_CV_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train and evaluate Ridge baseline on categorical features.

    Pipeline:
    1. Load train/test splits
    2. OneHotEncoding (fitted on train only)
    3. StandardScaler (fitted on train only)
    4. CV on train to find best alpha
    5. Train final model
    6. Evaluate on test

    Args:
    ----
        base_dir: Optional base directory for data files.
        target_col: Target column name.
        cv_splits: Number of CV folds.
        random_state: Random state for reproducibility.

    Returns:
    -------
        Dictionary with model, encoders, metrics, and alpha.

    """
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_train_test_data(base_dir, target_col)

    # CV to find best C with no leakage (fold-level preprocessing)
    best_C = find_best_C_no_leak(X_train, y_train, cv_splits, random_state)

    # Fit preprocessing on full train only, then transform train/test
    onehot_encoder, scaler, X_train_scaled, X_test_scaled = fit_preprocessing_pipeline(
        X_train, X_test
    )

    # Train final model
    model = train_logistic_model(X_train_scaled, y_train, best_C, random_state)

    # Evaluate on test
    y_test_proba = predict_proba_logistic(model, X_test_scaled)
    test_metrics = compute_metrics(y_test, y_test_proba)

    LOGGER.info("Ridge baseline test metrics: %s", test_metrics)

    return {
        "model": model,
        "onehot_encoder": onehot_encoder,
        "scaler": scaler,
        "C": best_C,
        "test_metrics": test_metrics,
    }
