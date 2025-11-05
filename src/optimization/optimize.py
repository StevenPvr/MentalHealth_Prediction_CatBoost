"""Hyperparameter optimization using Optuna with stratified cross-validation.

Cross-validation is intentionally performed during HPO only (not during the
final evaluation). We optimize CatBoost hyperparameters via Optuna, evaluating
an aggregate metric across K folds on training data while excluding the
held-out test split.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool  # type: ignore
from optuna.pruners import MedianPruner
from sklearn.metrics import log_loss, roc_auc_score  # type: ignore
from sklearn.model_selection import StratifiedKFold  # type: ignore

from .. import model as model_module
from .. import utils as utils_module
from ..constants import (
    DEFAULT_CV_N_SPLITS,
    DEFAULT_HPO_TRIALS,
    DEFAULT_OPTIMIZATION_METRIC,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_EARLY_STOPPING_ROUNDS,
)
from ..training.training import separate_features_target
from ..utils import (
    detect_categorical_columns,
    ensure_dir,
    fill_categorical_na,
    get_logger,
    save_json,
    splits_parquet_path,
)

LOGGER = get_logger(__name__)


def _detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Delegate to utils.detect_categorical_columns."""
    return detect_categorical_columns(df)


def _load_train_val(target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load non-test data and return features and target.

    The held-out test split is never used here to prevent leakage.
    """
    path = splits_parquet_path()
    LOGGER.info("Loading HPO training data from %s", path)
    df = pd.read_parquet(path)
    df = cast(pd.DataFrame, df[df["split"].isin(["train", "val"])]).copy()
    # Drop split column after filtering but remove any categorical levels
    # that are not present in the train/val subset. This avoids leaking
    # categories coming from the held-out test split into HPO preprocessing
    # or CV folds.
    df = df.drop(columns=["split"])
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].cat.remove_unused_categories()
    return separate_features_target(df, target_col=target_col)


def _iterate_folds_scores(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: Sequence[str],
    indices: Iterable[tuple[np.ndarray, np.ndarray]],
) -> Iterable[dict[str, float]]:
    """Yield per-fold metrics after training on the fold and scoring its holdout.

    Train on each fold with an evaluation set and early stopping,
    then score on the holdout fold.
    """
    for train_idx, test_idx in indices:
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        train_pool = Pool(X_train, y_train, cat_features=list(cat_features))
        val_pool = Pool(X_val, y_val, cat_features=list(cat_features))
        # Train with eval_set and early stopping to select the best iteration
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        y_val_bin = (y_val.astype(str) == "yes").astype(int)
        trees = getattr(model, "tree_count_", None)
        trees_int = int(trees) if isinstance(trees, int | np.integer) else 0
        yield {
            "logloss": float(log_loss(y_val_bin, y_pred_proba)),
            "auc": float(roc_auc_score(y_val_bin, y_pred_proba)),
            "trees": float(trees_int),
        }


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest a broader set of CatBoost hyperparameters."""
    depth = trial.suggest_int("depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-3, 100.0, log=True)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    iterations = trial.suggest_int("iterations", 200, 2000, log=True)
    return {
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "bagging_temperature": bagging_temperature,
        "iterations": iterations,
    }


def _create_study(metric: str) -> optuna.study.Study:
    """Create and configure an Optuna study for the requested metric."""
    direction = "maximize" if metric == "auc" else "minimize"
    sampler = optuna.samplers.RandomSampler(seed=DEFAULT_RANDOM_STATE)
    pruner = MedianPruner(n_warmup_steps=2)
    return optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)


def _build_objective(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: Sequence[str],
    fold_indices: Iterable[tuple[np.ndarray, np.ndarray]],
    metric: str,
) -> Callable[[optuna.Trial], float]:
    """Return the Optuna objective function used during HPO.

    The objective trains a model per fold and aggregates metrics.
    """

    def _objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        model = model_module.create_catboost_model()
        # Pass only the sampled hyperparameters; model keeps use_best_model=True
        model.set_params(**params)

        fold_metrics: list[dict[str, float]] = []
        for i, scores in enumerate(
            _iterate_folds_scores(model, X, y, cat_features, fold_indices), start=1
        ):
            fold_metrics.append(scores)
            auc_mean = float(np.mean([m["auc"] for m in fold_metrics]))
            logloss_mean = float(np.mean([m["logloss"] for m in fold_metrics]))
            trees_vals = [m.get("trees", 0.0) for m in fold_metrics]
            trees_mean = float(np.mean(trees_vals)) if any(trees_vals) else 0.0
            current_value = auc_mean if metric == "auc" else logloss_mean
            trial.report(current_value, step=i)
            if trial.should_prune():
                trial.set_user_attr(
                    "metrics",
                    {"auc_mean": auc_mean, "logloss_mean": logloss_mean, "trees_mean": trees_mean},
                )
                raise optuna.TrialPruned()

        auc_mean = float(np.mean([m["auc"] for m in fold_metrics]))
        logloss_mean = float(np.mean([m["logloss"] for m in fold_metrics]))
        trees_vals = [m.get("trees", 0.0) for m in fold_metrics]
        trees_mean = float(np.mean(trees_vals)) if any(trees_vals) else 0.0
        trial.set_user_attr(
            "metrics",
            {"auc_mean": auc_mean, "logloss_mean": logloss_mean, "trees_mean": trees_mean},
        )
        for k, v in params.items():
            trial.set_user_attr(f"param_{k}", v)
        return float(auc_mean) if metric == "auc" else float(logloss_mean)

    return _objective


def _summarize_study(study: optuna.study.Study) -> list[dict[str, Any]]:
    """Collect a compact summary across trials with aggregated metrics."""
    summaries: list[dict[str, Any]] = []
    for t in study.trials:
        metrics = t.user_attrs.get("metrics")
        if not metrics or not {"auc_mean", "logloss_mean"} <= set(metrics.keys()):
            continue
        summaries.append({"number": t.number, "value": t.value, **t.params, **metrics})
    return summaries


@dataclass
class OptimizationResult:
    """Container for HPO outcome and summary."""

    best_params: dict[str, Any]
    summaries: list[dict[str, Any]]
    best_iterations: int | None = None


def optimize_hyperparameters(
    n_splits: int = DEFAULT_CV_N_SPLITS,
    target_col: str = DEFAULT_TARGET_COLUMN,
    metric: str = DEFAULT_OPTIMIZATION_METRIC,
    n_trials: int = DEFAULT_HPO_TRIALS,
) -> OptimizationResult:
    """Run Optuna HPO with stratified CV and return results.

    - metric: "auc" (maximize) or "logloss" (minimize)
    - Prevents leakage by excluding the held-out test split
    """
    if metric not in {"auc", "logloss"}:
        raise ValueError("metric must be either 'auc' or 'logloss'")

    X, y = _load_train_val(target_col)
    X = fill_categorical_na(X)
    cat_features = _detect_categorical_columns(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    fold_indices = list(skf.split(X, y))

    study = _create_study(metric)
    objective = _build_objective(
        X=X,
        y=y,
        cat_features=cat_features,
        fold_indices=fold_indices,
        metric=metric,
    )

    LOGGER.info(
        "Starting Optuna study (trials=%d, metric=%s, folds=%d)", n_trials, metric, n_splits
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_trial.params)
    trees_mean = study.best_trial.user_attrs.get("metrics", {}).get("trees_mean")
    best_iterations = int(round(float(trees_mean))) if trees_mean else None
    LOGGER.info(
        "Best params: %s | best_value=%.4f | best_iterations=%s",
        best_params,
        study.best_value,
        str(best_iterations),
    )
    summaries = _summarize_study(study)
    return OptimizationResult(
        best_params=best_params, summaries=summaries, best_iterations=best_iterations
    )


def save_optimization_artifacts(result: OptimizationResult) -> dict[str, str]:
    """Persist HPO results under results/optimization as JSON files."""
    # Read RESULTS_DIR from utils at call-time to allow test monkeypatching.
    base_dir = utils_module.RESULTS_DIR / "optimization"
    ensure_dir(base_dir)
    best_path = base_dir / "best_params.json"
    history_path = base_dir / "cv_history.json"

    save_json(result.best_params, best_path)
    save_json(result.summaries, history_path)

    LOGGER.info("Saved best params to %s and CV history to %s", best_path, history_path)
    return {"best_params": str(best_path), "cv_history": str(history_path), "dir": str(base_dir)}
