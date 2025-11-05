"""Model evaluation helpers for the CatBoost pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, EFstrType, Pool  # type: ignore
from matplotlib.figure import Figure
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2_contingency  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.metrics import accuracy_score, log_loss, recall_score, roc_auc_score

from ..constants import DEFAULT_TARGET_COLUMN, PLOTS_DIR, SHAP_PLOT_FILENAME
from ..utils import (
    create_eval_run_directory,
    detect_categorical_columns,
    ensure_dir,
    fill_categorical_na,
    get_logger,
    model_path,
    render_eval_markdown,
    save_json,
    splits_parquet_path,
)

LOGGER = get_logger(__name__)
_SPLIT_CACHE: dict[str, pd.DataFrame] = {}


def load_split_dataframe(split: str, base_dir: str | None = None) -> pd.DataFrame:
    """Return a cached dataframe for the requested split."""
    if split not in _SPLIT_CACHE:
        path = splits_parquet_path(base_dir)
        LOGGER.info("Loading split '%s' from %s", split, path)
        full_df = pd.read_parquet(path)
        df_split = cast(pd.DataFrame, full_df[full_df["split"] == split]).copy()
        if "split" in df_split.columns:
            df_split = df_split.drop(columns=["split"])
        # Ensure categorical columns only contain categories present in this
        # split. When the combined splits artifact was saved the categorical
        # dtype may contain the union of categories across splits which can
        # leak information into downstream preprocessing (OneHotEncoder,
        # CatBoost's handling of categorical features, etc.). Removing unused
        # categories here prevents that subtle leakage channel.
        for col in df_split.select_dtypes(include=["category"]).columns:
            df_split[col] = df_split[col].cat.remove_unused_categories()

        _SPLIT_CACHE[split] = df_split

    return _SPLIT_CACHE[split].copy()


def load_model() -> CatBoostClassifier:
    """Load the trained CatBoost model from disk."""
    model = CatBoostClassifier()
    path = model_path()
    LOGGER.info("Loading model from %s", path)
    model.load_model(str(path))
    # Best-effort: annotate the model with the path it was loaded from.
    # Some C-extension classes (like CatBoostClassifier) may disallow
    # setting new attributes; ignore failures in that case.
    try:
        model.loaded_path = str(path)
    except Exception:
        pass
    return model


# Categorical detection centralized in utils.detect_categorical_columns


def evaluate_model(model: Any, target_col: str = DEFAULT_TARGET_COLUMN) -> dict[str, float]:
    """Evaluate the model on the test split."""
    df_test = load_split_dataframe("test")
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    X_test_prepared = fill_categorical_na(X_test)
    cat_features = detect_categorical_columns(X_test_prepared)
    test_pool = Pool(X_test_prepared, cat_features=cat_features)
    y_pred_proba = model.predict_proba(test_pool)[:, 1]

    y_true_bin = (y_test.astype(str) == "yes").astype(int)
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)

    # Robust AUC handling for single-class edge cases
    if y_true_bin.nunique() < 2:
        auc_value = float("nan")
    else:
        try:
            auc_value = float(roc_auc_score(y_true_bin, y_pred_proba))
        except ValueError:
            auc_value = float("nan")

    metrics = {
        "logloss": float(log_loss(y_true_bin, y_pred_proba, labels=[0, 1])),
        "auc": auc_value,
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "f1": float(f1_score(y_true_bin, y_pred_bin)),
        "recall": float(recall_score(y_true_bin, y_pred_bin)),
    }
    LOGGER.info("Evaluation metrics on test split: %s", metrics)
    return metrics


def _get_predictions(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    X_prepared = fill_categorical_na(X_test)
    cat_features = detect_categorical_columns(X_prepared)
    test_pool = Pool(X_prepared, cat_features=cat_features)
    return model.predict_proba(test_pool)[:, 1]


def _compute_overall_metrics(y_test: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_bin = (y_test.astype(str) == "yes").astype(int)
    # Guard against single-class edge case for AUC
    if y_true_bin.nunique() < 2:
        auc_value = float("nan")
    else:
        try:
            auc_value = float(roc_auc_score(y_true_bin, y_pred))
        except ValueError:
            auc_value = float("nan")

    return {
        "logloss": float(log_loss(y_true_bin, y_pred, labels=[0, 1])),
        "auc": auc_value,
    }


def _compute_group_auc(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float("nan")
    try:
        y_true_bin = (pd.Series(y_true).astype(str) == "yes").astype(int)
        return float(roc_auc_score(y_true_bin, y_pred))
    except ValueError:
        return float("nan")


def _compute_group_logloss(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    try:
        y_true_bin = (pd.Series(y_true).astype(str) == "yes").astype(int)
        return float(log_loss(y_true_bin, y_pred, labels=[0, 1]))
    except ValueError:
        return float("nan")


def _compute_metrics_by_group(
    y_test: pd.Series, y_pred: np.ndarray, groups: pd.Series
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for group in groups.unique():
        mask = groups == group
        metrics[str(group)] = {
            "auc": _compute_group_auc(y_test[mask], y_pred[mask]),
            "logloss": _compute_group_logloss(y_test[mask], y_pred[mask]),
            "count": float(mask.sum()),
        }
    return metrics


def _compute_auc_gap(metrics_by_group: dict[str, dict[str, float]]) -> float:
    aucs = [values["auc"] for values in metrics_by_group.values() if pd.notna(values["auc"])]
    return float(max(aucs) - min(aucs)) if len(aucs) >= 2 else float("nan")


def evaluate_fairness_by_group(
    model: Any,
    group_col: str = "gender",
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> dict[str, Any]:
    """Evaluate fairness metrics for a given group column on the test split."""
    df_test = load_split_dataframe("test")
    if group_col not in df_test.columns:
        raise ValueError(f"Group column '{group_col}' is missing from the test dataset.")

    X_test = df_test.drop(columns=[target_col])
    y_test = cast(pd.Series, df_test[target_col])
    groups = df_test[group_col].astype(str)

    y_pred = _get_predictions(model, X_test)
    overall = _compute_overall_metrics(y_test, y_pred)
    metrics_by_group = _compute_metrics_by_group(y_test, y_pred, groups)
    auc_gap = _compute_auc_gap(metrics_by_group)

    return {
        "overall": overall,
        "by_group": metrics_by_group,
        "gaps": {"auc_gap": auc_gap},
    }


@dataclass
class ShapSummary:
    """Container for SHAP statistics."""

    feature_names: list[str]
    mean_shap_values: np.ndarray
    ordered_feature_names: list[str]
    ordered_mean_shap_values: np.ndarray


def _compute_shap_values(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    cat_features = detect_categorical_columns(X_test)
    test_pool = Pool(X_test, cat_features=cat_features)
    shap_values = model.get_feature_importance(test_pool, type=EFstrType.ShapValues)
    shap_array = np.asarray(shap_values)
    return shap_array[:, :-1]


def compute_shap_summary(
    model: Any,
    target_col: str = DEFAULT_TARGET_COLUMN,
    split: str = "test",
) -> ShapSummary:
    """Compute SHAP mean impacts for the requested split."""
    df_split = load_split_dataframe(split)
    X_split = df_split.drop(columns=[target_col])
    X_prepared = fill_categorical_na(X_split)

    shap_values = _compute_shap_values(model, X_prepared)
    feature_names = list(X_split.columns)
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP values dimension does not match feature count: "
            f"{shap_values.shape[1]} vs {len(feature_names)}"
        )

    mean_shap = shap_values.mean(axis=0)
    order = np.argsort(-np.abs(mean_shap))
    return ShapSummary(
        feature_names=feature_names,
        mean_shap_values=mean_shap,
        ordered_feature_names=[feature_names[i] for i in order],
        ordered_mean_shap_values=mean_shap[order],
    )


def _create_shap_plot(mean_shap: np.ndarray, feature_names: list[str]) -> Figure:
    from matplotlib.patches import Patch

    colors = ["#2ecc71" if value > 0 else "#e74c3c" for value in mean_shap]
    figure, axis = plt.subplots(figsize=(10, max(5, 0.4 * len(feature_names))))
    axis.barh(
        range(len(feature_names)),
        mean_shap[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    axis.set_yticks(range(len(feature_names)))
    axis.set_yticklabels(feature_names[::-1], fontsize=10)
    axis.set_xlabel("Average SHAP impact", fontsize=11, fontweight="medium")
    axis.set_title(
        "Directional SHAP impact on treatment probability",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    axis.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axis.grid(axis="x", alpha=0.2, linestyle="-", linewidth=0.5)
    axis.set_axisbelow(True)
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.85, label="Increases probability"),
        Patch(facecolor="#e74c3c", alpha=0.85, label="Decreases probability"),
    ]
    axis.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=9)
    plt.tight_layout()
    return figure


def save_shap_summary_plot(
    model: Any,
    target_col: str = DEFAULT_TARGET_COLUMN,
    output_path: str | None = None,
    summary: ShapSummary | None = None,
) -> str:
    """Compute SHAP values on the test split and save the bar plot."""
    summary = summary or compute_shap_summary(model, target_col=target_col)
    figure = _create_shap_plot(summary.ordered_mean_shap_values, summary.ordered_feature_names)
    default_path = PLOTS_DIR / SHAP_PLOT_FILENAME
    output = Path(output_path) if output_path is not None else default_path
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    LOGGER.info("SHAP summary plot saved to %s", output)
    return str(output)


def _categorize_for_cramers(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype("object")
    if is_numeric_dtype(series):
        bins = min(10, series.nunique(dropna=True))
        return pd.qcut(series, q=bins, duplicates="drop").astype(str)
    return series.astype(str)


def build_shap_vs_cramers_table(
    summary: ShapSummary,
    target_series: pd.Series,
    base_df: pd.DataFrame,
) -> list[dict[str, str | float]]:
    """Build a comparison table of mean SHAP and Cramér's V per feature."""
    shap_series = pd.Series(summary.mean_shap_values, index=summary.feature_names, dtype="float64")
    table: list[dict[str, str | float]] = []

    for feature in summary.feature_names:
        if feature not in base_df.columns:
            continue
        feature_series = base_df[feature]
        data = pd.DataFrame({"feature": feature_series, "target": target_series}).dropna()
        cramers_v = float("nan")
        if not data.empty:
            prepared_feature = _categorize_for_cramers(data["feature"])
            contingency = pd.crosstab(prepared_feature, data["target"].astype(str))
            if contingency.size > 0 and min(contingency.shape) > 1:
                chi2 = chi2_contingency(contingency)[0]
                n = contingency.to_numpy().sum()
                min_dim = min(contingency.shape)
                if n > 0 and min_dim > 1:
                    cramers_v = float(math.sqrt(chi2 / (n * (min_dim - 1))))

        table.append(
            {
                "feature": feature,
                "mean_shap": float(shap_series.get(feature, float("nan"))),
                "cramers_v": cramers_v,
            }
        )

    table.sort(key=lambda row: abs(cast(float, row["mean_shap"])), reverse=True)
    return table


def save_eval_results(
    metrics: dict[str, float],
    fairness_gender: dict[str, Any] | None = None,
    fairness_country: dict[str, Any] | None = None,
    shap_plot_path: str | None = None,
    shap_vs_cramers_table: Sequence[Mapping[str, Any]] | None = None,
    run_dir: Path | None = None,
    model: Any | None = None,
) -> dict[str, str]:
    """Persist evaluation artifacts as JSON and Markdown.

    Args:
    ----
        metrics: Overall evaluation metrics.
        fairness_gender: Fairness metrics by gender.
        fairness_country: Fairness metrics by country.
        shap_plot_path: Path to the SHAP summary plot.
        shap_vs_cramers_table: Table comparing SHAP and Cramér's V.
        run_dir: Optional custom run directory.
        model: Trained model to extract hyperparameters from.

    Returns:
    -------
        Paths to saved artifacts.

    """
    base_dir, timestamp = create_eval_run_directory(run_dir)
    LOGGER.info("Saving evaluation results to %s", base_dir)

    json_payload: dict[str, Any] = {
        "metrics": metrics,
        "fairness_gender": fairness_gender,
        "fairness_country": fairness_country,
        "shap_plot_path": shap_plot_path,
        "shap_vs_cramers": list(shap_vs_cramers_table) if shap_vs_cramers_table else None,
        "timestamp": timestamp,
        "hyperparameters": model.get_params() if model else None,
    }
    json_path = base_dir / "eval_results.json"
    save_json(json_payload, json_path)

    fairness_sections = (
        ("Fairness by Gender", fairness_gender),
        ("Fairness by Country", fairness_country),
    )
    markdown = render_eval_markdown(
        metrics,
        fairness_sections,
        shap_plot_path,
        timestamp,
        shap_vs_cramers_table=shap_vs_cramers_table,
    )
    md_path = base_dir / "eval_results.md"
    ensure_dir(md_path.parent)
    md_path.write_text(markdown, encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "dir": str(base_dir),
    }
