"""Common utilities for the Mental Health prediction project."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.util import hash_pandas_object

from .constants import (
    CLEANED_DATASET_CSV_FILENAME,
    CLEANED_DATASET_FILENAME,
    DATA_DIR,
    DEFAULT_DATASET_FILENAME,
    ENCODERS_MAPPINGS_FILENAME,
    MODEL_DIR,
    MODEL_FILENAME,
    RESULTS_DIR,
    SPLITS_CSV_FILENAME,
    SPLITS_PARQUET_FILENAME,
)
from .logging_setup import configure_logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger for the project."""
    configure_logging()
    return logging.getLogger(name or "mental_health")


def ensure_dir(path: Path) -> None:
    """Create the given directory and its parents when missing."""
    path.mkdir(parents=True, exist_ok=True)


def get_data_dir(base: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the active data directory.

    Args:
    ----
        base: Optional override directory, typically injected by tests.

    Returns:
    -------
        Path pointing to the active data directory.

    """
    if base is not None:
        return Path(base)

    env_override = os.environ.get("MENTAL_HEALTH_DATA_DIR")
    if env_override:
        return Path(env_override)

    return DATA_DIR


def get_dataset_path(
    filename: str = DEFAULT_DATASET_FILENAME,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Build a dataset path relative to the active data directory."""
    return get_data_dir(base_dir) / filename


def dataset_csv_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the path to the raw CSV dataset."""
    return get_dataset_path(DEFAULT_DATASET_FILENAME, base_dir)


def cleaned_dataset_parquet_path(
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the path to the cleaned dataset parquet file."""
    return get_dataset_path(CLEANED_DATASET_FILENAME, base_dir)


def cleaned_dataset_csv_path(
    base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the path to the cleaned dataset CSV file."""
    return get_dataset_path(CLEANED_DATASET_CSV_FILENAME, base_dir)


def splits_parquet_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the path to the parquet split artefact."""
    return get_dataset_path(SPLITS_PARQUET_FILENAME, base_dir)


def splits_csv_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the path to the CSV split artefact."""
    return get_dataset_path(SPLITS_CSV_FILENAME, base_dir)


def encoders_mappings_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the path to the encoders mappings JSON file."""
    return get_dataset_path(ENCODERS_MAPPINGS_FILENAME, base_dir)


def model_path() -> Path:
    """Return the path to the persisted CatBoost model."""
    ensure_dir(MODEL_DIR)
    return MODEL_DIR / MODEL_FILENAME


def load_dataset(base_dir: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the raw dataset from disk."""
    return pd.read_csv(dataset_csv_path(base_dir))


def save_json(data: Any, path: Path) -> None:
    """Persist a JSON payload to disk using UTF-8 encoding."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns while keeping the first occurrence."""
    if df.empty or df.shape[1] <= 1:
        return df.copy()

    def _normalise_unhashable(value: Any) -> Any:
        if isinstance(value, list | tuple | set):
            return tuple(value)
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        return value

    def _series_signature(series: pd.Series) -> bytes:
        try:
            hashed = hash_pandas_object(series, index=False)
        except TypeError:
            # Fallback conversion for unhashable entries such as lists or dictionaries.
            hashed = hash_pandas_object(series.map(_normalise_unhashable), index=False)
        return hashed.to_numpy().tobytes()

    seen_hashes: dict[bytes, str] = {}
    columns_to_keep: list[str] = []

    for column in df.columns:
        col_hash = _series_signature(df[column])
        match = seen_hashes.get(col_hash)
        if match is not None and df[column].equals(df[match]):
            continue

        seen_hashes[col_hash] = column
        columns_to_keep.append(column)

    return df.loc[:, columns_to_keep].copy()


def fill_categorical_na(df: pd.DataFrame, placeholder: str = "missing") -> pd.DataFrame:
    """Replace missing values in categorical columns with a placeholder token."""
    result = df.copy()

    for column in result.select_dtypes(include=["category"]).columns:
        series = result[column]
        if placeholder not in series.cat.categories:
            series = series.cat.add_categories([placeholder])
        result[column] = series.fillna(placeholder)

    return result


def standardize_missing_tokens(df: pd.DataFrame, tokens: Iterable[str]) -> pd.DataFrame:
    """Replace the provided tokens by ``pd.NA`` to harmonise missing values."""
    token_list = list(tokens)
    if not token_list:
        return df.copy()

    result = df.copy()
    result.replace(token_list, pd.NA, inplace=True)
    return result


def normalize_text_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise textual columns by trimming spaces and lowering the case."""
    result = df.copy()
    result.columns = result.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    object_columns = result.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.replace(r"\s+", "_", regex=True)
        )

    return result


def write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Persist a DataFrame to CSV and ensure the destination directory exists."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Persist a DataFrame to Parquet and ensure the destination directory exists."""
    ensure_dir(path.parent)
    df.to_parquet(path, index=False, **kwargs)


def write_metrics_artifacts(
    metrics: Sequence[Mapping[str, Any]],
    subdir: str,
    base_filename: str,
) -> dict[str, Path]:
    """Persist metrics both as CSV and JSON under the results directory."""
    target_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    ensure_dir(target_dir)

    dataframe = pd.DataFrame(list(metrics))
    csv_path = target_dir / f"{base_filename}.csv"
    json_path = target_dir / f"{base_filename}.json"

    dataframe.to_csv(csv_path, index=False)
    save_json(list(metrics), json_path)

    return {"csv": csv_path, "json": json_path}


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of columns with pandas 'category' dtype.

    WHY: Multiple modules need to discover categorical features consistently
    for CatBoost and preprocessing. Centralizing this logic prevents subtle
    divergences and keeps the pipeline DRY.

    Args:
    ----
    df: Input DataFrame.

    Returns:
    -------
    List of column names whose dtype is 'category'.

    """
    return [column for column in df.columns if df[column].dtype.name == "category"]


# ---------------------------------------------------------------------------
# Reporting helpers used by the evaluation pipeline
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "logloss": "Logloss",
    "auc": "AUC",
    "accuracy": "Accuracy",
    "f1": "F1",
    "recall": "Recall",
    "auc_gap": "AUC gap",
}

METRIC_ORDER: Sequence[str] = ("logloss", "auc", "accuracy", "f1", "recall")


def create_eval_run_directory(run_dir: Path | None = None) -> tuple[Path, str]:
    """Create and return a timestamped evaluation directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = run_dir or (RESULTS_DIR / f"run_{timestamp}")
    ensure_dir(base_dir)
    return base_dir, timestamp


def _format_metric_line(name: str, value: float) -> str:
    label = METRIC_LABELS.get(name, name.title())
    return f"- {label}: {value:.4f}"


def _format_optional_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NaN"

    if math.isnan(numeric):
        return "NaN"

    return f"{numeric:.4f}"


def render_global_metrics_markdown(metrics: Mapping[str, float]) -> list[str]:
    """Return Markdown lines representing the global metrics."""
    lines = ["## Global Metrics", ""]
    for key in METRIC_ORDER:
        if key in metrics:
            lines.append(_format_metric_line(key, float(metrics[key])))
    remaining = sorted(set(metrics.keys()) - set(METRIC_ORDER))
    for key in remaining:
        lines.append(_format_metric_line(key, float(metrics[key])))
    return lines


def render_fairness_markdown(title: str, fairness: Mapping[str, Any] | None) -> list[str]:
    """Create a Markdown section describing the fairness metrics."""
    if fairness is None:
        return []

    overall = fairness.get("overall", {})
    gaps = fairness.get("gaps", {})
    by_group = fairness.get("by_group", {})

    lines = ["", f"## {title}", ""]
    if "auc" in overall:
        lines.append(f"- Overall AUC: {float(overall['auc']):.4f}")
    if "auc_gap" in gaps:
        lines.append(f"- AUC gap: {float(gaps['auc_gap']):.4f}")

    for group, metrics in sorted(by_group.items()):
        auc = float(metrics.get("auc", float("nan")))
        logloss = float(metrics.get("logloss", float("nan")))
        count = int(metrics.get("count", 0))
        lines.append(f"  - {group}: AUC={auc:.4f} | logloss={logloss:.4f} | n={count}")

    return lines


def _render_shap_vs_cramers_markdown(table: Sequence[Mapping[str, Any]]) -> list[str]:
    if not table:
        return []

    lines = [
        "",
        "## SHAP vs Cramér's V",
        "",
        "| Feature | Mean SHAP | Cramér's V |",
        "| --- | ---: | ---: |",
    ]

    for row in table:
        feature = str(row.get("feature", ""))
        mean_shap = _format_optional_float(row.get("mean_shap"))
        cramers_v = _format_optional_float(row.get("cramers_v"))
        lines.append(f"| {feature} | {mean_shap} | {cramers_v} |")

    return lines


def render_eval_markdown(
    metrics: Mapping[str, float],
    fairness_sections: Sequence[tuple[str, Mapping[str, Any] | None]],
    shap_plot_path: str | None,
    timestamp: str,
    *,
    shap_vs_cramers_table: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Assemble the different sections into a full Markdown report."""
    lines: list[str] = ["# Evaluation Results", "", f"- Timestamp: {timestamp}", ""]
    lines.extend(render_global_metrics_markdown(metrics))

    for title, payload in fairness_sections:
        lines.extend(render_fairness_markdown(title, payload))

    if shap_plot_path:
        lines.extend(["", "## SHAP", "", f"![SHAP Summary]({shap_plot_path})"])

    if shap_vs_cramers_table:
        lines.extend(_render_shap_vs_cramers_markdown(shap_vs_cramers_table))

    return "\n".join(lines) + "\n"
