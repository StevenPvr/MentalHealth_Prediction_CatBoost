"""Entry point for the data cleaning pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_cleaning.data_cleaning import (
    handle_missing_values,
    normalize_text,
    remove_duplicate_columns,
    save_cleaned_data,
)
from src.utils import get_logger, load_dataset, write_metrics_artifacts

LOGGER = get_logger(__name__)


def _append_metric(
    metrics: list[dict[str, Any]],
    step: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    note: str | None = None,
) -> None:
    metrics.append(
        {
            "step": step,
            "rows_before": len(before_df),
            "rows_after": len(after_df),
            "columns_before": len(before_df.columns),
            "columns_after": len(after_df.columns),
            "rows_delta": len(after_df) - len(before_df),
            "columns_delta": len(after_df.columns) - len(before_df.columns),
            "note": note or "",
        }
    )


def clean_data() -> None:
    """Execute the full data cleaning workflow."""
    metrics: list[dict[str, Any]] = []

    LOGGER.info("Loading raw dataset")
    df = load_dataset()
    _append_metric(metrics, "load_dataset", df, df, "Raw dataset")

    LOGGER.info("Removing duplicate columns")
    cleaned = remove_duplicate_columns(df)
    _append_metric(metrics, "remove_duplicate_columns", df, cleaned)

    LOGGER.info("Normalising missing tokens")
    with_missing_handled = handle_missing_values(cleaned)
    _append_metric(metrics, "handle_missing_values", cleaned, with_missing_handled)

    LOGGER.info("Normalising textual columns")
    normalised = normalize_text(with_missing_handled)
    _append_metric(metrics, "normalize_text", with_missing_handled, normalised)

    LOGGER.info("Saving cleaned dataset")
    save_cleaned_data(normalised)

    artifacts = write_metrics_artifacts(metrics, "data_cleaning", "clean_data_metrics")
    LOGGER.info(
        "Cleaning metrics stored at %s and %s",
        artifacts["csv"],
        artifacts["json"],
    )


if __name__ == "__main__":
    clean_data()
