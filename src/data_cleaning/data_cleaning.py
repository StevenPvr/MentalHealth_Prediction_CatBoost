"""Data cleaning helpers orchestrating calls to shared utilities."""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike

import pandas as pd

from ..constants import DEFAULT_MISSING_TOKENS
from ..utils import (
    cleaned_dataset_csv_path,
    cleaned_dataset_parquet_path,
    drop_duplicate_columns,
    get_logger,
    normalize_text_dataframe,
    standardize_missing_tokens,
    write_csv,
    write_parquet,
)

LOGGER = get_logger(__name__)


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns using the shared utility."""
    LOGGER.info("Removing duplicate columns on dataframe with shape %s", df.shape)
    result = drop_duplicate_columns(df)
    LOGGER.info("Duplicate columns removed: %d", df.shape[1] - result.shape[1])
    return result


def handle_missing_values(df: pd.DataFrame, tokens: Iterable[str] | None = None) -> pd.DataFrame:
    """Standardise missing tokens into ``pd.NA`` values."""
    tokens_to_use = DEFAULT_MISSING_TOKENS if tokens is None else tuple(tokens)
    LOGGER.info("Standardising missing tokens: using %d replacement tokens", len(tokens_to_use))
    return standardize_missing_tokens(df, tokens_to_use)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common text normalisation rules to string columns."""
    LOGGER.info("Normalising textual columns for dataframe with shape %s", df.shape)
    return normalize_text_dataframe(df)


def save_cleaned_data(df: pd.DataFrame, base_dir: str | PathLike[str] | None = None) -> None:
    """Persist the cleaned dataset both as CSV and Parquet files."""
    csv_path = cleaned_dataset_csv_path(base_dir)
    parquet_path = cleaned_dataset_parquet_path(base_dir)
    LOGGER.info("Saving cleaned dataset to %s and %s", csv_path.name, parquet_path.name)
    write_csv(df, csv_path)
    write_parquet(df, parquet_path)
    LOGGER.info("Cleaned dataset saved successfully")
