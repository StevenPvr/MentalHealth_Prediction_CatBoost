"""Data preparation utilities for the CatBoost training pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd  # type: ignore
from pandas import CategoricalDtype  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from ..constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TARGET_COLUMN,
    RARE_CATEGORY_LABEL,
    RARE_CATEGORY_THRESHOLD,
)
from ..utils import (
    cleaned_dataset_parquet_path,
    encoders_mappings_path,
    get_logger,
    save_json,
    splits_csv_path,
    splits_parquet_path,
    write_metrics_artifacts,
)

LOGGER = get_logger(__name__)


def load_cleaned_dataset(base_dir: str | None = None) -> pd.DataFrame:
    """Load the cleaned dataset parquet file."""
    path = cleaned_dataset_parquet_path(base_dir)
    LOGGER.info("Loading cleaned dataset from %s", path)
    return pd.read_parquet(path)


def shuffle_dataset(df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Return a shuffled copy of the dataframe."""
    LOGGER.info("Shuffling dataset with random_state=%d", random_state)
    return df.sample(frac=1.0, random_state=random_state)


def _remap_rare_categories(
    series: pd.Series, kept_categories: Iterable[object], other_label: str
) -> pd.Series:
    result = series.copy()
    is_categorical = isinstance(result.dtype, CategoricalDtype)

    if is_categorical and other_label not in result.cat.categories:
        result = result.cat.add_categories([other_label])

    mask_rare = result.notna() & ~result.isin(kept_categories)
    result = result.mask(mask_rare, other_label)

    if is_categorical:
        result = result.cat.remove_unused_categories()

    return result


def group_rare_categories_in_column(
    train: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    min_frequency: float = RARE_CATEGORY_THRESHOLD,
    other_label: str = RARE_CATEGORY_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Group rare categories based on frequency observed in the train split.

    Operates on train and test splits only.
    Returns train, test, and the mapping of column to kept categories.
    """
    if column not in train.columns:
        return train, test, {}

    train_out, test_out = train.copy(), test.copy()
    train_series = cast(pd.Series, train_out[column])
    observed = train_series.dropna()

    if observed.empty:
        return train_out, test_out, {}

    frequencies_series = cast(pd.Series, observed.value_counts(normalize=True))
    filtered_series = cast(pd.Series, frequencies_series[frequencies_series >= min_frequency])
    kept_categories = filtered_series.index.tolist()

    LOGGER.info("Grouping categories rarer than %.2f in column '%s'", min_frequency, column)

    train_out[column] = _remap_rare_categories(train_series, kept_categories, other_label)
    if column in test_out.columns:
        test_out[column] = _remap_rare_categories(
            cast(pd.Series, test_out[column]), kept_categories, other_label
        )

    mapping = {column: kept_categories}
    return train_out, test_out, mapping


def _record_metric(
    step: str, before_rows: int, after_rows: int, note: str | None = None
) -> dict[str, object]:
    return {
        "step": step,
        "rows_before": before_rows,
        "rows_after": after_rows,
        "rows_delta": after_rows - before_rows,
        "note": note or "",
    }


def _build_split_summary(
    train: pd.DataFrame, test: pd.DataFrame, *, note: str
) -> dict[str, object]:
    total_rows = len(train) + len(test)
    return {
        "step": "split_train_test",
        "rows_before": total_rows,
        "rows_after": total_rows,
        "rows_delta": 0,
        "train_rows": len(train),
        "test_rows": len(test),
        "note": note,
    }


def _split_with_stratification(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df[target_col] if target_col in df.columns else None
    first, second = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return cast(pd.DataFrame, first), cast(pd.DataFrame, second)


def split_train_test(
    df: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train (80%) and test (20%)."""
    metrics: list[dict[str, object]] = []
    shuffled = shuffle_dataset(df, random_state=random_state)
    metrics.append(
        _record_metric(
            "shuffle_dataset",
            len(df),
            len(shuffled),
            f"random_state={random_state}",
        )
    )

    train, test = _split_with_stratification(shuffled, 0.2, random_state, target_col)

    metrics.append(
        _build_split_summary(
            train,
            test,
            note=f"random_state={random_state}",
        )
    )

    LOGGER.info(
        "Split completed: train=%d rows, test=%d rows",
        len(train),
        len(test),
    )

    artifacts = write_metrics_artifacts(metrics, "data_preparation", "data_preparation_metrics")
    LOGGER.info(
        "Data preparation metrics stored at %s and %s",
        artifacts["csv"],
        artifacts["json"],
    )

    return train, test


def convert_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every column to the ``category`` dtype."""
    result = df.copy()
    for column in result.columns:
        result[column] = result[column].astype("category")
    return result


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parquet_path: str | None = None,
    csv_path: str | None = None,
) -> None:
    """Persist the concatenated splits with split metadata (train/test only)."""
    train_copy, test_copy = train.copy(), test.copy()
    mappings = {}

    if "country" in train_copy.columns:
        train_copy, test_copy, mapping = group_rare_categories_in_column(
            train_copy, test_copy, column="country"
        )
        mappings.update(mapping)

    # Save mappings if any
    if mappings:
        mappings_path = encoders_mappings_path()
        LOGGER.info("Saving encoders mappings to %s", mappings_path)
        save_json(mappings, mappings_path)

    for frame, label in ((train_copy, "train"), (test_copy, "test")):
        frame["split"] = label

    combined = pd.concat([train_copy, test_copy], ignore_index=True)
    combined = convert_to_categorical(combined)

    target_parquet = parquet_path or splits_parquet_path()
    target_csv = csv_path or splits_csv_path()
    LOGGER.info(
        "Saving dataset splits to %s and %s",
        target_parquet,
        target_csv,
    )
    combined.to_parquet(target_parquet, index=False)
    combined.to_csv(target_csv, index=False)
