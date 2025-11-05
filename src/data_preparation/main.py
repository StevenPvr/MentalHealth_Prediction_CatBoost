"""Entry point for the data preparation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import DEFAULT_RANDOM_STATE
from src.data_preparation.data_preparation import (
    load_cleaned_dataset,
    save_splits,
    split_train_test,
)
from src.utils import get_logger

LOGGER = get_logger(__name__)


def prepare_data() -> None:
    """Execute the dataset splitting workflow."""
    LOGGER.info("Loading cleaned dataset")
    df = load_cleaned_dataset()
    LOGGER.info("Dataset shape: %d rows, %d columns", len(df), len(df.columns))

    LOGGER.info("Splitting dataset into train/test")
    train, test = split_train_test(df, random_state=DEFAULT_RANDOM_STATE)
    LOGGER.info("Train rows: %d", len(train))
    LOGGER.info("Test rows: %d", len(test))

    LOGGER.info("Saving dataset splits")
    save_splits(train, test)


if __name__ == "__main__":
    prepare_data()
