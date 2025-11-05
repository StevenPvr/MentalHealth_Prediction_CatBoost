"""Tests to assert no categorical leakage when loading persisted splits.

These tests write a small `splits.parquet` artifact containing train/val/test
rows where the categorical column has category levels that are only present in
some splits. The pipeline code must remove unused categories on a per-split
basis to avoid leaking category levels from one split into preprocessing of
another.

Each test is executable directly (python test_leakage.py) to satisfy the
project's testing conventions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.baseline import logistic_regression
from src.training import training as training_module
from src.optimization import optimize as optimize_module


def _write_splits_parquet(tmp_path: Path) -> Path:
    # Create a small combined dataframe with categorical dtype whose categories
    # include values that occur only in some splits.
    df = pd.DataFrame(
        {
            "country": pd.Categorical(["A", "B", "A", "C"], categories=["A", "B", "C"]),
            "treatment": ["yes", "no", "no", "yes"],
            "split": ["train", "test", "train", "val"],
        }
    )

    target = tmp_path / "splits.parquet"
    df.to_parquet(target, index=False)
    return target


def test_baseline_load_splits_removes_unused_categories(tmp_path: Path) -> None:
    _write_splits_parquet(tmp_path)

    # Point the data directory resolver to tmp_path by env var so functions
    # that resolve the splits path without an explicit base_dir use this file.
    os.environ["MENTAL_HEALTH_DATA_DIR"] = str(tmp_path)

    train_df, test_df = logistic_regression.load_splits()

    # Train subset only contains 'A' values -> its categories must be reduced.
    train_cats = list(train_df["country"].cat.categories)
    assert "A" in train_cats
    assert "B" not in train_cats and "C" not in train_cats

    # Test subset only contains 'B' -> its categories must be reduced.
    test_cats = list(test_df["country"].cat.categories)
    assert "B" in test_cats
    assert "A" not in test_cats


def test_training_filter_train_removes_unused_categories() -> None:
    # Create a combined dataframe with unioned categories and run filter_train
    df = pd.DataFrame(
        {
            "country": pd.Categorical(["A", "B", "A"], categories=["A", "B"]),
            "split": ["train", "test", "train"],
            "treatment": ["yes", "no", "no"],
        }
    )

    filtered = training_module.filter_train(df)
    cats = list(filtered["country"].cat.categories)
    assert "A" in cats
    assert "B" not in cats


def test_optimize_load_train_val_removes_unused_categories(tmp_path: Path) -> None:
    # Write splits and ensure optimize._load_train_val uses our tmp dir via env
    _write_splits_parquet(tmp_path)
    os.environ["MENTAL_HEALTH_DATA_DIR"] = str(tmp_path)

    X, y = optimize_module._load_train_val(target_col="treatment")
    # The resulting X should not contain categories that only appeared in test
    for col in X.select_dtypes(include=["category"]).columns:
        # categories should be drawn only from train/val rows (here 'A' and 'C')
        cats = list(X[col].cat.categories)
        assert "B" not in cats


if __name__ == "__main__":
    import sys

    # Allow running the tests directly
    failures = 0
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                # tmp_path fixture not available when running directly; create temp dir
                from tempfile import TemporaryDirectory

                with TemporaryDirectory() as td:
                    from pathlib import Path

                    td_path = Path(td)
                    # Some tests expect a tmp_path arg
                    if func.__code__.co_argcount == 1:
                        func(td_path)
                    else:
                        func()
                print(f"{name}: OK")
            except AssertionError:
                print(f"{name}: FAIL")
                failures += 1
            except Exception as exc:
                print(f"{name}: ERROR: {exc}")
                failures += 1

    sys.exit(failures)
