"""Shared test helpers to keep tests concise and readable.

These helpers intentionally stay minimal to reduce duplication across tests
and standardize common assertions.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Mapping

import pandas as pd


def assert_contains_keys(mapping: Mapping[str, object], required: Iterable[str]) -> None:
    """Assert that all required keys are present in the mapping."""
    missing = set(required) - set(mapping.keys())
    assert not missing, f"Missing keys: {sorted(missing)}"


def assert_paths_exist(paths: Iterable[Path]) -> None:
    """Assert that all provided paths exist on disk."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    assert not missing, f"Missing files: {missing}"


def read_metrics_csv(path: Path) -> pd.DataFrame:
    """Load a CSV metrics artifact into a DataFrame."""
    return pd.read_csv(path)


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
