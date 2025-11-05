"""Tests unitaires pour le module d'optimisation (HPO)."""

from __future__ import annotations

import sys

# Bootstrapping pour exécution directe: ajouter la racine projet au sys.path
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.optimization import optimize as opt


@pytest.fixture
def hpo_splits() -> pd.DataFrame:
    """Jeu synthétique avec splits multiples et colonnes catégorielles."""

    rows: list[dict[str, Any]] = []
    genders = ["male", "female", "non-binary"]
    countries = ["us", "fr", "de"]
    for i in range(24):
        rows.append(
            {
                "gender": genders[i % len(genders)],
                "country": countries[i % len(countries)],
                "treatment": "yes" if i % 2 == 0 else "no",
                "split": "test" if i % 3 == 0 else ("val" if i % 3 == 1 else "train"),
            }
        )

    df = pd.DataFrame(rows)
    df["gender"] = pd.Categorical(df["gender"], categories=genders)
    df["country"] = pd.Categorical(df["country"], categories=countries)
    df["treatment"] = pd.Categorical(df["treatment"], categories=["yes", "no"])
    return df


def _write_splits(artifact_paths: dict[str, Path], df: pd.DataFrame) -> None:
    df.to_parquet(artifact_paths["splits_parquet_path"])


def test_optimize_hyperparameters_runs_cv(
    monkeypatch: pytest.MonkeyPatch,
    artifact_paths: dict[str, Path],
    hpo_splits: pd.DataFrame,
) -> None:
    _write_splits(artifact_paths, hpo_splits)

    # Capture used indices to ensure test split is not included
    train_val_indices = hpo_splits.index[hpo_splits["split"].isin(["train", "val"])]
    test_indices = set(hpo_splits.index[hpo_splits["split"] == "test"])

    class MockPool:
        def __init__(self, data, label=None, cat_features=None):
            if isinstance(data, pd.DataFrame):
                self.data = data.copy()
            else:
                self.data = data
            self.label = label
            self.cat_features = cat_features

    class MockStratifiedKFold:
        def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
            self.n_splits = n_splits

        def split(self, X: pd.DataFrame, y: pd.Series) -> Iterable[tuple[np.ndarray, np.ndarray]]:
            assert len(X) == len(train_val_indices)
            assert set(X.index) == set(train_val_indices)
            assert set(X.index).isdisjoint(test_indices)

            indices = np.arange(len(X))
            for fold in range(self.n_splits):
                test_mask = indices % self.n_splits == fold
                yield indices[~test_mask], indices[test_mask]

    @dataclass
    class MockModel:
        params: dict[str, Any] | None = None

        def set_params(self, **kwargs: Any) -> MockModel:
            self.params = dict(kwargs)
            return self

        def fit(self, X: Any, y: Any | None = None, **kwargs: Any) -> None:
            return None

        def predict_proba(self, X: Any) -> np.ndarray:
            # Only the number of samples matters for the test
            if hasattr(X, "data"):
                n_samples = len(X.data)
            elif isinstance(X, pd.DataFrame):
                n_samples = len(X)
            else:
                n_samples = 10
            # Deterministic probabilities for repeatability
            rng = np.random.default_rng(123)
            probs = rng.random(n_samples)
            probs = np.clip(probs, 1e-6, 1 - 1e-6)
            return np.column_stack([1 - probs, probs])

    monkeypatch.setattr(opt, "Pool", MockPool)
    monkeypatch.setattr(opt, "StratifiedKFold", MockStratifiedKFold)
    monkeypatch.setattr("src.model.create_catboost_model", lambda: MockModel())

    result = opt.optimize_hyperparameters(n_splits=2, n_trials=2)

    assert isinstance(result.best_params, dict)
    assert result.summaries and isinstance(result.summaries[0], dict)
    # Tous les résumés doivent contenir auc_mean/logloss_mean
    assert {"auc_mean", "logloss_mean"} <= set(result.summaries[0].keys())


def test_save_optimization_artifacts_writes_files(artifact_paths: dict[str, Path]) -> None:
    dummy = opt.OptimizationResult(best_params={"depth": 4}, summaries=[{"auc_mean": 0.7}])
    outputs = opt.save_optimization_artifacts(dummy)

    assert Path(outputs["best_params"]).exists()
    assert Path(outputs["cv_history"]).exists()
    assert Path(outputs["dir"]).exists()


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
