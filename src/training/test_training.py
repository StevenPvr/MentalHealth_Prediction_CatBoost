"""Tests Pytest pour le module ``training``."""

from __future__ import annotations

import sys

# Bootstrapping pour exécution directe: ajouter la racine projet au sys.path
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.constants import DEFAULT_EARLY_STOPPING_ROUNDS
from src.training import training as tr
from src.utils import fill_categorical_na


@pytest.fixture
def split_labels() -> tuple[str, str]:
    """Etiquettes standards des splits (sans split intermédiaire)."""

    return ("train", "test")


@pytest.fixture
def early_stopping_rounds() -> int:
    """Nombre d'itérations d'arrêt anticipé réduit pour accélérer les tests."""

    return min(5, DEFAULT_EARLY_STOPPING_ROUNDS)


@pytest.fixture
def splits_dataframe(split_labels: tuple[str, str]) -> pd.DataFrame:
    """Jeu de données synthétique avec colonnes catégorielles et valeurs manquantes."""

    data = []
    genders = ["male", "female", "non-binary", "prefer_not_to_say"]
    moods = ["calm", "anxious", "stressed"]
    for i in range(30):
        data.append(
            {
                "feature_gender": genders[i % len(genders)],
                "feature_mood": moods[i % len(moods)],
                "symptom_count": i % 5,
                "treatment": "yes" if i % 2 == 0 else "no",
                "split": split_labels[i % len(split_labels)],
            }
        )

    df = pd.DataFrame(data)
    df.loc[3, "feature_gender"] = np.nan
    df.loc[5, "feature_mood"] = np.nan

    df["feature_gender"] = pd.Categorical(df["feature_gender"], categories=genders)
    df["feature_mood"] = pd.Categorical(df["feature_mood"], categories=moods)
    df["treatment"] = pd.Categorical(df["treatment"], categories=["yes", "no"])

    return df


def test_load_splits_reads_parquet(
    artifact_paths: dict[str, Path], splits_dataframe: pd.DataFrame
) -> None:
    splits_dataframe.to_parquet(artifact_paths["splits_parquet_path"])

    loaded = tr.load_splits()

    pd.testing.assert_frame_equal(loaded, splits_dataframe)


def test_separate_features_target_returns_expected_shapes(
    splits_dataframe: pd.DataFrame,
) -> None:
    X, y = tr.separate_features_target(splits_dataframe, target_col="treatment")

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert "treatment" not in X.columns
    assert set(X.columns) == {"feature_gender", "feature_mood", "symptom_count", "split"}
    pd.testing.assert_series_equal(y, splits_dataframe["treatment"])


def test_filter_train_removes_non_train_rows(
    splits_dataframe: pd.DataFrame, split_labels: tuple[str, str]
) -> None:
    filtered = tr.filter_train(splits_dataframe)

    assert bool(filtered["split"].eq("train").all())


def test_fill_categorical_na_inserts_missing_category(splits_dataframe: pd.DataFrame) -> None:
    features = splits_dataframe[["feature_gender", "feature_mood"]].copy()
    features.loc[1, "feature_gender"] = np.nan
    features.loc[2, "feature_mood"] = np.nan

    cleaned = fill_categorical_na(pd.DataFrame(features))

    assert cleaned.isna().sum().sum() == 0
    assert "missing" in cleaned["feature_gender"].cat.categories
    assert "missing" in cleaned["feature_mood"].cat.categories
    assert (cleaned.loc[[1, 2], "feature_gender"] == "missing").any()
    assert (cleaned.loc[[1, 2], "feature_mood"] == "missing").any()


def test_fill_categorical_na_keeps_numeric_columns_untouched() -> None:
    df = pd.DataFrame(
        {
            "category": pd.Categorical(["a", None, "b"], categories=["a", "b"]),
            "numeric": [1.0, float("nan"), 3.0],
        }
    )

    cleaned = fill_categorical_na(df)

    assert cleaned["category"].dtype.name == "category"
    assert "missing" in cleaned["category"].cat.categories
    assert cleaned["category"].isna().sum() == 0
    assert cleaned["numeric"].dtype == df["numeric"].dtype
    assert cleaned["numeric"].isna().sum() == 1


@dataclass
class StubModel:
    """Implémentation minimale simulant ``CatBoostClassifier``."""

    fit_args: dict[str, Any] | None = None
    saved_path: str | None = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        cat_features: tuple[str, ...] | list[str] | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool | int | None = None,
    ) -> StubModel:
        self.fit_args = {
            "X_train": X_train.copy(),
            "y_train": y_train.copy(),
            "cat_features": tuple(cat_features or ()),
            "eval_set": (eval_set[0].copy(), eval_set[1].copy()) if eval_set else None,
            "early_stopping_rounds": early_stopping_rounds,
            "verbose": verbose,
        }
        return self

    def save_model(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("stub", encoding="utf-8")
        self.saved_path = path

    def is_fitted(self) -> bool:
        return self.fit_args is not None


def test_train_model_handles_categoricals(
    monkeypatch: pytest.MonkeyPatch,
    splits_dataframe: pd.DataFrame,
    early_stopping_rounds: int,
) -> None:
    stub = StubModel()

    def fake_create_model() -> StubModel:
        return stub

    monkeypatch.setattr(tr, "create_catboost_model", fake_create_model)

    train_df = splits_dataframe[splits_dataframe["split"] == "train"].copy()

    X_train, y_train = tr.separate_features_target(pd.DataFrame(train_df), target_col="treatment")
    model = tr.train_model(X_train, y_train)

    assert model is stub
    assert stub.is_fitted()
    assert stub.fit_args is not None
    assert all(cat in {"feature_gender", "feature_mood"} for cat in stub.fit_args["cat_features"])
    assert stub.fit_args["X_train"].isna().sum().sum() == 0
    # Uses internal validation split for early stopping
    assert stub.fit_args["eval_set"] is not None
    X_val, y_val = stub.fit_args["eval_set"]
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, pd.Series)
    assert stub.fit_args["early_stopping_rounds"] == DEFAULT_EARLY_STOPPING_ROUNDS


def test_save_model_writes_file(artifact_paths: dict[str, Path]) -> None:
    stub = StubModel()

    tr.save_model(stub)

    assert stub.saved_path == str(artifact_paths["model_path"])
    assert artifact_paths["model_path"].exists()


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
