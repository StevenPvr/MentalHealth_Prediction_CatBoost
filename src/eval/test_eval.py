"""Tests unitaires Pytest pour le module ``eval``."""

from __future__ import annotations
from pathlib import Path
import sys

# Ajouter la racine du projet au sys.path pour les imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pytest



from src.eval import eval as ev
from src.utils import fill_categorical_na


@pytest.fixture
def evaluation_splits() -> pd.DataFrame:
    """DataFrame synthétique avec split test, catégories rares et NaN."""

    rows: List[Dict[str, Any]] = []
    gender_cycle = ["male", "female", "non-binary", "unknown"]
    country_cycle = ["us", "fr", "rare", "de"]
    for i in range(18):
        rows.append(
            {
                "feature_gender": gender_cycle[i % len(gender_cycle)],
                "feature_country": country_cycle[i % len(country_cycle)],
                "feature_text": f"Symptom {i}",
                "treatment": "yes" if i % 2 == 0 else "no",
                "gender": gender_cycle[i % len(gender_cycle)],
                "split": "test" if i % 3 == 0 else ("val" if i % 3 == 1 else "train"),
            }
        )

    df = pd.DataFrame(rows)
    df.loc[2, "feature_gender"] = np.nan
    df.loc[4, "feature_country"] = np.nan

    gender_categories = ["male", "female", "non-binary", "unknown"]
    country_categories = ["us", "fr", "rare", "de"]

    df["feature_gender"] = pd.Categorical(df["feature_gender"], categories=gender_categories)
    df["feature_country"] = pd.Categorical(df["feature_country"], categories=country_categories)
    df["treatment"] = pd.Categorical(df["treatment"], categories=["yes", "no"])
    df["gender"] = pd.Categorical(df["gender"], categories=gender_categories)

    return df


def write_splits_for_eval(artifact_paths, df: pd.DataFrame) -> None:
    df.to_parquet(artifact_paths["splits_parquet_path"])


@dataclass
class DummyModel:
    """Stub minimal pour simuler ``CatBoostClassifier`` lors des tests."""

    probabilities: np.ndarray
    shap_values: np.ndarray
    loaded_path: str | None = None

    def load_model(self, path: str) -> None:
        self.loaded_path = path

    def predict_proba(self, pool: "DummyPool") -> np.ndarray:
        return self.probabilities

    def get_feature_importance(self, pool: "DummyPool", type: Any) -> np.ndarray:
        return self.shap_values

    def is_fitted(self) -> bool:
        return True


@dataclass
class DummyPool:
    data: pd.DataFrame
    cat_features: Iterable[str] | None = None


@pytest.fixture
def dummy_predictions(evaluation_splits: pd.DataFrame) -> np.ndarray:
    test_rows = evaluation_splits[evaluation_splits["split"] == "test"].shape[0]
    probs = np.linspace(0.1, 0.9, test_rows)
    return np.column_stack([1 - probs, probs])


@pytest.fixture
def dummy_model(dummy_predictions: np.ndarray) -> DummyModel:
    shap = np.ones((dummy_predictions.shape[0], 5))
    return DummyModel(probabilities=dummy_predictions, shap_values=shap)


def test_load_model_uses_configured_path(monkeypatch, artifact_paths, dummy_model: DummyModel) -> None:
    def fake_constructor() -> DummyModel:
        return dummy_model

    monkeypatch.setattr(ev, "CatBoostClassifier", fake_constructor)

    model = ev.load_model()

    assert model is dummy_model
    assert dummy_model.loaded_path == str(artifact_paths["model_path"])


def test_load_split_dataframe_caches(monkeypatch, artifact_paths, evaluation_splits: pd.DataFrame) -> None:
    write_splits_for_eval(artifact_paths, evaluation_splits)

    call_count = {"value": 0}

    def fake_read_parquet(path: str) -> pd.DataFrame:
        call_count["value"] += 1
        return evaluation_splits

    monkeypatch.setattr(ev.pd, "read_parquet", fake_read_parquet)

    first = ev.load_split_dataframe("test")
    second = ev.load_split_dataframe("test")

    assert call_count["value"] == 1
    assert "split" not in first.columns
    assert first is not second


def test_fill_categorical_na_handles_empty_dataframe() -> None:
    df = pd.DataFrame({"category": pd.Categorical([], categories=["yes", "no"])})

    cleaned = fill_categorical_na(df)

    assert cleaned.empty
    assert cleaned["category"].dtype.name == "category"
    assert "missing" in cleaned["category"].cat.categories


def test_cross_validate_model(monkeypatch, artifact_paths, evaluation_splits: pd.DataFrame) -> None:
    write_splits_for_eval(artifact_paths, evaluation_splits)
    
    # Mock du Pool pour capturer les données
    train_val_indices = evaluation_splits.index[evaluation_splits["split"].isin(["train", "val"])]
    test_indices = set(evaluation_splits.index[evaluation_splits["split"] == "test"])
    captured_cat_features: list[list[str]] = []

    class MockPool:
        def __init__(self, data, label=None, cat_features=None):
            if isinstance(data, pd.DataFrame):
                self.data = data.copy()
            else:
                self.data = data
            self.label = label
            self.cat_features = cat_features
            if isinstance(self.data, pd.DataFrame) and cat_features is not None:
                assert set(cat_features).issubset(self.data.columns.tolist())
                captured_cat_features.append(list(cat_features))

    class MockStratifiedKFold:
        def __init__(self, n_splits, shuffle, random_state):
            self.n_splits = n_splits

        def split(self, X, y):
            assert len(X) == len(train_val_indices)
            assert set(X.index) == set(train_val_indices)
            assert set(X.index).isdisjoint(test_indices)

            indices = np.arange(len(X))
            for fold in range(self.n_splits):
                test_mask = indices % self.n_splits == fold
                yield indices[~test_mask], indices[test_mask]

    # Mock du modèle CatBoost pour éviter l'entraînement réel
    class MockCatBoostClassifier:
        def __init__(self, *args, **kwargs):
            pass
        
        def set_params(self, **kwargs):
            # Accepte les paramètres mais ne fait rien
            return self
        
        def fit(self, X, y=None, **kwargs):
            # Accepte soit Pool (MockPool), soit X, y séparés
            pass
        
        def predict_proba(self, X):
            # Retourne des probas aléatoires mais valides
            if hasattr(X, 'data'):
                n = len(X.data)
            elif isinstance(X, pd.DataFrame):
                n = len(X)
            else:
                n = 10
            probs = np.random.rand(n)
            return np.column_stack([1 - probs, probs])

    # Mocker Pool dans le module eval
    monkeypatch.setattr(ev, "Pool", MockPool)
    monkeypatch.setattr(ev, "StratifiedKFold", MockStratifiedKFold)

    # Mocker create_catboost_model pour retourner notre mock
    def fake_create_model():
        return MockCatBoostClassifier()

    # Mocker dans le package model (pas model.model) car c'est depuis là qu'il est importé
    monkeypatch.setattr("src.model.create_catboost_model", fake_create_model)

    results = ev.cross_validate_model(n_splits=2, target_col="treatment")

    assert results["n_splits"] == 2
    assert "aggregated" in results
    assert "folds" in results
    assert len(results["folds"]) == 2
    assert "auc_mean" in results["aggregated"]
    assert "auc_std" in results["aggregated"]
    assert captured_cat_features, "Les features catégorielles devraient être détectées"
    for feature_list in captured_cat_features:
        assert all(feature in evaluation_splits.columns for feature in feature_list)


def test_evaluate_model_computes_metrics(
    monkeypatch, artifact_paths, evaluation_splits: pd.DataFrame, dummy_model: DummyModel
) -> None:
    write_splits_for_eval(artifact_paths, evaluation_splits)
    monkeypatch.setattr(ev, "Pool", lambda data, cat_features=None: DummyPool(data.copy(), cat_features))

    metrics = ev.evaluate_model(dummy_model, target_col="treatment")

    assert set(metrics.keys()) == {"logloss", "auc", "accuracy", "f1", "recall"}
    assert 0 <= metrics["auc"] <= 1
    assert metrics["logloss"] > 0


def test_evaluate_fairness_by_group(monkeypatch, artifact_paths, evaluation_splits: pd.DataFrame, dummy_model: DummyModel) -> None:
    write_splits_for_eval(artifact_paths, evaluation_splits)
    monkeypatch.setattr(ev, "Pool", lambda data, cat_features=None: DummyPool(data.copy(), cat_features))

    fairness = ev.evaluate_fairness_by_group(dummy_model, group_col="gender", target_col="treatment")

    assert {"overall", "by_group", "gaps"} <= fairness.keys()
    assert "auc_gap" in fairness["gaps"]


def test_save_shap_summary_plot_creates_file(
    monkeypatch,
    artifact_paths,
    evaluation_splits: pd.DataFrame,
    dummy_model: DummyModel,
) -> None:
    write_splits_for_eval(artifact_paths, evaluation_splits)
    monkeypatch.setattr(ev, "Pool", lambda data, cat_features=None: DummyPool(data.copy(), cat_features))
    monkeypatch.setattr(ev, "EFstrType", type("Enum", (), {"ShapValues": "shap"}))

    output_path = artifact_paths["results_dir"] / "shap_test.png"
    output = ev.save_shap_summary_plot(
        dummy_model,
        target_col="treatment",
        max_samples=5,
        output_path=str(output_path),
    )

    assert Path(output).exists()
    assert Path(output).stat().st_size > 0


def test_compute_cramers_v_table(monkeypatch, artifact_paths, evaluation_splits: pd.DataFrame, dummy_model: DummyModel) -> None:
    write_splits_for_eval(artifact_paths, evaluation_splits)
    monkeypatch.setattr(ev, "Pool", lambda data, cat_features=None: DummyPool(data.copy(), cat_features))
    monkeypatch.setattr(ev, "EFstrType", type("Enum", (), {"ShapValues": "shap"}))

    summary = ev.compute_shap_summary(dummy_model, target_col="treatment")
    table = ev.compute_cramers_v_table(summary, target_col="treatment", split="train")

    assert len(table) == len(summary.feature_names)
    assert all({"feature", "mean_shap", "cramers_v"} <= row.keys() for row in table)
    assert any(not np.isnan(row["cramers_v"]) for row in table)


def test_save_eval_results_writes_files(artifact_paths) -> None:
    metrics = {"logloss": 0.5, "auc": 0.8, "accuracy": 0.75, "f1": 0.7, "recall": 0.72}
    fairness = {"overall": {"auc": 0.8}, "gaps": {"auc_gap": 0.1}, "by_group": {}}
    shap_table = [{"feature": "gender", "mean_shap": 0.12, "cramers_v": 0.34}]

    outputs = ev.save_eval_results(
        metrics,
        fairness_gender=fairness,
        fairness_country=None,
        shap_plot_path="plot.png",
        shap_vs_cramers_table=shap_table,
    )

    assert Path(outputs["json"]).exists()
    assert Path(outputs["markdown"]).exists()
    assert Path(outputs["dir"]).exists()

    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "## Fairness by Gender" in markdown
    assert "- Overall AUC: 0.8000" in markdown
    assert "## SHAP vs Cramér's V" in markdown
    assert "| gender | 0.1200 | 0.3400 |" in markdown

    json_payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert json_payload["shap_vs_cramers"] == shap_table


