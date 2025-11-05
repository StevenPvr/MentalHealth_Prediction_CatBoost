"""Tests d'intégration pour la pipeline complète.

Ce test utilise UNIQUEMENT des données et composants simulés (mocks/dummies)
afin d'éviter tout accès aux données réelles et toute exécution coûteuse
(Optuna/CatBoost). On monkeypatch :
- l'HPO pour renvoyer des hyperparamètres minimaux,
- la création/chargement du modèle CatBoost par des stubs légers,
- le ``Pool``/``EFstrType`` pour produire des sorties factices.

L'objectif est de vérifier l'intégration bout‑à‑bout et la production des
artefacts, sans coût computationnel.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import numpy as np
import pandas as pd
import pytest

from src.baseline.main import ridge_baseline_pipeline
from src.data_preparation.main import prepare_data
from src.eval import eval as eval_module
from src.training.main import train_pipeline
from src.test_helpers import assert_contains_keys, assert_paths_exist


@pytest.fixture
def expected_splits() -> tuple[str, str]:
    """Etiquettes attendues pour les splits."""

    return ("train", "test")


@pytest.fixture
def shap_plot_path(artifact_paths: dict[str, Path]) -> Path:
    """Chemin attendu pour la figure SHAP de synthèse."""

    return artifact_paths["results_dir"] / "plots" / "shap_summary.png"


def _build_cleaned_dataset() -> pd.DataFrame:
    """Construit un jeu de données synthétique représentant le dataset nettoyé."""

    genders = ["male", "female", "non-binary"]
    countries = ["us", "fr", "de", "ca"]
    employment = ["yes", "no"]

    records = [
        {
            "treatment": "yes" if idx % 2 == 0 else "no",
            "gender": genders[idx % len(genders)],
            "country": countries[idx % len(countries)],
            "self_employed": employment[idx % len(employment)],
            "age_bracket": f"{20 + (idx % 5)}-{24 + (idx % 5)}",
        }
        for idx in range(120)
    ]
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Dummies et monkeypatches pour accélérer l'intégration
# ---------------------------------------------------------------------------


@dataclass
class _TrainingStubModel:
    """Modèle factice pour l'entraînement (écriture de fichier uniquement)."""

    tree_count_: int = 10

    def set_params(self, **_: object) -> _TrainingStubModel:
        return self

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        cat_features: tuple[str, ...] | list[str] | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int | None = None,
        verbose: bool | int | None = None,
    ) -> _TrainingStubModel:
        # Pas d'entraînement réel.
        _ = (X_train, y_train, cat_features, eval_set, early_stopping_rounds, verbose)
        return self

    def save_model(self, path: str) -> None:
        # Écrit un fichier minimal pour satisfaire le test.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("stub", encoding="utf-8")


@dataclass
class _DummyPool:
    data: pd.DataFrame
    cat_features: tuple[str, ...] | list[str] | None = None


@dataclass
class _EvalDummyModel:
    """Modèle factice pour l'évaluation (probas/SHAP dynamiques)."""

    loaded_path: str | None = None

    def load_model(self, path: str) -> None:  # signature de CatBoostClassifier
        self.loaded_path = path

    def predict_proba(self, pool: _DummyPool) -> np.ndarray:
        n = len(pool.data)
        if n == 0:
            return np.zeros((0, 2))
        probs = np.linspace(0.1, 0.9, n, dtype=float)
        return np.column_stack([1.0 - probs, probs])

    def get_feature_importance(self, pool: _DummyPool, type: object) -> np.ndarray:  # noqa: A003, W0622, W0613
        # Renvoie (n_samples, n_features + 1) comme CatBoost (dernière col = base value)
        n_samples = len(pool.data)
        n_features = pool.data.shape[1]
        if n_samples == 0 or n_features == 0:
            return np.zeros((max(n_samples, 1), max(n_features + 1, 1)))
        return np.ones((n_samples, n_features + 1), dtype=float)


@pytest.fixture
def patch_training_fast(monkeypatch: pytest.MonkeyPatch, artifact_paths: dict[str, Path]) -> None:
    """Accélère l'entraînement: mock HPO + modèle CatBoost factice."""

    # 1) Créer les artefacts d'optimisation dans le répertoire temporaire
    import json

    best_params = {"iterations": 50, "depth": 4, "learning_rate": 0.1}
    cv_history = [{"auc_mean": 0.7, "logloss_mean": 0.6}]

    opt_dir = artifact_paths["results_dir"] / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = opt_dir / "best_params.json"
    cv_history_path = opt_dir / "cv_history.json"

    with best_params_path.open("w", encoding="utf-8") as fh:
        json.dump(best_params, fh)

    with cv_history_path.open("w", encoding="utf-8") as fh:
        json.dump(cv_history, fh)

    # Patch RESULTS_DIR to point to the temporary results directory
    monkeypatch.setattr("src.training.main.RESULTS_DIR", artifact_paths["results_dir"])

    # 2) Modèle CatBoost factice utilisé par training.training
    import src.training.training as tr_mod

    monkeypatch.setattr(tr_mod, "create_catboost_model", _TrainingStubModel)


@pytest.fixture
def patch_eval_dummies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remplace CatBoost/Pool/Enum dans le module d'évaluation par des dummies."""

    # Patch du constructeur pour que load_model() retourne notre dummy
    monkeypatch.setattr(eval_module, "CatBoostClassifier", _EvalDummyModel)
    # Pool factice portant les données
    monkeypatch.setattr(
        eval_module, "Pool", lambda data, cat_features=None: _DummyPool(data, cat_features)
    )
    # EFstrType factice pour SHAP
    monkeypatch.setattr(eval_module, "EFstrType", type("Enum", (), {"ShapValues": "shap"}))


@pytest.fixture
def cleaned_dataset_written(artifact_paths: dict[str, Path]) -> Path:
    """Écrit un dataset nettoyé synthétique et retourne son chemin."""
    cleaned = _build_cleaned_dataset()
    path = artifact_paths["cleaned_dataset_path"]
    cleaned.to_parquet(path)
    return path


@pytest.fixture
def prepared_splits(cleaned_dataset_written: Path, artifact_paths: dict[str, Path]) -> pd.DataFrame:
    """Exécute la préparation et retourne le DataFrame combiné des splits."""
    prepare_data()
    return pd.read_parquet(artifact_paths["splits_parquet_path"])


@pytest.fixture
def trained_model(
    prepared_splits: pd.DataFrame,
    artifact_paths: dict[str, Path],
    patch_training_fast: None,  # noqa: ARG001
) -> Path:
    train_pipeline()
    return Path(artifact_paths["model_path"])


def test_prepare_writes_splits_and_preserves_rows(
    prepared_splits: pd.DataFrame,
    cleaned_dataset_written: Path,
    expected_splits: tuple[str, str, str],
) -> None:
    assert set(prepared_splits["split"].cat.categories) == set(expected_splits)
    cleaned = pd.read_parquet(cleaned_dataset_written)
    assert len(prepared_splits) == len(cleaned)


def test_training_writes_model(trained_model: Path) -> None:
    assert trained_model.exists()


def test_evaluation_writes_artifacts(
    artifact_paths: dict[str, Path],
    trained_model: Path,
    patch_eval_dummies: None,
    shap_plot_path: Path,
) -> None:
    model = eval_module.load_model()
    try:
        loaded_path = getattr(model, "loaded_path", None)
    except Exception:
        loaded_path = None
    if loaded_path is not None:
        assert loaded_path == str(trained_model)

    metrics = eval_module.evaluate_model(model)
    assert_contains_keys(metrics, ["logloss", "auc", "accuracy", "f1", "recall"])

    fairness_gender = eval_module.evaluate_fairness_by_group(model, group_col="gender")
    fairness_country = eval_module.evaluate_fairness_by_group(model, group_col="country")

    shap_path = eval_module.save_shap_summary_plot(model, output_path=str(shap_plot_path))
    assert Path(shap_path).exists()

    run_dir = artifact_paths["results_dir"] / "eval_run"
    outputs = eval_module.save_eval_results(
        metrics=metrics,
        fairness_gender=fairness_gender,
        fairness_country=fairness_country,
        shap_plot_path=shap_path,
        run_dir=run_dir,
    )
    assert_paths_exist([Path(outputs["json"]), Path(outputs["markdown"]), Path(outputs["dir"])])


def test_ridge_baseline_pipeline_writes_artifact(
    artifact_paths: dict[str, Path], prepared_splits: pd.DataFrame
) -> None:  # noqa: ARG001
    ridge_baseline_pipeline()
    result_files = list(artifact_paths["results_dir"].glob("run_*/ridge_baseline_results.json"))
    assert result_files, "ridge_baseline_results.json not found in results run directory"


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
