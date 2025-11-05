"""End-to-end test for the complete pipeline.

This test verifies the entire pipeline from data cleaning through evaluation.
It uses mocks for expensive operations (Optuna/CatBoost) to keep execution fast
while validating that all steps execute correctly and produce expected artifacts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import numpy as np
import pandas as pd
import pytest

from src.baseline.main import ridge_baseline_pipeline
from src.constants import DEFAULT_TARGET_COLUMN
from src.data_cleaning.main import clean_data
from src.data_preparation.main import prepare_data
from src.eval import eval as eval_module
from src.training.main import train_pipeline
from src.test_helpers import assert_contains_keys, assert_paths_exist


@pytest.fixture
def expected_splits() -> tuple[str, str]:
    """Expected split labels."""

    return ("train", "test")


@pytest.fixture
def shap_plot_path(artifact_paths: dict[str, Path]) -> Path:
    """Expected path for SHAP summary plot."""

    return artifact_paths["results_dir"] / "plots" / "shap_summary.png"


def _build_raw_dataset() -> pd.DataFrame:
    """Build synthetic raw dataset."""

    genders = ["male", "female", "non-binary"]
    countries = ["us", "fr", "de", "ca"]
    employment = ["yes", "no"]
    treatments = ["yes", "no"]

    records = [
        {
            "treatment": treatments[idx % 2],
            "gender": genders[idx % len(genders)],
            "country": countries[idx % len(countries)],
            "self_employed": employment[idx % len(employment)],
            "age_bracket": f"{20 + (idx % 5)}-{24 + (idx % 5)}",
            "notes": f"  Note {idx}  " if idx % 3 == 0 else f"note_{idx}",
        }
        for idx in range(200)
    ]
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Dummies and monkeypatches for fast execution
# ---------------------------------------------------------------------------


@dataclass
class _TrainingStubModel:
    """Stub model for training (file writing only)."""

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
        # No actual training.
        _ = (X_train, y_train, cat_features, eval_set, early_stopping_rounds, verbose)
        return self

    def save_model(self, path: str) -> None:
        # Write minimal file to satisfy test.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("stub", encoding="utf-8")


@dataclass
class _DummyPool:
    data: pd.DataFrame
    cat_features: tuple[str, ...] | list[str] | None = None


@dataclass
class _EvalDummyModel:
    """Dummy model for evaluation (dynamic probas/SHAP)."""

    loaded_path: str | None = None

    def load_model(self, path: str) -> None:
        self.loaded_path = path

    def get_params(self) -> dict[str, Any]:
        return {"dummy_param": "value"}

    def predict_proba(self, pool: _DummyPool) -> np.ndarray:
        n = len(pool.data)
        if n == 0:
            return np.zeros((0, 2))
        probs = np.linspace(0.1, 0.9, n, dtype=float)
        return np.column_stack([1.0 - probs, probs])

    def get_feature_importance(self, pool: _DummyPool, type: object) -> np.ndarray:  # noqa: A003, W0622, W0613
        # Return (n_samples, n_features + 1) like CatBoost (last col = base value)
        n_samples = len(pool.data)
        n_features = pool.data.shape[1]
        if n_samples == 0 or n_features == 0:
            return np.zeros((max(n_samples, 1), max(n_features + 1, 1)))
        return np.ones((n_samples, n_features + 1), dtype=float)


@pytest.fixture
def patch_training_fast(monkeypatch: pytest.MonkeyPatch, artifact_paths: dict[str, Path]) -> None:
    """Speed up training: mock HPO + fake CatBoost model."""

    # 1) Create the optimization artifacts in the temporary results directory
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

    # 2) Fake CatBoost model used by training.training
    import src.training.training as tr_mod

    monkeypatch.setattr(tr_mod, "create_catboost_model", _TrainingStubModel)


@pytest.fixture
def patch_eval_dummies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace CatBoost/Pool/Enum in eval module with dummies."""

    # Patch constructor so load_model() returns our dummy
    monkeypatch.setattr(eval_module, "CatBoostClassifier", _EvalDummyModel)
    # Fake Pool carrying data
    monkeypatch.setattr(
        eval_module, "Pool", lambda data, cat_features=None: _DummyPool(data, cat_features)
    )
    # Fake EFstrType for SHAP
    monkeypatch.setattr(eval_module, "EFstrType", type("Enum", (), {"ShapValues": "shap"}))


@pytest.fixture
def write_raw_dataset(artifact_paths: dict[str, Path]) -> Path:
    """Write a synthetic raw dataset and return its path."""
    raw_dataset = _build_raw_dataset()
    dataset_csv_path = artifact_paths["data_dir"] / "dataset.csv"
    raw_dataset.to_csv(dataset_csv_path, index=False)
    return dataset_csv_path


@pytest.fixture
def prepared_data(
    write_raw_dataset: Path, artifact_paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cleaning and preparation, return cleaned and split dataframes."""
    clean_data()
    prepare_data()
    cleaned_df = pd.read_parquet(artifact_paths["cleaned_dataset_path"])
    splits = pd.read_parquet(artifact_paths["splits_parquet_path"])
    return cleaned_df, splits


@pytest.fixture
def trained_model(
    prepared_data: tuple[pd.DataFrame, pd.DataFrame],
    artifact_paths: dict[str, Path],
    patch_training_fast: None,  # noqa: ARG001
) -> Path:
    """Train fast-patched model and return the model path."""
    train_pipeline()
    return artifact_paths["model_path"]


def test_e2e_clean_and_prepare_preserves_rows_and_metrics(
    artifact_paths: dict[str, Path], expected_splits: tuple[str, str], prepared_data
) -> None:
    cleaned_df, splits = prepared_data

    # Artifacts for cleaning and preparation
    metrics_clean = artifact_paths["results_dir"] / "data_cleaning" / "clean_data_metrics.csv"
    metrics_prep = (
        artifact_paths["results_dir"] / "data_preparation" / "data_preparation_metrics.csv"
    )
    assert_paths_exist(
        [
            artifact_paths["cleaned_dataset_path"],
            artifact_paths["splits_parquet_path"],
            metrics_clean,
            metrics_prep,
        ]
    )

    # Splits are labeled and preserve rows
    assert set(splits["split"].cat.categories) == set(expected_splits)
    assert len(splits) == len(cleaned_df)

    # Roughly 80/20
    train_count = int((splits["split"] == "train").sum())
    ratio = train_count / len(splits)
    assert 0.7 <= ratio <= 0.9


def test_e2e_training_produces_model_and_optimization_artifacts(
    artifact_paths: dict[str, Path], trained_model: Path
) -> None:
    assert trained_model.exists()
    opt_dir = artifact_paths["results_dir"] / "optimization"
    assert_paths_exist([opt_dir / "best_params.json", opt_dir / "cv_history.json"])


def test_e2e_evaluation_writes_artifacts_and_sections(
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
    assert_contains_keys(fairness_gender, ["overall", "by_group", "gaps"])
    assert_contains_keys(fairness_country, ["overall", "by_group", "gaps"])

    summary = eval_module.compute_shap_summary(model)
    shap_path = eval_module.save_shap_summary_plot(
        model, output_path=str(shap_plot_path), summary=summary
    )
    assert Path(shap_path).exists()

    df_test = eval_module.load_split_dataframe("test")
    shap_table = eval_module.build_shap_vs_cramers_table(
        summary, df_test[DEFAULT_TARGET_COLUMN], df_test
    )

    run_dir = artifact_paths["results_dir"] / "eval_run"
    artifacts = eval_module.save_eval_results(
        metrics=metrics,
        fairness_gender=fairness_gender,
        fairness_country=fairness_country,
        shap_plot_path=shap_path,
        shap_vs_cramers_table=shap_table,
        run_dir=run_dir,
    )

    assert_paths_exist(
        [Path(artifacts["json"]), Path(artifacts["markdown"]), Path(artifacts["dir"])]
    )

    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    assert "## Fairness by Gender" in markdown
    assert "## Fairness by Country" in markdown
    assert "## SHAP" in markdown
    if shap_table:
        assert "## SHAP vs Cramér's V" in markdown


def test_e2e_ridge_baseline_pipeline_writes_results(
    artifact_paths: dict[str, Path], prepared_data
) -> None:  # noqa: ARG001
    ridge_baseline_pipeline()
    result_files = list(artifact_paths["results_dir"].glob("run_*/ridge_baseline_results.json"))
    assert result_files, "ridge_baseline_results.json not found in results run directory"
    payload = __import__("json").loads(result_files[-1].read_text(encoding="utf-8"))
    assert_contains_keys(payload, ["C", "test_metrics"])


def test_e2e_full_pipeline_orchestration(
    write_raw_dataset: Path,
    artifact_paths: dict[str, Path],
    patch_training_fast: None,
    patch_eval_dummies: None,
    shap_plot_path: Path,
) -> None:
    """Test the full pipeline orchestration via main_global.run_pipeline."""
    from src.main_global import run_pipeline

    run_pipeline()

    # Check all expected artifacts are produced
    assert_paths_exist(
        [
            artifact_paths["cleaned_dataset_path"],
            artifact_paths["splits_parquet_path"],
            artifact_paths["model_path"],
            shap_plot_path,
        ]
    )

    # Check results directories have content
    eval_runs = list(artifact_paths["results_dir"].glob("run_*"))
    assert eval_runs, "No eval run directory found"

    baseline_runs = list(artifact_paths["results_dir"].glob("run_*"))
    assert baseline_runs, "No baseline run directory found"


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
