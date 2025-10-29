"""Fixtures Pytest globales garantissant l'isolement des artefacts disque."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture(scope="function", autouse=True)
def _patch_project_paths(tmp_path, monkeypatch):  # type: ignore
    """Redirige l'ensemble des chemins de sortie vers un répertoire temporaire."""

    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model_saved"
    results_dir = tmp_path / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    from src import utils as utils_module

    monkeypatch.setenv("MENTAL_HEALTH_DATA_DIR", str(data_dir))
    monkeypatch.setattr(utils_module, "DEFAULT_DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(utils_module, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(utils_module, "MODEL_SAVED_DIR", model_dir, raising=False)
    monkeypatch.setattr(utils_module, "RESULTS_DIR", results_dir, raising=False)

    monkeypatch.setattr(utils_module, "file_path", str(data_dir / "dataset.csv"), raising=False)
    monkeypatch.setattr(
        utils_module,
        "cleaned_dataset_path",
        str(data_dir / "dataset_cleaned.parquet"),
        raising=False,
    )
    monkeypatch.setattr(
        utils_module,
        "splits_parquet_path",
        str(data_dir / "splits.parquet"),
        raising=False,
    )
    monkeypatch.setattr(
        utils_module,
        "splits_csv_path",
        str(data_dir / "splits.csv"),
        raising=False,
    )
    monkeypatch.setattr(
        utils_module,
        "model_path",
        str(model_dir / "catboost_model.cbm"),
        raising=False,
    )

    from src.data_preparation import data_preparation as dp

    monkeypatch.setattr(dp, "cleaned_dataset_path", str(data_dir / "dataset_cleaned.parquet"), raising=False)
    monkeypatch.setattr(dp, "splits_parquet_path", str(data_dir / "splits.parquet"), raising=False)
    monkeypatch.setattr(dp, "splits_csv_path", str(data_dir / "splits.csv"), raising=False)

    from src.training import training as training_module

    monkeypatch.setattr(training_module, "splits_parquet_path", str(data_dir / "splits.parquet"), raising=False)
    monkeypatch.setattr(training_module, "model_path", str(model_dir / "catboost_model.cbm"), raising=False)
    monkeypatch.setattr(training_module, "MODEL_SAVED_DIR", model_dir, raising=False)

    from src.eval import eval as eval_module

    monkeypatch.setattr(eval_module, "splits_parquet_path", str(data_dir / "splits.parquet"), raising=False)
    monkeypatch.setattr(eval_module, "model_path", str(model_dir / "catboost_model.cbm"), raising=False)
    monkeypatch.setattr(eval_module, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(eval_module, "_SPLIT_CACHE", {}, raising=False)

    yield {
        "data_dir": data_dir,
        "model_dir": model_dir,
        "results_dir": results_dir,
        "cleaned_dataset_path": Path(utils_module.cleaned_dataset_path),
        "splits_parquet_path": Path(utils_module.splits_parquet_path),
        "splits_csv_path": Path(utils_module.splits_csv_path),
        "model_path": Path(utils_module.model_path),
    }


@pytest.fixture
def artifact_paths(_patch_project_paths):
    """Expose les chemins temporaires configurés par la fixture autouse."""

    return _patch_project_paths
