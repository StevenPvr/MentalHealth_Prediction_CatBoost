"""Global pytest fixtures to isolate file system artefacts."""

from __future__ import annotations

import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="function", autouse=True)
def _patch_project_paths(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[dict[str, Path], Any, None]:
    """Redirect all output paths to a temporary directory.

    Autouse fixtures are invoked by pytest implicitly and may appear unused to
    static analyzers. We suppress the false-positive to keep lint clean.
    """
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model_saved"
    results_dir = tmp_path / "results"
    for directory in (data_dir, model_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MENTAL_HEALTH_DATA_DIR", str(data_dir))

    from src import constants as constants_module
    from src import utils as utils_module

    monkeypatch.setattr(constants_module, "MODEL_DIR", model_dir, raising=False)
    monkeypatch.setattr(constants_module, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(constants_module, "LOG_DIR", results_dir / "log", raising=False)
    monkeypatch.setattr(constants_module, "PLOTS_DIR", results_dir / "plots", raising=False)
    monkeypatch.setattr(utils_module, "MODEL_DIR", model_dir, raising=False)
    monkeypatch.setattr(utils_module, "RESULTS_DIR", results_dir, raising=False)

    from src.eval import eval as eval_module
    from src.training import training as training_module

    monkeypatch.setattr(training_module, "model_path", lambda: model_dir / "catboost_model.cbm")
    monkeypatch.setattr(
        eval_module,
        "model_path",
        lambda: model_dir / "catboost_model.cbm",
        raising=False,
    )
    monkeypatch.setattr(eval_module, "_SPLIT_CACHE", {}, raising=False)
    monkeypatch.setattr(eval_module, "PLOTS_DIR", results_dir / "plots", raising=False)

    yield {
        "data_dir": data_dir,
        "model_dir": model_dir,
        "results_dir": results_dir,
        "cleaned_dataset_path": utils_module.cleaned_dataset_parquet_path(),
        "splits_parquet_path": utils_module.splits_parquet_path(),
        "splits_csv_path": utils_module.splits_csv_path(),
        "model_path": utils_module.model_path(),
    }


@pytest.fixture
def artifact_paths(_patch_project_paths):
    """Expose the temporary paths configured by the autouse fixture."""
    return _patch_project_paths


# ---------------------------------------------------------------------------
# Global warning filters to keep CI output clean and deterministic
# ---------------------------------------------------------------------------

# Some environments emit a RuntimeWarning about NumPy binary compatibility
# at import-time. Tests don't rely on compiled ufunc ABI details, so we
# silence it to avoid noisy logs.
warnings.filterwarnings(
    "ignore",
    message=r"numpy\.ufunc size changed.*",
    category=RuntimeWarning,
)
