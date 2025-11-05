"""Lance les tests unitaires et la pipeline baseline.

Ce script est exécutable directement (python -m src.test_global) et configure
le logging centralisé avant de déléguer l'exécution à pytest. Il cible les
tests unitaires uniquement en limitant la découverte aux modules concernés,
excluant ainsi les tests d'intégration et e2e. À la fin, il exécute également
la pipeline baseline (léger) pour garantir qu'un artefact de résultats est
produit sur les splits préparés.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from src.baseline.main import ridge_baseline_pipeline
from src.data_cleaning.main import clean_data
from src.data_preparation.main import prepare_data
from src.logging_setup import configure_logging
from src.utils import get_logger, splits_parquet_path


def run_unit_tests() -> int:
    """Configure logging and run all unit tests with pytest.

    Returns:
        Exit code returned by pytest (0 on success, non-zero on failures).
    """

    configure_logging()

    # Limit discovery to unit-test modules/directories to avoid e2e/integration.
    targets = [
        project_root / "src" / "baseline",
        project_root / "src" / "data_cleaning",
        project_root / "src" / "data_preparation",
        project_root / "src" / "eval",
        project_root / "src" / "model",
        project_root / "src" / "optimization",
        project_root / "src" / "training",
        project_root / "src" / "test_utils.py",
    ]

    args = [str(p) for p in targets]
    return pytest.main(args)


def run_baseline_pipeline() -> None:
    """Run a lightweight baseline pipeline after unit tests succeed.

    WHY: Ensures the baseline end-to-end workflow remains functional and
    produces a results artifact, without invoking heavy training.
    """

    logger = get_logger(__name__)
    logger.info("Running baseline pipeline after unit tests…")

    # Ensure splits exist; create them if missing.
    splits_path = splits_parquet_path()
    if not splits_path.exists():
        logger.info("Splits not found at %s; running clean/prepare first", splits_path)
        clean_data()
        prepare_data()

    ridge_baseline_pipeline()


def main() -> None:
    exit_code = run_unit_tests()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
