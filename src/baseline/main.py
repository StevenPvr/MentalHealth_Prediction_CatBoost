"""Entry point for the Ridge baseline pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.baseline import logistic_regression
from src.constants import DEFAULT_CV_N_SPLITS, DEFAULT_RANDOM_STATE
from src.utils import get_logger, save_json

LOGGER = get_logger(__name__)


def ridge_baseline_pipeline() -> None:
    """Execute the Ridge baseline training and evaluation workflow."""
    LOGGER.info("Starting Ridge baseline pipeline")
    LOGGER.info("Using CV splits: %d, random_state: %d", DEFAULT_CV_N_SPLITS, DEFAULT_RANDOM_STATE)

    result = logistic_regression.train_and_evaluate_ridge_baseline(
        cv_splits=DEFAULT_CV_N_SPLITS,
        random_state=DEFAULT_RANDOM_STATE,
    )

    LOGGER.info("Best C: %.4f", result["C"])
    LOGGER.info("Test metrics: %s", result["test_metrics"])

    # Save results
    from src.utils import create_eval_run_directory

    base_dir, timestamp = create_eval_run_directory()
    results_path = base_dir / "ridge_baseline_results.json"

    save_json(
        {
            "C": result["C"],
            "test_metrics": result["test_metrics"],
            "timestamp": timestamp,
        },
        results_path,
    )
    LOGGER.info("Results saved to %s", results_path)


if __name__ == "__main__":
    ridge_baseline_pipeline()
