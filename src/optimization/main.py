"""Entry point to run hyperparameter optimization with CV."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.optimization.optimize import optimize_hyperparameters, save_optimization_artifacts
from src.utils import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Run HPO and persist artifacts to the results directory."""
    result = optimize_hyperparameters()
    outputs = save_optimization_artifacts(result)
    LOGGER.info("Optimization artifacts saved: %s", outputs)


if __name__ == "__main__":
    main()
