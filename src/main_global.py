"""Orchestrateur global du pipeline.

Usage:
  - Pipeline:  python -m src.main_global
  - Pipeline:  python -m src.main_global pipeline
"""

import sys
from pathlib import Path

# S'assurer que la racine projet est dans le path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logging_setup import configure_logging, get_logger
from src.data_cleaning.main import clean_data
from src.data_preparation.main import prepare_data
from src.training.main import train_pipeline
from src.eval.main import eval_pipeline


def run_pipeline() -> None:
    logger = get_logger(__name__)

    logger.info("=== PIPELINE GLOBAL: CLEAN -> PREPARE -> TRAIN -> EVAL ===")
    clean_data()
    prepare_data()
    train_pipeline()
    eval_pipeline()


def main() -> None:
    # Par défaut lance le pipeline; si argument fourni, il doit être 'pipeline'
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].lower() == "pipeline"):
        configure_logging()
        run_pipeline()
    else:
        logger = get_logger(__name__)
        logger.error("Usage: python -m src.main [pipeline]")
        sys.exit(1)


if __name__ == "__main__":
    main()


