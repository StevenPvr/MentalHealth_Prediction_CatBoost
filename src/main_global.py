"""Global orchestrator for the data pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.baseline.main import ridge_baseline_pipeline
from src.data_cleaning.main import clean_data
from src.data_preparation.main import prepare_data
from src.eval.main import eval_pipeline
from src.logging_setup import configure_logging
from src.optimization.main import main as run_optimization
from src.training.main import train_pipeline
from src.utils import get_logger


def ask_for_optimization() -> bool:
    """Ask user if they want to run hyperparameter optimization."""
    while True:
        try:
            response = (
                input("Voulez-vous lancer l'optimisation des hyperparamètres CatBoost ? (y/n): ")
                .strip()
                .lower()
            )
            if response in ["y", "yes", "oui", "o"]:
                return True
            elif response in ["n", "no", "non"]:
                return False
            else:
                print("Réponse invalide. Tapez 'y' pour oui ou 'n' pour non.")
        except KeyboardInterrupt:
            print("\nAnnulation. Lancement sans optimisation.")
            return False


def run_pipeline(with_optimization: bool = False) -> None:
    """Execute the full pipeline: clean, prepare, [optimize], baseline, train, evaluate."""
    logger = get_logger(__name__)

    pipeline_steps = ["CLEAN", "PREPARE"]
    if with_optimization:
        pipeline_steps.append("OPTIMIZE")
    pipeline_steps.extend(["BASELINE", "TRAIN", "EVAL"])

    logger.info("=== PIPELINE GLOBAL: %s ===", " -> ".join(pipeline_steps))

    clean_data()
    prepare_data()

    if with_optimization:
        logger.info("Running hyperparameter optimization...")
        run_optimization()

    # Run lightweight baseline on prepared splits (no heavy training)
    ridge_baseline_pipeline()
    train_pipeline()
    eval_pipeline()


def main() -> None:
    """CLI entrypoint for the global pipeline."""
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].lower() == "pipeline"):
        configure_logging()
        # Ask user if they want optimization
        with_optimization = ask_for_optimization()
        run_pipeline(with_optimization=with_optimization)
    else:
        logger = get_logger(__name__)
        logger.error("Usage: python -m src.main_global [pipeline]")
        sys.exit(1)


if __name__ == "__main__":
    main()
