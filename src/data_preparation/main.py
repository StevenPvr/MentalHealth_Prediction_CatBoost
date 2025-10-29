"""Pipeline principal de préparation des données."""

import sys
from pathlib import Path

# Support exécution directe: ajouter la racine du projet au sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logging_setup import get_logger
from src.utils import RANDOM_STATE
from src.data_preparation.data_preparation import (
    load_cleaned_dataset,
    split_train_val_test,
    save_splits
)


def prepare_data() -> None:
    """
    Exécute le pipeline complet de préparation des données.
    
    Steps:
        1. Charge le dataset nettoyé
        2. Split en train (60%), val (20%), test (20%)
        3. Sauvegarde en Parquet
    """
    logger = get_logger(__name__)

    logger.info("🔄 Chargement du dataset nettoyé...")
    df = load_cleaned_dataset()
    logger.info("   ✓ %d lignes, %d colonnes", len(df), len(df.columns))

    logger.info("🔄 Split train/val/test (60/20/20)...")
    train, val, test = split_train_val_test(df, random_state=RANDOM_STATE)
    logger.info("   ✓ Train: %d lignes (%.1f%%)", len(train), len(train) / len(df) * 100)
    logger.info("   ✓ Val: %d lignes (%.1f%%)", len(val), len(val) / len(df) * 100)
    logger.info("   ✓ Test: %d lignes (%.1f%%)", len(test), len(test) / len(df) * 100)

    logger.info("💾 Sauvegarde des splits (1 fichier avec colonne 'split')...")
    logger.debug("   → Conversion en type catégoriel...")
    save_splits(train, val, test)
    logger.info("   ✓ splits.parquet (toutes colonnes = category)")
    logger.info("   ✓ splits.csv")

    logger.info("✅ Préparation terminée avec succès!")


if __name__ == "__main__":
    prepare_data()

