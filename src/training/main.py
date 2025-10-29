"""Pipeline principal d'entraînement du modèle."""

import sys
from pathlib import Path
from typing import cast
import pandas as pd

# Ajouter la racine du projet au sys.path pour les imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logging_setup import get_logger
from src.training.training import (
    load_splits,
    filter_train_val,
    separate_features_target,
    train_model,
    save_model
)


def train_pipeline() -> None:
    """
    Exécute le pipeline complet d'entraînement.
    
    Steps:
        1. Charge splits.parquet
        2. Filtre train et val
        3. Sépare features et target
        4. Entraîne le modèle avec early stopping
        5. Sauvegarde le modèle
    """
    logger = get_logger(__name__)

    logger.info("🔄 Chargement des données...")
    df = load_splits()
    logger.info("   ✓ %d lignes chargées", len(df))

    logger.info("🔄 Filtrage train/val...")
    df_train_val = filter_train_val(df)
    logger.info("   ✓ %d lignes (train + val)", len(df_train_val))

    logger.info("🔄 Séparation train et val...")
    train_df = cast(pd.DataFrame, df_train_val[df_train_val['split'] == 'train'].drop(columns=['split']))
    val_df = cast(pd.DataFrame, df_train_val[df_train_val['split'] == 'val'].drop(columns=['split']))
    logger.info("   ✓ Train: %d lignes", len(train_df))
    logger.info("   ✓ Val: %d lignes", len(val_df))

    logger.info("🔄 Séparation features/target...")
    X_train, y_train = separate_features_target(train_df)
    X_val, y_val = separate_features_target(val_df)
    logger.info("   ✓ Features: %d colonnes", len(X_train.columns))

    logger.info("🚀 Entraînement du modèle CatBoost...")
    logger.info("   Logs: learn = TRAIN | test = VALIDATION")
    separator = "=" * 60
    logger.info(separator)
    model = train_model(X_train, y_train, X_val, y_val, early_stopping_rounds=50)
    logger.info(separator)
    logger.info("   ✓ Modèle entraîné : %d arbres", model.tree_count_)

    logger.info("💾 Sauvegarde du modèle...")
    save_model(model)
    logger.info("   ✓ model_saved/catboost_model.cbm")

    logger.info("✅ Entraînement terminé avec succès!")


if __name__ == "__main__":
    train_pipeline()

