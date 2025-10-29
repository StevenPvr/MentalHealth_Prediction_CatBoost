"""Pipeline principal de nettoyage des données."""

from pathlib import Path
import sys

# Ajouter la racine du projet au sys.path pour les imports, cela permet des simplifier les imports relatifs.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logging_setup import get_logger
from src.utils import load_dataset, write_metrics_artifacts
from src.data_cleaning.data_cleaning import (
    remove_duplicate_columns,
    handle_missing_values,
    normalize_text,
    save_cleaned_data,
)


def clean_data() -> None:
    """
    Exécute le pipeline complet de nettoyage des données.
    
    Steps:
        1. Charge le dataset brut
        2. Supprime les colonnes dupliquées
        3. Gère les valeurs manquantes
        4. Sauvegarde en CSV et Parquet
    """
    logger = get_logger(__name__)

    metrics: list[dict[str, object]] = []

    def _record_step(step: str, before_df, after_df, note: str | None = None) -> None:
        rows_before = len(before_df)
        rows_after = len(after_df)
        cols_before = len(before_df.columns)
        cols_after = len(after_df.columns)

        metrics.append(
            {
                "step": step,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "columns_before": cols_before,
                "columns_after": cols_after,
                "rows_delta": rows_after - rows_before,
                "columns_delta": cols_after - cols_before,
                "note": note or "",
            }
        )

        logger.info(
            "   → %s : %d → %d lignes (%+d), %d → %d colonnes (%+d)%s",
            step,
            rows_before,
            rows_after,
            rows_after - rows_before,
            cols_before,
            cols_after,
            cols_after - cols_before,
            f" | {note}" if note else "",
        )

    logger.info("🔄 Chargement du dataset...")
    df = load_dataset()
    _record_step("load_dataset", df, df, "Dataset initial")

    logger.info("🔄 Suppression des colonnes dupliquées...")
    before = df
    df = remove_duplicate_columns(df)
    _record_step("remove_duplicate_columns", before, df, "Colonnes dupliquées supprimées")

    logger.info("🔄 Traitement des valeurs manquantes...")
    before = df
    df = handle_missing_values(df)
    _record_step("handle_missing_values", before, df, "Jetons normalisés en NaN")

    logger.info("🔄 Normalisation du texte...")
    before = df
    df = normalize_text(df)
    _record_step("normalize_text", before, df, "Colonnes/texte harmonisés")

    logger.info("💾 Sauvegarde des données nettoyées...")
    save_cleaned_data(df)
    logger.info("   ✓ dataset_cleaned.csv")
    logger.info("   ✓ dataset_cleaned.parquet")

    artifact_paths = write_metrics_artifacts(metrics, "data_cleaning", "clean_data_metrics")
    logger.info(
        "   ✓ Traces sauvegardées dans %s et %s",
        artifact_paths["csv"],
        artifact_paths["json"],
    )

    logger.info("✅ Nettoyage terminé avec succès!")


if __name__ == "__main__":
    clean_data()

