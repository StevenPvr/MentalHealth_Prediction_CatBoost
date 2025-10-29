"""Module de nettoyage orchestrant les appels aux utilitaires de données."""

from __future__ import annotations

from os import PathLike
from typing import Iterable

import pandas as pd

from ..utils import (
    drop_duplicate_columns,
    get_dataset_path,
    normalize_text_dataframe,
    standardize_missing_tokens,
    write_csv,
    write_parquet,
)

DEFAULT_MISSING_TOKENS: tuple[str, ...] = ("", " ", "NA", "N/A", "nan", "NaN", "null")


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes dupliquées via l'utilitaire dédié."""

    # Étape : déléguer la détection des doublons de colonnes
    return drop_duplicate_columns(df)


def handle_missing_values(df: pd.DataFrame, tokens: Iterable[str] | None = None) -> pd.DataFrame:
    """Standardise les valeurs manquantes grâce à l'utilitaire correspondant."""

    # Étape : harmoniser les marqueurs de valeurs absentes
    tokens_to_use = DEFAULT_MISSING_TOKENS if tokens is None else tuple(tokens)
    return standardize_missing_tokens(df, tokens_to_use)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Applique la normalisation textuelle fournie par l'utilitaire partagé."""

    # Étape : aligner casse et typographie des colonnes texte
    return normalize_text_dataframe(df)


def save_cleaned_data(df: pd.DataFrame, base_dir: str | PathLike[str] | None = None) -> None:
    """Sauvegarde le DataFrame nettoyé en CSV pour la traçabilité et en Parquet pour la performance."""

    # Étape : exporter les données nettoyées dans les formats standards
    csv_path = get_dataset_path("dataset_cleaned.csv", base_dir)
    parquet_path = get_dataset_path("dataset_cleaned.parquet", base_dir)
    write_csv(df, csv_path)
    write_parquet(df, parquet_path)
