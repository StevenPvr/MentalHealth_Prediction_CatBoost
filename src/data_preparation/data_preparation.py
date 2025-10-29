"""Module de préparation des données pour CatBoost."""

from __future__ import annotations

from typing import Tuple, cast

import pandas as pd
from pandas import CategoricalDtype
from sklearn.model_selection import train_test_split # type: ignore
from ..logging_setup import get_logger
from ..utils import (
    RANDOM_STATE,
    cleaned_dataset_path,
    splits_csv_path,
    splits_parquet_path,
    write_metrics_artifacts,
)


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Charge le dataset nettoyé depuis le fichier Parquet.
    
    Returns:
        DataFrame pandas contenant les données nettoyées
    """
    
    return pd.read_parquet(cleaned_dataset_path)


def shuffle_dataset(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Mélange aléatoirement les lignes du DataFrame (ordre uniquement).
    
    Args:
        df: DataFrame à mélanger
        random_state: Graine aléatoire pour reproductibilité
        
    Returns:
        DataFrame mélangé (mêmes lignes, ordre différent)
    """
    # Seul mélange nécessaire, chance que l'ordre reste identique est quasi nulle.
    shuffled = df.sample(frac=1.0, random_state=random_state)
    return shuffled


def group_rare_categories_in_column(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    min_frequency: float = 0.05,
    other_label: str = 'others'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Regroupe les catégories rares (< min_frequency) dans `other_label` en se basant UNIQUEMENT sur le split train.
    La transformation est appliquée ensuite à val et test avec la même règle (pour éviter tout leakage).

    Args:
        train: DataFrame du split train
        val: DataFrame du split val
        test: DataFrame du split test
        column: Nom de la colonne catégorielle à regrouper
        min_frequency: Seuil de fréquence minimale (0-1) dans train pour conserver une catégorie
        other_label: Libellé utilisé pour regrouper les catégories rares

    Returns:
        Tuple (train_out, val_out, test_out) avec la colonne traitée
    """
    if column not in train.columns:
        return train, val, test

    train_out = train.copy()
    val_out = val.copy()
    test_out = test.copy()

    train_series = cast(pd.Series, train_out[column]) # On appelle la colonne "country" plus tard afin de garder cette fonction générique.
    observed = train_series.dropna()
    if len(observed) == 0:
        return train_out, val_out, test_out

    frequencies = observed.value_counts(normalize=True)
    kept_categories = frequencies[frequencies >= min_frequency].index

    def _remap(series: pd.Series) -> pd.Series:
        """
        Fonction interne permmetant d'être appelé autant de fois que nécessaire pour traiter les différents splits.
        Permet d'autoriser l'ajout de la catégorie "others" si elle n'existe pas encore.
        Remplace les catégories rares par la catégorie "others".
        N'autorise plus la présence de catégories non utilisées.
        """
        result = series.copy()
        is_categorical = isinstance(result.dtype, CategoricalDtype)

        if is_categorical and other_label not in result.cat.categories:
            result = result.cat.add_categories([other_label])

        mask_rare = result.notna() & ~result.isin(kept_categories)
        result = result.mask(mask_rare, other_label)

        if is_categorical:
            result = result.cat.remove_unused_categories()

        return result

    train_out[column] = _remap(train_series)
    if column in val_out.columns:
        val_out[column] = _remap(cast(pd.Series, val_out[column]))
    if column in test_out.columns:
        test_out[column] = _remap(cast(pd.Series, test_out[column]))

    return train_out, val_out, test_out


def split_train_val_test(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    target_col: str = 'treatment'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divise le dataset en train (60%), val (20%), test (20%).
    
    Args:
        df: DataFrame à diviser
        random_state: Graine aléatoire pour reproductibilité (défaut: 42)
        target_col: Nom de la colonne cible pour la stratification (défaut: 'treatment')
        
    Returns:
        Tuple (train, val, test)
    """
    logger = get_logger(__name__)
    metrics: list[dict[str, object]] = []

    def _record_step(step: str, before_df: pd.DataFrame, after_df: pd.DataFrame, note: str | None = None) -> None:
        rows_before = len(before_df)
        rows_after = len(after_df)

        metrics.append(
            {
                "step": step,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_delta": rows_after - rows_before,
                "note": note or "",
            }
        )

        logger.info(
            "   → %s : %d → %d lignes (%+d)%s",
            step,
            rows_before,
            rows_after,
            rows_after - rows_before,
            f" | {note}" if note else "",
        )

    logger.info("🔄 Mélange aléatoire du dataset...")
    before = df
    df = shuffle_dataset(df, random_state=random_state)
    _record_step("shuffle_dataset", before, df, f"random_state={random_state}")

    # Split train+val (80%) et test (20%) avec stratification permettant d'avoir la target à la même proportion dans chaque split.
    strat_all = df[target_col] if target_col in df.columns else None
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=strat_all
    )
    train_val = cast(pd.DataFrame, train_val)
    test = cast(pd.DataFrame, test)
    
    # Split train (60%) et val (20% du total = 25% de train_val) avec stratification
    strat_tv = train_val[target_col] if target_col in train_val.columns else None
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=strat_tv
    )
    train = cast(pd.DataFrame, train)
    val = cast(pd.DataFrame, val)

    metrics.append(
        {
            "step": "split_train_val_test",
            "rows_before": len(df),
            "rows_after": len(train) + len(val) + len(test),
            "rows_delta": (len(train) + len(val) + len(test)) - len(df),
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "note": f"random_state={random_state}",
        }
    )

    logger.info(
        "   → split_train_val_test : train=%d, val=%d, test=%d (total=%d)",
        len(train),
        len(val),
        len(test),
        len(train) + len(val) + len(test),
    )

    artifact_paths = write_metrics_artifacts(metrics, "data_preparation", "data_preparation_metrics")
    logger.info(
        "   ✓ Traces sauvegardées dans %s et %s",
        artifact_paths["csv"],
        artifact_paths["json"],
    )

    return train, val, test


def convert_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit toutes les colonnes en type catégoriel.
    
    Args:
        df: DataFrame à convertir
        
    Returns:
        DataFrame avec colonnes catégorielles
    """
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype('category')
    return df


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    parquet_path: str | None = None,
    csv_path: str | None = None
) -> None:
    """
    Sauvegarde les datasets train/val/test en un seul fichier (Parquet et CSV).
    
    Ajoute une colonne 'split' avec les valeurs 'train', 'val', 'test'.
    Convertit toutes les colonnes en type catégoriel.
    
    Args:
        train: DataFrame d'entraînement
        val: DataFrame de validation
        test: DataFrame de test
    """
    # Ajouter la colonne 'split' à chaque DataFrame
    train_copy = train.copy()
    val_copy = val.copy()
    test_copy = test.copy()

    # Regrouper les pays rares (<5% dans train) en 'others' avant concaténation
    if 'country' in train_copy.columns:
        train_copy, val_copy, test_copy = group_rare_categories_in_column(
            train_copy, val_copy, test_copy, column='country', min_frequency=0.05, other_label='others'
        )
    
    train_copy['split'] = 'train'
    val_copy['split'] = 'val'
    test_copy['split'] = 'test'
    
    # Concaténer tous les splits
    df_combined = pd.concat([train_copy, val_copy, test_copy], ignore_index=True)
    
    # Convertir en catégoriel
    df_combined = convert_to_categorical(df_combined)
    
    # Sauvegarder en Parquet et CSV
    target_parquet = parquet_path if parquet_path is not None else splits_parquet_path
    target_csv = csv_path if csv_path is not None else splits_csv_path
    df_combined.to_parquet(target_parquet, index=False)
    df_combined.to_csv(target_csv, index=False)

