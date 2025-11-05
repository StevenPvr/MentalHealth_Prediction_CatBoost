"""Tests Pytest ciblant les utilitaires de nettoyage de données."""

from __future__ import annotations

import sys

# Permet l'exécution directe du fichier en ajoutant la racine projet au sys.path
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from collections.abc import Iterable

import pandas as pd
import pytest

from src.constants import (
    CLEANED_DATASET_CSV_FILENAME,
    CLEANED_DATASET_FILENAME,
    DEFAULT_MISSING_TOKENS,
)
from src.data_cleaning.data_cleaning import (
    handle_missing_values,
    normalize_text,
    remove_duplicate_columns,
    save_cleaned_data,
)
from src.utils import (
    drop_duplicate_columns,
    normalize_text_dataframe,
    standardize_missing_tokens,
    write_csv,
    write_parquet,
)


@pytest.fixture
def duplicate_dataframe() -> pd.DataFrame:
    """DataFrame contenant des colonnes dupliquées et des objets non hashables."""

    return pd.DataFrame(
        {
            "numeric": [1, 2, 3],
            "numeric_dup": [1, 2, 3],
            "list_objects": [["a"], ["b"], ["c"]],
            "list_objects_dup": [["a"], ["b"], ["c"]],
            "dict_objects": [{"x": 1}, {"x": 2}, {"x": 3}],
            "dict_objects_dup": [{"x": 1}, {"x": 2}, {"x": 3}],
        }
    )


@pytest.fixture
def large_dataframe() -> pd.DataFrame:
    """DataFrame synthétique avec de nombreuses colonnes afin de valider l'évolutivité."""

    base = {f"col_{i}": [i, i + 1] for i in range(64)}
    # Ajouter des doublons pour vérifier qu'ils sont éliminés même en présence de nombreuses colonnes
    base.update({f"dup_{i}": base[f"col_{i % 4}"] for i in range(64, 96)})
    return pd.DataFrame(base)


@pytest.fixture
def tokens() -> Iterable[str]:
    """Jeu de tokens personnalisés pour la standardisation des valeurs manquantes."""

    return {"", "??", "n/a", "--"}


@pytest.fixture
def tmp_dataframe() -> pd.DataFrame:
    """Petit DataFrame utilisé pour les tests d'écriture."""

    return pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})


def test_drop_duplicate_columns_handles_unhashables(duplicate_dataframe: pd.DataFrame) -> None:
    """Les colonnes contenant des listes ou dictionnaires sont correctement dédupliquées."""

    cleaned = drop_duplicate_columns(duplicate_dataframe)

    assert list(cleaned.columns) == ["numeric", "list_objects", "dict_objects"]
    pd.testing.assert_series_equal(cleaned["numeric"], duplicate_dataframe["numeric"])


def test_drop_duplicate_columns_large_frame(large_dataframe: pd.DataFrame) -> None:
    """La suppression des doublons fonctionne sur un nombre important de colonnes."""

    cleaned = drop_duplicate_columns(large_dataframe)

    # Les colonnes de base doivent être conservées et les doublons supprimés
    assert all(col in cleaned.columns for col in [f"col_{i}" for i in range(64)])
    assert not any(col.startswith("dup_") for col in cleaned.columns)


def test_standardize_missing_tokens_custom_set(tokens: Iterable[str]) -> None:
    """Les tokens personnalisés sont remplacés par des valeurs manquantes normalisées."""

    df = pd.DataFrame({"A": ["", "OK", "??"], "B": ["n/a", "--", "value"]})

    cleaned = standardize_missing_tokens(df, tokens)

    assert cleaned.isna().sum().sum() == 4
    assert pd.isna(cleaned.loc[1, "B"])


def test_handle_missing_values_defaults() -> None:
    """Le wrapper de data_cleaning applique la liste par défaut des tokens manquants."""

    df = pd.DataFrame({token: [token, token] for token in DEFAULT_MISSING_TOKENS})

    cleaned = handle_missing_values(df)

    assert cleaned.isna().all().all()


def test_remove_duplicate_columns_wrapper(duplicate_dataframe: pd.DataFrame) -> None:
    """Le module data_cleaning délègue la suppression des doublons à l'utilitaire."""

    cleaned = remove_duplicate_columns(duplicate_dataframe)

    assert list(cleaned.columns) == ["numeric", "list_objects", "dict_objects"]


def test_normalize_text_dataframe() -> None:
    """La normalisation du texte harmonise les colonnes et les valeurs."""

    df = pd.DataFrame(
        {
            " Mixed Case ": ["  Hello World  ", 'Test"Value', "Third'String"],
            "NUMERIC": [1, 2, 3],
        }
    )

    normalized = normalize_text_dataframe(df)

    assert list(normalized.columns) == ["mixed_case", "numeric"]
    assert normalized.loc[0, "mixed_case"] == "hello_world"
    assert normalized.loc[1, "mixed_case"] == "testvalue"


def test_normalize_text_wrapper() -> None:
    """La fonction du module data_cleaning délègue bien aux utilitaires."""

    df = pd.DataFrame({"COLUMN": [" VALUE "]})
    normalized = normalize_text(df)

    assert normalized.columns.tolist() == ["column"]
    assert normalized.iloc[0, 0] == "value"


def test_write_helpers(tmp_path: Path, tmp_dataframe: pd.DataFrame) -> None:
    """Les helpers d'écriture créent les fichiers attendus sans modifier le dépôt."""

    csv_path = tmp_path / "out.csv"
    parquet_path = tmp_path / "out.parquet"

    write_csv(tmp_dataframe, csv_path)
    write_parquet(tmp_dataframe, parquet_path)

    assert csv_path.exists()
    assert parquet_path.exists()

    reloaded_csv = pd.read_csv(csv_path)
    reloaded_parquet = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(reloaded_csv, tmp_dataframe)
    pd.testing.assert_frame_equal(reloaded_parquet, tmp_dataframe)


def test_save_cleaned_data_respects_base_dir(tmp_path: Path, tmp_dataframe: pd.DataFrame) -> None:
    """Le wrapper de sauvegarde utilise le répertoire fourni pour les tests."""

    save_cleaned_data(tmp_dataframe, base_dir=tmp_path)

    csv_path = tmp_path / CLEANED_DATASET_CSV_FILENAME
    parquet_path = tmp_path / CLEANED_DATASET_FILENAME

    assert csv_path.exists()
    assert parquet_path.exists()


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
