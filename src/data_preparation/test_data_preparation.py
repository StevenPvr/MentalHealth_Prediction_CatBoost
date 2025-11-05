"""Tests Pytest pour le module ``data_preparation``."""

from __future__ import annotations

import sys

# Bootstrapping pour exécution directe: ajouter la racine projet au sys.path
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break


import numpy as np
import pandas as pd
import pytest

from src.constants import DEFAULT_RANDOM_STATE, RARE_CATEGORY_LABEL, RARE_CATEGORY_THRESHOLD
from src.data_preparation import data_preparation as dp
from src.test_helpers import read_metrics_csv


@pytest.fixture
def random_state() -> int:
    """Expose le ``random_state`` par défaut du projet."""

    return DEFAULT_RANDOM_STATE


@pytest.fixture
def data_preparation_metrics_paths(artifact_paths) -> dict[str, Path]:
    """Fourni les chemins attendus pour les artefacts de métriques."""

    metrics_dir = artifact_paths["results_dir"] / "data_preparation"
    return {
        "csv": metrics_dir / "data_preparation_metrics.csv",
        "json": metrics_dir / "data_preparation_metrics.json",
    }


@pytest.fixture
def cleaned_dataset() -> pd.DataFrame:
    """Jeu de données synthétique riche en cas limites pour les tests."""

    records = []
    genders = ["male", "female", "non-binary", np.nan]
    countries = ["us", "fr", "de", "us", "fr", "de"]
    treatments = ["yes", "no"]

    for i in range(60):
        records.append(
            {
                "treatment": treatments[i % 2],
                "gender": genders[i % len(genders)],
                "country": countries[i % len(countries)],
                "history_length": float(i % 5),
                "notes": "  Mixed Case Value " if i % 7 == 0 else "ok",
            }
        )

    df = pd.DataFrame.from_records(records)
    df.loc[3, "notes"] = np.nan
    df.loc[5, "gender"] = np.nan
    df.loc[10, "country"] = "rare4"

    return df


@pytest.fixture
def categorical_splits(cleaned_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crée des splits train/test catégoriels avec raretés."""

    train = cleaned_dataset.iloc[:30].copy()
    test = cleaned_dataset.iloc[30:50].copy()

    train.loc[train.index[-4:], "country"] = ["rare_island", "rare_island", "rare2", "rare3"]
    test.loc[test.index[-2:], "country"] = ["rare2", "rare3"]

    for df in (train, test):
        df["country"] = pd.Categorical(df["country"])
        df["gender"] = pd.Categorical(df["gender"].fillna("unknown"))
        df["treatment"] = pd.Categorical(df["treatment"], categories=["yes", "no"])

    return train, test


def test_load_cleaned_dataset_reads_expected_frame(
    artifact_paths, cleaned_dataset: pd.DataFrame
) -> None:
    cleaned_dataset.to_parquet(artifact_paths["cleaned_dataset_path"])

    loaded = dp.load_cleaned_dataset()

    # Harmoniser les valeurs manquantes (None vs NaN) pour éviter le FutureWarning
    def _standardize_nulls(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        return out.where(pd.notna(out), np.nan)

    pd.testing.assert_frame_equal(_standardize_nulls(loaded), _standardize_nulls(cleaned_dataset))


@pytest.fixture
def simple_dataset_with_duplicate() -> pd.DataFrame:
    """Jeu de données simple contenant un doublon exact pour valider la préservation des lignes."""
    records = [
        {
            "treatment": "yes" if i % 2 == 0 else "no",
            "gender": f"g{i % 3}",
            "country": f"c{i % 4}",
            "history_length": float(i),
            "notes": f"note_{i}",
        }
        for i in range(10)
    ]
    records.append(records[0].copy())
    return pd.DataFrame.from_records(records)


def test_split_pipeline_preserves_all_rows(
    simple_dataset_with_duplicate: pd.DataFrame, random_state: int
) -> None:
    df = simple_dataset_with_duplicate
    original_len = len(df)
    train, test = dp.split_train_test(df, random_state=random_state, target_col="treatment")
    assert len(train) + len(test) == original_len


def test_split_pipeline_logs_metrics(
    artifact_paths,
    data_preparation_metrics_paths: dict[str, Path],
    simple_dataset_with_duplicate: pd.DataFrame,
    random_state: int,
) -> None:
    df = simple_dataset_with_duplicate
    original_len = len(df)
    _ = dp.split_train_test(df, random_state=random_state, target_col="treatment")

    metrics_csv = data_preparation_metrics_paths["csv"]
    assert metrics_csv.exists()

    metrics = read_metrics_csv(metrics_csv)
    shuffle_entry = metrics.loc[metrics["step"] == "shuffle_dataset"].iloc[0]
    assert int(shuffle_entry["rows_before"]) == original_len
    assert int(shuffle_entry["rows_after"]) == original_len
    assert int(shuffle_entry["rows_delta"]) == 0

    split_entry = metrics.loc[metrics["step"] == "split_train_test"].iloc[0]
    assert int(split_entry["rows_before"]) == original_len
    assert int(split_entry["rows_after"]) == original_len
    assert int(split_entry["rows_delta"]) == 0
    total_after_split = int(split_entry["train_rows"]) + int(split_entry["test_rows"])
    assert total_after_split == original_len


def test_shuffle_dataset_changes_order_but_preserves_rows(
    cleaned_dataset: pd.DataFrame, random_state: int
) -> None:
    shuffled = dp.shuffle_dataset(cleaned_dataset, random_state=random_state)

    assert set(shuffled.index) == set(cleaned_dataset.index)
    assert not shuffled.index.equals(cleaned_dataset.index)


def test_split_train_test_is_stratified(cleaned_dataset: pd.DataFrame, random_state: int) -> None:
    train, test = dp.split_train_test(
        cleaned_dataset, random_state=random_state, target_col="treatment"
    )

    assert len(train) == 48
    assert len(test) == 12

    def positive_rate(df: pd.DataFrame) -> float:
        return (df["treatment"] == "yes").mean()

    overall = positive_rate(pd.concat([train, test]))
    for subset in (train, test):
        assert abs(positive_rate(subset) - overall) <= 0.1


def test_convert_to_categorical_preserves_values(cleaned_dataset: pd.DataFrame) -> None:
    converted = dp.convert_to_categorical(cleaned_dataset)

    for column in converted.columns:
        assert converted[column].dtype.name == "category"
    assert set(converted["treatment"].cat.categories) == set(cleaned_dataset["treatment"].unique())


def test_group_rare_categories_in_column_relabels_rare_values(categorical_splits) -> None:
    train, test = categorical_splits

    t_out, te_out, mapping = dp.group_rare_categories_in_column(
        train,
        test,
        column="country",
        min_frequency=RARE_CATEGORY_THRESHOLD,
        other_label=RARE_CATEGORY_LABEL,
    )

    for df in (t_out, te_out):
        assert "rare2" not in set(df["country"])
        assert "rare3" not in set(df["country"])
        assert "others" in set(df["country"])


def test_save_splits_combines_dataframes(categorical_splits, artifact_paths) -> None:
    train, test = categorical_splits
    dp.save_splits(train, test)

    combined = pd.read_parquet(artifact_paths["splits_parquet_path"])
    csv_combined = pd.read_csv(artifact_paths["splits_csv_path"])

    assert set(combined["split"].cat.categories) == {"train", "test"}
    assert len(combined) == len(train) + len(test)
    assert bool(csv_combined["split"].isin(["train", "test"]).all())


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
