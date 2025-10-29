"""Tests unitaires Pytest pour le module ``data_preparation``."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import data_preparation as dp


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
def categorical_splits(cleaned_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crée des splits train/val/test catégoriels avec raretés."""

    train = cleaned_dataset.iloc[:30].copy()
    val = cleaned_dataset.iloc[30:40].copy()
    test = cleaned_dataset.iloc[40:50].copy()

    train.loc[train.index[-4:], "country"] = ["rare_island", "rare_island", "rare2", "rare3"]
    val.loc[val.index[-2:], "country"] = ["rare_island", "rare4"]
    test.loc[test.index[-2:], "country"] = ["rare2", "rare3"]

    for df in (train, val, test):
        df["country"] = pd.Categorical(df["country"])
        df["gender"] = pd.Categorical(df["gender"].fillna("unknown"))
        df["treatment"] = pd.Categorical(df["treatment"], categories=["yes", "no"])

    return train, val, test


def test_load_cleaned_dataset_reads_expected_frame(
    artifact_paths, cleaned_dataset: pd.DataFrame
) -> None:
    cleaned_dataset.to_parquet(artifact_paths["cleaned_dataset_path"])

    loaded = dp.load_cleaned_dataset()

    pd.testing.assert_frame_equal(loaded, cleaned_dataset)


def test_split_pipeline_preserves_all_rows_and_logs_metrics(artifact_paths) -> None:
    """Vérifie que le pipeline préserve toutes les lignes (y compris les doublons) et trace les métriques."""

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
    records.append(records[0].copy())  # Ajout d'un doublon exact (doit être conservé)

    df = pd.DataFrame.from_records(records)
    original_len = len(df)

    train, val, test = dp.split_train_val_test(df, random_state=99, target_col="treatment")

    # Vérifier que TOUTES les lignes sont préservées (y compris le doublon)
    assert len(train) + len(val) + len(test) == original_len

    metrics_csv = artifact_paths["results_dir"] / "data_preparation" / "data_preparation_metrics.csv"
    assert metrics_csv.exists()

    metrics = pd.read_csv(metrics_csv)

    # Vérifier que le shuffle ne supprime aucune ligne
    shuffle_entry = metrics.loc[metrics["step"] == "shuffle_dataset"].iloc[0]
    assert int(shuffle_entry["rows_before"]) == original_len
    assert int(shuffle_entry["rows_after"]) == original_len
    assert int(shuffle_entry["rows_delta"]) == 0

    # Vérifier que le split préserve toutes les lignes
    split_entry = metrics.loc[metrics["step"] == "split_train_val_test"].iloc[0]
    assert int(split_entry["rows_before"]) == original_len
    assert int(split_entry["rows_after"]) == original_len
    assert int(split_entry["rows_delta"]) == 0
    total_after_split = int(split_entry["train_rows"]) + int(split_entry["val_rows"]) + int(split_entry["test_rows"])
    assert total_after_split == original_len


def test_shuffle_dataset_changes_order_but_preserves_rows(cleaned_dataset: pd.DataFrame) -> None:
    shuffled = dp.shuffle_dataset(cleaned_dataset, random_state=123)

    assert set(shuffled.index) == set(cleaned_dataset.index)
    assert not shuffled.index.equals(cleaned_dataset.index)


def test_split_train_val_test_is_stratified(cleaned_dataset: pd.DataFrame) -> None:
    train, val, test = dp.split_train_val_test(cleaned_dataset, random_state=21, target_col="treatment")

    assert len(train) == 36
    assert len(val) == 12
    assert len(test) == 12

    def positive_rate(df: pd.DataFrame) -> float:
        return (df["treatment"] == "yes").mean()

    overall = positive_rate(pd.concat([train, val, test]))
    for subset in (train, val, test):
        assert abs(positive_rate(subset) - overall) <= 0.1


def test_convert_to_categorical_preserves_values(cleaned_dataset: pd.DataFrame) -> None:
    converted = dp.convert_to_categorical(cleaned_dataset)

    for column in converted.columns:
        assert converted[column].dtype.name == "category"
    assert set(converted["treatment"].cat.categories) == set(cleaned_dataset["treatment"].unique())


def test_group_rare_categories_in_column_relabels_rare_values(categorical_splits) -> None:
    train, val, test = categorical_splits

    t_out, v_out, te_out = dp.group_rare_categories_in_column(
        train, val, test, column="country", min_frequency=0.15, other_label="others"
    )

    for df in (t_out, v_out, te_out):
        assert "rare2" not in set(df["country"])
        assert "rare3" not in set(df["country"])
        assert "others" in set(df["country"])


def test_save_splits_combines_dataframes(categorical_splits, artifact_paths) -> None:
    train, val, test = categorical_splits
    dp.save_splits(train, val, test)

    combined = pd.read_parquet(artifact_paths["splits_parquet_path"])
    csv_combined = pd.read_csv(artifact_paths["splits_csv_path"])

    assert set(combined["split"].cat.categories) == {"train", "val", "test"}
    assert len(combined) == len(train) + len(val) + len(test)
    assert bool(csv_combined["split"].isin(["train", "val", "test"]).all())
