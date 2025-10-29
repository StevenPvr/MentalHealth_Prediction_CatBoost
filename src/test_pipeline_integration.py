"""Tests d'intégration couvrant le pipeline complet."""

from __future__ import annotations

from pathlib import Path

import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from src.data_preparation.main import prepare_data
from src.eval import eval as eval_module
from src.training.main import train_pipeline


def _build_cleaned_dataset() -> pd.DataFrame:
    """Construit un jeu de données synthétique représentant le dataset nettoyé."""

    genders = ["male", "female", "non-binary"]
    countries = ["us", "fr", "de", "ca"]
    employment = ["yes", "no"]

    records = [
        {
            "treatment": "yes" if idx % 2 == 0 else "no",
            "gender": genders[idx % len(genders)],
            "country": countries[idx % len(countries)],
            "self_employed": employment[idx % len(employment)],
            "age_bracket": f"{20 + (idx % 5)}-{24 + (idx % 5)}",
        }
        for idx in range(120)
    ]
    return pd.DataFrame.from_records(records)


def test_full_pipeline_produces_model_and_reports(artifact_paths) -> None:
    """Vérifie que les pipelines de préparation, d'entraînement et d'évaluation fonctionnent ensemble."""

    cleaned_dataset = _build_cleaned_dataset()
    cleaned_dataset.to_parquet(artifact_paths["cleaned_dataset_path"])

    # Préparation des données -> écrit splits.parquet/csv
    prepare_data()

    splits_path = Path(artifact_paths["splits_parquet_path"])
    assert splits_path.exists()

    splits = pd.read_parquet(splits_path)
    assert set(splits["split"].astype(str)) == {"train", "val", "test"}
    # Vérifier que TOUTES les lignes sont préservées (pas de suppression de doublons)
    assert len(splits) == len(cleaned_dataset)

    # Entraînement -> écrit le modèle CatBoost
    train_pipeline()
    model_path = Path(artifact_paths["model_path"])
    assert model_path.exists()

    # Évaluation -> métriques, fairness et artefacts
    model = eval_module.load_model()
    metrics = eval_module.evaluate_model(model)
    assert {"logloss", "auc", "accuracy", "f1", "recall"}.issubset(metrics.keys())

    fairness_gender = eval_module.evaluate_fairness_by_group(model, group_col="gender")
    fairness_country = eval_module.evaluate_fairness_by_group(model, group_col="country")

    shap_target = artifact_paths["results_dir"] / "plots" / "shap_summary.png"
    shap_path = eval_module.save_shap_summary_plot(model, output_path=str(shap_target))
    assert Path(shap_path).exists()

    run_dir = artifact_paths["results_dir"] / "eval_run"
    artifacts = eval_module.save_eval_results(
        metrics=metrics,
        fairness_gender=fairness_gender,
        fairness_country=fairness_country,
        shap_plot_path=shap_path,
        run_dir=run_dir,
    )

    assert Path(artifacts["json"]).exists()
    assert Path(artifacts["markdown"]).exists()
    assert Path(artifacts["dir"]).exists()


if __name__ == "__main__":
    import sys
    try:
        import pytest
    except Exception:
        print("Pytest non disponible. Installez-le dans le venv: pip install pytest")
        sys.exit(1)
    code = pytest.main([__file__, "-v"])
    sys.exit(code)