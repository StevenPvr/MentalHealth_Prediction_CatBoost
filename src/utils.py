"""Utilitaires communs pour le projet Mental Health."""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
from pandas.util import hash_pandas_object


# Chemin racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
MODEL_SAVED_DIR = PROJECT_ROOT / "model_saved"
RESULTS_DIR = PROJECT_ROOT / "results"

# Graine aléatoire globale pour la reproductibilité
# Modifier cette valeur pour obtenir des résultats différents
# Cette constante est utilisée pour :
# - Le split train/val/test (data_preparation)
# - Le shuffle des données (data_preparation)
# - L'initialisation du modèle CatBoost (model)
RANDOM_STATE = 42


def get_data_dir(base: str | os.PathLike[str] | None = None) -> Path:
    """Retourne le répertoire des données en tenant compte des surcharges de tests."""

    if base is not None:
        return Path(base)

    env_override = os.environ.get("MENTAL_HEALTH_DATA_DIR")
    if env_override:
        return Path(env_override)

    return DEFAULT_DATA_DIR


DATA_DIR = get_data_dir()


def get_dataset_path(filename: str = "dataset.csv", base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Construit un chemin de dataset relatif au dossier de données actif."""

    return get_data_dir(base_dir) / filename


file_path = str(get_dataset_path("dataset.csv"))
cleaned_dataset_path = str(get_dataset_path("dataset_cleaned.parquet"))

# Chemins des datasets avec splits
splits_parquet_path = str(get_dataset_path("splits.parquet"))
splits_csv_path = str(get_dataset_path("splits.csv"))

# Chemin du modèle sauvegardé
model_path = str(MODEL_SAVED_DIR / "catboost_model.cbm")


def load_dataset() -> pd.DataFrame:
    """
    Charge le dataset depuis le fichier CSV.

    Returns:
        DataFrame pandas contenant les données brutes
    """

    return pd.read_csv(file_path)


def ensure_dir(path: Path) -> None:
    """Crée récursivement le dossier s'il n'existe pas."""

    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    """Sauvegarde un objet JSON (UTF-8, indenté)."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes dupliquées en conservant la première occurrence."""

    if df.empty or df.shape[1] <= 1:
        return df.copy()

    def _normalise_unhashable(value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return tuple(value)
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        return value

    def _series_signature(series: pd.Series) -> bytes:
        try:
            hashed = hash_pandas_object(series, index=False)
        except TypeError:
            # Conversion de secours pour les objets non-hashables (listes, dicts, ...)
            hashed = hash_pandas_object(series.map(_normalise_unhashable), index=False)
        return hashed.to_numpy().tobytes()

    seen_hashes: dict[bytes, str] = {}
    columns_to_keep: list[str] = []

    for column in df.columns:
        col_hash = _series_signature(df[column])
        match = seen_hashes.get(col_hash)
        if match is not None and df[column].equals(df[match]):
            continue

        seen_hashes[col_hash] = column
        columns_to_keep.append(column)

    return df.loc[:, columns_to_keep].copy()


def fill_categorical_na(df: pd.DataFrame, placeholder: str = "missing") -> pd.DataFrame:
    """Remplace les ``NA`` dans les colonnes catégorielles par un jeton dédié.

    Le ``placeholder`` est ajouté aux catégories uniquement si nécessaire afin
    d'éviter les ``ValueError`` provenant de ``add_categories``.
    Les colonnes non catégorielles sont laissées inchangées.

    Args:
        df: DataFrame à nettoyer.
        placeholder: Valeur utilisée pour remplacer les valeurs manquantes.

    Returns:
        Copie du DataFrame avec les colonnes catégorielles nettoyées.
    """

    result = df.copy()

    for column in result.select_dtypes(include=["category"]).columns:
        series = result[column]
        if placeholder not in series.cat.categories:
            series = series.cat.add_categories([placeholder])
        result[column] = series.fillna(placeholder)

    return result


def standardize_missing_tokens(df: pd.DataFrame, tokens: Iterable[str]) -> pd.DataFrame:
    """Remplace les jetons fournis par ``pd.NA`` pour harmoniser les données manquantes."""

    token_list = list(tokens)
    if not token_list:
        return df.copy()

    result = df.copy()
    result.replace(token_list, pd.NA, inplace=True)
    return result


def normalize_text_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Met en minuscules les colonnes/valeurs texte et retire les espaces superflus."""

    result = df.copy()
    result.columns = result.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    object_columns = result.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace('"', "", regex=False)
            .str.replace("'", "", regex=False)
            .str.replace(r"\s+", "_", regex=True)
        )

    return result


def write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Écrit un DataFrame en CSV en garantissant la création du dossier cible."""

    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)


def write_parquet(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Écrit un DataFrame en Parquet en garantissant la création du dossier cible."""

    ensure_dir(path.parent)
    df.to_parquet(path, index=False, **kwargs)


def write_metrics_artifacts(
    metrics: Sequence[Mapping[str, Any]],
    subdir: str,
    base_filename: str,
) -> dict[str, Path]:
    """Sauvegarde un rapport de métriques en CSV et JSON dans ``results/``.

    Args:
        metrics: Séquence de dictionnaires contenant les métriques à tracer.
        subdir: Sous-répertoire de ``results`` où écrire les fichiers.
        base_filename: Nom de base (sans extension) pour les fichiers générés.

    Returns:
        Dictionnaire avec les chemins vers les fichiers ``csv`` et ``json``.
    """

    target_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    ensure_dir(target_dir)

    dataframe = pd.DataFrame(list(metrics))
    csv_path = target_dir / f"{base_filename}.csv"
    json_path = target_dir / f"{base_filename}.json"

    dataframe.to_csv(csv_path, index=False)
    save_json(list(metrics), json_path)

    return {"csv": csv_path, "json": json_path}


# ---------------------------------------------------------------------------
# Outils de factorisation pour l'écriture des résultats d'évaluation
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "logloss": "Logloss",
    "auc": "AUC",
    "accuracy": "Accuracy",
    "f1": "F1",
    "recall": "Recall",
    "auc_gap": "AUC gap",
}

METRIC_ORDER: Sequence[str] = ("logloss", "auc", "accuracy", "f1", "recall")


def create_eval_run_directory(run_dir: Path | None = None) -> tuple[Path, str]:
    """Construit le dossier d'exécution horodaté à utiliser pour un rapport d'évaluation."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = run_dir or (RESULTS_DIR / f"run_{timestamp}")
    ensure_dir(base_dir)
    return base_dir, timestamp


def _format_metric_line(name: str, value: float) -> str:
    label = METRIC_LABELS.get(name, name.title())
    return f"- {label}: {value:.4f}"


def _format_optional_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NaN"

    if math.isnan(numeric):
        return "NaN"

    return f"{numeric:.4f}"


def render_global_metrics_markdown(metrics: Mapping[str, float]) -> list[str]:
    """Retourne les lignes Markdown représentant les métriques globales."""

    lines = ["## Global Metrics", ""]
    for key in METRIC_ORDER:
        if key in metrics:
            lines.append(_format_metric_line(key, float(metrics[key])))
    remaining = sorted(set(metrics.keys()) - set(METRIC_ORDER))
    for key in remaining:
        lines.append(_format_metric_line(key, float(metrics[key])))
    return lines


def render_fairness_markdown(title: str, fairness: Mapping[str, Any] | None) -> list[str]:
    """Crée une section Markdown décrivant les métriques d'équité fournies."""

    if fairness is None:
        return []

    overall = fairness.get("overall", {})
    gaps = fairness.get("gaps", {})
    by_group = fairness.get("by_group", {})

    lines = ["", f"## {title}", ""]
    if "auc" in overall:
        lines.append(f"- Overall AUC: {float(overall['auc']):.4f}")
    if "auc_gap" in gaps:
        lines.append(f"- AUC gap: {float(gaps['auc_gap']):.4f}")

    for group, metrics in sorted(by_group.items()):
        auc = float(metrics.get("auc", float("nan")))
        logloss = float(metrics.get("logloss", float("nan")))
        count = int(metrics.get("count", 0))
        lines.append(f"  - {group}: AUC={auc:.4f} | logloss={logloss:.4f} | n={count}")

    return lines


def _render_shap_vs_cramers_markdown(
    table: Sequence[Mapping[str, Any]]
) -> list[str]:
    if not table:
        return []

    lines = ["", "## SHAP vs Cramér's V", "", "| Feature | Mean SHAP | Cramér's V |", "| --- | ---: | ---: |"]

    for row in table:
        feature = str(row.get("feature", ""))
        mean_shap = _format_optional_float(row.get("mean_shap"))
        cramers_v = _format_optional_float(row.get("cramers_v"))
        lines.append(f"| {feature} | {mean_shap} | {cramers_v} |")

    return lines


def render_eval_markdown(
    metrics: Mapping[str, float],
    fairness_sections: Sequence[tuple[str, Mapping[str, Any] | None]],
    shap_plot_path: str | None,
    timestamp: str,
    *,
    shap_vs_cramers_table: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Assemble les différentes sections en un rapport Markdown complet."""

    lines: list[str] = ["# Evaluation Results", "", f"- Timestamp: {timestamp}", ""]
    lines.extend(render_global_metrics_markdown(metrics))

    for title, payload in fairness_sections:
        lines.extend(render_fairness_markdown(title, payload))

    if shap_plot_path:
        lines.extend(["", "## SHAP", "", f"![SHAP Summary]({shap_plot_path})"])

    if shap_vs_cramers_table:
        lines.extend(_render_shap_vs_cramers_markdown(shap_vs_cramers_table))

    return "\n".join(lines) + "\n"
