"""Module d'évaluation du modèle CatBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Mapping, Sequence, cast
import pandas as pd
from catboost import CatBoostClassifier, Pool, EFstrType # type: ignore
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, recall_score # type: ignore
from sklearn.model_selection import StratifiedKFold # type: ignore
from ..utils import (
    create_eval_run_directory,
    ensure_dir,
    fill_categorical_na,
    model_path,
    render_eval_markdown,
    save_json,
    splits_parquet_path,
    RANDOM_STATE,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
import math
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2_contingency


_SPLIT_CACHE: dict[str, pd.DataFrame] = {}


def load_split_dataframe(split: str) -> pd.DataFrame:
    """Retourne une copie du DataFrame correspondant au ``split`` demandé.

    Les lectures disque sont mises en cache pour éviter de relire le fichier
    ``splits.parquet`` à chaque appel. La colonne ``split`` est retirée pour
    faciliter les traitements ultérieurs.
    """

    if split not in _SPLIT_CACHE:
        full_df = pd.read_parquet(splits_parquet_path)
        df_split = cast(pd.DataFrame, full_df[full_df["split"] == split]).copy()
        if "split" in df_split.columns:
            df_split = df_split.drop(columns=["split"])
        _SPLIT_CACHE[split] = df_split

    return _SPLIT_CACHE[split].copy()


def load_model() -> CatBoostClassifier:
    """
    Charge le modèle CatBoost entraîné depuis model_saved/.
    
    Returns:
        Modèle CatBoost chargé et prêt pour les prédictions
    """
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    return model


def evaluate_model(
    model: Any,
    target_col: str = 'treatment'
) -> Dict[str, float]:
    """
    Évalue le modèle sur le split test.
    
    Args:
        model: Modèle CatBoost entraîné
        target_col: Nom de la colonne cible (défaut: 'treatment')
        
    Returns:
        Dictionnaire contenant les métriques (logloss, auc)
    """
    df_test = load_split_dataframe('test')
    
    # Séparer features et target
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    # Gérer les NaN dans les colonnes catégorielles
    X_test = fill_categorical_na(X_test)
    
    # Prédictions (probabilités) avec Pool et cat_features explicites
    cat_features = [col for col in X_test.columns if X_test[col].dtype.name == 'category']
    test_pool = Pool(X_test, cat_features=cat_features)
    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    
    # Calculer les métriques
    logloss = float(log_loss(y_test, y_pred_proba))
    auc = float(roc_auc_score(y_test, y_pred_proba))
    # Binariser pour accuracy/F1 avec seuil 0.5, pos_label='yes'
    y_true_bin = (y_test.astype(str) == 'yes').astype(int)
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_true_bin, y_pred_bin))
    f1 = float(f1_score(y_true_bin, y_pred_bin))
    recall = float(recall_score(y_true_bin, y_pred_bin))
    
    return {
        'logloss': logloss,
        'auc': auc,
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall
    }


def _load_test_data(target_col: str, group_col: str) -> tuple[pd.DataFrame, pd.Series, str]:  # type: ignore
    """
    Charge les données de test et valide la présence de la colonne de groupe.
    
    Args:
        target_col: Nom de la colonne cible
        group_col: Nom de la colonne de groupe
        
    Returns:
        Tuple (X_test, y_test, group_col) - Features, cible, et colonne de groupe
        
    Raises:
        ValueError: Si la colonne de groupe est absente
    """
    df_test = load_split_dataframe('test')
    
    if group_col not in df_test.columns:
        raise ValueError(f"La colonne de groupe '{group_col}' est absente du dataset test.")
    
    X_test = df_test.drop(columns=[target_col])
    y_test = cast(pd.Series, df_test[target_col])
    
    return X_test, y_test, group_col


def cross_validate_model(
    n_splits: int = 5,
    target_col: str = 'treatment'
) -> Dict[str, Any]:
    """
    Évalue le modèle avec stratified K-fold cross-validation.
    
    Args:
        n_splits: Nombre de folds (défaut: 5)
        target_col: Colonne cible
        
    Returns:
        Dict avec métriques moyennes, écarts-types et résultats par fold
    """
    from ..model import create_catboost_model
    from ..training.training import separate_features_target
    
    # Charger toutes les données et conserver uniquement train + val
    df = pd.read_parquet(splits_parquet_path)
    df = cast(pd.DataFrame, df[df['split'].isin(['train', 'val'])])
    df = cast(pd.DataFrame, df.drop(columns=['split']))

    # Séparer features et target sur le sous-ensemble filtré
    X, y = separate_features_target(df, target_col=target_col)
    X = fill_categorical_na(X)

    # Cross-validation stratifiée
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    cat_features = [col for col in X.columns if X[col].dtype.name == 'category']
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_splits}...")
        
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Entraîner (désactiver use_best_model pour la CV car pas d'eval_set)
        model = create_catboost_model()
        model.set_params(use_best_model=False)
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        model.fit(train_pool, verbose=False)
        
        # Évaluer
        test_pool = Pool(X_test, cat_features=cat_features)
        y_pred_proba = model.predict_proba(test_pool)[:, 1]
        
        y_true_bin = (y_test.astype(str) == 'yes').astype(int)
        y_pred_bin = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'fold': fold,
            'logloss': float(log_loss(y_test, y_pred_proba)),
            'auc': float(roc_auc_score(y_test, y_pred_proba)),
            'accuracy': float(accuracy_score(y_true_bin, y_pred_bin)),
            'f1': float(f1_score(y_true_bin, y_pred_bin)),
            'recall': float(recall_score(y_true_bin, y_pred_bin)),
        }
        fold_results.append(metrics)
        print(f"    AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
    
    # Calculer moyennes et écarts-types
    metric_names = ['logloss', 'auc', 'accuracy', 'f1', 'recall']
    aggregated = {}
    
    for metric in metric_names:
        values = [r[metric] for r in fold_results]
        aggregated[f'{metric}_mean'] = float(np.mean(values))
        aggregated[f'{metric}_std'] = float(np.std(values))
    
    return {
        'n_splits': n_splits,
        'aggregated': aggregated,
        'folds': fold_results
    }


def _get_predictions(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    """
    Obtient les prédictions de probabilité du modèle.
    
    Args:
        model: Modèle CatBoost entraîné
        X_test: Features de test
        
    Returns:
        Array des probabilités pour la classe positive
    """
    X_test = fill_categorical_na(X_test)
    cat_features = [col for col in X_test.columns if X_test[col].dtype.name == 'category']
    test_pool = Pool(X_test, cat_features=cat_features)
    return model.predict_proba(test_pool)[:, 1]


def _compute_overall_metrics(y_test: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques globales sur l'ensemble du test.
    
    Args:
        y_test: Valeurs réelles de la cible
        y_pred_proba: Probabilités prédites
        
    Returns:
        Dictionnaire avec logloss et auc
    """
    return {
        'logloss': float(log_loss(y_test, y_pred_proba)),
        'auc': float(roc_auc_score(y_test, y_pred_proba))
    }


def _compute_group_auc(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule l'AUC pour un groupe, en gérant les cas d'erreur.
    
    Args:
        y_true: Valeurs réelles pour le groupe
        y_pred: Probabilités prédites pour le groupe
        
    Returns:
        AUC ou NaN si le calcul n'est pas possible
    """
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return float('nan')
    
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return float('nan')


def _compute_group_logloss(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule le logloss pour un groupe, en gérant les cas d'erreur.
    
    Args:
        y_true: Valeurs réelles pour le groupe
        y_pred: Probabilités prédites pour le groupe
        
    Returns:
        Logloss ou NaN si le calcul n'est pas possible
    """
    try:
        return float(log_loss(y_true, y_pred))
    except Exception:
        return float('nan')


def _compute_metrics_by_group(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    groups: pd.Series | pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Calcule les métriques pour chaque groupe.
    
    Args:
        y_test: Valeurs réelles de la cible
        y_pred_proba: Probabilités prédites
        groups: Série contenant les groupes
        
    Returns:
        Dictionnaire de métriques par groupe (auc, logloss, count)
    """
    metrics_by_group: Dict[str, Dict[str, float]] = {}
    groups_series = cast(pd.Series, groups)
    
    for g in groups_series.unique():
        idx = (groups_series == g)
        yt = y_test[idx]
        yp = y_pred_proba[idx]
        
        metrics_by_group[str(g)] = {
            'auc': _compute_group_auc(yt, yp),
            'logloss': _compute_group_logloss(yt, yp),
            'count': int(idx.sum())
        }
    
    return metrics_by_group


def _compute_auc_gap(metrics_by_group: Dict[str, Dict[str, float]]) -> float:
    """
    Calcule l'écart (gap) entre le max et le min des AUC par groupe.
    
    Args:
        metrics_by_group: Dictionnaire des métriques par groupe
        
    Returns:
        Écart d'AUC ou NaN si moins de 2 valeurs valides
    """
    valid_aucs = [m['auc'] for m in metrics_by_group.values() if pd.notna(m['auc'])]
    
    if len(valid_aucs) >= 2:
        return float(max(valid_aucs) - min(valid_aucs))
    
    return float('nan')


def evaluate_fairness_by_group(
    model: Any,
    group_col: str = 'gender',
    target_col: str = 'treatment'
) -> Dict[str, Any]:
    """
    Évalue des métriques par groupe (par genre et par pays) sur le split test, et calcule des écarts car forte différence de distributions au sein des groupes.
    
    Args:
        model: Modèle CatBoost entraîné
        group_col: Colonne de groupe sensible (défaut: 'gender')
        target_col: Colonne cible (défaut: 'treatment')
        
    Returns:
        Dictionnaire contenant les métriques globales, par groupe, et les écarts (gaps).
    """
    # Charger et préparer les données
    X_test, y_test, group_col = _load_test_data(target_col, group_col)
    
    # Obtenir les prédictions
    y_pred_proba = _get_predictions(model, X_test)
    
    # Calculer les métriques globales
    overall = _compute_overall_metrics(y_test, y_pred_proba)
    
    # Calculer les métriques par groupe
    # Note: on doit récupérer les groupes depuis X_test (qui a été nettoyé)
    # mais X_test ne contient plus target_col, donc on le récupère avant le drop
    df_test = load_split_dataframe('test')
    groups = df_test[group_col].astype(str)
    
    metrics_by_group = _compute_metrics_by_group(y_test, y_pred_proba, groups)
    
    # Calculer l'écart d'AUC
    auc_gap = _compute_auc_gap(metrics_by_group)
    
    return {
        'overall': overall,
        'by_group': metrics_by_group,
        'gaps': {
            'auc_gap': auc_gap
        }
    }


@dataclass
class ShapSummary:
    """Informations synthétiques sur les importances SHAP calculées."""

    feature_names: list[str]
    mean_shap_values: np.ndarray
    ordered_feature_names: list[str]
    ordered_mean_shap_values: np.ndarray


def compute_shap_summary(
    model: Any,
    target_col: str = 'treatment',
    split: str = 'test',
) -> ShapSummary:
    """Calcule les statistiques SHAP moyennes pour un split donné."""

    df_split = load_split_dataframe(split)

    if target_col not in df_split.columns:
        raise KeyError(f"La colonne cible '{target_col}' est absente du split {split}.")

    X_split = df_split.drop(columns=[target_col])
    X_split = fill_categorical_na(X_split)

    shap_values = _compute_shap_values(model, X_split)
    feature_names = list(X_split.columns)

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "Le nombre de colonnes SHAP ne correspond pas au nombre de features "
            f"({shap_values.shape[1]} vs {len(feature_names)})."
        )

    mean_shap = shap_values.mean(axis=0)
    order = np.argsort(-np.abs(mean_shap))

    return ShapSummary(
        feature_names=feature_names,
        mean_shap_values=mean_shap,
        ordered_feature_names=[feature_names[i] for i in order],
        ordered_mean_shap_values=mean_shap[order],
    )


def _compute_shap_values(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    """
    Calcule les SHAP values pour le dataset de test.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        
    Returns:
        Array des SHAP values (sans la colonne expected value)
    """
    cat_features = [col for col in X_test.columns if X_test[col].dtype.name == 'category']
    test_pool = Pool(X_test, cat_features=cat_features)
    
    shap_values = model.get_feature_importance(test_pool, type=EFstrType.ShapValues)
    shap_values = np.asarray(shap_values)
    # Dernière colonne = expected value → exclure
    return shap_values[:, :-1]


def _compute_mean_shap_impacts(
    shap_values: np.ndarray, 
    feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """
    Calcule l'impact moyen directionnel et ordonne par importance absolue.
    
    Args:
        shap_values: SHAP values brutes
        feature_names: Noms des features
        
    Returns:
        Tuple (mean_shap_sorted, feature_names_sorted)
    """
    mean_shap = shap_values.mean(axis=0)
    order = np.argsort(-np.abs(mean_shap))
    mean_shap_sorted = mean_shap[order]
    feature_names_sorted = [feature_names[i] for i in order]
    
    return mean_shap_sorted, feature_names_sorted


def _create_shap_plot(
    mean_shap_sorted: np.ndarray,
    feature_names_sorted: list[str]
) -> Figure:
    """
    Crée le plot SHAP avec impact directionnel.
    
    Args:
        mean_shap_sorted: Valeurs SHAP moyennes triées
        feature_names_sorted: Noms des features triés
        
    Returns:
        Figure matplotlib
    """
    from matplotlib.patches import Patch
    
    # Couleurs : positif (augmente prob.) vs négatif (diminue)
    colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in mean_shap_sorted]
    
    # Figure
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(feature_names_sorted))))
    
    # Barres horizontales inversées (plus importantes en haut)
    ax.barh(
        range(len(feature_names_sorted)), 
        mean_shap_sorted[::-1], 
        color=colors[::-1],
        edgecolor='white',
        linewidth=0.5,
        alpha=0.85
    )
    
    # Axes et labels
    ax.set_yticks(range(len(feature_names_sorted)))
    ax.set_yticklabels(feature_names_sorted[::-1], fontsize=10)
    ax.set_xlabel('Impact moyen sur la prédiction (SHAP)', fontsize=11, fontweight='medium')
    ax.set_title('Impact directionnel des variables sur la probabilité de traitement', 
                 fontsize=12, fontweight='bold', pad=15)
    
    # Ligne de référence à zéro
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Grille légère
    ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Légende
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.85, label='Augmente la probabilité'),
        Patch(facecolor='#e74c3c', alpha=0.85, label='Diminue la probabilité')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    
    return fig


def _categorize_for_cramers(series: pd.Series) -> pd.Series:
    """Transforme une série en catégories adaptées au calcul de Cramér."""

    if series.empty:
        return series.astype("object")

    non_na = series.dropna()
    if non_na.empty:
        return series.astype("object")

    if is_numeric_dtype(series):
        unique = non_na.nunique()
        if unique <= 1:
            return series.astype("object")

        bins = min(4, unique)
        try:
            categorized = pd.qcut(non_na, q=bins, duplicates='drop')
        except ValueError:
            if bins <= 1:
                return series.astype("object")
            categorized = pd.cut(non_na, bins=bins, include_lowest=True)

        result = series.astype("object").copy()
        result.loc[categorized.index] = categorized.astype(str)
        return result

    return series.astype(str)


def compute_cramers_v_table(
    summary: ShapSummary,
    target_col: str = 'treatment',
    split: str = 'train',
    fallback_split: str | None = 'val',
) -> list[dict[str, float | str]]:
    """Construit un tableau ``feature`` → (SHAP moyen, Cramér's V)."""

    candidate_splits: list[str] = [split]
    if fallback_split and fallback_split not in candidate_splits:
        candidate_splits.append(fallback_split)

    base_df: pd.DataFrame | None = None
    for candidate in candidate_splits:
        df_candidate = load_split_dataframe(candidate)
        if not df_candidate.empty:
            base_df = df_candidate
            break

    if base_df is None:
        raise ValueError("Aucun split disponible pour le calcul de Cramér's V.")

    if target_col not in base_df.columns:
        raise KeyError(f"La colonne cible '{target_col}' est absente du split {split}.")

    target_series = base_df[target_col].astype(str)
    shap_series = pd.Series(summary.mean_shap_values, index=summary.feature_names)

    table: list[dict[str, float | str]] = []

    for feature in summary.feature_names:
        if feature not in base_df.columns:
            continue

        feature_series = base_df[feature]
        data = pd.DataFrame({"feature": feature_series, "target": target_series}).dropna()

        cramers_v = float('nan')
        if not data.empty:
            prepared_feature = _categorize_for_cramers(data["feature"])
            contingency = pd.crosstab(prepared_feature, data["target"].astype(str))

            if contingency.size > 0 and min(contingency.shape) > 1:
                chi2 = chi2_contingency(contingency)[0]
                n = contingency.to_numpy().sum()
                min_dim = min(contingency.shape)
                if n > 0 and min_dim > 1:
                    cramers_v = float(math.sqrt(chi2 / (n * (min_dim - 1))))

        table.append(
            {
                "feature": feature,
                "mean_shap": float(shap_series.get(feature, float('nan'))),
                "cramers_v": cramers_v,
            }
        )

    table.sort(key=lambda row: abs(cast(float, row["mean_shap"])), reverse=True)
    return table


def save_shap_summary_plot(
    model: Any,
    target_col: str = 'treatment',
    max_samples: int = 500,
    output_path: str | None = None,
    summary: ShapSummary | None = None,
) -> str:
    """
    Calcule les SHAP values (CatBoost) sur l'ensemble du split test
    et sauvegarde un plot montrant l'impact directionnel moyen de chaque variable.
    
    Les valeurs SHAP positives augmentent la probabilité de traitement (yes),
    les valeurs négatives la diminuent (no).

    Args:
        model: Modèle CatBoost entraîné
        target_col: Nom de la cible
        max_samples: Paramètre déprécié, ignoré (on utilise tout le test set)
        output_path: Chemin de sortie du plot. Par défaut plots/shap_summary.png
        summary: Résumé SHAP pré-calculé (optionnel) pour éviter un nouveau calcul

    Returns:
        Chemin du fichier image sauvegardé
    """
    summary = summary or compute_shap_summary(model, target_col=target_col, split='test')
    mean_shap_sorted = summary.ordered_mean_shap_values
    feature_names_sorted = summary.ordered_feature_names

    # Créer le plot
    fig = _create_shap_plot(mean_shap_sorted, feature_names_sorted)

    # Sauvegarder
    default_path = Path(__file__).parent.parent.parent / 'plots' / 'shap_summary.png'
    out_path = Path(output_path) if output_path is not None else default_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return str(out_path)


def save_eval_results(
    metrics: Dict[str, float],
    fairness_gender: Dict[str, Any] | None = None,
    fairness_country: Dict[str, Any] | None = None,
    shap_plot_path: str | None = None,
    shap_vs_cramers_table: Sequence[Mapping[str, Any]] | None = None,
    run_dir: Path | None = None,
) -> Dict[str, str]:
    """
    Sauvegarde les résultats d'évaluation au format JSON et Markdown.

    Returns:
        Dict avec chemins vers les artefacts écrits
    """
    base_dir, timestamp = create_eval_run_directory(run_dir)

    # JSON
    json_payload: Dict[str, Any] = {
        "metrics": metrics,
        "fairness_gender": fairness_gender,
        "fairness_country": fairness_country,
        "shap_plot_path": shap_plot_path,
        "shap_vs_cramers": list(shap_vs_cramers_table) if shap_vs_cramers_table else None,
        "timestamp": timestamp,
    }
    json_path = base_dir / "eval_results.json"
    save_json(json_payload, json_path)

    fairness_sections = (
        ("Fairness by Gender", fairness_gender),
        ("Fairness by Country", fairness_country),
    )
    md_content = render_eval_markdown(
        metrics,
        fairness_sections,
        shap_plot_path,
        timestamp,
        shap_vs_cramers_table=shap_vs_cramers_table,
    )
    md_path = base_dir / "eval_results.md"
    ensure_dir(md_path.parent)
    md_path.write_text(md_content, encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "dir": str(base_dir),
    }




