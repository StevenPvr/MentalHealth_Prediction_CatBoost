"""Pipeline principal d'évaluation du modèle."""

import sys
from pathlib import Path

# Ajouter la racine du projet au sys.path pour les imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.logging_setup import get_logger
from src.eval.eval import (
    compute_cramers_v_table,
    compute_shap_summary,
    cross_validate_model,
    evaluate_fairness_by_group,
    evaluate_model,
    load_model,
    save_eval_results,
    save_shap_summary_plot,
)


def eval_pipeline() -> None:
    """
    Exécute le pipeline complet d'évaluation.
    
    Steps:
        1. Charge le modèle entraîné
        2. Évalue sur le split test
        3. Affiche les métriques (logloss, auc)
    """
    logger = get_logger(__name__)

    logger.info("🔄 Chargement du modèle...")
    model = load_model()
    logger.info("   ✓ Modèle chargé : %d arbres", model.tree_count_)

    logger.info("📊 Évaluation sur le split test...")
    metrics = evaluate_model(model, target_col='treatment')

    separator = "=" * 60
    logger.info(separator)
    logger.info("📈 RÉSULTATS")
    logger.info(separator)
    logger.info("   Logloss: %.4f", metrics['logloss'])
    logger.info("   AUC:     %.4f", metrics['auc'])
    logger.info("   Acc:     %.4f", metrics['accuracy'])
    logger.info("   F1:      %.4f", metrics['f1'])
    logger.info("   Recall:  %.4f", metrics['recall'])
    logger.info(separator)

    # Affichage simple du rapport d'équité par genre
    fairness_gender = evaluate_fairness_by_group(model, group_col='gender', target_col='treatment')
    logger.info("🎯 ÉQUITÉ PAR GENRE (test)")
    logger.info("- Global AUC:    %.4f", fairness_gender['overall']['auc'])
    logger.info("- AUC gap (max-min) entre groupes: %.4f", fairness_gender['gaps']['auc_gap'])
    for group, m in fairness_gender['by_group'].items():
        logger.info(
            "   · %s: AUC=%.4f | logloss=%.4f | n=%d",
            group,
            m['auc'],
            m['logloss'],
            m['count'],
        )

    # Affichage simple du rapport d'équité par country
    fairness_country = evaluate_fairness_by_group(model, group_col='country', target_col='treatment')
    logger.info("🌍 ÉQUITÉ PAR COUNTRY (test)")
    logger.info("- Global AUC:    %.4f", fairness_country['overall']['auc'])
    logger.info("- AUC gap (max-min) entre pays: %.4f", fairness_country['gaps']['auc_gap'])
    for group, m in fairness_country['by_group'].items():
        logger.info(
            "   · %s: AUC=%.4f | logloss=%.4f | n=%d",
            group,
            m['auc'],
            m['logloss'],
            m['count'],
        )

    # Cross-validation pour mesurer la stabilité
    logger.info("\n" + "=" * 60)
    logger.info("🔄 CROSS-VALIDATION (5-fold stratifié)")
    logger.info("=" * 60)
    cv_results = cross_validate_model(n_splits=5, target_col='treatment')
    agg = cv_results['aggregated']
    logger.info("AUC:      %.4f ± %.4f", agg['auc_mean'], agg['auc_std'])
    logger.info("Accuracy: %.4f ± %.4f", agg['accuracy_mean'], agg['accuracy_std'])
    logger.info("F1:       %.4f ± %.4f", agg['f1_mean'], agg['f1_std'])
    logger.info("Recall:   %.4f ± %.4f", agg['recall_mean'], agg['recall_std'])
    logger.info("Logloss:  %.4f ± %.4f", agg['logloss_mean'], agg['logloss_std'])
    
    logger.info("✅ Évaluation terminée avec succès!")

    # SHAP minimal (plot bar importances moyennes absolues)
    shap_path = None
    shap_table = None
    shap_summary = None
    try:
        shap_summary = compute_shap_summary(model, target_col='treatment')
        shap_path = save_shap_summary_plot(model, target_col='treatment', summary=shap_summary)
        shap_table = compute_cramers_v_table(shap_summary, target_col='treatment')
        logger.info("🖼️  SHAP plot sauvegardé: %s", shap_path)
        if shap_table:
            logger.info("📋  Table SHAP vs Cramér's V calculée (%d features)", len(shap_table))
    except Exception as e:
        logger.warning("⚠️  SHAP non généré (%s)", e, exc_info=True)

    # Sauvegarde des résultats consolidés
    try:
        artifacts = save_eval_results(
            metrics=metrics,
            fairness_gender=fairness_gender,
            fairness_country=fairness_country,
            shap_plot_path=shap_path,
            shap_vs_cramers_table=shap_table,
        )
        logger.info("💾 Résultats écrits dans: %s", artifacts['dir'])
        logger.info("   - JSON: %s", artifacts['json'])
        logger.info("   - Markdown: %s", artifacts['markdown'])
        # Copie du résumé dans la documentation pour versionning GitHub
        try:
            doc_path = Path(__file__).parent.parent.parent / 'documentation' / 'resultats.md'
            content = Path(artifacts['markdown']).read_text(encoding='utf-8')
            doc_path.write_text(content, encoding='utf-8')
            logger.info("   - Copie documentation: %s", doc_path)
        except Exception as e:
            logger.warning("   ⚠️  Copie du résumé dans documentation échouée (%s)", e, exc_info=True)
    except Exception as e:
        logger.exception("⚠️  Sauvegarde des résultats échouée (%s)", e)


if __name__ == "__main__":
    eval_pipeline()

