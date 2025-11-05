"""Entry point for the evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import DEFAULT_TARGET_COLUMN
from src.eval.eval import (
    build_shap_vs_cramers_table,
    compute_shap_summary,
    evaluate_fairness_by_group,
    evaluate_model,
    load_model,
    load_split_dataframe,
    save_eval_results,
    save_shap_summary_plot,
)
from src.utils import get_logger

LOGGER = get_logger(__name__)


def eval_pipeline() -> None:
    """Execute the evaluation workflow end-to-end."""
    model = load_model()
    metrics = evaluate_model(model)
    LOGGER.info("Evaluation metrics: %s", metrics)

    fairness_gender = evaluate_fairness_by_group(model, group_col="gender")
    fairness_country = evaluate_fairness_by_group(model, group_col="country")
    LOGGER.info(
        "Gender fairness AUC gap: %.4f",
        fairness_gender["gaps"].get("auc_gap", float("nan")),
    )
    LOGGER.info(
        "Country fairness AUC gap: %.4f",
        fairness_country["gaps"].get("auc_gap", float("nan")),
    )

    # Cross-validation is handled during hyperparameter optimization.

    shap_summary = compute_shap_summary(model)
    shap_plot_path = save_shap_summary_plot(model, summary=shap_summary)
    LOGGER.info("SHAP summary plot saved at %s", shap_plot_path)
    df_test = load_split_dataframe("test")
    shap_table = build_shap_vs_cramers_table(
        shap_summary,
        df_test[DEFAULT_TARGET_COLUMN],
        df_test,
    )

    save_eval_results(
        metrics=metrics,
        fairness_gender=fairness_gender,
        fairness_country=fairness_country,
        shap_plot_path=shap_plot_path,
        shap_vs_cramers_table=shap_table,
        model=model,
    )


if __name__ == "__main__":
    eval_pipeline()
