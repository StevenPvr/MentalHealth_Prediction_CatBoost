"""Model creation helpers wrapping the CatBoost classifier."""

from __future__ import annotations

from catboost import CatBoostClassifier  # type: ignore

from ..constants import DEFAULT_RANDOM_STATE


def create_catboost_model() -> CatBoostClassifier:
    """Instantiate a CatBoost classifier configured for binary classification."""
    return CatBoostClassifier(
        random_state=DEFAULT_RANDOM_STATE,
        verbose=True,
        loss_function="Logloss",
        eval_metric="Logloss",
        use_best_model=True,
    )
