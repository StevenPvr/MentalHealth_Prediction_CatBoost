"""Module de modélisation avec CatBoost."""

from catboost import CatBoostClassifier # type: ignore
from ..utils import RANDOM_STATE


def create_catboost_model() -> CatBoostClassifier:
    """
    Crée un modèle CatBoost pour données catégorielles.
    
    Loss: Logloss
    Métrique: Logloss sur train et val (early stopping sur val)
    Logs: learn = train, test = val
    
    Returns:
        CatBoostClassifier non entraîné
    """
    model = CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=True,
        loss_function='Logloss',
        eval_metric='Logloss',
        use_best_model=True
    )
    
    return model
