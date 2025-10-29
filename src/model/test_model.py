"""Tests pour le module model."""

import sys
from pathlib import Path

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from catboost import CatBoostClassifier # type: ignore
from src.model.model import create_catboost_model
from src.utils import RANDOM_STATE


def test_create_catboost_model():
    """Test de la création du modèle CatBoost."""
    model = create_catboost_model()
    
    # Vérifications
    assert isinstance(model, CatBoostClassifier), "Le modèle doit être une instance de CatBoostClassifier"
    params = model.get_params()
    assert params['random_state'] == RANDOM_STATE, f"Le random_state doit être {RANDOM_STATE}"
    assert params['verbose'] == True, "Le verbose doit être True"
    assert params['loss_function'] == 'Logloss', "La loss_function doit être Logloss"
    assert params['eval_metric'] == 'Logloss', "La métrique eval doit être Logloss"
    assert params['use_best_model'] == True, "use_best_model doit être True"
    
    print("✓ test_create_catboost_model: PASS")


if __name__ == "__main__":
    test_create_catboost_model()
    print("\n✅ Tous les tests passent!")

