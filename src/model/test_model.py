"""Tests pour le module de création du modèle CatBoost."""

from __future__ import annotations

import sys

# Bootstrapping pour exécution directe: ajouter la racine projet au sys.path
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from catboost import CatBoostClassifier  # type: ignore

from src.constants import DEFAULT_RANDOM_STATE
from src.model.model import create_catboost_model


def test_create_catboost_model() -> None:
    """Vérifie la configuration du modèle CatBoost retourné."""

    model = create_catboost_model()

    assert isinstance(model, CatBoostClassifier)
    params = model.get_params()
    assert params["random_state"] == DEFAULT_RANDOM_STATE
    assert params["verbose"] is True
    assert params["loss_function"] == "Logloss"
    assert params["eval_metric"] == "Logloss"
    assert params["use_best_model"] is True


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
