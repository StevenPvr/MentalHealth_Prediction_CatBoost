# Contribuer au projet

Merci de votre intérêt pour `MentalHealth_Prediction_CatBoost` ! Ce document résume les bonnes
pratiques attendues pour toute contribution.

## Environnement de développement

1. Créez un environnement virtuel Python 3.10.
2. Installez les dépendances de base et de développement :
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Qualité du code

Ce dépôt applique les principes KISS/DRY via des vérifications automatiques :

- **Formatage** : le code doit être formaté avec [Black](https://black.readthedocs.io) (configuré
  via `pyproject.toml`).
- **Linting** : exécutez [Ruff](https://docs.astral.sh/ruff/) pour détecter les doublons
  d'import, les styles incohérents et les simplifications possibles.
- **Typing** : validez les annotations à l'aide de [Pyright](https://microsoft.github.io/pyright/)
  pour prévenir les régressions liées aux types.

Commandes recommandées avant toute ouverture de PR :

```bash
ruff check src
black --check src
pyright
pytest
```

## Tests

- Ajoutez des tests ciblés pour tout nouveau comportement ou refactorisation.
- Conservez les tests existants verts (`pytest`).
- Utilisez les fixtures communes définies dans `src/conftest.py` pour isoler les ressources
  (fichiers temporaires, chemins de sortie, etc.).

## Git & Revue

- Travaillez dans une branche dédiée.
- Gardez des commits atomiques et explicitement liés aux changements.
- Décrivez brièvement vos modifications dans la PR (résumé + validations exécutées).

Merci et bonne contribution !
