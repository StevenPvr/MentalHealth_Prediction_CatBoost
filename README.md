# Mental Health – Modélisation et analyse

Ce dépôt rassemble une petite pipeline de A à Z pour prédire `treatment` (oui/non) à partir de variables catégorielles liées au mode de vie et à la santé mentale. On s’appuie sur CatBoost pour profiter d’un support natif des features catégorielles et d’une bonne robustesse sur ce type de données. Over-engineering et cas pratique à des fins pédagogique/d'apprentissage.

## Ce que vous trouverez ici

- Nettoyage et préparation des données (`src/data_cleaning`, `src/data_preparation`)
- Utilitaires partagés (`src/utils.py`) : `drop_duplicate_columns`, `standardize_missing_tokens`,
  `normalize_text_dataframe`, `write_csv`, `write_parquet`
- Entraînement d’un modèle CatBoost (`src/training`)
- Évaluation, fairness par groupes, et SHAP (`src/eval`)
- Orchestration d’un run complet (`src/main_global.py`)
- Résultats (rapports Markdown/JSON) dans `results/` et un résumé dans `documentation/resultats.md`

### Données

- Source: voir `documentation/methodologie.txt` (lien Kaggle)
- Fichiers principaux versionnés:
  - `data/dataset.csv`
  - `data/dataset_cleaned.csv`
  - `data/splits.csv`

### Démarrage rapide

1. Créez un environnement Python 3.10+ (testé sous Python 3.11).
2. Installez les dépendances d'exécution:

   ```bash
   pip install -r requirements.txt
   ```

   Pour disposer des outils de développement (tests, lint, notebooks), ajoutez:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. Lancez la pipeline complète:

   ```bash
   python -m src.main_global pipeline
   ```

   Cela enchaîne: nettoyage → préparation (splits) → entraînement → évaluation.


### Journalisation

- Toutes les étapes du pipeline utilisent le module standard `logging` via `src/logging_setup.py`.
- Les traces sont écrites à l'écran et dans `results/logs/pipeline.log` (créé automatiquement).
- Ajustez le niveau avec `MENTAL_HEALTH_LOG_LEVEL` (ex. `export MENTAL_HEALTH_LOG_LEVEL=DEBUG`). Pour renommer le fichier,
  définissez `MENTAL_HEALTH_LOG_FILE`.

### Traçabilité des transformations

- Le nettoyage (`python -m src.data_cleaning.main`) enregistre l'évolution du nombre de lignes/colonnes après chaque étape
  (`remove_duplicate_columns`, `handle_missing_values`, `normalize_text`) dans
  `results/data_cleaning/clean_data_metrics.csv` et `.json`.
- La préparation (`python -m src.data_preparation.main`) consigne de la même manière les effectifs avant/après
  `drop_duplicate_rows`, `shuffle_dataset` puis le split train/val/test dans
  `results/data_preparation/data_preparation_metrics.*` (avec les tailles précises de chaque split).
- Ces artefacts servent de garde-fous : ils permettent de confirmer qu'on conserve quasiment le volume initial de données.
  Une chute brutale (ex. un passage à ~17k lignes) n'est **pas** attendue et doit être investiguée via ces rapports et les logs
  associés.

### Tests

- Pour exécuter la suite de tests unitaires isolément:

  ```bash
  pytest
  ```

- Les fixtures Pytest redirigent automatiquement toutes les écritures vers des répertoires temporaires. Aucune exécution de
  pipeline complète ni artefact global (dans `data/`, `results/`, `model_saved/`…) n'est nécessaire pour valider les tests.
- Un workflow GitHub Actions (`install-check.yml`) vérifie la résolution des dépendances sur une installation propre.

### Résultats et interprétation

- Les métriques d'évaluation (Logloss, AUC, Accuracy, F1, Recall) ainsi que des analyses d'équité par genre/pays sont exportées dans `results/run_*/` au format JSON et Markdown.
- Une **cross-validation stratifiée 5-fold** est automatiquement exécutée sur les splits *train* + *val* uniquement afin de mesurer la stabilité du modèle (moyennes ± écarts-types) sans réutiliser le split *test*.
- Un résumé prêt à lire est copié dans `documentation/resultats.md` après chaque run.
- Un graphique SHAP (impact directionnel moyen par variable) est généré dans `plots/shap_summary.png`.

### Reproductibilité

- Les splits sont sauvegardés dans `data/splits.parquet` et `data/splits.csv`.
- Le modèle entraîné est stocké dans `model_saved/catboost_model.cbm`.
- Les versions exactes des dépendances d'exécution sont dans `requirements.txt` et les compléments de développement dans `requirements-dev.txt`.

### Structure du projet (extrait)

```text
src/
  data_cleaning/
  data_preparation/
  training/
  eval/
  main_global.py
data/
documentation/
plots/
results/
```

### Notes

- Pour des conseils sur la méthodologie, les limites et les pistes d’amélioration, consultez `documentation/methodologie.txt`.
- Transparence: voir `documentation/LLM_assistance_methodologie.txt` pour le périmètre d’usage du LLM.
- Les gros artefacts (modèles, logs CatBoost) et dossiers intermédiaires sont ignorés par Git, sauf certaines sorties utiles (CSV demandés, rapports). Vous pouvez ajuster `.gitignore` si besoin.
- Garantie mémoire: les utilitaires de nettoyage travaillent sur des copies minimales (hashs de colonnes,
  conversions paresseuses) pour rester compatibles avec des DataFrames volumineux tout en évitant les
  modifications en place.
