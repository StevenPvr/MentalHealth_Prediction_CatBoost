# Mental Health – Modélisation et analyse

Ce dépôt rassemble une pipeline de bout en bout pour prédire la variable `treatment` (oui/non) à partir de variables catégorielles relatives au mode de vie et à la santé mentale. CatBoost est utilisé pour son support natif des features catégorielles et sa robustesse sur ce type de données.

## Ce que vous trouverez ici

- **Pré-traitement** : nettoyage (`src/data_cleaning`) et préparation (`src/data_preparation`).
- **Utilitaires partagés** (`src/utils.py`) : fonctions de normalisation (`drop_duplicate_columns`, `standardize_missing_tokens`, `normalize_text_dataframe`, etc.) et gestion centralisée des chemins.
- **Modélisation** : optimisation d'hyperparamètres avec Optuna et cross-validation (`src/optimization`),
  entraînement CatBoost (`src/training`) et évaluation (`src/eval`) avec analyse d’équité et SHAP.
  Baseline avec régression logistique Ridge (`src/baseline`).
  Optuna utilise un pruner (MedianPruner) pour interrompre tôt les essais peu prometteurs.
- **Orchestration** : exécution de bout en bout via `src/main_global.py`.
- **Rapports** : synthèses Markdown/JSON dans `results/` et résumé lisible dans `documentation/resultats.md`.

## Données

- Source : consulter `documentation/methodologie.txt` (lien Kaggle et méthodologie complète).
- Artefacts versionnés :
  - `data/dataset.csv`
  - `data/dataset_cleaned.csv`
  - `data/splits.csv`

## Démarrage rapide

1. Créez un environnement Python 3.10+ (testé sous Python 3.11).
2. Installez les dépendances d’exécution :

   ```bash
   pip install -r requirements.txt
   ```

   Pour les outils de développement (tests, lint, notebooks) :

   ```bash
   pip install -r requirements-dev.txt
   ```

3. Lancez la pipeline complète :

   ```bash
   python -m src.main_global pipeline
   ```

   Le script orchestre automatiquement : nettoyage → préparation (splits) → entraînement → évaluation.

### Exécution ciblée

- Baseline Ridge : `python -m src.baseline.main`
- Nettoyage seul : `python -m src.data_cleaning.main`
- Préparation des données : `python -m src.data_preparation.main`
- Optimisation HPO seule (Optuna) : `python -m src.optimization.main`
- Entraînement : `python -m src.training.main`
- Évaluation : `python -m src.eval.main`

Chaque commande respecte la même gestion de chemins et écrit ses artefacts dans `results/`.

## Gestion des chemins et artefacts

- Les chemins racine sont centralisés dans `src/constants.py` (ex. `DATA_DIR`, `RESULTS_DIR`, `MODEL_DIR`).
- Les fonctions `src/utils.py` (ex. `get_data_dir`, `dataset_csv_path`) résolvent les fichiers sans concaténation de chaînes.
- L’environnement peut surcharger le dossier de données en définissant `MENTAL_HEALTH_DATA_DIR=/chemin/vers/donnees`.
- Les logs utilisent `MENTAL_HEALTH_LOG_LEVEL` et `MENTAL_HEALTH_LOG_FILE`; le fichier par défaut est `results/log/pipeline.log`.
- Les modèles sont sérialisés dans `model_saved/` ; les métriques, rapports et figures sont rangés sous `results/` (les figures par défaut sous `results/plots/`).

## Journalisation et traçabilité

- `src/logging_setup.py` configure le module standard `logging` pour toutes les étapes.
- Les journaux sont affichés en console et sauvegardés dans `results/logs/`.
- Les métriques de nettoyage sont écrites dans `results/data_cleaning/clean_data_metrics.{csv,json}`.
- Les métriques de préparation documentent les effectifs avant/après et les proportions de splits dans `results/data_preparation/data_preparation_metrics.{csv,json}`.
- Ces rapports servent de garde-fous : une diminution drastique du nombre de lignes doit conduire à inspecter ces fichiers et les logs associés.

## Tests

- Lancer la suite unitaire :

  ```bash
  pytest
  ```

- Les fixtures Pytest redirigent les écritures vers des répertoires temporaires ; aucun artefact global n’est nécessaire pour valider les tests.
- Le workflow GitHub Actions `install-check.yml` garantit la réplicabilité de l’installation.

## Résultats et interprétation

- Les métriques (Logloss, AUC, Accuracy, F1, Recall) et analyses d’équité par genre/pays sont exportées dans `results/run_*/` (JSON + Markdown).
- La **cross-validation stratifiée (5-fold)** est effectuée pendant l’optimisation des hyperparamètres (Optuna) sur les données d’entraînement (split *test* exclu), avec objectif de minimiser la Logloss. L’évaluation finale s’appuie uniquement sur le split *test*.
- Un résumé lisible est publié dans `documentation/resultats.md` après chaque run.
- Les graphiques SHAP sont sauvegardés par défaut dans `results/plots/` (ex. `results/plots/shap_summary.png`).

## Reproductibilité

- Les splits sont versionnés dans `data/splits.parquet` et `data/splits.csv`.
- Le modèle final est stocké dans `model_saved/catboost_model.cbm`.
- Les dépendances sont listées dans `requirements.txt` (exécution) et `requirements-dev.txt` (outillage supplémentaire).

## Organisation du dépôt

```text
src/
  constants.py
  data_cleaning/
  data_preparation/
  eval/
  training/
  main_global.py
notebook/
  stats_descriptive/
data/
documentation/
results/
model_saved/
```

- Les notebooks d’analyse exploratoire sont rangés sous `notebook/stats_descriptive/` et reflètent la version refactorisée des études uni/bivariées.
- Le dossier `results/` (incluant `results/plots/`) est créé à la volée ; vous pouvez le rediriger via `src/constants.py` ou des variables d’environnement (fixtures Pytest le redirigent vers des répertoires temporaires).

## Documentation complémentaire

- Méthodologie, limites et axes d’amélioration : `documentation/methodologie.txt`.
- Transparence : `documentation/LLM_assistance_methodologie.txt` détaille le rôle du LLM.
- Les gros artefacts (modèles, journaux CatBoost) sont exclus du contrôle de version, sauf quelques sorties utiles (CSV, rapports). Adaptez `.gitignore` si nécessaire.
- Les fonctions de nettoyage fonctionnent sur des copies minimales (hashs de colonnes, conversions paresseuses) pour maîtriser la consommation mémoire sur de grands DataFrames.
