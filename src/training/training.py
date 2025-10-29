"""Module d'entraînement du modèle CatBoost."""

from typing import Tuple, cast, Any
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier # type: ignore
from ..utils import fill_categorical_na, model_path, splits_parquet_path
from ..model import create_catboost_model


def load_splits() -> pd.DataFrame:
    """
    Charge le fichier splits.parquet.
    
    Returns:
        DataFrame contenant les données avec la colonne 'split'
    """
    df = pd.read_parquet(splits_parquet_path)
    
    return df


def separate_features_target(
    df: pd.DataFrame,
    target_col: str = 'treatment'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features (X) et la target (y).
    
    Args:
        df: DataFrame contenant features et target
        target_col: Nom de la colonne target (défaut: 'treatment')
        
    Returns:
        Tuple (X, y) avec X les features et y la target
    """
    X = df.drop(columns=[target_col])
    y = cast(pd.Series, df[target_col])
    
    return X, y


def filter_train_val(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garde uniquement les lignes avec split='train' ou split='val'.
    
    Args:
        df: DataFrame contenant la colonne 'split'
        
    Returns:
        DataFrame filtré avec uniquement train et val
    """
    df_filtered = cast(pd.DataFrame, df[df['split'].isin(['train', 'val'])])
    
    return df_filtered


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 50
) -> CatBoostClassifier:
    """
    Entraîne le modèle CatBoost avec early stopping.
    
    Args:
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        X_val: Features de validation
        y_val: Target de validation
        early_stopping_rounds: Nombre d'itérations sans amélioration (défaut: 50)
        
    Returns:
        Modèle CatBoost entraîné
    """
    model = create_catboost_model()
    
    # Gérer les NaN dans les colonnes catégorielles
    X_train = fill_categorical_na(X_train)
    X_val = fill_categorical_na(X_val)
    
    # Détecter automatiquement les colonnes catégorielles
    cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
    
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_stopping_rounds
    )
    
    return model


def save_model(model: Any, path: str | None = None) -> None:
    """
    Sauvegarde le modèle CatBoost au format .cbm.
    
    Args:
        model: Modèle CatBoost entraîné à sauvegarder
        path: Chemin de sortie alternatif (optionnel)
    """
    target_path = path if path is not None else model_path
    # Créer le dossier parent s'il n'existe pas
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    # Sauvegarder le modèle
    model.save_model(target_path)

