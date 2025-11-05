"""Unit tests for Ridge baseline module."""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrapping pour exécution directe: ajouter la racine projet au sys.path
for parent in Path(__file__).resolve().parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import numpy as np
import pandas as pd  # type: ignore
import pytest  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

from src.baseline import logistic_regression
from src.test_helpers import assert_contains_keys
from src.constants import DEFAULT_TARGET_COLUMN


def create_mock_splits(n_train: int = 100, n_test: int = 30) -> pd.DataFrame:
    """Create mock categorical splits for testing."""
    np.random.seed(42)
    splits = []

    # Train split
    train_data = {
        "feature1": np.random.choice(["A", "B", "C"], n_train),
        "feature2": np.random.choice(["X", "Y"], n_train),
        DEFAULT_TARGET_COLUMN: np.random.choice(["yes", "no"], n_train),
    }
    train_df = pd.DataFrame(train_data)
    train_df["split"] = "train"
    splits.append(train_df)

    # Test split
    test_data = {
        "feature1": np.random.choice(["A", "B", "C"], n_test),
        "feature2": np.random.choice(["X", "Y"], n_test),
        DEFAULT_TARGET_COLUMN: np.random.choice(["yes", "no"], n_test),
    }
    test_df = pd.DataFrame(test_data)
    test_df["split"] = "test"
    splits.append(test_df)

    return pd.concat(splits, ignore_index=True)


@pytest.fixture
def mock_splits_parquet(artifact_paths: dict[str, Path]) -> Path:
    """Create a mock splits.parquet file."""
    splits_df = create_mock_splits()
    path = artifact_paths["splits_parquet_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_parquet(path, index=False)
    return path


def test_load_splits(mock_splits_parquet: Path) -> None:
    """Test loading train and test splits."""
    train_df, test_df = logistic_regression.load_splits()

    assert len(train_df) == 100
    assert len(test_df) == 30
    assert "split" not in train_df.columns
    assert "split" not in test_df.columns
    assert DEFAULT_TARGET_COLUMN in train_df.columns
    assert DEFAULT_TARGET_COLUMN in test_df.columns


def test_separate_features_target() -> None:
    """Test separating features and target."""
    df = pd.DataFrame(
        {
            "feature1": ["A", "B", "C"],
            "feature2": ["X", "Y", "Z"],
            DEFAULT_TARGET_COLUMN: ["yes", "no", "yes"],
        }
    )
    X, y = logistic_regression.separate_features_target(df)

    assert len(X) == 3
    assert len(y) == 3
    assert DEFAULT_TARGET_COLUMN not in X.columns
    assert list(y) == ["yes", "no", "yes"]


def test_binarize_target() -> None:
    """Test target binarization."""
    y = pd.Series(["yes", "no", "yes", "maybe"])
    y_bin = logistic_regression.binarize_target(y)

    assert list(y_bin) == [1, 0, 1, 0]
    assert y_bin.dtype == int


def test_fit_onehot_encoder() -> None:
    """Test OneHotEncoder fitting."""
    X_train = pd.DataFrame(
        {
            "cat1": ["A", "B", "A"],
            "cat2": ["X", "Y", "X"],
        }
    )
    encoder = logistic_regression.fit_onehot_encoder(X_train)

    assert isinstance(encoder, OneHotEncoder)
    X_encoded = np.asarray(encoder.transform(X_train))
    assert X_encoded.shape[0] == 3
    assert X_encoded.shape[1] > 0


def test_transform_with_onehot() -> None:
    """Test OneHotEncoder transformation."""
    X_train = pd.DataFrame({"cat1": ["A", "B", "A"]})
    encoder = logistic_regression.fit_onehot_encoder(X_train)

    X_test = pd.DataFrame({"cat1": ["A", "B", "C"]})
    X_encoded = logistic_regression.transform_with_onehot(encoder, X_test)

    assert isinstance(X_encoded, np.ndarray)
    assert X_encoded.shape[0] == 3


def test_fit_scaler() -> None:
    """Test StandardScaler fitting."""
    X_train_encoded = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = logistic_regression.fit_scaler(X_train_encoded)

    assert isinstance(scaler, StandardScaler)
    X_scaled = np.asarray(scaler.transform(X_train_encoded))
    assert X_scaled.shape == (3, 2)


def test_transform_with_scaler() -> None:
    """Test StandardScaler transformation."""
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = logistic_regression.fit_scaler(X_train)

    X_test = np.array([[2.0, 3.0]])
    X_scaled = logistic_regression.transform_with_scaler(scaler, X_test)

    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape == (1, 2)


def test_find_best_C_no_leak() -> None:
    """Test finding best C via CV without leakage."""
    # Build small categorical dataframe
    np.random.seed(0)
    X_train = pd.DataFrame(
        {
            "cat1": np.random.choice(["A", "B", "C"], 60),
            "cat2": np.random.choice(["X", "Y"], 60),
        }
    )
    y_train = pd.Series(np.random.choice([0, 1], 60))

    alpha = logistic_regression.find_best_C_no_leak(X_train, y_train, cv_splits=3)

    assert isinstance(alpha, float)
    assert alpha > 0


def test_train_logistic_model() -> None:
    """Test training Logistic Regression model."""
    X_train = np.random.randn(50, 5)
    y_train = pd.Series(np.random.choice([0, 1], 50))
    C = 1.0

    model = logistic_regression.train_logistic_model(X_train, y_train, C)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_proba_logistic() -> None:
    """Test probability predictions from Logistic Regression."""
    X_train = np.random.randn(50, 5)
    y_train = pd.Series(np.random.choice([0, 1], 50))
    model = logistic_regression.train_logistic_model(X_train, y_train, C=1.0)

    X_test = np.random.randn(10, 5)
    proba = logistic_regression.predict_proba_logistic(model, X_test)

    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_compute_metrics() -> None:
    """Test metric computation."""
    y_true = pd.Series([1, 0, 1, 0, 1])
    y_pred_proba = np.array([[0.2, 0.8], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])

    metrics = logistic_regression.compute_metrics(y_true, y_pred_proba)

    assert "logloss" in metrics
    assert "auc" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "recall" in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_compute_metrics_single_class() -> None:
    """Test metric computation with single class (edge case)."""
    y_true = pd.Series([1, 1, 1, 1, 1])
    y_pred_proba = np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6], [0.2, 0.8]])

    metrics = logistic_regression.compute_metrics(y_true, y_pred_proba)

    assert "auc" in metrics
    assert np.isnan(metrics["auc"]) or isinstance(metrics["auc"], float)


def test_ridge_baseline_returns_components(mock_splits_parquet: Path) -> None:
    """Baseline returns trained model and preprocessing artifacts."""
    result = logistic_regression.train_and_evaluate_ridge_baseline()
    assert_contains_keys(
        result,
        ["model", "onehot_encoder", "scaler", "C", "test_metrics"],
    )


def test_ridge_baseline_metrics_keys_present(mock_splits_parquet: Path) -> None:
    """Baseline reports standard metric keys on the test split."""
    result = logistic_regression.train_and_evaluate_ridge_baseline()
    metrics = result["test_metrics"]
    assert_contains_keys(metrics, ["logloss", "auc", "accuracy", "f1", "recall"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
