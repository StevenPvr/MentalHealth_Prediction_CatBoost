"""Baseline models for comparison with CatBoost."""

from .logistic_regression import train_and_evaluate_ridge_baseline

__all__ = ["train_and_evaluate_ridge_baseline"]
