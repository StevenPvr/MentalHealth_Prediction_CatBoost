"""Expose CatBoost model factory at the package level."""

from __future__ import annotations

from .model import create_catboost_model as create_catboost_model

__all__ = ["create_catboost_model"]
