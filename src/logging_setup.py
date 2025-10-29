"""Configuration centralisée du logging pour la pipeline."""

from __future__ import annotations

import logging
import os
from logging.config import dictConfig
from pathlib import Path
from typing import Optional

from . import utils

_CONFIGURED = False


def _resolve_log_level() -> str:
    """Détermine le niveau de log à appliquer."""
    env_level = os.getenv("MENTAL_HEALTH_LOG_LEVEL") or os.getenv("LOG_LEVEL")
    if env_level:
        return env_level.upper()
    return "INFO"


def _resolve_log_file() -> Path:
    """Construit le chemin du fichier de log (et crée le dossier si besoin)."""
    log_dir = Path(utils.RESULTS_DIR) / "logs"
    utils.ensure_dir(log_dir)
    filename = os.getenv("MENTAL_HEALTH_LOG_FILE", "pipeline.log")
    return log_dir / filename


def configure_logging(force: bool = False) -> None:
    """Applique la configuration de logging du projet."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    log_level = _resolve_log_level()
    log_file = _resolve_log_file()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "filename": str(log_file),
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": log_level,
                "handlers": ["console", "file"],
            },
        }
    )

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Retourne un logger configuré pour le projet."""
    configure_logging()
    return logging.getLogger(name if name else "mental_health")


__all__ = ["configure_logging", "get_logger"]
