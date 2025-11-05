"""Central logging configuration for the pipeline."""

from __future__ import annotations

import os
from logging.config import dictConfig
from pathlib import Path

from .constants import DEFAULT_LOG_FILE, DEFAULT_LOG_LEVEL, LOG_DIR

_CONFIGURED = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_log_level() -> str:
    level = os.getenv("MENTAL_HEALTH_LOG_LEVEL") or os.getenv("LOG_LEVEL")
    return level.upper() if level else DEFAULT_LOG_LEVEL


def _resolve_log_file() -> Path:
    _ensure_dir(LOG_DIR)
    filename = os.getenv("MENTAL_HEALTH_LOG_FILE", DEFAULT_LOG_FILE)
    return LOG_DIR / filename


def configure_logging(force: bool = False) -> None:
    """Configure the project logging handlers."""
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


__all__ = ["configure_logging"]
