"""Public package interface for the project utilities.

We configure centralized logging at import time to ensure the shared
`pipeline.log` file exists from the very beginning of any run.
This aligns with AGENTS.md guidelines for early, centralized logging.
"""

from __future__ import annotations

from .logging_setup import configure_logging
from .utils import dataset_csv_path

# Ensure the logging system is configured as soon as the package is imported.
configure_logging()

file_path = str(dataset_csv_path())

__all__ = ["dataset_csv_path", "file_path"]
