"""Orchestrateur des tests unitaires.

Usage:
  - Tests:  python -m src.tests_main
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main() -> None:
    try:
        import pytest  # type: ignore
    except Exception:
        print("Pytest non disponible. Installez-le dans le venv: pip install pytest")
        sys.exit(1)
    code = pytest.main([str(project_root / "src")])
    sys.exit(code)


if __name__ == "__main__":
    main()


