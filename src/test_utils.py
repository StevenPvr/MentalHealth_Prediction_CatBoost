from __future__ import annotations

from pathlib import Path

import sys

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from src import utils


def test_create_eval_run_directory_uses_provided_path(tmp_path: Path) -> None:
    base_dir = tmp_path / "custom"
    base_dir.mkdir()

    resolved, timestamp = utils.create_eval_run_directory(base_dir)

    assert resolved == base_dir
    assert timestamp
    assert resolved.exists()


def test_create_eval_run_directory_generates_timestamp(artifact_paths) -> None:
    base_dir, timestamp = utils.create_eval_run_directory()

    assert base_dir.parent == artifact_paths["results_dir"]
    assert timestamp
    assert base_dir.exists()


@pytest.mark.parametrize(
    "metrics",
    [
        {"logloss": 0.1234, "auc": 0.9876, "custom": 1.2345},
    ],
)
def test_render_eval_markdown_structure(metrics) -> None:
    fairness_payload = {
        "overall": {"auc": 0.88},
        "gaps": {"auc_gap": 0.12},
        "by_group": {
            "group_a": {"auc": 0.9, "logloss": 0.4, "count": 10},
        },
    }

    shap_table = [
        {"feature": "gender", "mean_shap": 0.12, "cramers_v": 0.45},
        {"feature": "country", "mean_shap": -0.05, "cramers_v": 0.33},
    ]

    markdown = utils.render_eval_markdown(
        metrics,
        [("Fairness", fairness_payload), ("Other", None)],
        shap_plot_path="plot.png",
        timestamp="20240101_000000",
        shap_vs_cramers_table=shap_table,
    )

    assert "# Evaluation Results" in markdown
    assert "- Timestamp: 20240101_000000" in markdown
    assert "- Logloss: 0.1234" in markdown
    assert "- Overall AUC: 0.8800" in markdown
    assert "![SHAP Summary](plot.png)" in markdown
    assert "## SHAP vs Cramér's V" in markdown
    assert "| gender | 0.1200 | 0.4500 |" in markdown


if __name__ == "__main__":
    import sys
    try:
        import pytest
    except Exception:
        print("Pytest non disponible. Installez-le dans le venv: pip install pytest")
        sys.exit(1)
    code = pytest.main([__file__, "-v"])
    sys.exit(code)

