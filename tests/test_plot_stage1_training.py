from __future__ import annotations

import json
import sys

import pytest


pytest.importorskip("matplotlib")

from scripts.plot_stage1_training import main


def test_plot_stage1_training_supports_legacy_history(tmp_path, monkeypatch) -> None:
    history = [
        {
            "epoch": 1,
            "train_design_loss": 0.8,
            "normalized_design_mse": 0.7,
            "normalized_design_smooth_l1": 0.3,
        },
        {
            "epoch": 2,
            "train_design_loss": 0.5,
            "normalized_design_mse": 0.4,
            "normalized_design_smooth_l1": 0.2,
        },
    ]
    metrics = {
        "validation": {
            "normalized_design_mse": 0.4,
            "normalized_design_smooth_l1": 0.2,
        },
        "test": {
            "normalized_design_mse": 0.45,
            "normalized_design_smooth_l1": 0.22,
        },
    }
    (tmp_path / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["plot_stage1_training.py", "--run-dir", str(tmp_path)])

    main()

    assert (tmp_path / "stage1_training_curves.png").stat().st_size > 0
    assert (tmp_path / "stage1_training_curves.pdf").stat().st_size > 0
    summary = json.loads(
        (tmp_path / "stage1_training_curves_summary.json").read_text(encoding="utf-8")
    )
    assert summary["best_epoch"] == 2
    assert summary["best_validation_design_smooth_l1"] == pytest.approx(0.2)
