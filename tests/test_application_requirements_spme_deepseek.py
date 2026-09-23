from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_application_requirements_qwen_mlp_spme_deepseek.py"
)
SPEC = importlib.util.spec_from_file_location("application_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_projection_makes_both_electrodes_feasible() -> None:
    values = np.array([0.4, 0.3, 0.5, 0.7, 0.6, 1.0, 1.0])
    projected, report = MODULE.project_structural_feasibility(values, margin=1e-4)
    assert not MODULE.structural_feasibility(values)
    assert MODULE.structural_feasibility(projected)
    assert projected[0] + projected[3] == pytest.approx(0.9999)
    assert projected[1] + projected[4] < 1.0
    assert report["applied"] is True
    assert values[2] == projected[2]


def test_judge_validation_rejects_unsupported_status() -> None:
    with pytest.raises(MODULE.InvalidJudgeResponse):
        MODULE.validate_judgement(
            {
                "overall_verdict": "satisfied",
                "confidence": 0.9,
                "summary": "ok",
                "requirement_assessments": [
                    {"requirement": "容量", "status": "unknown", "evidence": "none"}
                ],
            }
        )


def test_judge_prompt_marks_product_requirements_not_evaluable() -> None:
    row = {
        "battery_description": "设计70Ah储能电芯，循环3000次。",
        "requirements": {"capacity_ah": 70, "cycle_count": 3000},
        "current_model_scope": {},
        "candidate_design": {"raw_structurally_feasible": True},
        "spme_evaluation": {"success": True},
    }
    instruction = MODULE.judge_instruction(row)
    rules = "".join(instruction["mandatory_scope_rules"])
    assert "循环寿命" in rules
    assert "not_evaluable" in rules
    assert "不等于" in rules


def test_extracts_fenced_json() -> None:
    parsed = MODULE.extract_json_object('```json\n{"overall_verdict":"insufficient_evidence"}\n```')
    assert parsed["overall_verdict"] == "insufficient_evidence"
