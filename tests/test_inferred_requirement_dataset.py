from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_inferred_requirement_dataset.py"
SPEC = importlib.util.spec_from_file_location("inferred_requirement", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(design_id: str, energy: float, retention: float, multiplier: float) -> dict:
    performance = {
        "capacity_1c_ah": energy / 3.5,
        "energy_1c_wh": energy,
        "average_voltage_1c_v": 3.5,
        "minimum_voltage_1c_v": 2.8,
        "energy_retention_5c": retention,
        "energy_retention_6c": retention * 0.9,
        "energy_5c_wh": energy * retention,
        "energy_6c_wh": energy * retention * 0.9,
    }
    return {
        "physical_design_id": design_id,
        "teacher_design": {
            "base_parameter_set": "Chen2020",
            "generation_mode": "regular",
            "parameter_updates": {
                "Positive particle diffusivity multiplier": {"multiplier": multiplier}
            },
        },
        "verified_performance": performance,
    }


def test_profiles_are_relative_and_contain_no_exact_performance() -> None:
    rows = [row("a", 10.0, 0.2, 0.5), row("b", 20.0, 0.8, 2.0)]
    references = MODULE.build_references(rows)
    low = MODULE.derive_requirement_profile(rows[0], references)
    high = MODULE.derive_requirement_profile(rows[1], references)
    assert low["energy_requirement_level"] == "中低"
    assert high["energy_requirement_level"] == "中高"
    assert "energy_1c_wh" not in MODULE.profile_tags(high)
    assert "20.0" not in MODULE.deterministic_request(high, 0)


def test_requirement_text_is_valid_and_demand_oriented() -> None:
    rows = [row("a", 10.0, 0.5, 2.0)]
    profile = MODULE.derive_requirement_profile(rows[0], MODULE.build_references(rows))
    text = MODULE.deterministic_request(profile, 0)
    assert MODULE.validate_description(text) == text
    assert any(word in text for word in MODULE.DEMAND_WORDS)


def test_percentile_rank_handles_single_reference() -> None:
    assert MODULE.percentile_rank(1.0, np.asarray([1.0])) == 0.5
