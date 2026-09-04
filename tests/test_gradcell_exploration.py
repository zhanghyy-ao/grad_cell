from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from gradcell.evaluation import scalarized_loss

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pareto_mask_rejects_dominated_and_duplicate_points() -> None:
    module = load_script("build_reference_front")
    energy = np.asarray([1.0, 2.0, 2.0, 3.0])
    power = np.asarray([4.0, 2.0, 2.0, 1.0])
    mask = module.pareto_mask(energy, power)
    assert mask.sum() == 3
    assert not mask[2]


def test_scalarized_loss_respects_preference_endpoints() -> None:
    bounds = {
        "energy_ideal": 10.0,
        "energy_nadir": 0.0,
        "high_rate_ideal": 10.0,
        "high_rate_nadir": 0.0,
        "retention_5c_min": 0.0,
        "retention_6c_min": 0.0,
        "constraint_weight": 2.0,
    }
    energy = np.asarray([10.0, 0.0])
    power = np.asarray([0.0, 10.0])
    assert (
        scalarized_loss(energy, power, power, np.ones(2), bounds)[0]
        < scalarized_loss(energy, power, power, np.ones(2), bounds)[1]
    )
    assert (
        scalarized_loss(energy, power, power, np.zeros(2), bounds)[1]
        < scalarized_loss(energy, power, power, np.zeros(2), bounds)[0]
    )


def test_evenly_spaced_indices_include_endpoints() -> None:
    module = load_script("verify_gradcell_dfn")
    indices = module.evenly_spaced_indices(11, 4)
    assert len(indices) == 4
    assert indices[0] == 0
    assert indices[-1] == 10


def test_scalarized_loss_penalizes_high_rate_constraint_violation() -> None:
    bounds = {
        "energy_ideal": 10.0,
        "energy_nadir": 0.0,
        "high_rate_ideal": 1.0,
        "high_rate_nadir": 0.0,
        "retention_5c_min": 0.6,
        "retention_6c_min": 0.5,
        "constraint_weight": 10.0,
    }
    energy = np.asarray([8.0, 8.0])
    retention_5c = np.asarray([0.7, 0.4])
    retention_6c = np.asarray([0.6, 0.3])
    loss = scalarized_loss(energy, retention_5c, retention_6c, np.full(2, 0.5), bounds)
    assert loss[0] < loss[1]
