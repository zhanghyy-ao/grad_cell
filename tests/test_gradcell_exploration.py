from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gradcell.benchmark import nominal_design_from_parameter_values
from gradcell.evaluation import hard_cutoff_metrics_from_physical_inputs, scalarized_loss

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


def test_unseen_midpoint_preferences_are_between_grid_points() -> None:
    module = load_script("evaluate_unseen_preferences")
    values = module.midpoint_preferences(5)
    assert np.allclose(values, [0.125, 0.375, 0.625, 0.875])
    assert not np.isin(values, np.linspace(0.0, 1.0, 5)).any()


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


def test_direct_physics_optimizer_reduces_independent_latent_losses() -> None:
    module = load_script("compare_initializer_physics_optimization")

    class QuadraticPhysicsModel:
        @staticmethod
        def evaluate(latent, preferences):
            target = preferences[:, None].expand_as(latent)
            loss = (latent - target).square().sum(dim=-1)
            batch = latent.shape[0]
            return SimpleNamespace(
                loss=loss,
                status=torch.ones(batch, dtype=torch.int64),
                energy=-loss,
                retention_5c=torch.ones(batch, dtype=latent.dtype),
                retention_6c=torch.ones(batch, dtype=latent.dtype),
            )

    preferences = torch.tensor([0.2, 0.8], dtype=torch.float64)
    initial = torch.zeros((2, 5), dtype=torch.float64)
    optimized, trace = module.optimize_latent_with_physics(
        QuadraticPhysicsModel(),
        initial,
        preferences,
        steps=20,
        learning_rate=0.1,
        max_step_norm=0.25,
        latent_limit=2.0,
    )
    assert trace[-1]["mean_soft_loss"] < trace[0]["mean_soft_loss"]
    assert torch.linalg.vector_norm(optimized[1] - initial[1]) > 0.0


def test_pairwise_report_treats_lower_loss_as_better() -> None:
    module = load_script("compare_initializer_physics_optimization")
    left = {
        "status": np.ones(2, dtype=np.int64),
        "loss": np.asarray([0.4, 0.2]),
        "energy_wh_kg": np.asarray([100.0, 110.0]),
        "energy_retention_5c": np.asarray([0.50, 0.51]),
        "energy_retention_6c": np.asarray([0.44, 0.45]),
    }
    right = {
        "status": np.ones(2, dtype=np.int64),
        "loss": np.asarray([0.3, 0.1]),
        "energy_wh_kg": np.asarray([101.0, 111.0]),
        "energy_retention_5c": np.asarray([0.51, 0.52]),
        "energy_retention_6c": np.asarray([0.45, 0.46]),
    }
    report = module.pairwise_report("initial", left, "optimized", right)
    assert report["right_better_loss_fraction"] == 1.0
    assert np.isclose(report["mean_loss_reduction"], 0.1)


def test_k0_physics_refinement_preference_satisfaction_requires_constraints() -> None:
    module = load_script("evaluate_k0_physics_refinement")
    initial = {
        "status": np.ones(2, dtype=np.int64),
        "loss": np.asarray([0.30, 0.30]),
        "energy_retention_5c": np.asarray([0.51, 0.51]),
        "energy_retention_6c": np.asarray([0.45, 0.45]),
    }
    optimized = {
        "status": np.ones(2, dtype=np.int64),
        "loss": np.asarray([0.20, 0.20]),
        "energy_retention_5c": np.asarray([0.52, 0.49]),
        "energy_retention_6c": np.asarray([0.46, 0.46]),
    }
    bounds = {"retention_5c_min": 0.50, "retention_6c_min": 0.44}
    report, diagnostics = module.summarize(
        initial,
        optimized,
        np.asarray([0.19, 0.19]),
        bounds,
        tolerance=1e-8,
        regret_threshold=0.02,
    )
    assert diagnostics["preference_satisfied"].tolist() == [True, False]
    assert report["hard_loss_nonworsening_fraction"] == 1.0
    assert report["preference_satisfaction_rate"] == 0.5


def test_k0_physics_refinement_rejects_mismatched_front() -> None:
    module = load_script("evaluate_k0_physics_refinement")
    front = {
        "metadata": {
            "source_model": "DFN",
            "capacity_formula": "chen2020_scaled",
            "capacity_multiplier": 1.25,
        }
    }
    config = {
        "physics_model": "SPMe",
        "capacity_formula": "chen2020_scaled",
        "capacity_multiplier": 1.25,
    }
    try:
        module.validate_provenance(front, config)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched reference model should be rejected")


def test_physical_nominal_design_uses_parameter_set_values_without_latent() -> None:
    values = {
        "Positive electrode porosity": 0.335,
        "Negative electrode porosity": 0.25,
        "Separator porosity": 0.47,
        "Positive electrode active material volume fraction": 0.665,
        "Negative electrode active material volume fraction": 0.75,
        "Nominal cell capacity [A.h]": 5.0,
    }
    design = nominal_design_from_parameter_values(values)
    assert design.physics_inputs.shape == (1, 8)
    assert np.allclose(design.physics_inputs[0, :5], [0.335, 0.25, 0.47, 0.665, 0.75])
    assert np.allclose(design.physics_inputs[0, 5:7], 1.0)
    assert design.initial_capacity_ah.tolist() == [5.0]
    assert design.stack_mass_kg[0] > 0.0


def test_physical_hard_cutoff_evaluator_rejects_misaligned_inputs() -> None:
    try:
        hard_cutoff_metrics_from_physical_inputs(
            np.zeros((2, 7)),
            np.ones(2),
            np.ones(2),
            "SPMe",
        )
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("invalid physical input shape should be rejected")
