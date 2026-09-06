import json

import pytest
import torch

from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer
from gradcell.training.trainer import train
from tests.test_model import ControlledFailureBackend


def build_model() -> GradCell:
    return GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=720.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=600.0)),
    ).double()


def test_refinement_validation_keeps_autograd_enabled():
    model = build_model()
    result = train(
        model,
        steps=1,
        batch_size=2,
        refinement_steps=1,
        validation_interval=1,
    )

    assert len(result.validation_losses) == 1
    assert torch.isfinite(torch.tensor(result.validation_losses)).all()


def test_training_restores_best_validation_state(tmp_path):
    model = build_model()
    checkpoint = tmp_path / "checkpoint.pt"
    result = train(
        model,
        steps=2,
        batch_size=2,
        validation_interval=1,
        checkpoint_path=checkpoint,
        # A very large delta guarantees that the first validation remains the best.
        min_delta=1e9,
    )

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert result.best_step == 1
    assert saved["best_model_state"] is not None
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), saved["best_model_state"][name])


def test_training_stops_when_all_physics_samples_fail():
    model = GradCell(
        DifferentiablePhysicsLayer(ControlledFailureBackend(fail_all=True)),
        DifferentiablePhysicsLayer(ControlledFailureBackend(fail_all=True)),
        DifferentiablePhysicsLayer(ControlledFailureBackend(fail_all=True)),
    ).double()
    with pytest.raises(RuntimeError, match="All 1C/5C/6C physics simulations failed"):
        train(model, steps=1, batch_size=2, validation_interval=0)


def test_joint_training_logs_k0_preservation_terms(tmp_path):
    model = build_model()

    def teacher(preference):
        return torch.ones(
            preference.shape[0], model.design_space.latent_dim,
            dtype=preference.dtype, device=preference.device,
        )

    log_path = tmp_path / "training.jsonl"
    train(
        model,
        steps=1,
        batch_size=2,
        refinement_steps=1,
        validation_interval=1,
        initial_loss_weight=0.3,
        initializer_teacher=teacher,
        initializer_distillation_weight=1.0,
        log_path=log_path,
    )

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["initial_loss"] > 0.0
    assert record["initializer_distillation"] > 0.0
    assert record["validation_initial_loss"] > 0.0
    assert record["validation_initializer_distillation"] > 0.0
