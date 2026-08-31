import torch

from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer
from gradcell.training.trainer import train


def build_model() -> GradCell:
    return GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=1200.0)),
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
