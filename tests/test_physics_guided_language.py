import torch

from gradcell.language.physics_guided import (
    DFNPerformanceSurrogate,
    SingleDesignPhysicsMLP,
    freeze_surrogate,
)


def test_single_design_is_bounded() -> None:
    model = SingleDesignPhysicsMLP(
        input_dim=16,
        design_lower=[-2.0] * 7,
        design_upper=[3.0] * 7,
        hidden_dim=32,
        num_blocks=1,
        dropout=0.0,
    )
    output = model(torch.randn(5, 16))
    assert output.shape == (5, 7)
    assert torch.all(output >= -2.0)
    assert torch.all(output <= 3.0)


def test_frozen_surrogate_returns_gradient_to_design_model() -> None:
    torch.manual_seed(7)
    design_model = SingleDesignPhysicsMLP(
        input_dim=16,
        design_lower=[-2.0] * 7,
        design_upper=[2.0] * 7,
        hidden_dim=32,
        num_blocks=1,
        dropout=0.0,
    )
    surrogate = freeze_surrogate(
        DFNPerformanceSurrogate(
            input_dim=7,
            output_dim=8,
            hidden_dim=32,
            num_blocks=1,
            dropout=0.0,
        )
    )
    prediction = surrogate(design_model(torch.randn(4, 16)))
    loss = torch.nn.functional.mse_loss(prediction, torch.randn(4, 8))
    loss.backward()

    design_gradients = [parameter.grad for parameter in design_model.parameters()]
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in design_gradients
    )
    assert all(parameter.grad is None for parameter in surrogate.parameters())
    assert all(not parameter.requires_grad for parameter in surrogate.parameters())
