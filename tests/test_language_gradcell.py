import torch
from torch import nn

from gradcell.language import GradCellLanguageCodec, LanguageGradCell, StructuredPreferenceTask
from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer


class TinyBackbone(nn.Module):
    hidden_size = 16

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, self.hidden_size)

    def forward(self, input_ids, attention_mask):
        return self.embedding(input_ids)


def _gradcell():
    return GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=720.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=600.0)),
    ).double()


def test_codec_roundtrip_and_schema():
    codec = GradCellLanguageCodec(bins=256, latent_limit=4.0)
    text = codec.serialize_task(StructuredPreferenceTask(0.7))
    assert "<PREFERENCE><LEVEL_178></PREFERENCE>" in text
    latent = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
    restored = codec.dequantize(codec.quantize(latent), dtype=latent.dtype)
    assert torch.allclose(restored, latent, atol=2 * 4.0 / 255)
    assert codec.serialize_design(latent).endswith("</DESIGN>")


def test_language_gradcell_physics_backward():
    model = LanguageGradCell(
        TinyBackbone(), _gradcell(), codec=GradCellLanguageCodec(bins=16)
    ).double()
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    preference = torch.tensor([0.2, 0.8], dtype=torch.float64)
    output = model(input_ids, attention_mask, preference, num_steps=1)
    assert output.level_logits.shape == (2, 5, 16)
    assert len(output.gradcell.steps) == 2
    loss = output.gradcell.final.loss.mean()
    loss.backward()
    assert model.continuous_head.weight.grad is not None
    assert torch.isfinite(model.continuous_head.weight.grad).all()
