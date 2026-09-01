import importlib.util
from pathlib import Path


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_electrolyte_v1_pipeline.py"
    spec = importlib.util.spec_from_file_location("electrolyte_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pipeline_no_longer_exposes_property_physics_weight_sweep():
    module = load_pipeline_module()
    assert not hasattr(module, "weight_slug")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "--physics-weights" not in source
    assert "--physics-validation-samples" in source
