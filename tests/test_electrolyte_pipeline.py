import importlib.util
from pathlib import Path


def load_pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_electrolyte_v1_pipeline.py"
    spec = importlib.util.spec_from_file_location("electrolyte_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_weight_slug_is_path_safe_and_stable():
    module = load_pipeline_module()
    assert module.weight_slug(0.0) == "0"
    assert module.weight_slug(0.1) == "0p1"
    assert module.weight_slug(1.0) == "1"
    assert module.weight_slug(2.5) == "2p5"
