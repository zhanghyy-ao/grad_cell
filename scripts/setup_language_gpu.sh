#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[physics,language-gpu,dev]"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch installation succeeded but no CUDA GPU is visible")
print("gpu:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
