#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-data/gradcell_lm/k0_distillation_s7.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-results/gradcell_lm/qwen3_8b_stage1.pt}"
EPOCHS="${EPOCHS:-3}"

mkdir -p "$(dirname "$OUTPUT_PATH")"
export TOKENIZERS_PARALLELISM=false

python scripts/train_language_distillation.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_PATH" \
  --model-name Qwen/Qwen3-8B \
  --load-in-4bit \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --gradient-accumulation 16
