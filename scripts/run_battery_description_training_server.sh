#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${QWEN_MODEL_NAME:-$PWD/models/Qwen3-8B}"
DATA="${BATTERY_DESCRIPTION_DATA:-data/multiset_dfn_language/battery_description_modes_v3.jsonl}"
EMBEDDINGS="${BATTERY_DESCRIPTION_EMBEDDINGS:-data/multiset_dfn_language/qwen_embeddings_modes_v3.npz}"
RESULT_ROOT="${BATTERY_DESCRIPTION_RESULTS:-results/battery_description_mlp_v3}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-8}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-64}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test -s "$DATA"
test -s "$MODEL_NAME/config.json"
mkdir -p "$(dirname "$EMBEDDINGS")" "$RESULT_ROOT"

if [[ ! -s "$EMBEDDINGS" ]]; then
  "$PYTHON_BIN" scripts/extract_paper_explore_embeddings.py \
    --data "$DATA" \
    --text-field battery_description \
    --model-name "$MODEL_NAME" \
    --output "$EMBEDDINGS" \
    --pooling mean \
    --max-length 1024 \
    --batch-size "$EMBED_BATCH_SIZE" \
    --dtype bfloat16 \
    --local-files-only
fi

for seed in 7 17 27; do
  "$PYTHON_BIN" scripts/train_battery_description_mlp.py \
    --data "$DATA" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$RESULT_ROOT/seed_${seed}" \
    --hidden-dim 512 \
    --num-blocks 2 \
    --dropout 0.1 \
    --batch-size "$MLP_BATCH_SIZE" \
    --epochs 300 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --early-stopping-patience 30 \
    --parameter-weight 1.0 \
    --parameter-set-weight 0.25 \
    --mode-weight 0.10 \
    --seed "$seed" \
    --device cuda
done

echo "Training completed. See $RESULT_ROOT/seed_*/metrics.json"
