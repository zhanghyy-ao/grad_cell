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
SOURCE_DATA="${DEEPSEEK_SOURCE_DATA:-data/multiset_dfn_language/battery_description_modes_v3.jsonl}"
DATA="${DEEPSEEK_2158_DATA:-data/multiset_dfn_language/deepseek_2158_strict_physics.jsonl}"
MANIFEST="${DEEPSEEK_2158_MANIFEST:-data/multiset_dfn_language/deepseek_2158_strict_physics_manifest.json}"
EMBEDDINGS="${DEEPSEEK_2158_EMBEDDINGS:-data/multiset_dfn_language/deepseek_2158_physics_qwen_embeddings.npz}"
RESULT_ROOT="${DIRECT_DFN_RESULTS:-results/deepseek_2158_direct_dfn}"
EXPECTED_RECORDS="${DEEPSEEK_EXPECTED_RECORDS:-2158}"
LANGUAGE_SOURCE="${DEEPSEEK_LANGUAGE_SOURCE:-deepseek-v4-flash}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-8}"
DFN_BATCH_SIZE="${DFN_BATCH_SIZE:-2}"
DFN_EPOCHS="${DFN_EPOCHS:-10}"
DFN_TIME_POINTS="${DFN_TIME_POINTS:-151}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test -s "$SOURCE_DATA"
test -s "$MODEL_NAME/config.json"
mkdir -p "$(dirname "$DATA")" "$(dirname "$EMBEDDINGS")" "$RESULT_ROOT"

"$PYTHON_BIN" scripts/prepare_deepseek_topk_dataset.py \
  --input "$SOURCE_DATA" \
  --output "$DATA" \
  --manifest "$MANIFEST" \
  --language-source "$LANGUAGE_SOURCE" \
  --expected-records "$EXPECTED_RECORDS" \
  --train-ratio 0.80 \
  --validation-ratio 0.10 \
  --test-ratio 0.10 \
  --seed 7

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
  "$PYTHON_BIN" scripts/train_battery_description_direct_dfn.py \
    --data "$DATA" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$RESULT_ROOT/seed_${seed}" \
    --hidden-dim 512 \
    --num-blocks 2 \
    --batch-size "$DFN_BATCH_SIZE" \
    --epochs "$DFN_EPOCHS" \
    --learning-rate 1e-4 \
    --design-weight 0.25 \
    --performance-weight 1.0 \
    --feasibility-weight 10.0 \
    --support-weight 1.0 \
    --time-points "$DFN_TIME_POINTS" \
    --rtol 1e-6 \
    --atol 1e-8 \
    --early-stopping-patience 3 \
    --seed "$seed" \
    --device cuda
done

echo "Direct PyBaMM DFN-gradient training completed: $RESULT_ROOT"
