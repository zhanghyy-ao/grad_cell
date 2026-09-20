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
RESULT_ROOT="${PHYSICS_GUIDED_RESULTS:-results/deepseek_2158_physics_guided}"
SURROGATE_DIR="$RESULT_ROOT/dfn_surrogate"
LANGUAGE_SOURCE="${DEEPSEEK_LANGUAGE_SOURCE:-deepseek-v4-flash}"
EXPECTED_RECORDS="${DEEPSEEK_EXPECTED_RECORDS:-2158}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-8}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-64}"
RUN_DFN_VERIFY="${RUN_DFN_VERIFY:-0}"
DFN_VERIFY_SEED="${DFN_VERIFY_SEED:-7}"

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

"$PYTHON_BIN" scripts/train_dfn_performance_surrogate.py \
  --data "$DATA" \
  --output-dir "$SURROGATE_DIR" \
  --hidden-dim 256 \
  --num-blocks 3 \
  --batch-size "$MLP_BATCH_SIZE" \
  --epochs 500 \
  --early-stopping-patience 50 \
  --seed 7 \
  --device cuda

for seed in 7 17 27; do
  "$PYTHON_BIN" scripts/train_battery_description_physics_guided.py \
    --data "$DATA" \
    --embeddings "$EMBEDDINGS" \
    --surrogate-checkpoint "$SURROGATE_DIR/best_model.pt" \
    --output-dir "$RESULT_ROOT/seed_${seed}" \
    --hidden-dim 512 \
    --num-blocks 2 \
    --batch-size "$MLP_BATCH_SIZE" \
    --epochs 300 \
    --learning-rate 1e-3 \
    --design-weight 0.25 \
    --performance-weight 1.0 \
    --feasibility-weight 10.0 \
    --support-weight 1.0 \
    --performance-warmup-epochs 20 \
    --early-stopping-patience 30 \
    --seed "$seed" \
    --device cuda
done

if [[ "$RUN_DFN_VERIFY" == "1" ]]; then
  "$PYTHON_BIN" scripts/verify_language_designs_dfn.py \
    --predictions "$RESULT_ROOT/seed_${DFN_VERIFY_SEED}/test_predictions.jsonl" \
    --config configs/multiset_dfn_language.yaml \
    --output "$RESULT_ROOT/seed_${DFN_VERIFY_SEED}/test_predictions_dfn.jsonl" \
    --report "$RESULT_ROOT/seed_${DFN_VERIFY_SEED}/dfn_replay_metrics.json" \
    --batch-size 8
fi

echo "Physics-guided single-design training completed: $RESULT_ROOT"
