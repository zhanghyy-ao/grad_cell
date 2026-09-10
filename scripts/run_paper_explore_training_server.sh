#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-8B}"
DATA="${PAPER_DATA:-data/paper_explore_2000/dataset.jsonl}"
EMBEDDINGS="${PAPER_EMBEDDINGS:-data/paper_explore_2000/qwen3_8b_embeddings.npz}"
RESULT_ROOT="${PAPER_RESULT_ROOT:-results/paper_explore_2000/qwen_mlp}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-8}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-32}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-0}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
test -s "$DATA"
mkdir -p "$(dirname "$EMBEDDINGS")" "$RESULT_ROOT"

embedding_args=(
  --data "$DATA"
  --model-name "$MODEL_NAME"
  --output "$EMBEDDINGS"
  --pooling mean
  --max-length 512
  --batch-size "$EMBED_BATCH_SIZE"
  --dtype bfloat16
)
if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  embedding_args+=(--load-in-4bit)
fi

echo "[1/3] Extract frozen Qwen embeddings"
if [[ -s "$EMBEDDINGS" ]]; then
  echo "skip existing $EMBEDDINGS"
else
  "$PYTHON_BIN" scripts/extract_paper_explore_embeddings.py "${embedding_args[@]}"
fi

echo "[2/3] MLP smoke test"
"$PYTHON_BIN" scripts/train_paper_explore_mlp.py \
  --data "$DATA" --embeddings "$EMBEDDINGS" \
  --output-dir "$RESULT_ROOT/smoke" \
  --hidden-dim 128 --num-blocks 1 --batch-size 16 \
  --epochs 5 --early-stopping-patience 3 \
  --max-train-samples 100 --seed 7 --device cuda

test -s "$RESULT_ROOT/smoke/best_model.pt"
test -s "$RESULT_ROOT/smoke/metrics.json"

echo "[3/3] Train formal MLP seeds"
for seed in 7 17 27; do
  "$PYTHON_BIN" scripts/train_paper_explore_mlp.py \
    --data "$DATA" --embeddings "$EMBEDDINGS" \
    --output-dir "$RESULT_ROOT/seed_${seed}" \
    --hidden-dim 512 --num-blocks 2 --dropout 0.1 \
    --batch-size "$MLP_BATCH_SIZE" --epochs 300 \
    --learning-rate 1e-3 --weight-decay 1e-4 \
    --early-stopping-patience 30 --gradient-clip 1.0 \
    --latent-weight 1.0 --performance-weight 0.3 \
    --feasibility-weight 0.3 --unsupported-weight 0.1 \
    --seed "$seed" --device cuda
done

echo "Training completed. Metrics are under $RESULT_ROOT/seed_*/metrics.json"
