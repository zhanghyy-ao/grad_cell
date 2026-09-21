#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

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
RESULT_ROOT="${THREE_STAGE_RESULTS:-results/deepseek_2158_three_stage}"
EXPECTED_RECORDS="${DEEPSEEK_EXPECTED_RECORDS:-2158}"
LANGUAGE_SOURCE="${DEEPSEEK_LANGUAGE_SOURCE:-deepseek-v4-flash}"
SEEDS="${THREE_STAGE_SEEDS:-7}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-50}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2}"
SPME_TIME_POINTS="${SPME_TIME_POINTS:-101}"
STAGE3_CORRECTION_EPOCHS="${STAGE3_CORRECTION_EPOCHS:-1}"
STAGE3_CORRECTION_TRAIN_RECORDS="${STAGE3_CORRECTION_TRAIN_RECORDS:-4}"
STAGE3_CORRECTION_VALIDATION_RECORDS="${STAGE3_CORRECTION_VALIDATION_RECORDS:-2}"
STAGE3_CORRECTION_TEST_RECORDS="${STAGE3_CORRECTION_TEST_RECORDS:-2}"
STAGE3_AUDIT_RECORDS="${STAGE3_AUDIT_RECORDS:-16}"
STAGE3_DFN_TIME_POINTS="${STAGE3_DFN_TIME_POINTS:-301}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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
    --batch-size "${EMBED_BATCH_SIZE:-8}" \
    --dtype bfloat16 \
    --local-files-only
fi

for seed in $SEEDS; do
  SEED_ROOT="$RESULT_ROOT/seed_${seed}"
  STAGE1_DIR="$SEED_ROOT/stage1_design_supervision"
  STAGE2_DIR="$SEED_ROOT/stage2_spme_online"
  STAGE3_DIR="$SEED_ROOT/stage3_dfn_correction"
  AUDIT_DIR="$SEED_ROOT/stage3_dfn_acceptance"

  echo "[Stage 1/3] supervised design-parameter training (seed=$seed)"
  "$PYTHON_BIN" scripts/train_battery_description_stage1_supervised.py \
    --data "$DATA" \
    --embeddings "$EMBEDDINGS" \
    --output-dir "$STAGE1_DIR" \
    --hidden-dim 512 \
    --num-blocks 2 \
    --batch-size "$STAGE1_BATCH_SIZE" \
    --epochs "$STAGE1_EPOCHS" \
    --seed "$seed" \
    --device cuda

  echo "[Stage 2/3] SPMe online physics-gradient training (seed=$seed)"
  "$PYTHON_BIN" -X faulthandler scripts/train_battery_description_direct_dfn.py \
    --data "$DATA" \
    --embeddings "$EMBEDDINGS" \
    --initial-checkpoint "$STAGE1_DIR/best_model.pt" \
    --output-dir "$STAGE2_DIR" \
    --physics-model SPMe \
    --hidden-dim 512 \
    --num-blocks 2 \
    --batch-size "$STAGE2_BATCH_SIZE" \
    --epochs "$STAGE2_EPOCHS" \
    --learning-rate 1e-4 \
    --design-weight 0.25 \
    --performance-weight 1.0 \
    --feasibility-weight 10.0 \
    --support-weight 1.0 \
    --time-points "$SPME_TIME_POINTS" \
    --rtol 1e-6 \
    --atol 1e-8 \
    --training-voltage-floor-v 2.0 \
    --minimum-physics-success-rate 0.8 \
    --seed "$seed" \
    --device cuda

  echo "[Stage 3/3] sampled DFN sensitivity correction (seed=$seed)"
  "$PYTHON_BIN" -X faulthandler scripts/train_battery_description_direct_dfn.py \
    --data "$DATA" \
    --embeddings "$EMBEDDINGS" \
    --initial-checkpoint "$STAGE2_DIR/best_model.pt" \
    --output-dir "$STAGE3_DIR" \
    --physics-model DFN \
    --hidden-dim 512 \
    --num-blocks 2 \
    --batch-size 1 \
    --epochs "$STAGE3_CORRECTION_EPOCHS" \
    --learning-rate 1e-5 \
    --design-weight 0.25 \
    --performance-weight 1.0 \
    --feasibility-weight 10.0 \
    --support-weight 1.0 \
    --max-train-records "$STAGE3_CORRECTION_TRAIN_RECORDS" \
    --max-validation-records "$STAGE3_CORRECTION_VALIDATION_RECORDS" \
    --max-test-records "$STAGE3_CORRECTION_TEST_RECORDS" \
    --time-points 151 \
    --rtol 1e-6 \
    --atol 1e-8 \
    --training-voltage-floor-v 2.0 \
    --minimum-physics-success-rate 0.5 \
    --early-stopping-patience 1 \
    --seed "$seed" \
    --device cuda

  echo "[Stage 3/3] independent strict DFN acceptance (seed=$seed)"
  "$PYTHON_BIN" -X faulthandler scripts/audit_battery_description_stage3_dfn.py \
    --data "$DATA" \
    --embeddings "$EMBEDDINGS" \
    --checkpoint "$STAGE3_DIR/best_model.pt" \
    --output-dir "$AUDIT_DIR" \
    --max-records "$STAGE3_AUDIT_RECORDS" \
    --relative-tolerance "${STAGE3_RELATIVE_TOLERANCE:-0.10}" \
    --minimum-dfn-success-rate "${STAGE3_MINIMUM_DFN_SUCCESS_RATE:-0.95}" \
    --minimum-all-metrics-within-tolerance-rate "${STAGE3_MINIMUM_ALL_WITHIN_RATE:-0.20}" \
    --time-points "$STAGE3_DFN_TIME_POINTS" \
    --rtol 1e-8 \
    --atol 1e-10 \
    --seed "$((seed + 7000))" \
    --device cpu
done

echo "Three-stage training and DFN acceptance completed: $RESULT_ROOT"
