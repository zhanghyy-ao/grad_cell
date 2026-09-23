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
PROMPT_DATA="${APPLICATION_PROMPT_DATA:-data/application_prompts/application_battery_prompts_deepseek_s7.jsonl}"
CHECKPOINT="${DESIGN_CHECKPOINT:-results/deepseek_2158_three_stage/seed_7/stage2_spme_online/best_model.pt}"
OUTPUT_DIR="${APPLICATION_EVAL_OUTPUT_DIR:-results/application_requirements_spme_deepseek}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

test -s "$PROMPT_DATA"
test -s "$MODEL_NAME/config.json"
test -s "$CHECKPOINT"
mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "${APPLICATION_EVAL_MAX_RECORDS:-}" ]]; then
  EXTRA_ARGS+=(--max-records "$APPLICATION_EVAL_MAX_RECORDS")
fi

"$PYTHON_BIN" -X faulthandler \
  scripts/evaluate_application_requirements_qwen_mlp_spme_deepseek.py \
  --data "$PROMPT_DATA" \
  --model-name "$MODEL_NAME" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT_DIR/evaluations.jsonl" \
  --report "$OUTPUT_DIR/report.json" \
  --batch-size "${APPLICATION_EVAL_BATCH_SIZE:-4}" \
  --max-length "${APPLICATION_EVAL_MAX_LENGTH:-1024}" \
  --spme-reference-capacity-ah "${SPME_REFERENCE_CAPACITY_AH:-5.0}" \
  --time-points "${SPME_TIME_POINTS:-151}" \
  --rtol "${SPME_RTOL:-1e-6}" \
  --atol "${SPME_ATOL:-1e-8}" \
  --current-ramp-time-s "${SPME_CURRENT_RAMP_TIME_S:-1.0}" \
  --retries "${DEEPSEEK_RETRIES:-3}" \
  --timeout-s "${DEEPSEEK_TIMEOUT_S:-300}" \
  --local-files-only \
  --require-deepseek-success \
  "${EXTRA_ARGS[@]}"

echo "Evaluation completed: $OUTPUT_DIR"
