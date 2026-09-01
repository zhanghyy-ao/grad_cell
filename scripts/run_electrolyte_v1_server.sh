#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-7}"
PHYSICS_SAMPLES="${PHYSICS_SAMPLES:-32}"
PHYSICS_VALIDATION_SAMPLES="${PHYSICS_VALIDATION_SAMPLES:-8}"
PHYSICS_TEST_SAMPLES="${PHYSICS_TEST_SAMPLES:-8}"
EPOCHS="${EPOCHS:-100}"
RUN_NAME="${RUN_NAME:-electrolyte_physics_only_dfn_s${SEED}}"
DATA_PATH="data/${RUN_NAME}.npz"
RESULT_DIR="results/${RUN_NAME}"
export PYTHONPATH="${PYTHONPATH:-src}"

echo "[1/5] Tests and static checks"
"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m ruff check src scripts tests

echo "[2/5] DOI-isolated synthetic PyBaMM DFN voltage targets"
"${PYTHON_BIN}" scripts/prepare_electrolyte_v1_data.py \
  --physics-backend dfn --physics-samples "${PHYSICS_SAMPLES}" \
  --physics-validation-samples "${PHYSICS_VALIDATION_SAMPLES}" \
  --physics-test-samples "${PHYSICS_TEST_SAMPLES}" \
  --time-points 41 --probe-horizon-s 600 --current-a 5 \
  --seed "${SEED}" --output "${DATA_PATH}"

echo "[3/5] Analytic VJP check"
"${PYTHON_BIN}" scripts/validate_electrolyte_v1_gradients.py --backend analytic --eps 1e-6

echo "[4/5] PyBaMM DFN sensitivity epsilon sweep"
for eps_value in 1e-3 3e-4 1e-4 3e-5 1e-5; do
  "${PYTHON_BIN}" scripts/validate_electrolyte_v1_gradients.py \
    --backend dfn --time-points 21 --probe-horizon-s 300 --eps "${eps_value}"
done

echo "[5/5] Physics-only end-to-end training"
"${PYTHON_BIN}" scripts/train_electrolyte_v1.py \
  --data "${DATA_PATH}" --physics-backend dfn \
  --physics-samples "${PHYSICS_SAMPLES}" --epochs "${EPOCHS}" \
  --physics-batch-size 2 --seed "${SEED}" --output-dir "${RESULT_DIR}"

echo "Completed: ${RESULT_DIR}"
