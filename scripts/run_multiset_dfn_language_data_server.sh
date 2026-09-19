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
CONFIG="${MULTISET_CONFIG:-configs/multiset_dfn_language.yaml}"
RUN_PHYSICS="${RUN_PHYSICS:-1}"
WITH_DEEPSEEK="${WITH_DEEPSEEK:-1}"

if [[ "$RUN_PHYSICS" == "1" ]]; then
  echo "[1/2] Generate or resume the five-parameter-set DFN archive"
  "$PYTHON_BIN" scripts/generate_multiset_dfn_archive.py --config "$CONFIG"
else
  echo "[1/2] Skip DFN generation because RUN_PHYSICS=$RUN_PHYSICS"
fi

echo "[2/2] Build battery-description-to-parameter-design records"
dataset_args=(--config "$CONFIG")
if [[ "$WITH_DEEPSEEK" == "1" ]]; then
  : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in .env}"
  dataset_args+=(--with-deepseek --require-deepseek-success)
fi
"$PYTHON_BIN" scripts/build_multiset_dfn_language_dataset.py "${dataset_args[@]}"

echo "Completed. See data/multiset_dfn_language/ for archives and manifests."
