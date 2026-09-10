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
SHARD_COUNT="${SHARD_COUNT:-4}"
SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD:-2048}"
TIME_POINTS="${TIME_POINTS:-101}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/paper_explore_2000}"
RESULT_ROOT="${RESULT_ROOT:-results/paper_explore_2000}"

mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT"

echo "[1/4] Generate SPMe physics shards"
pids=()
archives=()
for ((shard=0; shard<SHARD_COUNT; shard++)); do
  seed=$((101 + shard))
  archive="$OUTPUT_ROOT/physics_archive_spme_${SAMPLES_PER_SHARD}_s${seed}.npz"
  archives+=("$archive")
  if [[ -s "$archive" ]]; then
    echo "skip existing $archive"
    continue
  fi
  "$PYTHON_BIN" scripts/generate_supervised_data.py \
    --backend pybamm --model SPMe \
    --capacity-formula chen2020_scaled \
    --samples "$SAMPLES_PER_SHARD" --batch-size 8 \
    --sampler sobol --latent-limit 4 --time-points "$TIME_POINTS" \
    --seed "$seed" --output "$archive" \
    --run-dir "$RESULT_ROOT/archive_s${seed}" \
    > "$RESULT_ROOT/archive_s${seed}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[2/4] Build empirical Pareto front"
"$PYTHON_BIN" scripts/build_reference_front_3d.py \
  --data "${archives[@]}" \
  --output "$OUTPUT_ROOT/pareto_front_spme_8192.npz" \
  --min-retention-5c 0.0 --min-retention-6c 0.0 \
  --capacity-formula chen2020_scaled

echo "[3/4] Build 2000 canonical language tasks"
"$PYTHON_BIN" scripts/generate_paper_explore_dataset.py \
  --config configs/paper_explore_2000.yaml

echo "[4/4] Optional DeepSeek rewrite"
if [[ "${WITH_DEEPSEEK:-0}" == "1" ]]; then
  : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before enabling DeepSeek rewrite}"
  "$PYTHON_BIN" scripts/generate_paper_explore_dataset.py \
    --config configs/paper_explore_2000.yaml --with-deepseek
else
  echo "Skipped. Run again with WITH_DEEPSEEK=1 after setting DEEPSEEK_API_KEY."
fi

echo "Workflow completed. See $OUTPUT_ROOT/manifest.json"
