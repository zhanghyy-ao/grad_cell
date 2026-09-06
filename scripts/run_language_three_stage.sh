#!/usr/bin/env bash
set -euo pipefail

K0_CHECKPOINT="${K0_CHECKPOINT:-results/gradcell_exploration/k0_s7/model.pt}"
REFERENCE_FRONT="${REFERENCE_FRONT:-results/gradcell_exploration/reference/pareto_front_1c5c6c.npz}"
DATA="${DATA:-data/gradcell_lm/k0_distillation_s7.jsonl}"
RUN_ROOT="${RUN_ROOT:-results/gradcell_lm}"

python scripts/generate_language_design_data.py \
  --checkpoint "$K0_CHECKPOINT" --reference-front "$REFERENCE_FRONT" \
  --samples 4096 --output "$DATA"

python scripts/train_language_stage1_semantic.py \
  --data "$DATA" --output-dir "$RUN_ROOT/stage1_s7" --load-in-4bit

python scripts/train_language_stage2_k0_physics.py \
  --stage1-dir "$RUN_ROOT/stage1_s7" --reference-front "$REFERENCE_FRONT" \
  --output "$RUN_ROOT/stage2_k0_s7.pt" --backend pybamm --physics-model SPMe \
  --load-in-4bit --steps 1000 --batch-size 1

python scripts/train_language_stage3_k3_refiner.py \
  --stage1-dir "$RUN_ROOT/stage1_s7" --stage2-checkpoint "$RUN_ROOT/stage2_k0_s7.pt" \
  --reference-front "$REFERENCE_FRONT" --output "$RUN_ROOT/stage3_k3_s7.pt" \
  --backend pybamm --physics-model SPMe --load-in-4bit --steps 300 \
  --refinement-steps 3
