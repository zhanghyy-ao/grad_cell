from __future__ import annotations

import argparse
from pathlib import Path

from gradcell.language.physics_training import (
    PhysicsTrainingConfig,
    build_gradcell,
    load_language_model,
    train_physics_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Qwen K=0 PyBaMM training.")
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--reference-front", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="pybamm")
    parser.add_argument("--physics-model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    gradcell = build_gradcell(args.backend, args.physics_model, args.reference_front)
    model, tokenizer = load_language_model(
        stage1_dir=args.stage1_dir,
        gradcell=gradcell,
        load_in_4bit=args.load_in_4bit,
    )
    for parameter in model.gradcell.parameters():
        parameter.requires_grad_(False)
    for parameter in model.projector.parameters():
        parameter.requires_grad_(True)
    for parameter in model.continuous_head.parameters():
        parameter.requires_grad_(True)
    for parameter in model.level_head.parameters():
        parameter.requires_grad_(False)
    config = PhysicsTrainingConfig(
        stage=2,
        steps=args.steps,
        batch_size=args.batch_size,
        refinement_steps=0,
        learning_rate=args.learning_rate,
        validation_interval=args.validation_interval,
    )
    train_physics_stage(model, tokenizer, config, args.output)


if __name__ == "__main__":
    main()
