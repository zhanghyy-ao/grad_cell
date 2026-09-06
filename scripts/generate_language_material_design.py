from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gradcell.language import StructuredPreferenceTask
from gradcell.language.physics_training import (
    build_gradcell,
    load_language_model,
    load_physics_checkpoint,
    tokenize_preferences,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a physics-verified GradCell JSON design.")
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-front", type=Path)
    parser.add_argument("--preference", type=float, required=True)
    parser.add_argument("--target-energy", type=float, default=150.0)
    parser.add_argument("--min-retention-5c", type=float, default=0.50)
    parser.add_argument("--min-retention-6c", type=float, default=0.44)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--backend", choices=("toy", "pybamm"), default="pybamm")
    parser.add_argument("--physics-model", choices=("SPMe", "DFN"), default="SPMe")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    StructuredPreferenceTask(
        args.preference,
        target_energy=args.target_energy,
        min_retention_5c=args.min_retention_5c,
        min_retention_6c=args.min_retention_6c,
    )

    gradcell = build_gradcell(args.backend, args.physics_model, args.reference_front)
    model, tokenizer = load_language_model(
        stage1_dir=args.stage1_dir,
        gradcell=gradcell,
        load_in_4bit=args.load_in_4bit,
    )
    load_physics_checkpoint(model, args.checkpoint)
    model.eval()
    physics_parameter = next(model.gradcell.parameters())
    preference = torch.tensor(
        [args.preference], dtype=physics_parameter.dtype, device=physics_parameter.device
    )
    targets = torch.tensor(
        [[args.target_energy, args.min_retention_5c, args.min_retention_6c]],
        dtype=physics_parameter.dtype,
        device=physics_parameter.device,
    )
    language_device = model.backbone.model.get_input_embeddings().weight.device
    tokens = tokenize_preferences(
        tokenizer, model.codec, preference, language_device, targets=targets
    )
    with torch.enable_grad():
        output = model(
            tokens["input_ids"],
            tokens["attention_mask"],
            preference,
            num_steps=args.refinement_steps,
            targets=targets,
        )
    print(model.render_best_json(output, targets=targets)[0])


if __name__ == "__main__":
    main()
