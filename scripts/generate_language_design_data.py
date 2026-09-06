from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gradcell.language import GradCellLanguageCodec, StructuredPreferenceTask
from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer


def build_teacher() -> GradCell:
    return GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=720.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=600.0)),
    ).double()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a GradCell K=0 checkpoint to JSONL.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bins", type=int, default=256)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    teacher = build_teacher()
    teacher_state = teacher.state_dict()
    transferred = {
        name: value
        for name, value in state.items()
        if name.startswith(("task_encoder.", "initializer.")) and name in teacher_state
    }
    if not transferred:
        raise RuntimeError("checkpoint contains no compatible K=0 task encoder/initializer tensors")
    teacher_state.update(transferred)
    teacher.load_state_dict(teacher_state)
    teacher.eval()

    generator = torch.Generator().manual_seed(args.seed)
    preferences = torch.rand(args.samples, generator=generator, dtype=torch.float64)
    # Always retain exact endpoints for evaluation and representation checks.
    preferences[0], preferences[1] = 0.0, 1.0
    codec = GradCellLanguageCodec(bins=args.bins)
    with torch.no_grad():
        latents = teacher.initializer(teacher.task_encoder(preferences))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (preference, latent) in enumerate(zip(preferences, latents)):
            task = StructuredPreferenceTask(float(preference))
            record = {
                "id": f"preference_{index:06d}",
                "preference": task.preference,
                "task_text": codec.serialize_task(task),
                "teacher_latent": latent.tolist(),
                "teacher_design_text": codec.serialize_design(latent),
                "teacher_levels": codec.quantize(latent).tolist(),
                "source_checkpoint": str(args.checkpoint),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "samples": args.samples, "bins": args.bins}))


if __name__ == "__main__":
    main()
