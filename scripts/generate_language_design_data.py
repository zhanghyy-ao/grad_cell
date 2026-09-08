from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gradcell.language import (
    GradCellLanguageCodec,
    MaterialDesignJSONCodec,
    StructuredPreferenceTask,
)
from gradcell.design import DesignSpace
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
    parser.add_argument(
        "--reference-front",
        type=Path,
        help="When provided, build diverse feasible goal tasks and select their front oracle.",
    )
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
    json_design_space = teacher.design_space
    task_targets = torch.tensor([150.0, 0.50, 0.44], dtype=torch.float64).expand(
        args.samples, -1
    ).clone()
    if args.reference_front is None:
        with torch.no_grad():
            latents = teacher.initializer(teacher.task_encoder(preferences))
    else:
        with np.load(args.reference_front, allow_pickle=False) as arrays:
            front_latent = arrays["latent"].copy()
            energy = arrays["energy_wh_kg"].copy()
            retention_5c = arrays["energy_retention_5c"].copy()
            retention_6c = arrays["energy_retention_6c"].copy()
            front_metadata = json.loads(str(arrays["metadata"]))
        json_design_space = DesignSpace(
            capacity_formula=front_metadata.get("capacity_formula", "chen2020_scaled"),
            capacity_multiplier=float(front_metadata.get("capacity_multiplier", 1.0)),
        )
        rng = np.random.default_rng(args.seed)
        selected = []
        generated_targets = []
        for preference in preferences.numpy():
            anchor = int(rng.integers(0, len(front_latent)))
            targets = np.asarray(
                [
                    energy[anchor] - rng.uniform(0.0, 2.0),
                    retention_5c[anchor] - rng.uniform(0.005, 0.025),
                    retention_6c[anchor] - rng.uniform(0.005, 0.025),
                ]
            )
            violation = (
                np.maximum(targets[0] - energy, 0.0) / 160.0
                + 5.0 * np.maximum(targets[1] - retention_5c, 0.0)
                + 5.0 * np.maximum(targets[2] - retention_6c, 0.0)
            )
            margin = np.minimum(retention_5c - targets[1], retention_6c - targets[2])
            improvement = -(preference * energy / 160.0 + (1.0 - preference) * margin)
            selected.append(int(np.argmin(violation + 0.05 * improvement)))
            generated_targets.append(targets)
        latents = torch.from_numpy(front_latent[np.asarray(selected)]).double()
        task_targets = torch.from_numpy(np.asarray(generated_targets)).double()
    json_codec = MaterialDesignJSONCodec(json_design_space)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (preference, latent, targets) in enumerate(
            zip(preferences, latents, task_targets)
        ):
            task = StructuredPreferenceTask(
                float(preference),
                target_energy=float(targets[0]),
                min_retention_5c=float(targets[1]),
                min_retention_6c=float(targets[2]),
            )
            record = {
                "id": f"preference_{index:06d}",
                "preference": task.preference,
                "targets": targets.tolist(),
                "task_text": codec.serialize_task(task),
                "teacher_latent": latent.tolist(),
                "teacher_design_text": codec.serialize_design(latent),
                "target_json": json_codec.dumps_latent(latent.unsqueeze(0)),
                "teacher_levels": codec.quantize(latent).tolist(),
                "source_checkpoint": str(args.checkpoint),
            }
            serialized = json.dumps(record, ensure_ascii=False)
            forbidden_terms = ("np_ratio", "physics_loss")
            present = [term for term in forbidden_terms if term in serialized]
            if present:
                raise RuntimeError(
                    f"training record {index} contains forbidden internal fields: {present}"
                )
            handle.write(serialized + "\n")
    print(json.dumps({"output": str(args.output), "samples": args.samples, "bins": args.bins}))


if __name__ == "__main__":
    main()
