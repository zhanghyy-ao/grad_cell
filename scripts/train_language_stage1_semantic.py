from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from gradcell.language import (
    GradCellLanguageCodec,
    LanguageDesignDataset,
    LanguageGradCell,
    MaterialDesignJSONCodec,
    differentiable_design_penalty,
)
from gradcell.language.model import QwenBackbone
from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer


def placeholder_gradcell() -> GradCell:
    return GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=720.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=600.0)),
    )


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: tagged-task to valid design JSON.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--schema-penalty-weight", type=float, default=2.0)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--level-weight", type=float, default=0.25)
    parser.add_argument("--feasibility-penalty-weight", type=float, default=10.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError('Install `pip install -e ".[language-gpu]"`') from exc

    torch.manual_seed(args.seed)
    dataset = LanguageDesignDataset(args.data)
    if any(not record.get("target_json") for record in dataset.records):
        raise ValueError("Stage 1 requires target_json in every JSONL record")
    validation_size = max(1, round(0.1 * len(dataset)))
    train_set, validation_set = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = QwenBackbone(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    gradcell = placeholder_gradcell()
    for parameter in gradcell.parameters():
        parameter.requires_grad_(False)
    codec = GradCellLanguageCodec(bins=args.bins)
    model = LanguageGradCell(backbone, gradcell, codec=codec, freeze_backbone=False)
    device = backbone.model.get_input_embeddings().weight.device
    model.projector.to(device)
    model.continuous_head.to(device)
    model.level_head.to(device)

    def collate(records: list[dict]) -> dict:
        prompts = [record["task_text"] + "\n" for record in records]
        targets = [record["target_json"] + tokenizer.eos_token for record in records]
        combined = [prompt + target for prompt, target in zip(prompts, targets)]
        encoded = tokenizer(
            combined,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = encoded.input_ids.clone()
        schema_labels = torch.full_like(labels, -100)
        for row, (prompt, target) in enumerate(zip(prompts, targets)):
            prompt_length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            labels[row, :prompt_length] = -100
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            for offset, token_id in enumerate(target_ids):
                position = prompt_length + offset
                if position >= labels.shape[1]:
                    break
                piece = tokenizer.decode([token_id])
                if any(character in piece for character in '{}":,') or any(
                    character.isalpha() for character in piece
                ):
                    schema_labels[row, position] = labels[row, position]
        prompt_batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded.input_ids.to(device),
            "attention_mask": encoded.attention_mask.to(device),
            "labels": labels.to(device),
            "schema_labels": schema_labels.to(device),
            "prompt_ids": prompt_batch.input_ids.to(device),
            "prompt_mask": prompt_batch.attention_mask.to(device),
            "teacher_latent": torch.stack([record["teacher_latent"] for record in records]).to(device),
        }

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation_set, args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = []
        for step, batch in enumerate(train_loader, start=1):
            causal = backbone.causal_forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            schema_penalty = shifted_cross_entropy(causal.logits, batch["schema_labels"])
            _, latent, level_logits = model.propose(batch["prompt_ids"], batch["prompt_mask"])
            teacher = batch["teacher_latent"].to(dtype=latent.dtype)
            latent_loss = F.mse_loss(latent, teacher)
            level_loss = F.cross_entropy(
                level_logits.flatten(0, 1), codec.quantize(teacher).flatten()
            )
            decoded = gradcell.design_space(latent.to(next(gradcell.parameters()).device))
            feasibility_penalty = differentiable_design_penalty(
                decoded, gradcell.design_space
            ).mean().to(device)
            loss = (
                causal.loss
                + args.schema_penalty_weight * schema_penalty
                + args.latent_weight * latent_loss
                + args.level_weight * level_loss
                + args.feasibility_penalty_weight * feasibility_penalty
            )
            (loss / args.gradient_accumulation).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals.append(float(loss.detach()))

        model.eval()
        validation_latent = []
        valid_json = 0
        generated_json = 0
        json_codec = MaterialDesignJSONCodec(gradcell.design_space)
        with torch.no_grad():
            for batch in validation_loader:
                _, latent, _ = model.propose(batch["prompt_ids"], batch["prompt_mask"])
                validation_latent.append(
                    float(F.mse_loss(latent, batch["teacher_latent"].to(latent.dtype)))
                )
                generated = backbone.generate(
                    input_ids=batch["prompt_ids"],
                    attention_mask=batch["prompt_mask"],
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                prompt_width = batch["prompt_ids"].shape[1]
                for row in generated[:, prompt_width:]:
                    generated_json += 1
                    try:
                        json_codec.loads(tokenizer.decode(row, skip_special_tokens=True))
                    except (ValueError, json.JSONDecodeError):
                        continue
                    valid_json += 1
        record = {
            "epoch": epoch,
            "train_total_loss": sum(totals) / len(totals),
            "validation_latent_mse": sum(validation_latent) / len(validation_latent),
            "validation_valid_json_rate": valid_json / max(generated_json, 1),
        }
        history.append(record)
        print(json.dumps(record))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_adapter(str(args.output_dir / "qwen_adapter"))
    torch.save(
        {
            "projector": model.projector.state_dict(),
            "continuous_head": model.continuous_head.state_dict(),
            "level_head": model.level_head.state_dict(),
            "model_name": args.model_name,
            "bins": args.bins,
            "history": history,
        },
        args.output_dir / "language_heads.pt",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"stage": 1, "history": history}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
