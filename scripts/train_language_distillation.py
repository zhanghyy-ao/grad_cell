from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from gradcell.language import GradCellLanguageCodec, LanguageDesignDataset, LanguageGradCell
from gradcell.language.model import QwenBackbone
from gradcell.models import GradCell
from gradcell.physics import AnalyticToyBackend, DifferentiablePhysicsLayer


def placeholder_gradcell() -> GradCell:
    return GradCell(
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=3600.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=720.0)),
        DifferentiablePhysicsLayer(AnalyticToyBackend(horizon_s=600.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 Qwen GradCell latent distillation.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError('Install `pip install -e ".[language]"` before training') from exc

    torch.manual_seed(args.seed)
    dataset = LanguageDesignDataset(args.data)
    if len(dataset) < 2:
        parser.error("the dataset must contain at least two records")
    validation_size = max(1, round(len(dataset) * args.validation_fraction))
    training_size = len(dataset) - validation_size
    train_set, validation_set = random_split(
        dataset, [training_size, validation_size], generator=torch.Generator().manual_seed(args.seed)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = QwenBackbone(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    codec = GradCellLanguageCodec(bins=args.bins)
    gradcell = placeholder_gradcell()
    for parameter in gradcell.parameters():
        parameter.requires_grad_(False)
    model = LanguageGradCell(
        backbone,
        gradcell,
        codec=codec,
        freeze_backbone=not args.use_lora,
    )
    head_device = backbone.model.get_input_embeddings().weight.device
    model.projector.to(head_device)
    model.continuous_head.to(head_device)
    model.level_head.to(head_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    def collate(records: list[dict]) -> dict:
        encoded = tokenizer(
            [record["task_text"] for record in records],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded.input_ids.to(head_device),
            "attention_mask": encoded.attention_mask.to(head_device),
            "latent": torch.stack([record["teacher_latent"] for record in records]).to(head_device),
        }

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, collate_fn=collate)
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            _, continuous, logits = model.propose(batch["input_ids"], batch["attention_mask"])
            levels = codec.quantize(batch["latent"])
            token_loss = F.cross_entropy(logits.flatten(0, 1), levels.flatten())
            latent_loss = F.mse_loss(continuous, batch["latent"])
            expected = codec.dequantize(
                (torch.softmax(logits, -1) * torch.arange(args.bins, device=head_device)).sum(-1),
                dtype=continuous.dtype,
            )
            alignment = F.mse_loss(expected, continuous)
            loss = token_loss + latent_loss + args.alignment_weight * alignment
            (loss / args.gradient_accumulation).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())

        model.eval()
        validation = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                _, continuous, _ = model.propose(batch["input_ids"], batch["attention_mask"])
                validation += float(F.mse_loss(continuous, batch["latent"]))
        record = {
            "epoch": epoch + 1,
            "train_loss": running / len(train_loader),
            "validation_latent_mse": validation / len(validation_loader),
        }
        history.append(record)
        print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": model.projector.state_dict(),
            "continuous_head": model.continuous_head.state_dict(),
            "level_head": model.level_head.state_dict(),
            "model_name": args.model_name,
            "load_in_4bit": args.load_in_4bit,
            "use_lora": args.use_lora,
            "lora_rank": args.lora_rank if args.use_lora else None,
            "bins": args.bins,
            "history": history,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
