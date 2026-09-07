from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import random_split

from gradcell.language import LanguageDesignDataset, MaterialDesignJSONCodec
from gradcell.language.model import QwenBackbone


def error_category(error: Exception) -> str:
    message = str(error)
    if "schema" in message or "fields" in message:
        return "schema_or_fields"
    if "outside" in message or "feasible interval" in message:
        return "physical_bounds"
    if "complete JSON" in message:
        return "incomplete_json"
    return "json_syntax"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Stage-1 LoRA JSON generator.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-invalid", type=int, default=3)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError('Install `pip install -e ".[language-gpu]"`') from exc

    dataset = LanguageDesignDataset(args.data)
    validation_size = max(1, round(0.1 * len(dataset)))
    _, validation_set = random_split(
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
        adapter_path=str(args.stage1_dir / "qwen_adapter"),
    )
    backbone.eval()
    device = backbone.model.get_input_embeddings().weight.device
    codec = MaterialDesignJSONCodec()
    counts: Counter[str] = Counter()
    examples = 0
    limit = min(args.max_samples, len(validation_set))

    for index in range(limit):
        prompt = validation_set[index]["task_text"] + "\n"
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = backbone.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        if "</DESIGN>" in text:
            counts["stopped"] += 1
            text = text.split("</DESIGN>", 1)[0]
        try:
            codec.loads(text)
            counts["valid"] += 1
        except (ValueError, json.JSONDecodeError) as exc:
            category = error_category(exc)
            counts[category] += 1
            if examples < args.show_invalid:
                print(json.dumps({"sample": index, "error": str(exc), "text": text}))
                examples += 1

    summary = {
        "samples": limit,
        "valid_json_rate": counts["valid"] / max(limit, 1),
        "design_stop_rate": counts["stopped"] / max(limit, 1),
        "failure_counts": {
            key: value for key, value in counts.items() if key not in {"valid", "stopped"}
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
