from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LanguageDesignDataset(Dataset):
    """JSONL dataset of structured tasks and teacher GradCell latents."""

    def __init__(self, path: str | Path) -> None:
        self.records = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not {"task_text", "preference", "teacher_latent"} <= record.keys():
                    raise ValueError(f"invalid record at line {line_number}")
                if len(record["teacher_latent"]) != 5:
                    raise ValueError(f"teacher_latent must have five values at line {line_number}")
                self.records.append(record)
        if not self.records:
            raise ValueError("language design dataset is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "task_text": record["task_text"],
            "preference": torch.tensor(record["preference"], dtype=torch.float32),
            "teacher_latent": torch.tensor(record["teacher_latent"], dtype=torch.float32),
        }
