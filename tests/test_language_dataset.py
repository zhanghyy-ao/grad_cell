import json

import pytest

from gradcell.language import LanguageDesignDataset


def test_language_design_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_text": "<TASK></TASK><DESIGN>",
                "preference": 0.5,
                "teacher_latent": [0, 0, 0, 0, 0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = LanguageDesignDataset(path)
    assert len(dataset) == 1
    assert dataset[0]["teacher_latent"].shape == (5,)


def test_language_design_dataset_rejects_bad_latent(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"task_text": "x", "preference": 0.5, "teacher_latent": [0]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="five values"):
        LanguageDesignDataset(path)
