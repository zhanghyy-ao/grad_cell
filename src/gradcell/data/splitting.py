from __future__ import annotations

import numpy as np


def group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic 70/15/15 row indices without splitting a group."""
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three groups are required for train/validation/test splits")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_end = max(1, int(0.7 * len(unique)))
    validation_end = min(len(unique) - 1, max(train_end + 1, int(0.85 * len(unique))))
    partitions = (
        unique[:train_end],
        unique[train_end:validation_end],
        unique[validation_end:],
    )
    indices = tuple(np.flatnonzero(np.isin(groups, partition)) for partition in partitions)
    if any(len(index) == 0 for index in indices):
        raise ValueError("Group split produced an empty partition")
    return indices
