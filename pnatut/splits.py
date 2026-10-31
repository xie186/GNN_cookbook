"""Split by whole cells so that overlapping patches from one cell never straddle train/val/test."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def split_cells(cell_ids: Sequence[str], fractions=(0.6, 0.2, 0.2), seed: int = 0) -> Dict[str, str]:
    """Seeded assignment of each cell id to 'train' / 'val' / 'test' (grouped split, one group = one cell)."""
    ids = sorted(cell_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_tr = int(round(fractions[0] * len(ids)))
    n_va = int(round(fractions[1] * len(ids)))
    out = {}
    for pos, i in enumerate(perm):
        out[ids[i]] = "train" if pos < n_tr else ("val" if pos < n_tr + n_va else "test")
    return out


def patch_splits(patch_cell_ids: Sequence[str], split_map: Dict[str, str]) -> np.ndarray:
    return np.array([split_map[c] for c in patch_cell_ids])


def assert_no_cell_overlap(patch_cell_ids: Sequence[str], split_arr: np.ndarray) -> pd.DataFrame:
    """Raises if any cell has patches in two splits; returns the per-split cell/patch counts."""
    df = pd.DataFrame({"cell_id": list(patch_cell_ids), "split": split_arr})
    n_splits_per_cell = df.groupby("cell_id")["split"].nunique()
    bad = n_splits_per_cell[n_splits_per_cell > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} cells appear in more than one split, e.g. {bad.index[:3].tolist()}")
    summary = df.groupby("split").agg(n_cells=("cell_id", "nunique"), n_patches=("cell_id", "size"))
    return summary.reindex(["train", "val", "test"])
