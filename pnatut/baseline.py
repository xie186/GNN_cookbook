"""The transparent baseline: every patch (and every cell) as a vector of protein proportions.

The vector lives in the same global marker vocabulary for every patch, so patches with
different numbers of molecules are directly comparable. Grouping patches by composition
(k-means) is a *descriptive* baseline, not a biological validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def proportions(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).sum())


def bray_curtis(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).sum() / (a + b).sum())


def compare_two(comp_a: np.ndarray, comp_b: np.ndarray, vocab: Sequence[str], top: int = 10) -> pd.DataFrame:
    """Side-by-side proportions of the markers that differ most, plus the three distances as attrs."""
    df = pd.DataFrame({"patch_A": comp_a, "patch_B": comp_b}, index=list(vocab))
    df["abs_difference"] = (df["patch_A"] - df["patch_B"]).abs()
    df = df.sort_values("abs_difference", ascending=False).head(top)
    df.attrs["l1_distance"] = l1_distance(comp_a, comp_b)
    df.attrs["cosine_similarity"] = cosine_similarity(comp_a, comp_b)
    df.attrs["bray_curtis"] = bray_curtis(comp_a, comp_b)
    return df


@dataclass
class CompositionGroups:
    """k-means on patch proportion vectors, fitted on training patches only."""

    n_groups: int
    seed: int = 0
    model: Optional[KMeans] = None

    def fit(self, train_comp: np.ndarray) -> "CompositionGroups":
        self.model = KMeans(n_clusters=self.n_groups, n_init=10, random_state=self.seed).fit(train_comp)
        return self

    def predict(self, comp: np.ndarray) -> np.ndarray:
        return self.model.predict(comp)

    def profiles(self, vocab: Sequence[str], top: int = 5) -> pd.DataFrame:
        """Mean composition of each group and its top markers."""
        c = self.model.cluster_centers_
        rows = []
        for g in range(self.n_groups):
            order = np.argsort(-c[g])[:top]
            rows.append({"group": g, "top_markers": ", ".join(f"{vocab[j]} ({c[g, j]:.2f})" for j in order)})
        return pd.DataFrame(rows).set_index("group")


def per_cell_fractions(labels: np.ndarray, patch_cell_ids: Sequence[str], n_groups: int,
                       prefix: str = "group") -> pd.DataFrame:
    """Fraction of each cell's *sampled patches* that fall in each group.

    The denominator is the number of patches sampled for that cell (column ``n_patches``),
    so these are frequencies under the anchor-sampling scheme, not surface areas.
    """
    df = pd.DataFrame({"cell_id": list(patch_cell_ids), "label": labels})
    tab = pd.crosstab(df["cell_id"], df["label"]).reindex(columns=range(n_groups), fill_value=0)
    n = tab.sum(axis=1)
    frac = tab.div(n, axis=0)
    frac.columns = [f"{prefix}_{g}" for g in frac.columns]
    frac.insert(0, "n_patches", n)
    return frac
