"""Reproducible per-cell anchor sampling and exact K-hop patches with ``k_hop_subgraph``.

A *patch* is the induced subgraph of all molecules within K graph steps of an *anchor*
molecule. K is a graph distance: K=2 means "neighbours of neighbours", and can contain
dozens of molecules. Patches whose node count falls outside ``[min_nodes, max_nodes]`` are
rejected and counted; nothing is silently truncated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from pnatut.graphs import CellGraph, one_hot
from pnatut.pyg_compat import Data, k_hop_subgraph


@dataclass
class Patch:
    cell_id: str
    sample_id: str
    cell_index: int
    k: int
    anchor_cell_local: int
    anchor_original_umi: int
    subset: torch.Tensor        # sorted cell-local node ids in the patch
    mapping: int                # position of the anchor inside ``subset``
    edge_index: torch.Tensor    # patch-local, both directions
    marker_id: torch.Tensor     # [n_nodes]
    hop: torch.Tensor           # [n_nodes] graph distance from the anchor
    pool: torch.Tensor          # [n_nodes] 0 = UMI1, 1 = UMI2

    @property
    def n_nodes(self) -> int:
        return int(self.subset.numel())

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.size(1) // 2)

    def to_data(self, n_markers: int) -> Data:
        """PyG ``Data`` with one-hot ``x`` (plus a spare, all-zero mask column) and labels ``y``."""
        x = torch.zeros(self.n_nodes, n_markers + 1)
        x[:, :n_markers] = one_hot(self.marker_id, n_markers)
        return Data(x=x, edge_index=self.edge_index, y=self.marker_id.clone(), hop=self.hop, pool=self.pool,
                    anchor_local=torch.tensor([self.mapping]), num_nodes=self.n_nodes)


def extract_patch(cell: CellGraph, anchor: int, k: int) -> Patch:
    """Exact K-hop induced subgraph around ``anchor`` using PyG's ``k_hop_subgraph``."""
    ei_full, n = cell.data.edge_index, cell.num_nodes
    subset, ei, mapping, _ = k_hop_subgraph(int(anchor), k, ei_full, relabel_nodes=True, num_nodes=n)
    hop = torch.full((subset.numel(),), k, dtype=torch.long)
    pos = {int(v): i for i, v in enumerate(subset.tolist())}
    for h in range(k - 1, -1, -1):  # nested smaller balls overwrite with their (smaller) distance
        s_h, _, _, _ = k_hop_subgraph(int(anchor), h, ei_full, num_nodes=n)
        for v in s_h.tolist():
            hop[pos[v]] = h
    return Patch(cell.cell_id, cell.sample_id, -1, k, int(anchor), int(cell.nodes["original_umi"].iloc[int(anchor)]),
                 subset, int(mapping[0]), ei, cell.data.marker_id[subset].clone(), hop, cell.data.pool[subset].clone())


@dataclass
class PatchSet:
    k: int
    patches: List[Patch]
    rejections: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __len__(self) -> int:
        return len(self.patches)

    def table(self, vocab: Optional[Sequence[str]] = None) -> pd.DataFrame:
        rows = []
        for i, p in enumerate(self.patches):
            r = {"patch": i, "cell_id": p.cell_id, "sample_id": p.sample_id, "cell_index": p.cell_index, "k": p.k,
                 "anchor_cell_local": p.anchor_cell_local, "anchor_umi": p.anchor_original_umi,
                 "anchor_marker_id": int(p.marker_id[p.mapping]), "n_nodes": p.n_nodes, "n_edges": p.n_edges}
            for h in range(p.k + 1):
                r[f"n_hop{h}"] = int((p.hop == h).sum())
            if vocab is not None:
                r["anchor_marker"] = vocab[r["anchor_marker_id"]]
            rows.append(r)
        return pd.DataFrame(rows).set_index("patch")

    def composition(self, n_markers: int) -> np.ndarray:
        """Patch-by-marker proportions [n_patches, n_markers], rows sum to 1."""
        out = np.zeros((len(self.patches), n_markers), dtype=np.float32)
        for i, p in enumerate(self.patches):
            cnt = np.bincount(p.marker_id.numpy(), minlength=n_markers)
            out[i] = cnt / cnt.sum()
        return out

    def counts(self, n_markers: int) -> np.ndarray:
        out = np.zeros((len(self.patches), n_markers), dtype=np.int64)
        for i, p in enumerate(self.patches):
            out[i] = np.bincount(p.marker_id.numpy(), minlength=n_markers)
        return out


def anchor_candidates(cell: CellGraph, cell_seed: int, n_candidates: int) -> np.ndarray:
    """A fixed, seeded permutation prefix of the cell's node ids (same candidates for every K)."""
    rng = np.random.default_rng(cell_seed)
    return rng.permutation(cell.num_nodes)[:n_candidates]


def sample_patches(cells: List[CellGraph], k: int, n_anchors_per_cell: int = 40, seed: int = 0,
                   min_nodes: int = 5, max_nodes: int = 150, oversample: int = 3, show_progress: bool = False) -> PatchSet:
    """For each cell draw candidate anchors reproducibly and keep the first ``n_anchors_per_cell``
    whose exact K-hop patch has ``min_nodes <= n_nodes <= max_nodes``.

    Per-cell seeds come from ``numpy.random.SeedSequence(seed).spawn(len(cells))`` so the
    candidate list of a cell does not depend on which other cells are present. Rejections
    are recorded per cell with the reason (too small / too large) and whether the candidate
    pool was exhausted before the quota was reached.
    """
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(len(cells))]
    patches, rej = [], []
    it = enumerate(cells)
    if show_progress:
        from tqdm.auto import tqdm
        it = tqdm(list(it), desc=f"k={k}")
    for ci, cell in it:
        cands = anchor_candidates(cell, seeds[ci], n_anchors_per_cell * oversample)
        acc = small = large = tried = 0
        for a in cands:
            tried += 1
            p = extract_patch(cell, int(a), k)
            if p.n_nodes < min_nodes:
                small += 1
                continue
            if p.n_nodes > max_nodes:
                large += 1
                continue
            p.cell_index = ci
            patches.append(p)
            acc += 1
            if acc == n_anchors_per_cell:
                break
        rej.append({"cell_id": cell.cell_id, "k": k, "n_candidates_tried": tried, "n_accepted": acc,
                    "n_rejected_too_small": small, "n_rejected_too_large": large,
                    "quota_reached": acc == n_anchors_per_cell, "cell_n_nodes": cell.num_nodes})
    return PatchSet(k, patches, pd.DataFrame(rej).set_index("cell_id"))


def coverage_overlap(ps: PatchSet, cells: List[CellGraph]) -> pd.DataFrame:
    """Per cell: fraction of molecules covered by >=1 patch, by >=2 patches, and mean pairwise Jaccard."""
    by_cell: Dict[int, List[Patch]] = {}
    for p in ps.patches:
        by_cell.setdefault(p.cell_index, []).append(p)
    rows = []
    for ci, plist in by_cell.items():
        n = cells[ci].num_nodes
        hits = np.zeros(n, dtype=np.int64)
        sets = []
        for p in plist:
            idx = p.subset.numpy()
            hits[idx] += 1
            sets.append(set(idx.tolist()))
        jac = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j])
                jac.append(len(sets[i] & sets[j]) / u if u else 0.0)
        rows.append({"cell_id": cells[ci].cell_id, "k": ps.k, "n_patches": len(plist), "cell_n_nodes": n,
                     "nodes_in_any_patch": int((hits > 0).sum()), "coverage": float((hits > 0).mean()),
                     "frac_nodes_in_2plus_patches": float((hits > 1).mean()),
                     "mean_pairwise_jaccard": float(np.mean(jac)) if jac else np.nan,
                     "max_pairwise_jaccard": float(np.max(jac)) if jac else np.nan})
    return pd.DataFrame(rows).set_index("cell_id")


def patch_report(ps: PatchSet, cells: List[CellGraph], vocab: Sequence[str]) -> pd.DataFrame:
    """One-row summary used to compare K values side by side."""
    t = ps.table(vocab)
    co = coverage_overlap(ps, cells)
    r = ps.rejections
    return pd.DataFrame({
        "n_patches": [len(ps)],
        "n_nodes_median": [t["n_nodes"].median()], "n_nodes_q90": [t["n_nodes"].quantile(0.9)], "n_nodes_max": [t["n_nodes"].max()],
        "n_edges_median": [t["n_edges"].median()],
        "frac_cells_quota_reached": [r["quota_reached"].mean()],
        "frac_candidates_rejected_small": [r["n_rejected_too_small"].sum() / r["n_candidates_tried"].sum()],
        "frac_candidates_rejected_large": [r["n_rejected_too_large"].sum() / r["n_candidates_tried"].sum()],
        "coverage_median": [co["coverage"].median()], "frac_nodes_in_2plus_patches_median": [co["frac_nodes_in_2plus_patches"].median()],
        "mean_pairwise_jaccard_median": [co["mean_pairwise_jaccard"].median()],
        "distinct_markers_per_patch_median": [np.median([(np.bincount(p.marker_id.numpy()) > 0).sum() for p in ps.patches])],
    }, index=[f"K={ps.k}"])
