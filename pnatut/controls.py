"""Controls that tell composition apart from topology, plus invariance checks.

* ``shuffle_labels_within_cell``: permute marker labels among the molecules of a cell.
  Keeps every marker count and the whole graph; destroys any relation between a
  molecule's label and where it sits.
* ``rewire_preserving_degrees``: re-pair the UMI1 and UMI2 endpoints at random
  (bipartite configuration model). Keeps node count, labels, bipartiteness and each
  molecule's degree (up to merged duplicates); destroys *which* molecules are adjacent.
* ``permutation_invariance``: a graph encoder must give the same patch embedding when
  the nodes are listed in a different order.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from pnatut.graphs import CellGraph
from pnatut.patches import Patch, PatchSet, extract_patch
from pnatut.pyg_compat import Data, coalesce


def shuffle_labels_within_cell(cell: CellGraph, seed: int) -> CellGraph:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cell.num_nodes)
    new = copy.copy(cell)
    new.nodes = cell.nodes.copy()
    new.nodes["marker"] = cell.nodes["marker"].values[perm]
    new.nodes["marker_id"] = cell.nodes["marker_id"].values[perm]
    new.data = cell.data.clone()
    new.data.marker_id = cell.data.marker_id[perm].clone()
    new.qc = dict(cell.qc, control="within_cell_label_shuffle")
    return new


def rewire_preserving_degrees(cell: CellGraph, seed: int) -> CellGraph:
    """Random re-pairing of UMI1->UMI2 edge endpoints; duplicates created by chance are merged."""
    rng = np.random.default_rng(seed)
    ei = cell.data.edge_index
    pool = cell.data.pool
    fwd = ei[:, pool[ei[0]] == 0]                     # one copy of each undirected edge: UMI1 -> UMI2
    src, dst = fwd[0].numpy(), fwd[1].numpy()
    dst_perm = dst[rng.permutation(len(dst))]          # every UMI2 keeps its degree, every UMI1 keeps its degree
    new_fwd = torch.stack([torch.as_tensor(src), torch.as_tensor(dst_perm)])
    both = torch.cat([new_fwd, new_fwd.flip(0)], dim=1)
    ei_c = coalesce(both, num_nodes=cell.num_nodes)
    new = copy.copy(cell)
    new.data = cell.data.clone()
    new.data.edge_index = ei_c
    new.data.edge_weight = torch.ones(ei_c.size(1))
    new.qc = dict(cell.qc, control="degree_preserving_rewire",
                  n_edges_after_rewire=int(ei_c.size(1) // 2),
                  frac_edges_lost_to_merged_duplicates=float(1 - (ei_c.size(1) // 2) / len(src)))
    return new


def re_extract(ps: PatchSet, cells_perturbed: Sequence[CellGraph]) -> PatchSet:
    """Same anchors and K on perturbed cells (label shuffle keeps the node sets; rewiring changes them)."""
    out = []
    for p in ps.patches:
        q = extract_patch(cells_perturbed[p.cell_index], p.anchor_cell_local, p.k)
        q.cell_index = p.cell_index
        out.append(q)
    return PatchSet(ps.k, out, ps.rejections)


@torch.no_grad()
def permutation_invariance(model, data: Data, seed: int = 0) -> Dict[str, float]:
    """Permute node order, remap edges, compare node (un-permuted) and pooled embeddings."""
    model.eval()
    n = data.num_nodes
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    x2 = data.x[perm]
    ei2 = inv[data.edge_index]
    out1 = model(data.x, data.edge_index, torch.zeros(n, dtype=torch.long))
    out2 = model(x2, ei2, torch.zeros(n, dtype=torch.long))
    return {"max_abs_diff_node_embeddings": float((out1["z"] - out2["z"][inv]).abs().max()),
            "max_abs_diff_patch_embedding": float((out1["graph"] - out2["graph"]).abs().max())}


def seed_reproducibility(make_and_train, seed: int) -> Dict[str, float]:
    """Train twice with the same seed and report the largest weight difference (should be 0)."""
    m1 = make_and_train(seed)
    m2 = make_and_train(seed)
    diffs = [float((p1 - p2).abs().max()) for p1, p2 in zip(m1.parameters(), m2.parameters())]
    return {"max_abs_weight_difference": max(diffs)}
