"""A tiny molecular graph you can check by hand (section 3 of the tutorial).

Eight "molecules" carry four distinct protein markers, several of them repeated, and are
joined by eight proximity edges. The functions here build exactly the tables the notebook
shows: molecule table, edge table, shared vocabulary, one-hot features, ``edge_index`` and
the PyG ``Data`` object, plus a k-hop neighbourhood report.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from pnatut.pyg_compat import Data, k_hop_subgraph

# original molecule id -> marker label (repeated labels on purpose)
TOY_MOLECULES: List[Tuple[str, str]] = [
    ("mol_101", "CD3"), ("mol_102", "CD3"), ("mol_103", "CD4"), ("mol_104", "CD45"),
    ("mol_105", "HLA-ABC"), ("mol_106", "CD45"), ("mol_107", "CD3"), ("mol_108", "HLA-ABC"),
]
# undirected proximity edges written once, by original ids
TOY_EDGES: List[Tuple[str, str]] = [
    ("mol_101", "mol_103"), ("mol_101", "mol_104"), ("mol_102", "mol_104"), ("mol_103", "mol_105"),
    ("mol_104", "mol_106"), ("mol_106", "mol_107"), ("mol_107", "mol_108"), ("mol_105", "mol_108"),
]


def make_vocabulary(markers) -> List[str]:
    """One shared protein vocabulary: the sorted set of distinct marker labels (column order)."""
    return sorted(set(markers))


def molecule_table(molecules=TOY_MOLECULES) -> pd.DataFrame:
    """original_id, local_index (0..n-1, the row position) and marker label."""
    df = pd.DataFrame(molecules, columns=["original_id", "marker"])
    df.insert(1, "local_index", np.arange(len(df)))
    return df


def edge_table(edges=TOY_EDGES) -> pd.DataFrame:
    return pd.DataFrame(edges, columns=["source_id", "target_id"])


def one_hot_features(mol_df: pd.DataFrame, vocab: List[str]) -> torch.Tensor:
    """x[i, j] = 1 if molecule i carries vocabulary entry j else 0 -> shape [n_molecules, n_vocab]."""
    col = {m: j for j, m in enumerate(vocab)}
    x = torch.zeros(len(mol_df), len(vocab))
    for i, m in enumerate(mol_df["marker"]):
        x[i, col[m]] = 1.0
    return x


def edge_index_from_table(edge_df: pd.DataFrame, id_to_local: Dict[str, int], undirected: bool = True) -> torch.Tensor:
    """Translate original ids to local indices; for an undirected graph store both directions."""
    src = torch.tensor([id_to_local[s] for s in edge_df["source_id"]], dtype=torch.long)
    dst = torch.tensor([id_to_local[t] for t in edge_df["target_id"]], dtype=torch.long)
    ei = torch.stack([src, dst])
    if undirected:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    return ei


def toy_graph():
    """Return (mol_df, edge_df, vocab, x, edge_index, data) for the tiny example."""
    mol_df = molecule_table()
    edge_df = edge_table()
    vocab = make_vocabulary(mol_df["marker"])
    x = one_hot_features(mol_df, vocab)
    id_to_local = dict(zip(mol_df["original_id"], mol_df["local_index"]))
    edge_index = edge_index_from_table(edge_df, id_to_local)
    y = torch.tensor([vocab.index(m) for m in mol_df["marker"]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return mol_df, edge_df, vocab, x, edge_index, data


def neighborhood_report(data, mol_df: pd.DataFrame, anchor_local: int, k: int) -> Dict[str, object]:
    """Everything ``k_hop_subgraph`` returns, translated back to original ids.

    Returns a dict with the sorted ``subset`` (cell-local indices), the local renumbering
    table (original id, old index, new index, hop distance), the anchor's new index
    (``mapping``), the induced edges in both numberings, and the feature rows ``x[subset]``.
    """
    subset, ei_local, mapping, edge_mask = k_hop_subgraph(anchor_local, k, data.edge_index,
                                                          relabel_nodes=True, num_nodes=data.num_nodes)
    # hop distance = smallest h whose h-hop subset contains the node
    hop = {int(anchor_local): 0}
    for h in range(1, k + 1):
        s_h, _, _, _ = k_hop_subgraph(anchor_local, h, data.edge_index, num_nodes=data.num_nodes)
        for n in s_h.tolist():
            hop.setdefault(n, h)
    renum = pd.DataFrame({
        "original_id": mol_df["original_id"].values[subset.numpy()],
        "marker": mol_df["marker"].values[subset.numpy()],
        "cell_local_index": subset.numpy(),
        "patch_local_index": np.arange(len(subset)),
        "hop": [hop[int(n)] for n in subset],
    })
    ei_orig = data.edge_index[:, edge_mask]
    edges = pd.DataFrame({
        "src_cell_local": ei_orig[0].numpy(), "dst_cell_local": ei_orig[1].numpy(),
        "src_patch_local": ei_local[0].numpy(), "dst_patch_local": ei_local[1].numpy(),
    })
    edges["src_id"] = mol_df["original_id"].values[edges["src_cell_local"]]
    edges["dst_id"] = mol_df["original_id"].values[edges["dst_cell_local"]]
    return {"subset": subset, "renumbering": renum, "anchor_new_index": int(mapping[0]),
            "edge_index_local": ei_local, "edges": edges, "x_rows": data.x[subset], "edge_mask": edge_mask}
