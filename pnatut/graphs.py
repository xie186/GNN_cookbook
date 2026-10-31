"""Turn one cell's edge rows into a PyG graph, with the sanity checks a careful analyst runs first.

A PNA edge joins a UMI1 molecule (``umi1``, ``marker_1``) to a UMI2 molecule (``umi2``,
``marker_2``). The two UMI pools are different oligo sets, so the measured graph is
bipartite. Nodes are *molecules*; the node feature is the protein label of that molecule.
We keep the node identity as (pool, umi) so an accidental id collision between the pools
could never merge two molecules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from pnatut.pyg_compat import Data, coalesce, contains_self_loops, is_undirected


@dataclass
class CellGraph:
    cell_id: str
    sample_id: str
    nodes: pd.DataFrame   # original_umi, pool, marker, marker_id, local_index
    data: Data            # marker_id [N], pool [N], edge_index [2, 2E] (both directions), edge_weight [2E]
    qc: Dict[str, object]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_undirected_edges(self) -> int:
        return int(self.data.edge_index.size(1) // 2)


def graph_qc(cell_edges: pd.DataFrame) -> Dict[str, object]:
    """Checks on the raw edge rows of one cell, before any simplification."""
    e = cell_edges
    out: Dict[str, object] = {"n_edge_rows": int(len(e))}
    out["n_missing_marker"] = int(e["marker_1"].isna().sum() + e["marker_2"].isna().sum())
    out["n_invalid_endpoint"] = int(e["umi1"].isna().sum() + e["umi2"].isna().sum())
    out["n_duplicate_pairs"] = int(e.duplicated(subset=["umi1", "umi2"]).sum())
    out["n_self_loops_same_id"] = int((e["umi1"] == e["umi2"]).sum())  # would only matter if pools shared ids
    u1, u2 = set(e["umi1"].tolist()), set(e["umi2"].tolist())
    out["n_umi1_nodes"], out["n_umi2_nodes"] = len(u1), len(u2)
    out["n_id_collisions_between_pools"] = len(u1 & u2)
    m1 = e.groupby("umi1")["marker_1"].nunique()
    m2 = e.groupby("umi2")["marker_2"].nunique()
    out["n_umis_with_multiple_markers"] = int((m1 > 1).sum() + (m2 > 1).sum())
    out["read_count_min"], out["read_count_max"] = int(e["read_count"].min()), int(e["read_count"].max())
    return out


def build_cell_graph(cell_edges: pd.DataFrame, vocab: List[str], cell_id: str, sample_id: str) -> CellGraph:
    """Nodes = distinct (pool, umi); edges = rows, stored in both directions; features = marker id.

    Simplification made explicit: ``read_count``/``uei_count`` are kept as ``edge_weight`` for
    inspection but the tutorial's message passing treats the graph as a *simple, unweighted,
    undirected* graph. Duplicate (umi1, umi2) rows, if any, would be merged by ``coalesce``.
    """
    qc = graph_qc(cell_edges)
    marker_index = {m: i for i, m in enumerate(vocab)}
    u1 = np.sort(cell_edges["umi1"].unique())
    u2 = np.sort(cell_edges["umi2"].unique())
    n1 = len(u1)
    nodes = pd.DataFrame({
        "original_umi": np.concatenate([u1, u2]),
        "pool": ["UMI1"] * n1 + ["UMI2"] * len(u2),
    })
    nodes["local_index"] = np.arange(len(nodes))
    # marker of each molecule: taken from the edge rows (each UMI carries exactly one marker; qc checks this)
    m_u1 = cell_edges.drop_duplicates("umi1").set_index("umi1")["marker_1"].reindex(u1).values
    m_u2 = cell_edges.drop_duplicates("umi2").set_index("umi2")["marker_2"].reindex(u2).values
    nodes["marker"] = np.concatenate([m_u1, m_u2])
    nodes["marker_id"] = nodes["marker"].map(marker_index).astype(int)
    pos1 = pd.Series(np.arange(n1), index=u1)
    pos2 = pd.Series(np.arange(n1, n1 + len(u2)), index=u2)
    src = torch.as_tensor(pos1.reindex(cell_edges["umi1"].values).values, dtype=torch.long)
    dst = torch.as_tensor(pos2.reindex(cell_edges["umi2"].values).values, dtype=torch.long)
    w = torch.as_tensor(cell_edges["read_count"].values.astype(np.float32))
    ei = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    ew = torch.cat([w, w])
    ei_c, ew_c = coalesce(ei, ew, num_nodes=len(nodes))
    qc["n_edges_after_coalesce_undirected"] = int(ei_c.size(1) // 2)
    # connectivity
    A = sp.coo_matrix((np.ones(ei_c.size(1)), (ei_c[0].numpy(), ei_c[1].numpy())), shape=(len(nodes),) * 2)
    n_comp, labels = sp.csgraph.connected_components(A.tocsr(), directed=False)
    qc["n_connected_components"] = int(n_comp)
    qc["largest_component_fraction"] = float(np.bincount(labels).max() / len(nodes))
    deg = np.bincount(ei_c[1].numpy(), minlength=len(nodes))
    qc["n_isolated_nodes"] = int((deg == 0).sum())
    qc["is_undirected"] = bool(is_undirected(ei_c, num_nodes=len(nodes)))
    qc["has_self_loops"] = bool(contains_self_loops(ei_c))
    qc["mean_degree"] = float(deg.mean())
    data = Data(marker_id=torch.as_tensor(nodes["marker_id"].values, dtype=torch.long),
                pool=torch.as_tensor((nodes["pool"] == "UMI2").values.astype(np.int64)),
                edge_index=ei_c, edge_weight=ew_c, num_nodes=len(nodes))
    return CellGraph(cell_id, sample_id, nodes, data, qc)


def one_hot(marker_id: torch.Tensor, n_markers: int) -> torch.Tensor:
    """Float one-hot matrix [N, n_markers] (the ``x`` of the tiny example, built on demand for real patches)."""
    return torch.nn.functional.one_hot(marker_id, num_classes=n_markers).float()


def build_all(subset, show_progress: bool = False) -> List[CellGraph]:
    """Build a :class:`CellGraph` for every cell of a :class:`pnatut.pxl_subset.PNASubset`."""
    groups = dict(tuple(subset.edges.groupby("component", sort=False)))
    out = []
    it = subset.cell_ids
    if show_progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="cells")
    for cid in it:
        out.append(build_cell_graph(groups[cid], subset.vocab, cid, str(subset.cells.loc[cid, "sample"])))
    return out


def qc_table(cells: List[CellGraph]) -> pd.DataFrame:
    rows = []
    for c in cells:
        r = {"cell_id": c.cell_id, "n_nodes": c.num_nodes, "n_edges": c.num_undirected_edges}
        r.update(c.qc)
        rows.append(r)
    return pd.DataFrame(rows).set_index("cell_id")


def cell_composition(cells: List[CellGraph], n_markers: int) -> np.ndarray:
    """Whole-cell marker counts from the graph nodes: [n_cells, n_markers]."""
    out = np.zeros((len(cells), n_markers), dtype=np.int64)
    for i, c in enumerate(cells):
        out[i] = np.bincount(c.data.marker_id.numpy(), minlength=n_markers)
    return out
