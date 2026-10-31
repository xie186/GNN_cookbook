"""Patch embeddings, nearest neighbours and overlapping reference-centred neighbourhoods.

VIMA defines one *microniche per patch*: the soft membership of every other patch is the
probability that a random walk on the fingerprint kNN graph, started at that patch, sits
on the anchor after ``t`` steps (CNA's ``nam``). Per-sample abundance is the mean
membership over the sample's patches. Here we keep that definition but anchor only ``R``
reference patches chosen from the training split, so the tables stay readable. A patch
therefore belongs to *several* neighbourhoods with different weights; the weights of one
patch are not constrained to sum to one.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from pnatut.pyg_compat import Batch, Data


@torch.no_grad()
def embed_patches(model, datas: Sequence[Data], batch_size: int = 256) -> np.ndarray:
    """Mean-pooled node embeddings per patch, no masking: [n_patches, latent]."""
    model.eval()
    out = []
    for s in range(0, len(datas), batch_size):
        b = Batch.from_data_list(list(datas[s:s + batch_size]))
        out.append(model(b.x, b.edge_index, b.batch)["graph"].numpy())
    return np.concatenate(out)


def l2_normalise(E: np.ndarray) -> np.ndarray:
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return l2_normalise(A) @ l2_normalise(B).T


def nearest_patches(E: np.ndarray, query: int, k: int = 5, exclude_same_cell: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The k most cosine-similar patches to ``query`` (optionally excluding its own cell)."""
    sims = cosine_similarity_matrix(E[query:query + 1], E)[0]
    order = np.argsort(-sims)
    rows = []
    for j in order:
        if j == query:
            continue
        if exclude_same_cell is not None and exclude_same_cell[j] == exclude_same_cell[query]:
            continue
        rows.append({"patch": int(j), "cosine_similarity": float(sims[j])})
        if len(rows) == k:
            break
    return pd.DataFrame(rows).set_index("patch")


def knn_graph(E: np.ndarray, k: int = 15) -> sp.csr_matrix:
    """Symmetric kNN connectivity (cosine) over all patches; weights are similarities in (0, 1]."""
    D = kneighbors_graph(l2_normalise(E), n_neighbors=k, mode="distance", metric="cosine", include_self=False)
    W = D.copy()
    W.data = 1.0 - W.data / 2.0        # cosine distance in [0, 2] -> similarity-like weight in [0, 1]
    W.data = np.maximum(W.data, 1e-6)
    W = W.maximum(W.T)                 # make it symmetric: an edge if either is in the other's k list
    return W.tocsr()


def choose_references(E_train: np.ndarray, train_index: np.ndarray, n_refs: int, seed: int = 0) -> np.ndarray:
    """Reference patches = training patches closest to k-means centres fitted on training embeddings."""
    km = KMeans(n_clusters=n_refs, n_init=10, random_state=seed).fit(E_train)
    refs, taken = [], np.zeros(len(E_train), dtype=bool)
    for c in km.cluster_centers_:
        d = ((E_train - c) ** 2).sum(1)
        d[taken] = np.inf                        # never pick the same patch twice
        j = int(np.argmin(d))
        taken[j] = True
        refs.append(int(train_index[j]))
    return np.array(refs)


def random_walk_membership(W: sp.csr_matrix, refs: np.ndarray, n_steps: int = 3, self_weight: float = 1.0) -> np.ndarray:
    """M[p, r] = probability that a ``n_steps``-step random walk started at patch p is at reference r.

    Transition matrix as in CNA/VIMA: T = (W + self_weight * I) D^-1 with D the column sums,
    i.e. a walker leaves a node along its weighted edges, keeping a self-loop of weight 1.
    Starting from e_p, the mass after t steps is T^t e_p; its entry at r is M[p, r].
    Computed for all p at once by propagating from the references with T^T.
    """
    n = W.shape[0]
    A = W + self_weight * sp.eye(n, format="csr")
    colsum = np.asarray(A.sum(axis=0)).ravel()
    T = A @ sp.diags(1.0 / colsum)                   # column-stochastic
    M = np.zeros((n, len(refs)))
    M[refs, np.arange(len(refs))] = 1.0              # unit mass at each reference
    TT = T.T.tocsr()
    for _ in range(n_steps):
        M = TT @ M                                   # (T^t)^T e_r  ->  row p holds P(walk from p is at r)
    return M


def membership_sets(M: np.ndarray, top_m: int = 100) -> np.ndarray:
    """Thresholded, overlapping sets: patch p is 'in' neighbourhood r if it is among the top_m by weight."""
    S = np.zeros_like(M, dtype=bool)
    for r in range(M.shape[1]):
        S[np.argsort(-M[:, r])[:top_m], r] = True
    return S


def per_cell_membership(M: np.ndarray, patch_cell_ids: Sequence[str], cell_table: pd.DataFrame,
                        prefix: str = "nbhd", relative_to_uniform: bool = False) -> pd.DataFrame:
    """Mean membership weight of each cell's sampled patches in each neighbourhood (VIMA's NAM entry).

    Denominator = number of patches sampled for that cell (``n_patches``). Rows need not sum to one.
    ``sample_id`` is carried along from ``cell_table``. With ``relative_to_uniform`` the means are
    multiplied by the total number of patches, so 1 = what a uniform spread of the walk would give.
    """
    scale = M.shape[0] if relative_to_uniform else 1.0
    df = pd.DataFrame(M * scale, columns=[f"{prefix}_{r}" for r in range(M.shape[1])])
    df["cell_id"] = list(patch_cell_ids)
    g = df.groupby("cell_id")
    out = g.mean(numeric_only=True)
    out.insert(0, "n_patches", g.size())
    out.insert(0, "sample_id", cell_table.loc[out.index, "sample"].values)
    return out


def per_cell_set_fractions(S: np.ndarray, patch_cell_ids: Sequence[str], cell_table: pd.DataFrame,
                           prefix: str = "nbhd") -> pd.DataFrame:
    """Fraction of each cell's patches that are members of each thresholded set (rows need not sum to 1)."""
    return per_cell_membership(S.astype(float), patch_cell_ids, cell_table, prefix)


def hard_clusters(E_train: np.ndarray, E_all: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Baseline: k-means fitted on training embeddings, every patch gets exactly one label."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(E_train)
    return km.predict(E_all)
