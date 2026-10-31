"""A small shared GraphSAGE encoder for masked-marker prediction, built from PyG layers.

Input per node: one-hot marker (length V) plus one extra "mask" column (length V+1 total).
Two ``SAGEConv`` layers turn each node's own label and its neighbours' labels into a
``latent``-dimensional node embedding. A linear head predicts the hidden marker of masked
nodes. ``global_mean_pool`` averages node embeddings into one fixed-length patch vector.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import torch
from torch import Tensor, nn

from pnatut.pyg_compat import SAGEConv, global_mean_pool


class MaskedMarkerSAGE(nn.Module):
    def __init__(self, n_markers: int, hidden: int = 64, latent: int = 32, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_markers, self.hidden, self.latent = n_markers, hidden, latent
        self.mask_column = n_markers              # index of the extra one-hot column meaning "label hidden"
        dims = [n_markers + 1] + [hidden] * (n_layers - 1) + [latent]
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(latent, n_markers)  # masked-label classifier on node embeddings

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Node embeddings z: [num_nodes, latent]. ReLU between layers, none after the last."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.dropout(torch.relu(h))
        return h

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None) -> Dict[str, Tensor]:
        z = self.encode(x, edge_index)
        out = {"z": z, "logits": self.head(z)}
        if batch is not None:
            out["graph"] = global_mean_pool(z, batch)   # [num_graphs, latent]
        return out


def apply_mask(x: Tensor, mask: Tensor, mask_column: int) -> Tensor:
    """Return a copy of ``x`` where masked rows are replaced by the pure mask indicator.

    The original label of a masked node survives only in the target ``y``; the feature row
    carries no trace of it (all marker columns zero, mask column one).
    """
    xm = x.clone()
    xm[mask] = 0.0
    xm[mask, mask_column] = 1.0
    return xm


def count_parameters(model: nn.Module) -> pd.DataFrame:
    rows = [{"parameter": n, "shape": list(p.shape), "count": p.numel()} for n, p in model.named_parameters()]
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"parameter": "TOTAL", "shape": "", "count": int(df["count"].sum())}
    return df
