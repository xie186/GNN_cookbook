"""Minimal pure-PyTorch stand-ins for the PyTorch Geometric (PyG) API used in the tutorial.

This module is only imported by :mod:`pnatut.pyg_compat` when ``torch_geometric`` cannot be
imported. Every function mirrors the signature and the return convention of the PyG
function of the same name (PyG 2.6), so notebook code written against PyG runs unchanged.
The implementations are deliberately short and readable; they are not optimised.

Conventions shared with PyG
---------------------------
* ``edge_index`` is a ``LongTensor`` of shape ``[2, num_edges]``; row 0 = source, row 1 =
  target. Messages flow source -> target. An undirected graph stores both directions.
* ``batch`` is a ``LongTensor`` of shape ``[num_nodes]`` giving the graph id of each node
  inside a mini-batch.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------------------
class Data:
    """A single graph: a bag of named tensors, like ``torch_geometric.data.Data``.

    Only the behaviour the tutorial relies on is implemented: attribute storage,
    ``num_nodes`` / ``num_edges``, ``keys()``, ``to()``, ``clone()`` and a PyG-style repr.
    """

    def __init__(self, x: Optional[Tensor] = None, edge_index: Optional[Tensor] = None,
                 edge_attr: Optional[Tensor] = None, y: Optional[Tensor] = None, **kwargs):
        self._store = {}
        for k, v in dict(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, **kwargs).items():
            if v is not None:
                self._store[k] = v

    # attribute access -----------------------------------------------------------------
    def __getattr__(self, key):
        store = self.__dict__.get("_store", {})
        if key in store:
            return store[key]
        raise AttributeError(f"'Data' has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "_store":
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __contains__(self, key):
        return key in self._store

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def to_dict(self):
        return dict(self._store)

    # sizes ----------------------------------------------------------------------------
    @property
    def num_nodes(self) -> Optional[int]:
        if "num_nodes" in self._store:
            return int(self._store["num_nodes"])
        if "x" in self._store:
            return int(self._store["x"].size(0))
        if "edge_index" in self._store and self._store["edge_index"].numel() > 0:
            return int(self._store["edge_index"].max()) + 1
        return None

    @num_nodes.setter
    def num_nodes(self, value):
        self._store["num_nodes"] = int(value)

    @property
    def num_edges(self) -> int:
        return int(self._store["edge_index"].size(1)) if "edge_index" in self._store else 0

    @property
    def num_node_features(self) -> int:
        return int(self._store["x"].size(1)) if "x" in self._store and self._store["x"].dim() > 1 else 0

    def is_undirected(self) -> bool:
        return is_undirected(self.edge_index, num_nodes=self.num_nodes)

    def has_self_loops(self) -> bool:
        return contains_self_loops(self.edge_index)

    # utilities ------------------------------------------------------------------------
    def to(self, device):
        out = Data()
        for k, v in self._store.items():
            out._store[k] = v.to(device) if torch.is_tensor(v) else v
        return out

    def clone(self):
        out = Data()
        for k, v in self._store.items():
            out._store[k] = v.clone() if torch.is_tensor(v) else v
        return out

    def __repr__(self) -> str:
        parts = []
        for k, v in self._store.items():
            if torch.is_tensor(v):
                parts.append(f"{k}={list(v.shape)}")
            elif isinstance(v, (int, float, str, bool)):
                parts.append(f"{k}={v!r}")
            else:
                parts.append(f"{k}={type(v).__name__}")
        return f"{type(self).__name__}({', '.join(parts)})"


class Batch(Data):
    """Several graphs stacked into one big disconnected graph (``Batch.from_data_list``).

    Node tensors are concatenated along dim 0; ``edge_index`` entries are shifted by the
    cumulative node count so that edges still point at the right nodes. ``batch[i]`` is the
    graph id of node ``i`` and ``ptr`` holds the node offsets of each graph.
    """

    @classmethod
    def from_data_list(cls, data_list: Sequence[Data], follow_batch=None, exclude_keys=None) -> "Batch":
        exclude = set(exclude_keys or [])
        out = cls()
        keys = [k for k in data_list[0].keys() if k not in exclude]
        num_nodes = [int(d.num_nodes) for d in data_list]
        ptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(num_nodes), 0).tolist()), dtype=torch.long)
        batch = torch.repeat_interleave(torch.arange(len(data_list)), torch.tensor(num_nodes))
        for k in keys:
            vals = [d[k] for d in data_list]
            if k == "edge_index":
                out._store[k] = torch.cat([v + ptr[i] for i, v in enumerate(vals)], dim=1)
            elif torch.is_tensor(vals[0]):
                v0 = vals[0]
                if v0.dim() == 0:
                    out._store[k] = torch.stack(vals)
                elif k in ("anchor_local", "anchor_index", "mapping") and v0.numel() == 1:
                    # per-graph node pointers must be shifted like edge_index
                    out._store[k] = torch.cat([v.view(-1) + ptr[i] for i, v in enumerate(vals)])
                else:
                    out._store[k] = torch.cat(vals, dim=0)
            else:
                out._store[k] = vals  # python objects (ids, strings) are kept as a list
        out._store["batch"] = batch
        out._store["ptr"] = ptr
        out._store["num_nodes"] = int(ptr[-1])
        out._num_graphs = len(data_list)
        return out

    def __setattr__(self, key, value):
        if key in ("_store", "_num_graphs"):
            object.__setattr__(self, key, value)
        else:
            self._store[key] = value

    @property
    def num_graphs(self) -> int:
        return self._num_graphs


# ---------------------------------------------------------------------------------------
# Utilities (torch_geometric.utils)
# ---------------------------------------------------------------------------------------
def maybe_num_nodes(edge_index: Tensor, num_nodes: Optional[int] = None) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0


def degree(index: Tensor, num_nodes: Optional[int] = None, dtype=None) -> Tensor:
    """Number of occurrences of each node id in ``index`` (e.g. in-degree from ``edge_index[1]``)."""
    n = maybe_num_nodes(index.view(1, -1), num_nodes)
    out = torch.zeros(n, dtype=dtype or torch.long, device=index.device)
    return out.scatter_add_(0, index, torch.ones_like(index, dtype=out.dtype))


def coalesce(edge_index: Tensor, edge_attr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             reduce: str = "sum"):
    """Sort edges by (source, target) and merge duplicates (summing ``edge_attr`` by default)."""
    n = maybe_num_nodes(edge_index, num_nodes)
    key = edge_index[0] * n + edge_index[1]
    uniq, inv = torch.unique(key, return_inverse=True)
    ei = torch.stack([uniq // n, uniq % n])
    if edge_attr is None:
        return ei
    if reduce == "sum":
        attr = torch.zeros((uniq.numel(),) + tuple(edge_attr.shape[1:]), dtype=edge_attr.dtype)
        attr.index_add_(0, inv, edge_attr)
    else:
        raise NotImplementedError(reduce)
    return ei, attr


def to_undirected(edge_index: Tensor, edge_attr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
                  reduce: str = "add"):
    """Add the reverse of every edge and coalesce, so each undirected edge appears twice."""
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    if edge_attr is None:
        return coalesce(ei, None, num_nodes)
    attr = torch.cat([edge_attr, edge_attr], dim=0)
    return coalesce(ei, attr, num_nodes, reduce="sum" if reduce == "add" else reduce)


def is_undirected(edge_index: Tensor, edge_attr=None, num_nodes: Optional[int] = None) -> bool:
    n = maybe_num_nodes(edge_index, num_nodes)
    a = coalesce(edge_index, None, n)
    b = coalesce(edge_index.flip(0), None, n)
    return a.shape == b.shape and bool(torch.equal(a, b))


def contains_self_loops(edge_index: Tensor) -> bool:
    return bool((edge_index[0] == edge_index[1]).any())


def remove_self_loops(edge_index: Tensor, edge_attr: Optional[Tensor] = None):
    keep = edge_index[0] != edge_index[1]
    return edge_index[:, keep], (None if edge_attr is None else edge_attr[keep])


def subgraph(subset: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None,
             relabel_nodes: bool = False, num_nodes: Optional[int] = None, return_edge_mask: bool = False):
    """Edges whose two endpoints are both in ``subset`` (the *induced* subgraph)."""
    n = maybe_num_nodes(edge_index, num_nodes)
    if subset.dtype == torch.bool:
        node_mask = subset
        subset = node_mask.nonzero().view(-1)
    else:
        node_mask = torch.zeros(n, dtype=torch.bool)
        node_mask[subset] = True
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    ei = edge_index[:, edge_mask]
    ea = None if edge_attr is None else edge_attr[edge_mask]
    if relabel_nodes:
        mapping = torch.full((n,), -1, dtype=torch.long)
        mapping[subset] = torch.arange(subset.numel())
        ei = mapping[ei]
    if return_edge_mask:
        return ei, ea, edge_mask
    return ei, ea


def k_hop_subgraph(node_idx: Union[int, List[int], Tensor], num_hops: int, edge_index: Tensor,
                   relabel_nodes: bool = False, num_nodes: Optional[int] = None,
                   flow: str = "source_to_target", directed: bool = False):
    """Nodes within ``num_hops`` graph steps of ``node_idx`` and the induced edges.

    Same algorithm and return order as PyG: ``(subset, edge_index, mapping, edge_mask)``.
    ``subset`` is sorted; ``mapping`` gives the position of each seed node inside ``subset``;
    with ``relabel_nodes=True`` the returned ``edge_index`` uses positions in ``subset``.
    """
    n = maybe_num_nodes(edge_index, num_nodes)
    assert flow in ("source_to_target", "target_to_source")
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index
    node_mask = torch.zeros(n, dtype=torch.bool, device=row.device)
    edge_mask = torch.zeros(row.size(0), dtype=torch.bool, device=row.device)
    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device).view(-1)
    subsets = [node_idx]
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[: node_idx.numel()]
    node_mask.fill_(False)
    node_mask[subset] = True
    if not directed:
        edge_mask = node_mask[row] & node_mask[col]
    ei = edge_index[:, edge_mask]
    if relabel_nodes:
        mapping = torch.full((n,), -1, dtype=torch.long, device=row.device)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        ei = mapping[ei]
    return subset, ei, inv, edge_mask


# ---------------------------------------------------------------------------------------
# Layers and pooling (torch_geometric.nn)
# ---------------------------------------------------------------------------------------
def _scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = torch.zeros((dim_size,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, dtype=src.dtype, device=src.device).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))


def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = torch.zeros((dim_size,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


def _scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = torch.full((dim_size,) + tuple(src.shape[1:]), float("-inf"), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(0, index.view(-1, 1).expand_as(src), src, reduce="amax", include_self=True)
    out[torch.isinf(out)] = 0
    return out


class SAGEConv(nn.Module):
    r"""GraphSAGE layer (Hamilton et al., 2017), as in ``torch_geometric.nn.SAGEConv``:

    .. math::
        \mathbf{h}'_i = \mathbf{W}_1 \cdot \mathrm{mean}_{j \in \mathcal{N}(i)} \mathbf{h}_j
                        + \mathbf{W}_2 \cdot \mathbf{h}_i + \mathbf{b}

    ``lin_l`` (with bias) acts on the aggregated neighbour message, ``lin_r`` (no bias) on
    the node's own features ("root weight"). Only ``aggr='mean'|'sum'|'max'`` are supported.
    """

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean", normalize: bool = False,
                 root_weight: bool = True, project: bool = False, bias: bool = True):
        super().__init__()
        if project:
            raise NotImplementedError("project=True is not implemented in the fallback")
        self.in_channels, self.out_channels, self.aggr = in_channels, out_channels, aggr
        self.normalize, self.root_weight = normalize, root_weight
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        if root_weight:
            self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, size=None) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)
        msg = x[src]                                   # one message per edge: the source node's features
        if self.aggr == "mean":
            agg = _scatter_mean(msg, dst, n)
        elif self.aggr in ("sum", "add"):
            agg = _scatter_sum(msg, dst, n)
        elif self.aggr == "max":
            agg = _scatter_max(msg, dst, n)
        else:
            raise ValueError(self.aggr)
        out = self.lin_l(agg)
        if self.root_weight:
            out = out + self.lin_r(x)
        if self.normalize:
            out = nn.functional.normalize(out, p=2.0, dim=-1)
        return out

    def __repr__(self):
        return f"{type(self).__name__}({self.in_channels}, {self.out_channels}, aggr={self.aggr})"


def global_mean_pool(x: Tensor, batch: Optional[Tensor], size: Optional[int] = None) -> Tensor:
    """Average the node vectors of each graph -> one vector per graph (``[num_graphs, F]``)."""
    if batch is None:
        return x.mean(dim=0, keepdim=True)
    size = int(batch.max()) + 1 if size is None else size
    return _scatter_mean(x, batch, size)


def global_add_pool(x: Tensor, batch: Optional[Tensor], size: Optional[int] = None) -> Tensor:
    if batch is None:
        return x.sum(dim=0, keepdim=True)
    size = int(batch.max()) + 1 if size is None else size
    return _scatter_sum(x, batch, size)


def global_max_pool(x: Tensor, batch: Optional[Tensor], size: Optional[int] = None) -> Tensor:
    if batch is None:
        return x.max(dim=0, keepdim=True).values
    size = int(batch.max()) + 1 if size is None else size
    return _scatter_max(x, batch, size)
