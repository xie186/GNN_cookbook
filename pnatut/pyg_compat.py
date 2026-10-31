"""Import the PyTorch Geometric API used by the tutorial, falling back to pure PyTorch.

Everything in the tutorial is written against the real PyG API (``torch_geometric`` 2.x):
``Data``, ``Batch``, ``k_hop_subgraph``, ``subgraph``, ``to_undirected``, ``coalesce``,
``degree``, ``SAGEConv`` and the ``global_*_pool`` read-outs. When ``torch_geometric`` is
importable it is used verbatim. When it is not (as on the machine where this tutorial was
first executed: no network access, no cached wheel), :mod:`pnatut._pyg_fallback` provides
drop-in re-implementations with identical signatures and return conventions.

``BACKEND`` tells you which one is active; the notebook prints it. ``tests/test_tut_pyg_compat.py``
checks the fallback against hand-computed results and, when PyG is installed, against PyG itself.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only when PyG is installed
    import torch_geometric  # noqa: F401
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import SAGEConv, global_add_pool, global_max_pool, global_mean_pool
    from torch_geometric.utils import (coalesce, contains_self_loops, degree, is_undirected, k_hop_subgraph,
                                       remove_self_loops, subgraph, to_undirected)

    BACKEND = f"torch_geometric {torch_geometric.__version__}"
    USING_PYG = True
except ImportError:
    from pnatut._pyg_fallback import (Batch, Data, SAGEConv, coalesce, contains_self_loops, degree,  # noqa: F401
                                      global_add_pool, global_max_pool, global_mean_pool, is_undirected,
                                      k_hop_subgraph, remove_self_loops, subgraph, to_undirected)

    BACKEND = "pure-torch fallback (pnatut._pyg_fallback): torch_geometric is not installed"
    USING_PYG = False

__all__ = ["BACKEND", "USING_PYG", "Data", "Batch", "SAGEConv", "global_mean_pool", "global_add_pool",
           "global_max_pool", "k_hop_subgraph", "subgraph", "to_undirected", "coalesce", "degree",
           "is_undirected", "contains_self_loops", "remove_self_loops"]
