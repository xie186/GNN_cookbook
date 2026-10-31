"""Small matplotlib helpers with one fixed categorical palette (identity never cycles)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# fixed categorical order (validated palette from the dataviz reference); "Other" is grey
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#9a9a94"
SEQUENTIAL = "Blues"


def marker_colors(markers_in_order: Sequence[str]) -> Dict[str, str]:
    """Assign the fixed palette to up to 8 markers (in the given order); everything else is grey."""
    cols = {}
    for i, m in enumerate(markers_in_order):
        cols[m] = CATEGORICAL[i] if i < len(CATEGORICAL) else OTHER
    return cols


def draw_molecular_graph(edge_index, labels: Sequence[str], node_ids: Optional[Sequence[str]] = None,
                         ax=None, colors: Optional[Dict[str, str]] = None, highlight: Optional[int] = None,
                         hop: Optional[Sequence[int]] = None, seed: int = 0, title: str = "",
                         show_ids: bool = True, node_size: int = 700, font_size: int = 7):
    """Draw an undirected molecular graph: node colour = marker, ring = anchor, label = id + marker."""
    ei = edge_index.numpy() if hasattr(edge_index, "numpy") else np.asarray(edge_index)
    n = len(labels)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    und = {(min(a, b), max(a, b)) for a, b in zip(ei[0], ei[1]) if a != b}
    G.add_edges_from(und)
    pos = nx.spring_layout(G, seed=seed)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if colors is None:
        order = pd.Series(labels).value_counts().index.tolist()
        colors = marker_colors(order)
    node_cols = [colors.get(l, OTHER) for l in labels]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b8b8b2", width=1.2)
    edgecolors = ["#0b0b0b" if (highlight is not None and i == highlight) else "white" for i in range(n)]
    widths = [2.5 if (highlight is not None and i == highlight) else 0.8 for i in range(n)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_cols, node_size=node_size, edgecolors=edgecolors, linewidths=widths)
    if show_ids:
        txt = {}
        for i in range(n):
            parts = []
            if node_ids is not None:
                parts.append(str(node_ids[i]))
            parts.append(str(labels[i]))
            if hop is not None:
                parts.append(f"hop {int(hop[i])}")
            txt[i] = "\n".join(parts)
        nx.draw_networkx_labels(G, pos, labels=txt, ax=ax, font_size=font_size, font_color="#0b0b0b")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    return ax, pos


def legend_for(colors: Dict[str, str], ax, title: str = "marker", **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=m) for m, c in colors.items()]
    ax.legend(handles=handles, title=title, fontsize=7, title_fontsize=8, frameon=False, **kw)


def plot_history(history: pd.DataFrame, ax=None, title: str = "training curves"):
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(history.index, history["train_loss"], color=CATEGORICAL[0], lw=2, label="train loss")
    ax[0].plot(history.index, history["val_loss"], color=CATEGORICAL[1], lw=2, label="validation loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("masked cross-entropy"); ax[0].legend(frameon=False); ax[0].set_title(title, fontsize=10)
    ax[1].plot(history.index, history["val_accuracy"], color=CATEGORICAL[2], lw=2, label="val accuracy")
    ax[1].plot(history.index, history["val_balanced_accuracy"], color=CATEGORICAL[3], lw=2, label="val balanced accuracy")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("fraction correct (masked nodes)"); ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    return ax


def composition_bars(comp_rows: Dict[str, np.ndarray], vocab: Sequence[str], top: int = 12, ax=None, title: str = ""):
    """Grouped horizontal bars of the ``top`` markers (by the first row's proportions)."""
    first = next(iter(comp_rows.values()))
    order = np.argsort(-first)[:top]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 0.3 * top + 1))
    h = 0.8 / len(comp_rows)
    for i, (name, row) in enumerate(comp_rows.items()):
        ax.barh(np.arange(top) + i * h, row[order], height=h * 0.9, color=CATEGORICAL[i % len(CATEGORICAL)], label=name)
    ax.set_yticks(np.arange(top) + h * (len(comp_rows) - 1) / 2)
    ax.set_yticklabels([vocab[j] for j in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("proportion of molecules")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def size_histograms(tables: Dict[str, pd.DataFrame], column: str = "n_nodes", ax=None, bins: int = 40):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.2))
    for i, (name, t) in enumerate(tables.items()):
        ax.hist(t[column], bins=bins, histtype="step", lw=2, color=CATEGORICAL[i], label=name)
    ax.set_xlabel(column.replace("_", " ")); ax.set_ylabel("number of patches"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
