"""Read a Pixelgen PNA ``.pxl`` file through the documented pixelator API and cache a subset.

Only public pixelator calls are used (``pixelator.read_pna`` -> ``PNAPixelDataset``,
``.adata()``, ``.filter(components=...)``, ``.edgelist().to_polars()``, ``.metadata()``).
The chosen cells, their edges, the panel table, the count matrix and a provenance record
are written as parquet/JSON so the rest of the tutorial never needs the 3 GB file again.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# What the official sources say (verified 2026-10-29; see README "Data provenance").
OFFICIAL_DATASET = {
    "dataset_name": "pna062-unstim-pbmcs",
    "download_api": "pixelator.pna.pixeldataset.download.DownloadableDatasets.download_dataset('pna062-unstim-pbmcs', output_path=...)",
    "registry_url_pixelator_0_30_0": ("https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/pna-datasets/v2/"
                                       "sha-079762b-2025-07-02-74062c09/pixelator/PNA062_unstim_PBMCs_1000cells_S02_S2.layout.pxl?download=1"),
    "paper_page": "https://software.pixelgen.com/datasets/pna-methods-paper",
    "paper_page_file": "PNA062_unstim_PBMCs_1000cells_S02_S2.layout.pxl",
    "paper_page_size": "3.63 GiB",
    "paper_page_md5": "94ee29f330a3921460aacb5d5535e4af",
    "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0), as stated on the paper dataset page",
    "citation": ("Karlsson et al., 'Cells as Single-Molecule Protein Proximity Networks', bioRxiv 2025.06.19.660329, "
                 "doi:10.1101/2025.06.19.660329; data (c) Pixelgen Technologies AB"),
    "sample_sheet": ("PixelgenTechnologies/pixelgen-PNA-paper data/sample_metadata.csv: sample PNA062_unstim_PBMCs_1000cells_S02, "
                     "alias Resting_PBMC, condition 'Resting PBMCs', cell_number 1000; no donor column"),
}


def _without_proxy_env():
    """pixelator opens its own DuckDB session which aborts on an unparsable ``http_proxy``."""
    return {k: os.environ.pop(k) for k in list(os.environ) if k.lower() in ("http_proxy", "https_proxy", "all_proxy")}


def open_pna(pxl_path):
    """``pixelator.read_pna`` with the sandbox proxy variables temporarily removed."""
    from pixelator import read_pna

    saved = _without_proxy_env()
    try:
        return read_pna(str(pxl_path))
    finally:
        os.environ.update(saved)


def file_checksums(path, block: int = 1 << 24) -> Dict[str, object]:
    """size in bytes, md5 and sha256 of a file (streamed)."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return {"size_bytes": size, "size_gib": round(size / 2**30, 3), "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def choose_cells(obs: pd.DataFrame, n_cells: int, seed: int, tau_types: Sequence[str] = ("normal",),
                 min_n_edges: int = 0, max_isotype_fraction: Optional[float] = None) -> List[str]:
    """Selection rule: pixelator QC pass, then a seeded draw without replacement from the id-sorted list."""
    keep = obs["tau_type"].isin(list(tau_types)) & (obs["n_edges"] >= min_n_edges)
    if max_isotype_fraction is not None:
        keep &= obs["isotype_fraction"] <= max_isotype_fraction
    eligible = sorted(obs.index[keep].astype(str))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(n_cells, len(eligible)), replace=False)
    return sorted(chosen.tolist())


@dataclass
class PNASubset:
    cells: pd.DataFrame       # one row per chosen component (cell); index = component id
    edges: pd.DataFrame       # component, umi1, umi2, marker_1, marker_2, read_count, uei_count
    markers: pd.DataFrame     # panel table; index = marker id in panel order (the shared vocabulary)
    counts: pd.DataFrame      # cells x markers antibody counts from pixelator's AnnData
    provenance: dict

    @property
    def vocab(self) -> List[str]:
        return list(self.markers.index)

    @property
    def cell_ids(self) -> List[str]:
        return list(self.cells.index)

    def edges_of(self, cell_id: str) -> pd.DataFrame:
        return self.edges[self.edges["component"] == cell_id]


CELL_COLUMNS = ["sample", "n_umi1", "n_umi2", "n_edges", "n_umi", "reads_in_component", "n_antibodies",
                "tau", "tau_type", "isotype_fraction", "intracellular_fraction"]
EDGE_COLUMNS = ["component", "umi1", "umi2", "marker_1", "marker_2", "read_count", "uei_count"]
MARKER_COLUMNS = ["control", "full_name", "uniprot_id", "antibody_count", "antibody_pct"]


def extract_subset(pxl_path, out_dir, n_cells: int = 100, seed: int = 0, tau_types=("normal",),
                   min_n_edges: int = 0, checksum: bool = True, overwrite: bool = False) -> PNASubset:
    """Pick ``n_cells`` QC-passing cells reproducibly and cache them under ``out_dir``."""
    pxl_path, out_dir = Path(pxl_path), Path(out_dir)
    if (out_dir / "provenance.json").exists() and not overwrite:
        return load_subset(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pg = open_pna(pxl_path)
    adata = pg.adata()
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    chosen = choose_cells(obs, n_cells, seed, tau_types, min_n_edges)
    cells = obs.loc[chosen, [c for c in CELL_COLUMNS if c in obs.columns]].copy()
    cells.index.name = "component"
    # documented, lazy filter -> only the chosen components are materialised
    saved = _without_proxy_env()
    try:
        el = pg.filter(components=chosen).edgelist().to_polars().to_pandas()
    finally:
        os.environ.update(saved)
    edges = el[[c for c in EDGE_COLUMNS if c in el.columns]].copy()
    for c in ("component", "marker_1", "marker_2"):
        edges[c] = edges[c].astype(str)
    markers = adata.var[[c for c in MARKER_COLUMNS if c in adata.var.columns]].copy()
    markers.index = markers.index.astype(str)
    markers.index.name = "marker_id"
    X = adata[chosen].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    counts = pd.DataFrame(X, index=chosen, columns=markers.index).astype(np.int64)
    counts.index.name = "component"

    import pixelator
    prov = {
        "source_file": str(pxl_path.resolve()), "source_file_name": pxl_path.name,
        "pixelator_metadata": pg.metadata(), "pixelator_version_used_to_read": pixelator.__version__,
        "official": OFFICIAL_DATASET,
        "selection_rule": {"n_cells_requested": n_cells, "seed": seed, "tau_types": list(tau_types), "min_n_edges": min_n_edges,
                           "rule": "pixelator QC columns: tau_type in tau_types and n_edges >= min_n_edges; eligible ids sorted; "
                                   "numpy.random.default_rng(seed).choice(eligible, n_cells, replace=False); ids then sorted"},
        "n_cells_in_file": int(len(obs)), "n_cells_eligible": int(obs["tau_type"].isin(list(tau_types)).sum()),
        "n_cells_chosen": len(chosen), "chosen_cells": chosen,
        "n_edges_in_subset": int(len(edges)), "n_markers": int(len(markers)),
        "control_markers": list(markers.index[markers["control"].astype(bool)]) if "control" in markers else [],
        "extracted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
    }
    if checksum:
        prov["source_checksums"] = file_checksums(pxl_path)
    cells.to_parquet(out_dir / "cells.parquet")
    edges.to_parquet(out_dir / "edges.parquet", index=False)
    markers.to_parquet(out_dir / "markers.parquet")
    counts.to_parquet(out_dir / "counts.parquet")
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return PNASubset(cells, edges, markers, counts, prov)


def load_subset(out_dir) -> PNASubset:
    out_dir = Path(out_dir)
    prov = json.loads((out_dir / "provenance.json").read_text())
    return PNASubset(pd.read_parquet(out_dir / "cells.parquet"), pd.read_parquet(out_dir / "edges.parquet"),
                     pd.read_parquet(out_dir / "markers.parquet"), pd.read_parquet(out_dir / "counts.parquet"), prov)
