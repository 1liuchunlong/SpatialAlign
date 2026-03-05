"""
Preprocessing helpers for SpatialAlign.

The original notebook performs several data‑handling steps such as
attaching a precomputed UMAP layout stored in a CSV file.  We expose
those steps here as reusable functions without touching the original
notebook code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def attach_umap_from_csv(
    adata_st,
    umap_csv_path: str | Path,
    umap_key: str = "X_umap",
) :
    """
    Attach 2D UMAP coordinates stored in a CSV file to ``adata_st.obsm``.

    Parameters
    ----------
    adata_st
        Spatial AnnData object.
    umap_csv_path
        Path to a CSV file with index matching ``adata_st.obs_names`` and
        at least two columns named ``UMAP1`` and ``UMAP2``.
    umap_key
        Key under which to store the coordinates in ``adata_st.obsm``.

    Returns
    -------
    adata_st_subset
        A copy of ``adata_st`` restricted to the shared cells and with
        ``adata_st_subset.obsm[umap_key]`` populated.
    """
    umap_csv_path = Path(umap_csv_path)
    umap_df = pd.read_csv(umap_csv_path, index_col=0)

    # Intersect cells between AnnData and the UMAP dataframe
    shared = adata_st.obs_names.intersection(umap_df.index)
    adata_st = adata_st[shared].copy()

    adata_st.obsm[umap_key] = umap_df.loc[shared, ["UMAP1", "UMAP2"]].to_numpy(
        dtype=np.float32
    )
    return adata_st


__all__ = ["attach_umap_from_csv"]

