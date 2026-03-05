"""
High‑level SpatialAlign pipeline.

This module wraps the sequence of steps that were previously executed
manually inside the notebook ``0729/nsclc.ipynb`` into a single reusable
function ``spatial_align``:

1. Train an scRNA‑seq classifier (stage 1).
2. Use the trained classifier to pseudo‑label the ST data.
3. Optionally attach precomputed 2D UMAP coordinates to the ST AnnData.
4. Train the spatial encoder with prototype‑guided contrastive learning
   (stage 2).

All heavy computations are delegated to the existing modules
(``train_sc_0930.py``, ``pseudo_labeling.py``, ``trainer_stage2_v3.py``);
this package does not change their behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .utils import add_project_root_to_sys_path, ensure_dir
from .preprocess import attach_umap_from_csv
from .model import (
    run_stage1_sc_classifier,
    run_pseudo_labeling,
    run_stage2_spatial_encoder,
)


# Make sure imports used in the underlying training code can be resolved
add_project_root_to_sys_path()


@dataclass
class SpatialAlignConfig:
    """
    Configuration for a SpatialAlign run.

    Parameters
    ----------
    adata_sc_path
        Path to the scRNA‑seq AnnData ``.h5ad`` file.
    adata_st_path
        Path to the spatial transcriptomics AnnData ``.h5ad`` file.
    output_dir
        Directory where checkpoints and any optional outputs are written.
    label_key
        Observation column containing cell‑type labels.
    umap_csv_path
        Optional CSV with UMAP1/UMAP2 coordinates for the ST cells,
        matching the logic in ``nsclc.ipynb``.
    stage1_checkpoint
        Path where the scRNA classifier checkpoint will be saved
        (defaults to ``output_dir / \"mlp_stage1.pt\"``).
    gat_checkpoint
        Path where the GAT/ST encoder checkpoint will be saved
        (defaults to ``output_dir / \"gat_st.pt\"``).
    mlp_stage2_save_path
        Path where the updated MLP checkpoint from stage 2 is saved.
        By default this reuses ``stage1_checkpoint``, matching the
        original notebook.
    """

    adata_sc_path: str
    adata_st_path: str
    output_dir: str
    label_key: str = "celltype"
    umap_csv_path: Optional[str] = None
    stage1_checkpoint: Optional[str] = None
    gat_checkpoint: Optional[str] = None
    mlp_stage2_save_path: Optional[str] = None


def spatial_align(config: SpatialAlignConfig):
    """
    Run the full SpatialAlign pipeline described above.

    Returns
    -------
    adata_sc, adata_st
        The processed scRNA and spatial AnnData objects after stage‑2
        training (including pseudo‑labels and optional UMAP layout on
        the ST object).
    """
    out_dir = ensure_dir(config.output_dir)

    # Resolve default checkpoint paths (mirror nsclc.ipynb behaviour)
    stage1_ckpt = (
        Path(config.stage1_checkpoint)
        if config.stage1_checkpoint is not None
        else out_dir / "mlp_stage1.pt"
    )

    gat_ckpt = (
        Path(config.gat_checkpoint)
        if config.gat_checkpoint is not None
        else out_dir / "gat_st.pt"
    )

    mlp_stage2_path = (
        Path(config.mlp_stage2_save_path)
        if config.mlp_stage2_save_path is not None
        else stage1_ckpt
    )

    # ---- Stage 1: train scRNA classifier and prepare aligned AnnData ----
    adata_sc, adata_st, _prototypes = run_stage1_sc_classifier(
        adata_sc_path=config.adata_sc_path,
        adata_st_path=config.adata_st_path,
        model_save_path=str(stage1_ckpt),
        label_key=config.label_key,
    )

    # ---- Pseudo‑labelling of ST data ----
    adata_st = run_pseudo_labeling(adata_st, str(stage1_ckpt))

    # ---- Optional: attach UMAP coordinates from CSV ----
    if config.umap_csv_path is not None:
        adata_st = attach_umap_from_csv(adata_st, config.umap_csv_path)

    # ---- Stage 2: train spatial encoder ----
    adata_st = run_stage2_spatial_encoder(
        stage1_checkpoint_path=str(stage1_ckpt),
        adata_sc=adata_sc,
        adata_st=adata_st,
        gat_checkpoint_path=str(gat_ckpt),
        mlp_stage2_save_path=str(mlp_stage2_path),
    )

    return adata_sc, adata_st


__all__ = ["SpatialAlignConfig", "spatial_align"]

