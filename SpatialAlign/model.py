"""
Model components for SpatialAlign.

为了让 `SpatialAlign` 目录在单独上传到 GitHub 时可以自洽运行，
这里直接复制并内置了原来分散在

* ``train_sc_0930.py`` – scRNA‑seq classifier (stage 1)
* ``pseudo_labeling.py`` – ST 伪标签推断
* ``trainer_stage2_v3.py`` – 空间 encoder 训练 (stage 2)

中的核心函数，然后通过三个统一的入口函数对外暴露：

* ``run_stage1_sc_classifier`` – 包一层 `train_model`
* ``run_pseudo_labeling`` – 包一层 `pseudoing_label`
* ``run_stage2_spatial_encoder`` – 包一层 `train_for_stage2`
"""

from __future__ import annotations

from .train_sc_0930_impl import train_model
from .pseudo_labeling_impl import pseudoing_label
from .trainer_stage2_v3_impl import train_for_stage2


def run_stage1_sc_classifier(
    adata_sc_path: str,
    adata_st_path: str,
    model_save_path: str,
    label_key: str = "celltype",
):
    """
    Wrapper around :func:`train_sc_0930.train_model`.

    Returns the processed scRNA AnnData, spatial AnnData and per‑class
    prototypes exactly as the original function does.
    """
    return train_model(
        adata_sc_path=adata_sc_path,
        adata_st_path=adata_st_path,
        model_save_path=model_save_path,
        label_key=label_key,
    )


def run_pseudo_labeling(adata_st, stage1_checkpoint_path: str):
    """
    Wrapper around :func:`pseudo_labeling.pseudoing_label`.
    """
    return pseudoing_label(adata_st, stage1_checkpoint_path)


def run_stage2_spatial_encoder(
    stage1_checkpoint_path: str,
    adata_sc,
    adata_st,
    gat_checkpoint_path: str,
    mlp_stage2_save_path: str,
):
    """
    Wrapper around :func:`trainer_stage2_v3.train_for_stage2`.
    """
    return train_for_stage2(
        stage1_checkpoint_path,
        adata_sc,
        adata_st,
        gat_checkpoint_path,
        mlp_stage2_save_path,
    )


__all__ = [
    "run_stage1_sc_classifier",
    "run_pseudo_labeling",
    "run_stage2_spatial_encoder",
]
