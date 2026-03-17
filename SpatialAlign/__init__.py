"""
SpatialAlign: Spatial transcriptomics cell type annotation via scRNA-seq alignment.
"""
from .train_sc_stage1 import train_model
from .pseudo_labeling_impl import pseudoing_label
from .train_stage2 import train_for_stage2

__all__ = ["train_model", "pseudoing_label", "train_for_stage2"]
