"""
Utility helpers for the SpatialAlign package.

These helpers are intentionally lightweight and only wrap generic logic
such as path handling. All modelling code lives in the existing
repository modules (e.g. ``train_sc_0930.py``, ``pseudo_labeling.py``,
``trainer_stage2_v3.py``) which we call from this package without
modifying them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def add_project_root_to_sys_path() -> Path:
    """
    Ensure the project root (the folder containing this package) is on ``sys.path``.

    This mirrors the behaviour in the original notebooks where
    ``sys.path.append("..")`` is used to import training utilities that
    live next to the notebooks.
    """
    # ``SpatialAlign/SpatialAlign/utils.py`` -> go up two levels to project root
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def ensure_dir(path: os.PathLike | str) -> Path:
    """
    Create ``path`` as a directory if it does not already exist and
    return it as a :class:`pathlib.Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = ["add_project_root_to_sys_path", "ensure_dir"]

