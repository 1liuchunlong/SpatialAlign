import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np
from sklearn.preprocessing import LabelEncoder
from itertools import cycle
from .ultils import location_to_edge
from typing import Optional

class SCRNADataset(Dataset):
    """带真实标签的 scRNA 数据集"""
    def __init__(self, adata: sc.AnnData, label_key: str, le: LabelEncoder):
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        self.X = torch.from_numpy(X.astype(np.float32))
        raw = adata.obs[label_key].values
        self.le = le
        self.y  = torch.from_numpy(self.le.transform(raw)).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


class STSpatialDataset(Dataset):
    """
    ST 数据集；每个 sample 返回 (表达 x, 标签/占位 y, 空间坐标 loc)
    - 有标签模式：传入 pseudo_label_key 和 LabelEncoder le
    - 无标签模式：pseudo_label_key=None 或该列不存在；y 用 -1 占位
    """
    def __init__(self,
                 adata: sc.AnnData,
                 pseudo_label_key: Optional[str] = None,
                 le: Optional[LabelEncoder] = None):
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        self.X = torch.from_numpy(X.astype(np.float32))
        self.coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
        self.has_label = (pseudo_label_key is not None) and (pseudo_label_key in adata.obs)

        if self.has_label:
            if le is None:
                raise ValueError("有标签模式下需要提供 LabelEncoder le。")
            raw = adata.obs[pseudo_label_key].astype(str).values
            self.y = torch.from_numpy(le.transform(raw)).long()
        else:
            self.y = None

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        y = self.y[i] if self.y is not None else None
        return self.X[i], y, self.coords[i]


class spatialCollate:
    def __init__(self, knn=1):
        self.knn = knn

    def __call__(self, batch):
        fits, supervs, locs = zip(*batch)
        if isinstance(fits[0], torch.Tensor):
            Xs = torch.stack(fits)
        else:
            Xs = torch.from_numpy(np.vstack(fits))
        if supervs[0] is not None:
            if isinstance(supervs[0], torch.Tensor):
                ys = torch.stack(supervs).long().unsqueeze(-1)
            else:
                ys = torch.from_numpy(np.vstack(supervs)).long().unsqueeze(-1)
        else:
            ys = None
        locs = np.vstack(locs)
        edge_idx = location_to_edge(locs, self.knn)
        return Xs, ys, edge_idx
