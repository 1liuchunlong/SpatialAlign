import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import numpy as np

# —— 1.1 新的 Encoder Backbone —— 
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dims=[512,256,256], p_drop=0.2):
        super().__init__()
        # 只构建到最后一个特征维度（不含分类层）
        dims = [in_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)       # [B, D_emb]


class KAN_Encoder(nn.Module):
    """
    Kolmogorov–Arnold Network (KAN) 编码器
    支持并行 grid 条子映射分支，每条分支可自定义层宽度
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dims: list = [512, 256, 256],
                 grid: int = 2,
                 p_drop: float = 0.2):
        super().__init__()
        self.grid = grid
        # 并行构造 grid 条子映射，每条子映射内部使用 hidden_dims
        self.branches = nn.ModuleList()
        for _ in range(grid):
            layers = []
            dims = [in_dim] + hidden_dims
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p_drop))
            self.branches.append(nn.Sequential(*layers))
        # 拼接后投射到最后一个 hidden_dims 大小
        total_dim = hidden_dims[-1] * grid
        self.project = nn.Sequential(
            nn.Linear(total_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(p_drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        outs = [branch(x) for branch in self.branches]  # 每条输出 [B, hidden_dims[-1]]
        cat = torch.cat(outs, dim=1)                    # [B, hidden_dims[-1] * grid]
        z = self.project(cat)                           # [B, hidden_dims[-1]]
        return z


# —— 1.2 新的 Classifier Head —— 
class ClassifierHead(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, h):
        return self.fc(h)        # [B, C]
