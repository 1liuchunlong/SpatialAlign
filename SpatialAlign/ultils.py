import random
import numpy as np
import pandas as pd
import torch
import pynndescent
from scipy.sparse import csr_matrix
from scipy import sparse
import torch_geometric.utils
import scanpy as sc


def get_knn(location, n_neighbors):
    index = pynndescent.NNDescent(location, n_neighbors=n_neighbors)
    neighbor_idx, neighbor_dist = index.neighbor_graph
    return neighbor_idx, neighbor_dist

def knn_to_adj(knn_idx, knn_dist):
    n_samples, n_neighbors = knn_idx.shape
    row = np.repeat(np.arange(n_samples), n_neighbors)
    col = knn_idx.flatten()
    data = np.ones(n_samples * n_neighbors)
    adj_matrix = csr_matrix((data, (row, col)), shape=(n_samples, n_samples))
    return adj_matrix

def location_to_edge(location, n_neighbors):
    idx, dist = get_knn(location, n_neighbors)
    adj = knn_to_adj(idx, dist)
    edge = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
    return edge

def augment_rare_cells(adata, rare_types=None, n_aug=3, noise_std=0.05, mix_ratio=0.5):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X = X.astype(np.float32)
    obs = adata.obs.copy().reset_index(drop=True)
    var = adata.var.copy()

    X_new, labels_new = [], []

    if rare_types is None:
        num_classes  = adata.obs['celltype'].nunique()
        total_cells  = adata.n_obs
        rare_threshold = total_cells / (num_classes * 5)
        type_counts  = adata.obs['celltype'].value_counts()
        rare_types   = type_counts[type_counts < rare_threshold].index.tolist()

    for ctype in rare_types:
        mask    = (obs['celltype'] == ctype).values
        X_rare  = X[mask]

        for i in range(X_rare.shape[0]):
            x_orig = X_rare[i]
            for _ in range(n_aug):
                x_aug = x_orig.copy()
                drop_idx = np.random.choice(len(x_aug),
                                            size=int(0.05 * len(x_aug)),
                                            replace=False)
                x_aug[drop_idx] = 0.0
                noise = np.random.normal(0, noise_std, size=x_aug.shape).astype(np.float32)
                x_aug = x_aug + noise
                j = np.random.randint(0, X_rare.shape[0])
                x_other = X_rare[j]
                x_aug = mix_ratio * x_aug + (1 - mix_ratio) * x_other
                x_aug = np.clip(x_aug, 0.0, None)
                X_new.append(x_aug)
                labels_new.append(ctype)

    X_new   = np.stack(X_new, axis=0).astype(np.float32)
    obs_new = pd.DataFrame({'celltype': labels_new})
    X_all   = np.vstack([X, X_new]).astype(np.float32)
    obs_all = pd.concat([obs, obs_new], ignore_index=True)
    adata_aug = sc.AnnData(X_all, obs=obs_all, var=var)
    print(f"扩充后样本数: {adata_aug.n_obs} (新增 {len(X_new)} 个样本)")
    return adata_aug
