import torch
import scanpy as sc
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .dnn import Encoder, ClassifierHead
import torch.nn.functional as F


class InferenceDataset(Dataset):
    """仅返回输入 x，不包含标签 y"""

    def __init__(self, X):
        # X: numpy array 或者 torch.Tensor，shape = (n_cells, G)
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X.astype(np.float32))
        else:
            self.X = X.float()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx]


def pseudoing_label(adata_st, sc_stage1_checkpoint_path):
    # —— 2) 读取 ST 数据和 checkpoint ——
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint (model + LabelEncoder + class names)
    ckpt = torch.load(sc_stage1_checkpoint_path, map_location=device)
    le = ckpt['label_encoder']   # 训练时保存进 checkpoint
    class_names = ckpt['class_names']     # list of strings
    marker_genes = ckpt['marker_genes']

    X = adata_st.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_norm = X.astype(np.float32)  # shape = (n_cells, n_genes)

    # —— 构造 DataLoader（只要 X_norm） ——
    inf_ds = InferenceDataset(X_norm)
    inf_loader = DataLoader(inf_ds, batch_size=2048, shuffle=False, num_workers=4)

    # Instantiate model exactly as in training
    # —— 5) 实例化并 load 参数 ——
    hidden_dims = [128, 64, 64]
    encoder = Encoder(
        in_dim=len(marker_genes),
        hidden_dims=hidden_dims,
        p_drop=0.2
    ).to(device)

    classifier = ClassifierHead(
        feat_dim=hidden_dims[-1],
        num_classes=len(class_names)
    ).to(device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    classifier.load_state_dict(ckpt['classifier_state_dict'])
    encoder.eval()
    classifier.eval()

    # —— 批量推理 ——
    all_preds = []
    all_confs = []
    with torch.no_grad():
        for xb in inf_loader:
            xb = xb.to(device)
            h = encoder(xb)
            logits = classifier(h)
            probs = F.softmax(logits, dim=1)        # [B, C]
            conf, preds = probs.max(dim=1)          # conf: [B], preds: [B]
            all_preds.append(preds.cpu().numpy())
            all_confs.append(conf.cpu().numpy())

    # 拼成一维数组
    all_preds = np.concatenate(all_preds)
    all_confs = np.concatenate(all_confs)          # (n_cells,)

    # 1) 把整数编码映射成字符串标签
    pred_labels = [class_names[p] for p in all_preds]

    # 2) 赋值给 adata.obs
    adata_st.obs['pseudo_label'] = pd.Categorical(
        pred_labels,
        categories=class_names  # 保持类别顺序
    )
    adata_st.obs['pseudo_confidence'] = all_confs

    # —— 6) 评估和写回 ——
    gt_all = adata_st.obs['celltype']
    pr_all = adata_st.obs['pseudo_label']
    # 训练时见过的类别
    seen = set(class_names)  # 或者 set(le.classes_)
    mask = gt_all.isin(seen)

    # 用掩码过滤真实值和预测值
    gt = gt_all[mask].values
    pr = pr_all[mask].values

    # 4) 计算 Accuracy & Report
    acc = accuracy_score(gt, pr)
    print(f"Filtered Accuracy: {acc:.4f}")

    print(classification_report(
        gt, pr,
        labels=list(seen),    # 指定报告里只显示 seen 这些类
        target_names=list(seen),
        zero_division=0,      # 如果某类在 pr 中零预测，就不会报错
        digits=4
    ))
    return adata_st

