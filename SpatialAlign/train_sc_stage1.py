import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, classification_report
import scanpy as sc
from .dnn import Encoder, ClassifierHead
import numpy as np
from .losses import FocalLoss
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from .ultils import augment_rare_cells
from scipy.sparse import csr_matrix
from torch.utils.data import WeightedRandomSampler
import random


def set_seed(seed=2026):
    """设置所有随机源的种子以确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 确定性与性能设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# —— 1) Dataset 封装 ——
class AnnDataset(Dataset):
    def __init__(self, adata, has_label=True, label_key=None, le=None):
        self.X = torch.from_numpy(
            adata.X.astype(np.float32).A
            if hasattr(adata.X, "A") else adata.X
        ).float()
        self.has_label = has_label
        if has_label:
            raw = adata.obs[label_key].values
            if le is None:
                le = LabelEncoder().fit(raw)
            self.y = torch.from_numpy(le.transform(raw)).long()
            self.le = le
        else:
            self.y = None

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.has_label:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def get_prototype_embeddings(encoder, adata_sc, label_key, device):
    # 通过编码器获得每个细胞的嵌入
    encoder.eval()
    cell_embeddings = []
    labels = adata_sc.obs[label_key].values

    with torch.no_grad():
        for x in adata_sc.X:
            x_tensor = torch.tensor(x).float().to(device)
            h = encoder(x_tensor)  # 通过编码器得到嵌入
            cell_embeddings.append(h.cpu().numpy())  # 将嵌入保存为 NumPy 数组

    cell_embeddings = np.array(cell_embeddings)

    # 计算每个细胞类型的原型嵌入（按标签聚合）
    unique_labels = np.unique(labels)
    prototypes = {}

    for label in unique_labels:
        # 获取属于该标签的所有嵌入
        label_mask = (labels == label)
        label_embeddings = cell_embeddings[label_mask]

        # 计算该标签的原型嵌入（可以是均值）
        prototype_embedding = np.mean(label_embeddings, axis=0)
        prototypes[label] = prototype_embedding

    return prototypes


"""
    adata_sc_path 输入 sc 的
"""


def train_model(
    adata_sc_path,
    adata_st_path,
    model_save_path,
    label_key='celltype',
):
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # —— 2) 构建 DataLoader ——
    adata_sc = sc.read_h5ad(adata_sc_path)
    # 可选：少样本类增强
    # adata_sc = augment_rare_cells(adata_sc)

    adata_st = sc.read_h5ad(adata_st_path)

    # 可选：归一化/对数化/scale（按你的数据类型开启）
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    # sc.pp.scale(adata_sc, max_value=10)
    # sc.pp.scale(adata_st, max_value=10)

    # 基因对齐
    shared_genes = adata_st.var.index.intersection(adata_sc.var.index)
    shared_genes = np.array(shared_genes)
    adata_sc = adata_sc[:, shared_genes].copy()
    adata_st = adata_st[:, shared_genes].copy()

    # 如果需要稠密（多数情况不用）
    if isinstance(adata_sc.X, csr_matrix):
        adata_sc.X = adata_sc.X.todense()
    if isinstance(adata_st.X, csr_matrix):
        adata_st.X = adata_st.X.todense()

    # # 保存对齐后的数据（方便复现）
    # adata_sc.write(adata_sc_save_path)
    # adata_st.write(adata_st_sava_path)

    # 数据集与编码器
    full_sc = AnnDataset(adata_sc, has_label=True, label_key=label_key)
    train_classes = full_sc.le.classes_
    mask = adata_st.obs[label_key].isin(train_classes)
    adata_st = adata_st[mask].copy()
    full_st = AnnDataset(adata_st, has_label=True, label_key=label_key)

    # 类别权重（不平衡处理）
    labels = full_sc.y.numpy()
    C = len(full_sc.le.classes_)
    cnt = Counter(labels)
    class_counts = np.array([cnt[i] for i in range(C)], dtype=np.float32)
    total = class_counts.sum()
    class_weights = total / (C * np.sqrt(class_counts))
    class_weights = torch.from_numpy(class_weights).to(device)

    # DataLoader（建议 drop_last=True 提升 SupCon 稳定性）
    loader_sc = DataLoader(full_sc, batch_size=2048, shuffle=True, num_workers=4, drop_last=True)
    loader_st = DataLoader(full_st, batch_size=2048, shuffle=False, num_workers=4)

    # —— 3) 模型 + 优化器 + Loss ——
    hidden_dims = [128, 64, 64]
    encoder = Encoder(in_dim=adata_sc.shape[1], hidden_dims=hidden_dims, p_drop=0.2).to(device)
    classifier = ClassifierHead(feat_dim=hidden_dims[-1], num_classes=C).to(device)

    criterion_cls = FocalLoss(gamma=2.0, weight=class_weights)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=3e-4,
        weight_decay=1e-6
    )

    # swanlab_init()
    best_val_acc = 0.0
    patience = 10
    wait = 0
    n_epochs = 7

    # —— 4) 训练循环 ——
    for epoch in range(1, n_epochs + 1):
        encoder.train()
        classifier.train()
        total_loss = 0.0
        total_cls = 0.0  # NEW: 记录分类损失
        all_preds, all_trues = [], []

        # —— Training ——
        for x, y in loader_sc:
            x, y = x.to(device), y.to(device)

            # 编码
            h = encoder(x)  # [B, D]
            logits = classifier(h)  # [B, C]

            # 分类损失
            loss_cls = criterion_cls(logits, y)

            # 联合损失
            loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_cls += loss_cls.item() * bs
            preds = logits.argmax(1).detach().cpu().numpy()
            all_preds.append(preds)
            all_trues.append(y.detach().cpu().numpy())

        avg_loss = total_loss / len(full_sc)
        avg_cls = total_cls / len(full_sc)
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        train_acc = accuracy_score(all_trues, all_preds)

        # —— Validation ——
        encoder.eval()
        classifier.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x_val, y_val in loader_st:
                x_val, y_val = x_val.to(device), y_val.to(device)
                h_val = encoder(x_val)
                logits_val = classifier(h_val)
                preds_val = logits_val.argmax(1).cpu().numpy()
                val_preds.append(preds_val)
                val_trues.append(y_val.cpu().numpy())

        val_acc = accuracy_score(np.concatenate(val_trues), np.concatenate(val_preds))

        print(
            f"Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.4f} "
            f"(cls {avg_cls:.4f}  "
            f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        # —— Early stopping on val_acc ——
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'label_encoder': full_sc.le,
                'class_names': full_sc.le.classes_.tolist(),
                'marker_genes': adata_sc.var_names.tolist(),
            }, model_save_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered based on validation accuracy.")
                break

    # 假设模型训练完毕，encoder已训练好，adata_sc是训练时使用的scRNA-seq数据
    prototypes = get_prototype_embeddings(encoder, adata_sc, label_key, device)

    # 打印每个细胞类型的原型嵌入
    for label, prototype in prototypes.items():
        print(f"Label: {label}, Prototype: {prototype}")

    return adata_sc, adata_st, prototypes

