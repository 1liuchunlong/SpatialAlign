import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import scanpy as sc

from mydatasets import SCRNADataset, STSpatialDataset, spatialCollate
from dnn import Encoder, ClassifierHead
from gat_encoder import GATEncoder
from losses import cross_modal_supcon_with_queue  # 已在 losses 里实现
from sklearn.metrics.pairwise import cosine_similarity
import ot
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    classification_report,
)
import random


def def_cycle(loader):
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)


def set_seed(seed=2026):
    """设置所有随机源的种子以确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 确定性与性能设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_sc_prototypes(adata_sc, sc_encoder, le, device, normalize=False):
    """
    返回:
      proto_raw: (K, D) 每个类别的均值向量
      counts:    (K,)   各类别样本数
    """
    sc_encoder.eval()
    X = adata_sc.X
    if not isinstance(X, np.ndarray):  # 兼容 csr_matrix
        X = X.A
    X = torch.from_numpy(X).float().to(device)
    Z = sc_encoder(X)  # (N_sc, D)

    # 把 obs['celltype'] 映射成整数标签
    y_str = adata_sc.obs['celltype'].to_numpy()
    y_int = torch.as_tensor(le.transform(y_str), device=device, dtype=torch.long)

    K = len(le.classes_)
    D = Z.size(1)
    proto_sum = torch.zeros(K, D, device=device)
    counts = torch.zeros(K, device=device)

    for k in range(K):
        idx = (y_int == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:  # 没有该类
            continue
        proto_sum[k] = Z.index_select(0, idx).mean(dim=0)
        counts[k] = idx.numel()

    proto_raw = proto_sum
    if normalize:
        proto_raw = F.normalize(proto_raw, dim=1)

    return proto_raw, counts


def build_sc_label_index(adata_sc, label_key='celltype'):
    sc_labels = adata_sc.obs[label_key].to_numpy()
    uniq = np.unique(sc_labels)
    return {lab: np.where(sc_labels == lab)[0] for lab in uniq}


# ===================== 成对队列（环形） =====================
def init_pair_queue(device, feat_dim=64, size=32768):
    q = {
        'size': size,
        'feat_dim': feat_dim,
        # 只用负样 bank（若未来要用正样，可保留这些键）
        'neg_sc': torch.randn(size, feat_dim, device=device),
        'neg_st': torch.randn(size, feat_dim, device=device),
        'neg_y': torch.zeros(size, dtype=torch.long, device=device),
        'neg_ptr': 0,
    }
    return q


def _ring_enqueue(buf, ptr_name, x):
    B = x.shape[0]
    size = buf['size']
    ptr = buf[ptr_name]
    end = ptr + B
    if end <= size:
        return ptr, slice(ptr, end), end % size
    first = size - ptr
    return ptr, (slice(ptr, size), slice(0, B - first)), (B - first)


@torch.no_grad()
def enqueue_neg_pairs(buf, zr_neg, zs_neg, y_neg):
    # zr_neg/zs_neg: [K,D]（raw 特征），y_neg: [K]
    zr_neg = zr_neg.detach()
    zs_neg = zs_neg.detach()
    y_neg = y_neg.detach()
    _, seg, new_ptr = _ring_enqueue(buf, 'neg_ptr', zr_neg)
    if isinstance(seg, tuple):
        s1, s2 = seg
        cut = s1.stop - s1.start
        buf['neg_sc'][s1] = zr_neg[:cut]
        buf['neg_sc'][s2] = zr_neg[cut:]
        buf['neg_st'][s1] = zs_neg[:cut]
        buf['neg_st'][s2] = zs_neg[cut:]
        buf['neg_y'][s1] = y_neg[:cut]
        buf['neg_y'][s2] = y_neg[cut:]
    else:
        buf['neg_sc'][seg] = zr_neg
        buf['neg_st'][seg] = zs_neg
        buf['neg_y'][seg] = y_neg
    buf['neg_ptr'] = new_ptr


@torch.no_grad()
def sample_from_pairQ_neg_only(pairQ, K_neg_global=128, use_window=True):
    """仅采负样 bank 的小配额（进入分母）。"""
    neg_bank = None
    N_total = int(pairQ['neg_sc'].size(0))
    if N_total > 0 and K_neg_global > 0:
        N_take = min(N_total, K_neg_global)
        device = pairQ['neg_sc'].device
        if use_window:
            ptr = pairQ.get('neg_sample_ptr', 0)
            idx = (torch.arange(N_take, device=device) + ptr) % N_total
            pairQ['neg_sample_ptr'] = int((ptr + N_take) % N_total)
        else:
            idx = torch.randint(0, N_total, (N_take,), device=device)
        neg_bank = {
            'r': pairQ['neg_sc'].index_select(0, idx),
            's': pairQ['neg_st'].index_select(0, idx),
        }
    return neg_bank


# ===================== 稀有 ST 初始化：分层 Top‑K 负样 =====================
@torch.no_grad()
def init_neg_bank_with_rare_ST(
    zs_rare, ys_rare, le,
    zr_all_raw, zr_all_n, sc_label2idx_t,
    pairQ,
    t_per_class=5, cap_per_anchor=1000
):
    zs_rare_n = F.normalize(zs_rare, dim=1)
    all_labs = list(sc_label2idx_t.keys())

    neg_sc_feats, neg_st_feats, neg_labels = [], [], []
    for i in range(zs_rare_n.size(0)):
        st_feat_n = zs_rare_n[i]
        st_feat = zs_rare[i]
        y_int = ys_rare[i].item()
        lab_name = le.classes_[y_int]

        neg_idx_chunks = []
        for other_lab in all_labs:
            if other_lab == lab_name:  # 只选异类
                continue
            idx_diff = sc_label2idx_t.get(other_lab, None)
            if idx_diff is None or idx_diff.numel() == 0:
                continue
            zr_diff_n = zr_all_n.index_select(0, idx_diff)  # [Nc,D]
            sim_c = torch.mv(zr_diff_n, st_feat_n)
            t = min(t_per_class, sim_c.numel())
            if t > 0:
                top_local = torch.topk(sim_c, k=t, largest=True).indices
                neg_idx_chunks.append(idx_diff.index_select(0, top_local))

        if len(neg_idx_chunks) == 0:
            continue

        idx_neg_all = torch.cat(neg_idx_chunks, dim=0)
        if idx_neg_all.numel() > cap_per_anchor:
            zr_neg_all_n = zr_all_n.index_select(0, idx_neg_all)
            sim_all = torch.mv(zr_neg_all_n, st_feat_n)
            keep = torch.topk(sim_all, k=cap_per_anchor, largest=True).indices
            idx_neg_all = idx_neg_all.index_select(0, keep)

        k_neg = int(idx_neg_all.numel())
        if k_neg > 0:
            neg_sc_feats.append(zr_all_raw.index_select(0, idx_neg_all))
            neg_st_feats.append(st_feat.expand(k_neg, -1))
            neg_labels.append(ys_rare[i].expand(k_neg))

    added = 0
    if len(neg_sc_feats) > 0:
        enqueue_neg_pairs(
            pairQ,
            torch.cat(neg_sc_feats, dim=0),
            torch.cat(neg_st_feats, dim=0),
            torch.cat(neg_labels, dim=0)
        )
        added = int(torch.cat(neg_labels, dim=0).numel())
    return added


# ===================== 在线 hard negative mining（每 step 可调用） =====================
@torch.no_grad()
def mine_and_enqueue_hard_negs_for_batch(
    zs, ys, le,
    zr_all_raw, zr_all_n, sc_label2idx_t,
    pairQ, t_per_class=2, cap_per_anchor=64
):
    zs_n = F.normalize(zs, dim=1)
    all_labs = list(sc_label2idx_t.keys())
    neg_sc_list, neg_st_list, neg_y_list = [], [], []

    for i in range(zs_n.size(0)):
        st_feat_n = zs_n[i]
        st_feat = zs[i]
        y_int = ys[i].item()
        lab_name = le.classes_[y_int]

        neg_idx_chunks = []
        for other_lab in all_labs:
            if other_lab == lab_name:
                continue
            idx_diff = sc_label2idx_t.get(other_lab, None)
            if idx_diff is None or idx_diff.numel() == 0:
                continue
            zr_diff_n = zr_all_n.index_select(0, idx_diff)
            sim_c = torch.mv(zr_diff_n, st_feat_n)
            t = min(t_per_class, sim_c.numel())
            if t > 0:
                top_local = torch.topk(sim_c, k=t, largest=True).indices
                neg_idx_chunks.append(idx_diff.index_select(0, top_local))

        if len(neg_idx_chunks) == 0:
            continue

        idx_neg_all = torch.cat(neg_idx_chunks, dim=0)
        if idx_neg_all.numel() > cap_per_anchor:
            zr_neg_all_n = zr_all_n.index_select(0, idx_neg_all)
            sim_all = torch.mv(zr_neg_all_n, st_feat_n)
            keep = torch.topk(sim_all, k=cap_per_anchor, largest=True).indices
            idx_neg_all = idx_neg_all.index_select(0, keep)

        k_neg = int(idx_neg_all.numel())
        if k_neg > 0:
            neg_sc_list.append(zr_all_raw.index_select(0, idx_neg_all))
            neg_st_list.append(st_feat.expand(k_neg, -1))
            neg_y_list.append(ys[i].expand(k_neg))

    if len(neg_sc_list) > 0:
        enqueue_neg_pairs(
            pairQ,
            torch.cat(neg_sc_list, dim=0),
            torch.cat(neg_st_list, dim=0),
            torch.cat(neg_y_list, dim=0)
        )
        return int(torch.cat(neg_y_list, dim=0).numel())
    return 0


# >>> NEW: UOT 伪标签刷新
@torch.no_grad()
def get_prototype_embeddings(encoder, adata_sc, device, le, label_key='celltype'):
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

    # 使用 le.classes_ 保证原型嵌入的顺序与 label_encoder 一致
    ordered_prototypes = []
    for class_name in le.classes_:
        ordered_prototypes.append(prototypes[class_name])

    return np.array(ordered_prototypes)


@torch.no_grad()
def refresh_pseudo_labels_cosine(
    adata_sc, adata_st,
    sc_encoder,
    st_encoder, le, device,
    label_key='celltype'
):
    """
    用 cosine 相似度 + 原型，给 ST 重算伪标签 & 置信度
    """

    # 1) sc 端类别原型 (K, D)
    sc_embeddings = get_prototype_embeddings(sc_encoder, adata_sc, device, le)

    # 2) 算所有 ST 的 embedding（一次性）
    spatial_collate = spatialCollate(knn=3)
    ds_st = STSpatialDataset(
        adata=adata_st,
        pseudo_label_key=None,
        le=le
    )
    loader_st = DataLoader(
        ds_st, batch_size=512, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=spatial_collate
    )

    st_encoder.eval()
    all_st_embeddings = []

    with torch.no_grad():
        for x_st, y_st, edge_idx in loader_st:
            x_st = x_st.to(device)
            edge_idx = edge_idx.to(device)
            zs = st_encoder(x_st, edge_idx)  # (B, D)
            all_st_embeddings.append(zs.cpu().numpy())

    all_st_embeddings = np.concatenate(all_st_embeddings, axis=0)  # (N_st, D)

    # 3) cosine 相似度 → 伪标签 + 置信度
    sim = cosine_similarity(all_st_embeddings, sc_embeddings)  # (N_st, K)
    pseudo_labels = np.argmax(sim, axis=1)
    confidences = np.max(sim, axis=1)

    # 4) 写回 AnnData
    adata_st.obs['pseudo_label'] = le.inverse_transform(pseudo_labels)
    adata_st.obs['pseudo_confidence'] = confidences

    # 优先用 label_key，如果没有就试 'celltype'
    real_labels = None
    if label_key in adata_st.obs:
        real_labels = adata_st.obs[label_key]
    elif 'celltype' in adata_st.obs:
        real_labels = adata_st.obs['celltype']

    if real_labels is not None:
        # 注意保证 index 对齐
        y_true = real_labels
        y_pred = adata_st.obs['pseudo_label'].loc[y_true.index]

        acc = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average='weighted')
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

        print(f"[cosine] Pseudo-label Accuracy (ACC): {acc:.4f}")
        print(f"[cosine] Weighted F1: {f1_w:.4f}")
        print(f"[cosine] ARI: {ari:.4f}")
        print(f"[cosine] AMI: {ami:.4f}")
        print(f"[cosine] NMI: {nmi:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, digits=4))
    else:
        print("Warning: real_labels is None，既没有找到 label_key，也没有 'celltype'。")

    return adata_st


def refresh_pseudo_labels_uot(
    adata_sc, adata_st,
    sc_encoder,
    st_encoder, le, device,
    label_key="celltype",
    epsilon=0.005,
    reg_m=0.1,
):
    """
    用 OT（cosine cost + unbalanced Sinkhorn）基于 sc 原型，给 ST 重算伪标签 & 置信度

    - 左边缘 a: ST spots 均匀分布
    - 右边缘 b: 来自 adata_sc 中 label_key 的类型频数 (type prior)
    - cost: 1 - cosine( z_st, prototype )
    - 用 unbalanced OT 得到 gamma，然后对每一行归一化成 fraction
      -> 行内最大 fraction 作为 pseudo_confidence
    """

    # ===================== 1) sc 端类别原型 =====================
    # sc_embeddings: [K, D]，假设顺序与 le.classes_ 对齐
    sc_embeddings = get_prototype_embeddings(sc_encoder, adata_sc, device, le)  # [K, D]

    if isinstance(sc_embeddings, torch.Tensor):
        sc_embeddings = sc_embeddings.detach().cpu().numpy()
    sc_embeddings = np.asarray(sc_embeddings)

    type_names = np.asarray(le.classes_)  # [K]

    # type prior b：用 adata_sc 中的真实 celltype 频数
    type_counts = (
        adata_sc.obs[label_key]
        .value_counts()
        .reindex(type_names)
        .fillna(0)
    )
    b = type_counts.to_numpy().astype(np.float64)
    if b.sum() <= 0:
        # 极端保护：如果全是 0，就退回均匀 prior
        b = np.ones_like(b, dtype=np.float64)
    b = b / b.sum()  # [K]

    # ===================== 2) ST 端 embedding（一次性） =====================
    spatial_collate = spatialCollate(knn=3)
    ds_st = STSpatialDataset(
        adata=adata_st,
        pseudo_label_key=None,
        le=le
    )
    loader_st = DataLoader(
        ds_st, batch_size=512, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=spatial_collate
    )

    st_encoder.eval()
    all_st_embeddings = []

    with torch.no_grad():
        for x_st, y_st, edge_idx in loader_st:
            x_st = x_st.to(device)
            edge_idx = edge_idx.to(device)
            zs = st_encoder(x_st, edge_idx)  # (B, D)
            all_st_embeddings.append(zs.cpu().numpy())

    all_st_embeddings = np.concatenate(all_st_embeddings, axis=0)  # (N_st, D)
    all_st_embeddings = all_st_embeddings.astype(np.float64)
    n_spots = all_st_embeddings.shape[0]

    # ===================== 3) 构造 OT 的 cost 与边缘 =====================
    # cost = 1 - cosine，归一化到 [0, 1]
    sim = cosine_similarity(all_st_embeddings, sc_embeddings)  # (N_st, K)
    M = 1.0 - sim
    M = M.astype(np.float64)
    M = M / (M.max() + 1e-8)

    # 左边缘 a: ST spot 均匀分布
    a = np.ones(n_spots, dtype=np.float64)
    a = a / a.sum()

    # ===================== 4) 跑 unbalanced OT =====================
    gamma = ot.unbalanced.sinkhorn_knopp_unbalanced(
        a, b, M,
        reg=epsilon,
        reg_m=reg_m,
    )  # [N_st, K]

    # 行归一化成 fraction：每个 spot 的 cell-type 组成
    row_sum = gamma.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    frac = gamma / row_sum  # [N_st, K]

    # ===================== 5) 写回 AnnData：伪标签 & 置信度 =====================
    adata_st = adata_st.copy()

    # 每一行最大 fraction 对应的类型作为伪标签
    pseudo_idx = frac.argmax(axis=1)  # [N_st]
    pseudo_labels = type_names[pseudo_idx]  # label 字符串
    confidences = frac.max(axis=1)  # 最大 fraction

    adata_st.obs['pseudo_label'] = pseudo_labels
    adata_st.obs['pseudo_confidence'] = confidences

    # ===================== 6) 可选评估 =====================
    real_labels = None
    if label_key in adata_st.obs:
        real_labels = adata_st.obs[label_key]
    elif 'celltype' in adata_st.obs:
        real_labels = adata_st.obs['celltype']

    if real_labels is not None:
        # 注意保证 index 对齐
        y_true = real_labels
        y_pred = adata_st.obs['pseudo_label'].loc[y_true.index]

        acc = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average='weighted')
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

        print(f"[OT] Pseudo-label Accuracy (ACC): {acc:.4f}")
        print(f"[OT] Weighted F1: {f1_w:.4f}")
        print(f"[OT] ARI: {ari:.4f}")
        print(f"[OT] AMI: {ami:.4f}")
        print(f"[OT] NMI: {nmi:.4f}")
        print("Classification report:")
        print(classification_report(y_true, y_pred, digits=4))
    else:
        print("Warning: real_labels is None，既没有找到 label_key，也没有 'celltype'。")

    return adata_st


def print_confidence_stats(adata_st, epoch):
    """
    按类别打印 pseudo_confidence 的分布情况：
    - 样本数
    - 均值、标准差
    - 分位数：p25, p50, p75
    """
    if 'pseudo_label' not in adata_st.obs or 'pseudo_confidence' not in adata_st.obs:
        print(f"[Epoch {epoch+1}] pseudo_label / pseudo_confidence 不在 adata_st.obs 中，跳过统计")
        return

    preds = adata_st.obs['pseudo_label'].to_numpy()
    conf = adata_st.obs['pseudo_confidence'].to_numpy()

    print(f"\n===== Epoch {epoch+1} pseudo_confidence stats by class =====")
    for lab in np.unique(preds):
        idx = (preds == lab)
        conf_c = conf[idx]
        if conf_c.size == 0:
            continue
        mean = conf_c.mean()
        std = conf_c.std()
        p25, p50, p75 = np.percentile(conf_c, [25, 50, 75])
        cmin, cmax = conf_c.min(), conf_c.max()

        print(
            f"Class {lab:>15s} | n={conf_c.size:6d} | "
            f"mean={mean:.3f}, std={std:.3f}, "
            f"p25={p25:.3f}, p50={p50:.3f}, p75={p75:.3f}, "
            f"min={cmin:.3f}, max={cmax:.3f}"
        )
    print("========================================================\n")


# ===================== 主训练函数 =====================
def train_for_stage2(
    mlp_stage1_path, adata_sc, adata_st,
    gat_pt_savepath, mlp_stage2_save_path,
    epochs=3,
    K_NEG_BANK=256,
    T_NEG_INIT=5, CAP_INIT=2000,
    T_NEG_ONLINE=2, CAP_ONLINE=128
):
    set_seed()
    device = torch.device('cuda')

    # ---- 载入 ckpt / 数据 ----
    ckpt = torch.load(mlp_stage1_path, map_location=device)
    class_names = ckpt['class_names']
    marker_genes = ckpt['marker_genes']
    le = ckpt['label_encoder']

    # ---- sc encoder/head（冻结）----
    sc_encoder = Encoder(in_dim=len(marker_genes), hidden_dims=[128, 64, 64], p_drop=0.2).to(device)
    sc_encoder.load_state_dict(ckpt['encoder_state_dict'])
    for p in sc_encoder.parameters():
        p.requires_grad = False

    sc_head = ClassifierHead(64, len(class_names)).to(device)
    sc_head.load_state_dict(ckpt['classifier_state_dict'])
    for p in sc_head.parameters():
        p.requires_grad = False

    # ---- ST encoder/head（可训练）----
    st_encoder = GATEncoder(
        d_input=adata_st.shape[1],
        d_hidden=64,
        d_latent=64,
        num_heads=4,
        n_layers=1,
        dropout=0.1,
        residual=True
    ).to(device)

    st_head = ClassifierHead(64, len(class_names)).to(device)
    optimizer = optim.Adam(list(st_encoder.parameters()) + list(st_head.parameters()), lr=5e-4)

    # ---- CATF 伪标签 + 过滤 ----
    Q_START, Q_END = 10.0, 5.0
    N_MIN = 20
    CONF_MIN = 0.30

    spatial_collate_fn = spatialCollate(knn=3)

    def build_st_loader(epoch=0):
        print_confidence_stats(adata_st, epoch)
        new_preds = adata_st.obs['pseudo_label'].to_numpy()
        new_conf = adata_st.obs['pseudo_confidence'].to_numpy()

        # ---- 动态调整过滤阈值 q_base（每个 epoch 都更新）----
        if epochs <= 1:
            # 只有 1 轮的话，就直接用 Q_END 或 Q_START，看你需求
            q_base = Q_END
        else:
            # epoch 从 0 到 epochs-1
            t = epoch / (epochs - 1)  # t ∈ [0, 1]
            q_base = (1 - t) * Q_START + t * Q_END

        q_base = float(np.clip(q_base, 0.0, 100.0))
        mask = np.zeros(len(adata_st), dtype=bool)
        class_stats = {}
        rare_idx_bucket = []

        for label in np.unique(new_preds):
            # 该类所有样本
            idx_all = np.where(new_preds == label)[0]
            if len(idx_all) == 0:
                continue

            # 先不立即用 CONF_MIN 过滤，先判断是否稀有类
            n_total = len(idx_all)  # 该类原始总数

            if n_total < N_MIN:  # 稀有类：不进行 CONF_MIN 过滤，直接全部保留
                mask[idx_all] = True
                class_stats[label] = (n_total, n_total)
                rare_idx_bucket.extend(idx_all.tolist())
                continue

            # 非稀有类：先做 CONF_MIN 过滤得到候选集
            idx = idx_all[new_conf[idx_all] >= CONF_MIN]
            if len(idx) == 0:
                # 非稀有类在 conf>=CONF_MIN 后一个都没有，直接略过
                class_stats[label] = (0, n_total)
                continue

            # 下面的 n_c / conf_c 都是基于“候选集”（conf>=CONF_MIN）来算
            conf_c = new_conf[idx]

            # 这里其实 n_c 一定 >=1，且因为非稀有类已经跳过 N_MIN 检查
            # 所以直接进入百分位数筛选
            cutoff = np.percentile(conf_c, q_base)
            keep_idx = idx[conf_c >= cutoff]

            mask[keep_idx] = True
            class_stats[label] = (len(keep_idx), n_total)

        adata_high = adata_st[mask].copy()
        ds_st = STSpatialDataset(adata_high, pseudo_label_key='pseudo_label', le=le)
        loader = DataLoader(
            ds_st, batch_size=512, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=spatial_collate_fn
        )

        kept_num = int(mask.sum())
        kept_ratio = kept_num / len(mask) * 100.0
        print(
            f"[CATF] Epoch {epoch+1}: kept {kept_num}/{len(mask)} ({kept_ratio:.2f}%), "
            f"q_base={q_base:.1f}%, conf_min={CONF_MIN:.2f}"
        )
        for label, (kept, total) in class_stats.items():
            print(f"  Class {label}: kept {kept}/{total}")

        return loader, np.array(rare_idx_bucket, dtype=int)

    # ---- 建索引 & 队列 & 首次 CATF ----
    sc_label2idx = build_sc_label_index(adata_sc, label_key='celltype')
    pairQ = init_pair_queue(device=device, feat_dim=64, size=32768)

    loader_st, rare_idx_prev = build_st_loader(epoch=0)

    # 预编码全部 sc 特征（raw + norm）——用于负样挖掘
    with torch.no_grad():
        X_sc_all = adata_sc.X
        if not isinstance(X_sc_all, np.ndarray):
            X_sc_all = X_sc_all.A
        zr_all_raw = sc_encoder(torch.from_numpy(X_sc_all).float().to(device))  # [N_sc, D] raw
    zr_all_n = F.normalize(zr_all_raw, dim=1)

    sc_label2idx_t = {
        lab: torch.as_tensor(idxs, device=device, dtype=torch.long)
        for lab, idxs in sc_label2idx.items() if len(idxs) > 0
    }

    # —— 新增：计算 sc 端类别原型（K×D）——
    proto_sc_raw, _ = compute_sc_prototypes(adata_sc, sc_encoder, le, device, normalize=False)
    proto_sc_raw.requires_grad_(False)

    # 稀有 ST 做一次性负样初始化
    if rare_idx_prev is not None and rare_idx_prev.size > 0:
        adata_rare = adata_st[rare_idx_prev].copy()
        tmp_ds = STSpatialDataset(adata_rare, pseudo_label_key='pseudo_label', le=le)
        tmp_dl = DataLoader(
            tmp_ds, batch_size=512, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=spatialCollate(knn=3)
        )
        zs_list, ys_list = [], []
        with torch.no_grad():
            st_encoder.eval()
            for Xs_cpu, ys_cpu, edge_idx_cpu in tmp_dl:
                zs_list.append(st_encoder(Xs_cpu.to(device), edge_idx_cpu.to(device)))
                ys_list.append(ys_cpu.squeeze(-1).to(device))
            st_encoder.train()
        zs_rare = torch.cat(zs_list, 0)
        ys_rare = torch.cat(ys_list, 0)

        added0 = init_neg_bank_with_rare_ST(
            zs_rare, ys_rare, le,
            zr_all_raw, zr_all_n, sc_label2idx_t,
            pairQ, t_per_class=T_NEG_INIT, cap_per_anchor=CAP_INIT
        )
        print(
            f"[PairQ][INIT StratTopK-NEG] +{added0} neg "
            f"(t_per_cls={T_NEG_INIT}, cap={CAP_INIT})"
        )

    # ===================== 训练循环 =====================
    for epoch in range(epochs):
        st_encoder.train()
        st_head.train()

        total_loss = total_cross = 0.0

        for Xs_cpu, ys_cpu, edge_idx_cpu in loader_st:
            Xs = Xs_cpu.to(device)
            ys = ys_cpu.squeeze(-1).to(device)  # (B,)
            edge_idx = edge_idx_cpu.to(device)

            # —— 用类别原型替代逐步采样的 sc 特征 ——
            with torch.no_grad():
                zr = proto_sc_raw.index_select(0, ys)  # (B, D)
            yr = ys  # 两模态正对齐的标签一致

            zs = st_encoder(Xs, edge_idx)
            logits_s = st_head(zs)

            # —— 在线 hard negative mining（环形入队）——
            _ = mine_and_enqueue_hard_negs_for_batch(
                zs, ys, le,
                zr_all_raw, zr_all_n, sc_label2idx_t,
                pairQ, t_per_class=T_NEG_ONLINE, cap_per_anchor=CAP_ONLINE
            )

            # —— 从队列取一小撮负样进分母 ——
            neg_bank = sample_from_pairQ_neg_only(pairQ, K_NEG_BANK, use_window=True)

            # —— 对比 + 分类 ——
            cross_loss = cross_modal_supcon_with_queue(
                zr, zs, yr, ys, tau=0.1,
                pos_bank=None, neg_bank=neg_bank
            )

            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_cross += float(cross_loss.item())

        n_batches = len(loader_st)
        print(
            f"Epoch {epoch+1:02d} | avg total={total_loss/n_batches:.4f} "
            f"| cross={total_cross/n_batches:.4f}"
        )
        if (epoch + 1) >= 2:
            adata_st = refresh_pseudo_labels_uot(
                adata_sc=adata_sc, adata_st=adata_st,
                sc_encoder=sc_encoder,
                st_encoder=st_encoder,
                le=le, device=device
            )
            loader_st, rare_idx_prev = build_st_loader(epoch=epoch)

    # ===================== 保存 =====================
    torch.save({
        'epoch': epoch + 1,
        'gat_state_dict': st_encoder.state_dict(),
        'marker_genes': marker_genes,
        'class_names': class_names
    }, gat_pt_savepath)

    torch.save({
        'encoder_state_dict': sc_encoder.state_dict(),
        'label_encoder': le,
        'class_names': class_names,
        'marker_genes': marker_genes
    }, mlp_stage2_save_path)

    print(f" saved GATEncoder checkpoint at epoch {epoch+1} to {gat_pt_savepath}")
    return adata_st

