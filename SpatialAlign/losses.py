import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce)
        loss = (1 - p_t) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def cross_modal_supcon_with_queue(
    zr, zs, yr, ys, tau=0.2, eps=1e-12,
    pos_bank=None,   # {'r': zr_pos [P,D], 's': zs_pos [P,D], 'y': y_pos [P]}
    neg_bank=None    # {'r': zr_neg [Nr,D] (可选), 's': zs_neg [Ns,D] (可选)}
):
    """
    跨模态 SupCon（支持额外的"队列正样对"和"负样 bank"）
    """
    device = zr.device
    zr = F.normalize(zr, dim=1)
    zs = F.normalize(zs, dim=1)

    Z = torch.cat([zr, zs], dim=0)                  # [2B, D]
    Y = torch.cat([yr, ys], dim=0).to(device)       # [2B]
    M = torch.cat([torch.zeros_like(yr), torch.ones_like(ys)], dim=0).to(device)  # 0:r, 1:s
    N = Z.size(0)

    # ---- in-batch 相似度 & 分母（不含自对角）----
    sim_in = (Z @ Z.t()) / tau                      # [2B, 2B]
    eye = torch.eye(N, device=device)
    exp_in = torch.exp(sim_in) * (1 - eye)          # 去掉自对角
    denom = exp_in.sum(dim=1, keepdim=True)         # [2B,1]

    # ---- in-batch 跨模态正例掩码（同类 & 异模态）----
    pos_mask_in = (M.unsqueeze(1) != M.unsqueeze(0)) & (Y.unsqueeze(1) == Y.unsqueeze(0))
    pos_mask_in = pos_mask_in.float()
    pos_mask_in.fill_diagonal_(0.0)

    log_prob_in = (sim_in - torch.log(denom + eps))   # [2B, 2B]

    # ---- 负样 bank：只加到分母 ----
    if neg_bank is not None:
        if 'r' in neg_bank and neg_bank['r'] is not None and neg_bank['r'].numel() > 0:
            zr_neg = F.normalize(neg_bank['r'].to(device), dim=1)    # [Nr, D]
            denom += torch.exp((Z @ zr_neg.t()) / tau).sum(dim=1, keepdim=True)
        if 's' in neg_bank and neg_bank['s'] is not None and neg_bank['s'].numel() > 0:
            zs_neg = F.normalize(neg_bank['s'].to(device), dim=1)    # [Ns, D]
            denom += torch.exp((Z @ zs_neg.t()) / tau).sum(dim=1, keepdim=True)

    # 分母变了，in-batch 的 log_prob 也要用新分母重算
    log_prob_in = (sim_in - torch.log(denom + eps))

    # ---- 额外正样对（成对队列）----
    add_num = torch.zeros((N, 1), device=device)
    add_cnt = torch.zeros((N, 1), device=device)

    if pos_bank is not None and all(k in pos_bank for k in ('r', 's', 'y')):
        zr_pos = pos_bank['r']
        zs_pos = pos_bank['s']
        y_pos  = pos_bank['y']
        if zr_pos is not None and zs_pos is not None and y_pos is not None \
           and zr_pos.numel() > 0 and zs_pos.numel() > 0 and y_pos.numel() > 0:
            zr_pos = F.normalize(zr_pos.to(device), dim=1)
            zs_pos = F.normalize(zs_pos.to(device), dim=1)
            y_pos  = y_pos.to(device)

            idx_r = (M == 0).nonzero(as_tuple=False).squeeze(1)
            if idx_r.numel() > 0 and zs_pos.numel() > 0:
                sim_rp = (Z[idx_r] @ zs_pos.t()) / tau
                match_r = (Y[idx_r].unsqueeze(1) == y_pos.unsqueeze(0)).float()
                log_denom_r = torch.log(denom[idx_r] + eps)
                log_prob_rp = sim_rp - log_denom_r
                add_num[idx_r] += (log_prob_rp * match_r).sum(dim=1, keepdim=True)
                add_cnt[idx_r] += match_r.sum(dim=1, keepdim=True)
                denom[idx_r] += torch.exp(sim_rp).sum(dim=1, keepdim=True)

            idx_s = (M == 1).nonzero(as_tuple=False).squeeze(1)
            if idx_s.numel() > 0 and zr_pos.numel() > 0:
                sim_sp = (Z[idx_s] @ zr_pos.t()) / tau
                match_s = (Y[idx_s].unsqueeze(1) == y_pos.unsqueeze(0)).float()
                log_denom_s = torch.log(denom[idx_s] + eps)
                log_prob_sp = sim_sp - log_denom_s
                add_num[idx_s] += (log_prob_sp * match_s).sum(dim=1, keepdim=True)
                add_cnt[idx_s] += match_s.sum(dim=1, keepdim=True)
                denom[idx_s] += torch.exp(sim_sp).sum(dim=1, keepdim=True)

    # ---- 聚合 ----
    pos_sum_in = (log_prob_in * pos_mask_in).sum(dim=1, keepdim=True)
    pos_cnt_in = (pos_mask_in.sum(dim=1, keepdim=True))

    pos_sum_all = pos_sum_in + add_num
    pos_cnt_all = pos_cnt_in + add_cnt

    valid = (pos_cnt_all.squeeze(1) > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    mean_log_prob_pos = pos_sum_all[valid] / (pos_cnt_all[valid] + eps)
    loss = - mean_log_prob_pos.mean()
    return loss
