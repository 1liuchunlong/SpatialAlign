# SpatialAlign

Spatial transcriptomics cell type annotation via scRNA-seq alignment.

## 复现 nsclc_1128 工作流

本仓库完全复现 `0729/nsclc_1128.ipynb` 的完整流程，包含：

1. **Stage 1**: 训练 scRNA-seq 分类器 (MLP)
2. **伪标签推断**: 使用 Stage 1 模型对空间转录组数据进行细胞类型预测
3. **Stage 2**: 基于伪标签训练空间 encoder (GAT)，使用 UOT 刷新伪标签

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据

数据已迁移至 `data/` 目录，本地完全独立、开箱即用：

- `sc_nsclc.h5ad` - scRNA-seq 数据
- `lung9_rep1_15types.h5ad` - 空间转录组数据
- `X_umap_df.csv` - 预计算的 UMAP 坐标（用于可视化）

> 若从 GitHub 克隆，因大文件未纳入版本库，需自行将上述数据放入 `data/` 目录。

## 运行

在 `SpatialAlign/` 根目录下启动 Jupyter，运行 `demo-nsclc.ipynb`：

```bash
cd SpatialAlign
jupyter notebook demo-nsclc.ipynb
```

## 目录结构

```
SpatialAlign/
├── data/                 # 数据文件（已包含）
│   ├── sc_nsclc.h5ad
│   ├── lung9_rep1_15types.h5ad
│   └── X_umap_df.csv
├── demo-nsclc.ipynb      # 主复现 notebook
├── requirements.txt
├── README.md
└── SpatialAlign/         # Python 包
    ├── __init__.py
    ├── train_sc_stage1.py    # Stage 1 训练
    ├── pseudo_labeling_impl.py  # 伪标签推断
    ├── train_stage2.py       # Stage 2 训练
    ├── dnn.py               # Encoder / ClassifierHead
    ├── losses.py            # FocalLoss, cross_modal_supcon_with_queue
    ├── mydatasets.py        # SCRNADataset, STSpatialDataset
    ├── gat_encoder.py       # GATEncoder
    └── ultils.py            # location_to_edge, augment_rare_cells
```
