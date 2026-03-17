import torch
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, d_input, num_heads=4, n_layers=1, dropout=0.0, residual=True):
        super(GATLayer, self).__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.conv_layers = nn.ModuleList()
        self.n_layers = n_layers

        for i in range(self.n_layers):
            self.conv_layers.append(GATConv(d_input, d_input, heads=num_heads, concat=False))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if self.residual:
            resid = x
        for layer in self.conv_layers:
            x = layer(x, edge_index)
            x = self.dropout(F.leaky_relu(x))
        if self.residual:
            x += resid
        return x


class GATEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent, num_heads=1, n_layers=1, dropout=0.0, residual=True):
        super().__init__()

        self.pre_fc = nn.Sequential(nn.Linear(d_input, d_hidden, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_hidden, d_latent, bias=True))
        self.post_fc = nn.Linear(d_latent, d_latent, bias=True)
        self.gat_layer = GATLayer(d_input=d_latent, num_heads=num_heads, n_layers=n_layers, dropout=dropout, residual=residual)

    def forward(self, x, e):
        z = self.pre_fc(x)
        z = self.gat_layer(z, e)
        z = self.post_fc(z)
        return z
