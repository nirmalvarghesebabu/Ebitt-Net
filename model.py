"""EBiTT-Net architecture (Eqs. 23-39); defaults follow Table 2."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from .graph import batch_edges


class EBiTTNet(nn.Module):
    def __init__(self, n_genes, edge_index, edge_attr, graph_hidden=64, gat_heads=4,
                 gat_layers=2, d_model=128, tf_heads=4, tf_layers=2, d_ff=256, dropout=0.2):
        super().__init__()
        self.k = n_genes
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_attr", edge_attr)
        # Eq. 23: node init h0 = phi(x)
        self.node_proj = nn.Linear(1, graph_hidden)
        head_dim = graph_hidden // gat_heads
        self.gats = nn.ModuleList([
            GATConv(graph_hidden, head_dim, heads=gat_heads, edge_dim=1, dropout=dropout)
            for _ in range(gat_layers)])
        self.act = nn.ELU()
        # Eq. 29: expression branch (+ learned gene-identity embedding: an addition, see README)
        self.expr_proj = nn.Linear(1, d_model)
        self.gene_emb = nn.Embedding(n_genes, d_model)
        # Eq. 30: gene-level fusion
        self.fuse = nn.Linear(d_model + graph_hidden, d_model)
        layer = nn.TransformerEncoderLayer(d_model, tf_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=False)
        self.transformer = nn.TransformerEncoder(layer, tf_layers)
        self.classifier = nn.Linear(d_model, 1)  # Eq. 39 (logit; sigmoid applied in loss/inference)

    def forward(self, x):                       # x: [B, k] standardised log-CPM
        B, k = x.shape
        xe = x.unsqueeze(-1)
        h = self.node_proj(xe).reshape(B * k, -1)
        ei, ea = batch_edges(self.edge_index, self.edge_attr, B, k)
        for gat in self.gats:
            h = self.act(gat(h, ei, ea))
        h = h.reshape(B, k, -1)
        z_expr = self.expr_proj(xe) + self.gene_emb.weight.unsqueeze(0)
        z0 = self.fuse(torch.cat([z_expr, h], dim=-1))
        zt = self.transformer(z0).mean(dim=1)   # pool to fixed-dim vector
        return self.classifier(zt).squeeze(-1)  # logits [B]
