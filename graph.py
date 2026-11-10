"""Fold-specific biological interaction graph (Eqs. 18-21)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch


def load_edges(path):
    df = pd.read_csv(path)
    df.columns = ["gene_a", "gene_b", "score"][: len(df.columns)]
    if "score" not in df:
        df["score"] = 1.0
    return df


def build_graph(selected_genes, edges_df, tau=0.4):
    """A_ij = s_ij if s_ij >= tau else 0 (Eq. 20). Selected genes with no edge stay as isolated nodes.

    `score` must be on a 0-1 scale (divide STRING combined_score by 1000).
    Returns edge_index [2,E] (both directions), edge_attr [E,1].
    """
    idx = {g: i for i, g in enumerate(selected_genes)}
    e = edges_df[edges_df.gene_a.isin(idx) & edges_df.gene_b.isin(idx) & (edges_df.score >= tau)]
    e = e[e.gene_a != e.gene_b].drop_duplicates(["gene_a", "gene_b"])
    src = [idx[a] for a in e.gene_a] + [idx[b] for b in e.gene_b]
    dst = [idx[b] for b in e.gene_b] + [idx[a] for a in e.gene_a]
    w = list(e.score) * 2
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(w, dtype=torch.float32).reshape(-1, 1)
    n_iso = len(selected_genes) - len(set(src))
    return edge_index, edge_attr, {"nodes": len(selected_genes), "edges": len(src) // 2, "isolated": n_iso}


def batch_edges(edge_index, edge_attr, batch_size, k):
    """Replicate the shared fold graph for every sample in a batch."""
    off = (torch.arange(batch_size) * k).view(-1, 1, 1)
    ei = (edge_index.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1)
    ea = edge_attr.repeat(batch_size, 1)
    return ei, ea
