"""Post-hoc Integrated Gradients (Eqs. 46-51). Never used in training."""
from __future__ import annotations
import numpy as np
import torch


def integrated_gradients(model, x, baseline=None, steps=50, device="cpu"):
    """IG of the model logit w.r.t. each selected gene. x: [k]. Baseline default: zeros
    (= the training-fold mean expression, because inputs are standardised)."""
    model.eval()
    x = torch.tensor(x, dtype=torch.float32, device=device)
    base = torch.zeros_like(x) if baseline is None else torch.tensor(baseline, dtype=torch.float32, device=device)
    alphas = torch.arange(1, steps + 1, device=device, dtype=torch.float32).view(-1, 1) / steps
    pts = (base + alphas * (x - base)).requires_grad_(True)       # [m,k]
    out = model(pts).sum()
    grads, = torch.autograd.grad(out, pts)
    return ((x - base) * grads.mean(0)).detach().cpu().numpy()    # Eq. 47


def normalised_importance(ig, eps=1e-12):
    a = np.abs(ig)
    return a / (a.sum() + eps)                                    # Eq. 49


def pathway_contributions(importance, genes, gmt_path):
    """PC_k = sum_i Gimp_i * W_ik (Eq. 50) with binary membership from a GMT file."""
    import pandas as pd
    imp = dict(zip(genes, importance))
    rows = []
    with open(gmt_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            members = set(parts[2:]) & imp.keys()
            if members:
                rows.append((parts[0], len(members), sum(imp[g] for g in members)))
    return pd.DataFrame(rows, columns=["pathway", "n_genes", "contribution"]).sort_values(
        "contribution", ascending=False)
