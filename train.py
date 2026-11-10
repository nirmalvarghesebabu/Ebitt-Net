"""Model fitting (Eqs. 41-44). Weight decay supplies the L2 term. Early stopping watches TRAINING loss only."""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(model, X_train, y_train, lr=1e-4, weight_decay=1e-4, batch_size=4,
        epochs=100, patience=15, device="cpu", seed=42):
    set_seed(seed)
    model.to(device).train()
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=device)
        tot = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:                    # avoid degenerate single-sample batch
                continue
            opt.zero_grad()
            loss = lossf(model(X[idx]), y[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        tot /= len(X); hist.append(tot)
        if tot < best - 1e-6:
            best, bad, best_state = tot, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), hist


@torch.no_grad()
def predict_proba(model, X, device="cpu"):
    model.eval()
    return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=device))).cpu().numpy()
