"""Leakage-controlled LOOCV (Algorithm 1).   python -m ebittnet.run --help"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut

from .data import load_primary_dataset
from .preprocessing import FoldPreprocessor
from .feature_selection import select_biomarkers
from .graph import load_edges, build_graph
from .model import EBiTTNet
from .train import fit, predict_proba, set_seed
from .explain import integrated_gradients, normalised_importance, pathway_contributions
from .metrics import summarise


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--counts", required=True); p.add_argument("--metadata", required=True)
    p.add_argument("--edges", required=True, help="CSV gene_a,gene_b,score (score in 0-1)")
    p.add_argument("--gmt"); p.add_argument("--id-map")
    p.add_argument("--sample-col", default="sample"); p.add_argument("--condition-col", default="condition")
    p.add_argument("--control-label", default="burned_uninfected"); p.add_argument("--case-label", default="burned_mrsa")
    p.add_argument("--top-k", type=int, default=200); p.add_argument("--tau", type=float, default=0.4)
    p.add_argument("--min-count", type=int, default=10); p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patience", type=int, default=15); p.add_argument("--ig-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permute-labels", action="store_true",
                   help="sanity check: shuffle labels BEFORE the CV loop; accuracy should fall to chance")
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    a = parse(); os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(a.seed)
    counts, y, names = load_primary_dataset(a.counts, a.metadata, a.sample_col, a.condition_col,
                                            a.control_label, a.case_label, a.id_map)
    if a.permute_labels:
        y = np.random.RandomState(a.seed).permutation(y)
    edges = load_edges(a.edges)

    preds, sel_sets, attr_rows, pw_rows, graph_info = [], [], [], [], []
    for r, (tr, te) in enumerate(LeaveOneOut().split(counts)):
        t0 = time.time()
        ctr, cte = counts.iloc[tr], counts.iloc[te]
        ytr = y[tr]
        pre = FoldPreprocessor(a.min_count, a.min_samples).fit(ctr)                  # train-only
        genes, _ = select_biomarkers(pre.logcpm(ctr), ytr, top_k=a.top_k)             # train-only DE
        ei, ea, ginfo = build_graph(genes, edges, a.tau); ginfo["fold"] = r; graph_info.append(ginfo)
        Xtr = pre.transform(ctr)[genes].values; Xte = pre.transform(cte)[genes].values
        model = EBiTTNet(len(genes), ei, ea)
        model, hist = fit(model, Xtr, ytr, a.lr, a.weight_decay, a.batch_size, a.epochs, a.patience, device, a.seed)
        p = float(predict_proba(model, Xte, device)[0])
        ig = integrated_gradients(model, Xte[0], steps=a.ig_steps, device=device)     # after fitting
        imp = normalised_importance(ig)
        preds.append({"fold": r, "sample": names[te[0]], "y_true": int(y[te[0]]), "prob": p,
                      "y_pred": int(p > 0.5), "epochs": len(hist), "final_train_loss": hist[-1]})
        sel_sets.append(set(genes))
        attr_rows += [{"fold": r, "gene": g, "ig": float(v), "importance": float(w)} for g, v, w in zip(genes, ig, imp)]
        if a.gmt:
            pw = pathway_contributions(imp, genes, a.gmt); pw["fold"] = r; pw_rows.append(pw)
        print(f"fold {r}: held out {names[te[0]]} y={y[te[0]]} p={p:.3f} genes={len(genes)} "
              f"graph={ginfo} ({time.time()-t0:.1f}s)")

    pdf = pd.DataFrame(preds); pdf.to_csv(f"{a.out}/predictions.csv", index=False)
    m = summarise(pdf.y_true, pdf.prob); m["note"] = ("N=8 LOOCV; exploratory; pooled AUC mixes fold-specific models; "
                                                      "not evidence of generalisation")
    json.dump(m, open(f"{a.out}/metrics.json", "w"), indent=2)
    pd.DataFrame(graph_info).to_csv(f"{a.out}/graph_stats.csv", index=False)

    K = len(sel_sets); allg = sorted(set().union(*sel_sets))
    stab = pd.DataFrame({"gene": allg, "selection_frequency": [sum(g in s for s in sel_sets) / K for g in allg]})
    stab.sort_values("selection_frequency", ascending=False).to_csv(f"{a.out}/selection_frequency.csv", index=False)  # Eq. 52
    att = pd.DataFrame(attr_rows); att.to_csv(f"{a.out}/attributions_per_fold.csv", index=False)
    summ = att.groupby("gene").agg(mean_importance=("importance", "mean"), sd_importance=("importance", "std"),
                                   n_folds=("fold", "nunique")).sort_values("mean_importance", ascending=False)
    summ.to_csv(f"{a.out}/attribution_summary.csv")
    if pw_rows:
        pd.concat(pw_rows).groupby("pathway")["contribution"].mean().sort_values(ascending=False)\
            .to_csv(f"{a.out}/pathway_contributions.csv")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
