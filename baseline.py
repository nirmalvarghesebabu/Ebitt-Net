"""Leakage-free logistic-regression baseline with the same LOOCV + in-fold selection.
Any deep model should be compared against this before claiming a benefit.
    python -m ebittnet.baseline --counts ... --metadata ...
"""
from __future__ import annotations
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from .data import load_primary_dataset
from .preprocessing import FoldPreprocessor
from .feature_selection import select_biomarkers
from .metrics import summarise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--counts", required=True); p.add_argument("--metadata", required=True)
    p.add_argument("--id-map"); p.add_argument("--control-label", default="burned_uninfected")
    p.add_argument("--case-label", default="burned_mrsa"); p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--permute-labels", action="store_true"); p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    counts, y, _ = load_primary_dataset(a.counts, a.metadata, control_label=a.control_label,
                                        case_label=a.case_label, id_map_path=a.id_map)
    if a.permute_labels:
        y = np.random.RandomState(a.seed).permutation(y)
    probs = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(counts):
        pre = FoldPreprocessor().fit(counts.iloc[tr])
        genes, _ = select_biomarkers(pre.logcpm(counts.iloc[tr]), y[tr], a.top_k)
        clf = LogisticRegression(C=0.1, max_iter=1000).fit(pre.transform(counts.iloc[tr])[genes], y[tr])
        probs[te] = clf.predict_proba(pre.transform(counts.iloc[te])[genes])[:, 1]
    print(json.dumps(summarise(y, probs), indent=2))


if __name__ == "__main__":
    main()
