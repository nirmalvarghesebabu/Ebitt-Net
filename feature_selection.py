"""Training-fold-specific differential-expression ranking and biomarker selection (Eqs. 13-17).

DE is a Welch t-test on log2(CPM+1) with Benjamini-Hochberg correction. For a
DESeq2-faithful analysis, replace `rank_genes` with PyDESeq2/DESeq2 *run on the
training fold only* - never on all samples.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def rank_genes(logcpm_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    a = logcpm_train.values[y_train == 1]
    b = logcpm_train.values[y_train == 0]
    lfc = a.mean(0) - b.mean(0)  # log2 fold change on log2 scale (Eq. 13)
    t, p = stats.ttest_ind(a, b, equal_var=False, axis=0, nan_policy="omit")
    p = np.nan_to_num(p, nan=1.0)
    padj = multipletests(p, method="fdr_bh")[1]
    return pd.DataFrame({"lfc": lfc, "p": p, "padj": padj}, index=logcpm_train.columns)


def select_biomarkers(logcpm_train, y_train, top_k=200, lfc_min=np.log2(1.5)):
    """S_r = Select(G_Tr, R^(r)): |LFC| >= log2(1.5) pre-filter (Eq. 15), then rank by p, then |LFC|.

    With n=7 training samples padj rarely reaches 0.05, so selection is by rank
    (top_k), not by significance. Reported honestly in the README.
    """
    res = rank_genes(logcpm_train, y_train)
    res["abs_lfc"] = res["lfc"].abs()
    cand = res[res["abs_lfc"] >= lfc_min]
    if len(cand) < top_k:           # relax if too few pass the fold-change filter
        cand = res
    cand = cand.sort_values(["p", "abs_lfc"], ascending=[True, False])  # deterministic tie handling
    sel = cand.head(top_k).index.tolist()
    return sorted(sel), res
