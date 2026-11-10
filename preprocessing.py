"""Training-fold-only preprocessing (Eqs. 7-11). Nothing here sees held-out data."""
from __future__ import annotations
import numpy as np
import pandas as pd


class FoldPreprocessor:
    """Count filtering -> CPM -> log2(CPM+1); standardisation fitted on train only."""

    def __init__(self, min_count=10, min_samples=3):
        self.Tc, self.Ts = min_count, min_samples   # Eqs. (7)-(8)

    def fit(self, counts_train: pd.DataFrame):
        n_expr = (counts_train.values >= self.Tc).sum(axis=0)
        self.genes_ = counts_train.columns[n_expr >= self.Ts]
        logcpm = self._logcpm(counts_train[self.genes_])
        self.mean_ = logcpm.mean(0)
        self.std_ = logcpm.std(0, ddof=0).replace(0, 1.0)
        return self

    @staticmethod
    def _logcpm(c: pd.DataFrame) -> pd.DataFrame:
        # library size over the retained genes (Eq. 9, M' retained genes)
        cpm = c.div(c.sum(axis=1), axis=0) * 1e6
        return np.log2(cpm + 1.0)

    def logcpm(self, counts: pd.DataFrame) -> pd.DataFrame:
        return self._logcpm(counts[self.genes_])

    def transform(self, counts: pd.DataFrame) -> pd.DataFrame:
        return (self.logcpm(counts) - self.mean_) / self.std_
