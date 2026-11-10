"""Loading the GSE210640 primary comparison (burned vs burned + MRSA biofilm)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def load_primary_dataset(counts_path, metadata_path, sample_col="sample",
                         condition_col="condition", control_label="burned_uninfected",
                         case_label="burned_mrsa", id_map_path=None):
    """Return (counts[N x M DataFrame, raw ints], y[N], sample_names).

    counts file: genes x samples (TSV/CSV auto-detected). Only the two primary
    groups are kept; labels follow Eq. (1): 0 = control, 1 = MRSA biofilm.
    """
    counts = pd.read_csv(counts_path, sep=None, engine="python", index_col=0)
    meta = pd.read_csv(metadata_path)
    meta = meta[meta[condition_col].isin([control_label, case_label])].copy()
    missing = [s for s in meta[sample_col] if s not in counts.columns]
    if missing:
        raise ValueError(f"Samples in metadata missing from counts: {missing}")
    counts = counts[meta[sample_col].tolist()]

    if id_map_path:
        m = pd.read_csv(id_map_path).drop_duplicates("gene_id").set_index("gene_id")["symbol"]
        counts = counts.rename(index=m.to_dict())
        counts = counts[~counts.index.isna()]
    # unambiguous identifiers only: sum duplicated rows
    counts = counts.groupby(level=0).sum()
    if counts.isna().any().any():
        raise ValueError("Missing values in count matrix; handle explicitly.")

    y = (meta[condition_col] == case_label).astype(int).to_numpy()
    X = counts.T  # samples x genes
    if len(y) != 8 or y.sum() != 4:
        print(f"[warn] expected 8 samples (4/4); got N={len(y)}, positives={y.sum()}")
    return X, y, meta[sample_col].tolist()
