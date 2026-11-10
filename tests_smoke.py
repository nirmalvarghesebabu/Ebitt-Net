"""Synthetic smoke test (no GEO data needed): python tests_smoke.py"""
import numpy as np, pandas as pd, torch
from ebittnet.preprocessing import FoldPreprocessor
from ebittnet.feature_selection import select_biomarkers
from ebittnet.graph import build_graph
from ebittnet.model import EBiTTNet
from ebittnet.train import fit, predict_proba
from ebittnet.explain import integrated_gradients

rng = np.random.RandomState(0)
G = [f"G{i}" for i in range(300)]
counts = pd.DataFrame(rng.poisson(200, (8, 300)), columns=G)
y = np.array([0]*4 + [1]*4); counts.iloc[4:, :20] *= 3
pre = FoldPreprocessor().fit(counts.iloc[:7])
genes, _ = select_biomarkers(pre.logcpm(counts.iloc[:7]), y[:7], top_k=30)
edges = pd.DataFrame({"gene_a": genes[:-1], "gene_b": genes[1:], "score": 0.9})
ei, ea, info = build_graph(genes, edges, 0.4)
m, _ = fit(EBiTTNet(len(genes), ei, ea), pre.transform(counts.iloc[:7])[genes].values, y[:7], epochs=5)
x = pre.transform(counts.iloc[7:])[genes].values
print("prob", predict_proba(m, x), "IG shape", integrated_gradients(m, x[0]).shape, info)
