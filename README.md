# EBiTT-Net — Explainable Bio-Informed Transcriptomic Transformer Network

Exploratory implementation of the framework in *"Explainable Bio-Informed Transcriptomic Learning for
Identifying Host-Response Signatures Associated with MRSA Biofilm Infection"* (GSE210640, human epidermal organoids).

> **Read this first.** This is a **reconstruction** from the manuscript. The original code, checkpoints and
> predictions are not available, so the paper's reported 97.84% accuracy / AUC 0.985 and the baseline table
> **are not reproduced or verified by this repo**. The primary task has **N = 8** samples (4 vs 4); results from
> any run are exploratory and are not evidence of generalisation or clinical performance.

## Pipeline (Algorithm 1)
For each leave-one-out fold, using **training samples only**:
count filtering → log2(CPM+1) + standardisation → Welch-t/BH differential ranking → top-k biomarker selection →
interaction graph (edge score ≥ τ) → GAT (2 layers, 4 heads, 64-d) → gene-level fusion → Transformer
(2 layers, 4 heads, 128-d, FFN 256, GELU, dropout 0.2) → logit. Then predict the held-out sample and compute
post-hoc Integrated Gradients (never used in training). Defaults match Table 2 (AdamW, lr 1e-4, wd 1e-4,
batch 4, 100 epochs, patience 15 on training loss, seed 42).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch-geometric 2.4 needs a torch-2.1-matching install; see PyG docs
python tests_smoke.py               # synthetic end-to-end check
```

## Data
1. Download gene-level counts for **GSE210640** from GEO (`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE210640`).
2. Make `data/counts.tsv` (genes × samples) and `data/metadata.csv` (`sample,condition`) with the labels
   `burned_uninfected` and `burned_mrsa` (or pass `--control-label/--case-label`). Other conditions are ignored.
3. Provide `data/edges.csv` (`gene_a,gene_b,score`, score 0–1, e.g. STRING combined_score/1000) using the same
   identifiers as the counts (use `--id-map` to convert Ensembl → symbol). Optional `data/pathways.gmt`.
   **Record the database, version/date and τ you use** — the manuscript does not specify them.

## Run
```bash
python -m ebittnet.run --counts data/counts.tsv --metadata data/metadata.csv \
    --edges data/edges.csv --gmt data/pathways.gmt --id-map data/id_map.csv --out results

# sanity checks you should run before believing any number
python -m ebittnet.baseline --counts data/counts.tsv --metadata data/metadata.csv            # logistic-regression baseline
python -m ebittnet.run ... --permute-labels --out results_perm                                # should drop to ~chance
```
Outputs: `predictions.csv`, `metrics.json`, `selection_frequency.csv` (Eq. 52), `attributions_per_fold.csv`,
`attribution_summary.csv`, `pathway_contributions.csv`, `graph_stats.csv`.

## Implementation notes / deviations (be transparent in any write-up)
- **DE ranking** uses Welch t-test + BH on log-CPM, not DESeq2 (paper's source study used DESeq2). Swap in PyDESeq2
  *inside the fold* if you want it. With 7 training samples few genes reach padj < 0.05, so selection is by rank (`--top-k`).
- A **learned gene-identity embedding** is added to the expression projection (Eq. 29); the paper does not specify one.
- Early stopping monitors training loss (as in Table 2); there is no validation set at this sample size.
- Pooled AUC across LOOCV folds mixes predictions from different models; report the per-sample predictions too.
- Unspecified in the paper and chosen here: top-k=200, τ=0.4, filter thresholds (10 counts in ≥3 train samples), IG baseline = 0 (training-fold mean), 50 IG steps.

## Known inconsistencies in the manuscript (not replicated)
- Table 6 lists 20,000 input genes, but the described pipeline selects a small fold-specific set.
- Table 5's last two rows differ only by post-hoc IG, which cannot change predictions.
- Figure 3 (forest plot with aOR/q-values for ISG15, IFIT1, CCL5…) does not match the caption or Table 3 (IG scores for IL6, CXCL8, TNF…).
- Baseline models (Table 4) have no code or protocol here; add them with the same in-fold selection before comparing.

## License / data
Add a LICENSE of your choice. Data are public via GEO; do not commit raw data.
