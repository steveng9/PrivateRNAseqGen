# DP Synthetic RNA-seq via Hierarchical Marginal Selection + Private-PGM

**Status**: Design phase — not yet implemented  
**Competition**: CAMDA 2026, Track 1 (bulk RNA-seq); Track 2 (scRNA-seq) as stretch goal  
**Deadline**: May 15, 2026

---

## Overview

This project implements a differentially private synthetic RNA-seq data generator for the
CAMDA 2026 competition. The core idea: select highly informative 1-, 2-, 3-, and 4-way
gene marginals using a hierarchical pruning strategy, then fit a Private-PGM graphical
model with calibrated Gaussian noise to generate synthetic data.

**Motivation**: Last year's Track 1 winner used Private-PGM with a naive marginal set
(1-way genes + 2-way gene×label). Our hypothesis is that smarter marginal selection —
including higher-order gene-gene correlations — produces better synthetic data quality at
the same or lower ε.

---

## Core Design (Summary)

1. **Gene + marginal selection** (hierarchical, on private training data, no DP budget):
   - 1-way: S=1000 most variable genes
   - 2-way: top R=150 most correlated pairs (from ~700 gene pool)
   - 3-way: top Q=30 triples from genes in top pairs
   - 4-way: top P=7 quads from genes in top triples
   - Total: 1187 marginals fed into Private-PGM

2. **Private-PGM fitting**: calibrated Gaussian noise on marginal measurements;
   sweep ε ∈ {1, 2, 5, 7, 10}

3. **Sample + post-process**: draw from fitted graphical model; reformat to integer counts

See [`docs/PLAN_private_pgm_generator.md`](docs/PLAN_private_pgm_generator.md) for the
full design, open questions, implementation plan, and risk table.

---

## DP Posture

This pipeline does **not** carry an end-to-end formal (ε, δ)-DP guarantee, because
marginal selection uses the private training data without spending formal budget. The
Private-PGM stage provides meaningful empirical privacy protection and is expected to
pass CAMDA's privacy evaluation. A fully formal DP version (using a public reference
such as TCGA for marginal selection) is noted in the plan as a future extension.

---

## Repo Structure

```
docs/
  PLAN_private_pgm_generator.md   Full design doc
src/                              (to be populated)
  generators/
    private_pgm/
      discretize.py
      marginal_selection.py
      pgm_wrapper.py
experiments/                      (to be populated)
  track1_bulk/
  track2_scrna/
```

---

## Key References

- McKenna et al. (2019). "Graphical-model based estimation and inference for differential
  privacy." ICML.
- McKenna et al. (2021). "Winning the NIST Contest: A scalable and general approach to
  differentially private synthetic data." TPDP.
- Dwork et al. (2014). "The Algorithmic Foundations of Differential Privacy."
- CAMDA 2025 Track 1 winning submission.
