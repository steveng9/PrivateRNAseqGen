# Plan: DP Synthetic Generator via Hierarchical Marginal Selection + Private-PGM

**Status**: Design phase — not yet implemented
**Goal**: Build a synthetic data generator with practical DP-like guarantees for CAMDA 2026 submission
**Tracks**: Primarily Track 1 (bulk RNA-seq); Track 2 (scRNA-seq) as stretch goal
**Deadline**: May 15, 2026
**Last updated**: 2026-03-16

---

## Motivation

Last year's Track 1 winner used Private-PGM with ε=7, with a very naive marginal set:
- 1-way marginals of the 1000 most important genes
- 2-way marginals of those 1000 genes × cancer type label

This won despite being simple. The hypothesis is: **if we select more informative
marginals — including higher-order (3-way, 4-way) correlations — we can achieve better
synthetic data quality at the same or lower ε**.

The key insight from scDesign2: preserving 2-way gene-gene covariances (on ~300 genes)
produces realistic synthetic data. We want to do the same thing, but (a) at a larger
scale, (b) including higher-order interactions, and (c) with noise from Private-PGM
providing empirical privacy protection.

---

## DP Posture (Important Caveat)

**Marginal selection will be done on the private training data** (not a separate public
reference), without spending formal DP budget. This means:

- The pipeline does **not** have an end-to-end formal (ε, δ)-DP guarantee.
- The Private-PGM stage still adds calibrated Gaussian noise to all marginal measurements,
  providing meaningful empirical privacy protection and passing CAMDA's privacy evaluation.
- This is acceptable for the competition. It would not be acceptable for a rigorous
  privacy conference paper — at best a workshop paper or extended abstract.
- If a truly formal DP version is desired later, marginal selection could be switched to
  use a public reference dataset (e.g., TCGA for bulk) and all privacy budget would then
  go to PGM measurement.

---

## Core Design

### Step 1: Gene + Marginal Selection (on private training data, no DP budget)

#### Hierarchical pruned selection:

1. **1-way marginals**: Select S=1000 most informative genes by variance or HVG criterion.

2. **2-way marginals**: Compute pairwise MI/correlation on a pool of ~500–700 top genes.
   - C(700, 2) = 244,650 pairs — trivially feasible.
   - Select top R=150 most informative pairs.

3. **3-way marginals**: Take the genes involved in the top R=150 pairs. Compute all
   3-way combinations among them.
   - Typical unique gene count from 150 pairs: ~100–200 genes.
   - C(150, 3) = 551,300 triples — feasible.
   - Select top Q=30 most informative triples.

4. **4-way marginals**: Take genes involved in the top Q=30 triples. Compute all 4-way
   combinations.
   - Typical unique gene count from 30 triples: ~20–40 genes.
   - C(30, 4) = 27,405 quads — feasible.
   - Select top P=7 most informative quads.

**Total marginals fed into Private-PGM**: S + R + Q + P = 1000 + 150 + 30 + 7 = 1187

### Step 2: Measure Marginals with DP Noise (Private-PGM)

- Apply Private-PGM to the private training data using the pre-selected marginals.
- Private-PGM adds calibrated Gaussian noise to each marginal measurement, then
  finds the maximum-entropy distribution consistent with the noisy measurements.
- Sweep ε ∈ {1, 2, 5, 7, 10}; compare quality and privacy scores.

### Step 3: Sample Synthetic Data

- Draw synthetic samples from the fitted Private-PGM graphical model.
- Post-process to match expected data format (e.g., integer counts for RNA-seq).

---

## Privacy Budget Allocation

How to split ε across marginal orders is a key open design question. Options:

### Option A: Equal budget per marginal
Each of the 1187 marginals gets ε/1187. Problem: 4-way marginals have much larger
count tables (K^4 cells) and will be dominated by noise.

### Option B: Equal budget per order
ε/4 per order. Problem: very unequal per-marginal budget (ε/4000 for 1-way vs. ε/28
for 4-way).

### Option C: Budget weighted by order count (recommended starting point)
Example: 50% to 1-way, 25% to 2-way, 15% to 3-way, 10% to 4-way.
Gives each marginal more budget at higher orders, which have larger count tables.

### Option D: Proportional to count table size
Budget per marginal ∝ K^k (so higher-order marginals get proportionally more).
Ensures roughly equal signal-to-noise ratio across orders.

**Recommendation**: Start with Option C, then tune. Option D is theoretically principled
and worth testing.

---

## What "Most Informative" Means

### For 2-way marginals:
- **Mutual information (MI)** is the most principled choice — captures non-linear
  dependencies, robust to expression range differences.
- **Absolute Spearman correlation**: fast, rank-based, robust to outliers. Good proxy.
- **Absolute Pearson**: fastest, but misses non-linear structure.
- Start with Spearman for speed; switch to MI if quality is insufficient.

### For 3-way and 4-way marginals:
- **Total correlation**: TC(X1;...;Xk) = Σ H(Xi) - H(X1,...,Xk). Measures how much
  joint structure exceeds the sum of marginal structures. Well-defined and principled.
- **Conditional MI proxy**: rank 3-way triples by MI(X3; {X1,X2}), i.e., how much
  the third gene adds given the other two. Fast to compute from pairwise MI via chain
  rule approximation.
- **Sum-of-pairwise-MI proxy**: rank triples/quads by sum of all pairwise MIs within
  the group. Cheapest to compute; ignores true higher-order structure but may be
  sufficient in practice.
- **Recommendation**: use sum-of-pairwise-MI as a fast proxy first (reuses 2-way
  computation). If quality is poor, switch to total correlation (requires discretization
  and plug-in entropy estimation).

---

## Discretization

Private-PGM works on discrete attributes. RNA-seq data is counts with high dynamic
range, often zero-inflated.

### Proposed approach:
- Bin each gene into K discrete levels.
- For zero-inflated data (especially scRNA-seq): treat 0 as a dedicated bin, then
  quantile-bin non-zero values into K-1 remaining levels.
- Apply the same binning scheme consistently across marginal selection and PGM fitting.

### Count table sizes at K=8:
- 1-way: 8 cells
- 2-way: 64 cells
- 3-way: 512 cells
- 4-way: 4,096 cells — manageable

### Open question: optimal K
More bins → better fidelity but more noise per cell. Start K=8; test K=4 and K=16.
For bulk RNA-seq (less sparsity), K=16 may be feasible. For scRNA-seq, K=4 or K=8.

---

## Evaluation Metrics (CAMDA 2026)

### Track 1 (Bulk RNA-seq):
1. **Utility**: Downstream ML performance
   - Cancer subtype prediction (TCGA-BRCA)
   - Cancer tissue-of-origin (TCGA COMBINED)
2. **Fidelity**: Statistical preservation of original data properties
3. **Biological plausibility**: Conservation of gene co-expression structure
4. **Privacy**: Assessment of privacy preservation (exact method unspecified by CAMDA)

*Note*: Track 1 COMBINED dataset has a provided reference; BRCA does not.

### Track 2 (scRNA-seq):
1. **Statistical**: SCC (HVG ranking), MMD (distributional similarity), LISI (mixing),
   ARI (clustering similarity), ARI (cluster vs. ground truth)
2. **Non-statistical**: UMAP visual quality, CellTypist classification accuracy (using
   `Immune_All_High` model), random forest distinguishability

---

## Applying to scRNA-seq (Track 2) — Stretch Goal

Additional challenges vs. bulk:

1. **Cell type heterogeneity**: Gene correlations differ by cell type. Options:
   - Include cell type as an attribute in the PGM.
   - Run pipeline separately per cell type (splits ε or uses per-type budget).
   - Model cell type composition separately (Dirichlet/multinomial), then condition.
   *Recommended*: include cell type as a PGM attribute — simplest.

2. **Extreme sparsity**: ~80-95% zeros. Zero-bin dominates marginals. Use the
   zero-dedicated-bin approach in discretization (see above).

3. **DP unit: cell vs. donor**: The privacy-sensitive unit is the *donor*, not the cell.
   If using formal DP, need to aggregate cells per donor or use group privacy.
   For this competition (informal DP), cell-level DP is simpler and acceptable.

4. **HVG selection**: Use `sc.pp.highly_variable_genes` (`min_mean=0.0125`,
   `max_mean=3`, `min_disp=0.5`) as the gene pool for marginal selection.

**Decision**: Implement Track 1 first. Extend to Track 2 only if Track 1 works and time
permits.

---

## Implementation Plan

### Phase 1: Core pipeline for bulk RNA-seq (weeks 1–2)
- [ ] Set up `src/generators/private_pgm/` directory
- [ ] Implement discretization module (with zero-inflated option)
- [ ] Implement hierarchical marginal selection (1-way → 2-way → 3-way → 4-way)
- [ ] Check Private-PGM Python library availability (`ryan112358/private-pgm` on GitHub)
- [ ] Wrap or port Private-PGM fitting + sampling
- [ ] End-to-end smoke test on a small synthetic dataset

### Phase 2: Track 1 evaluation (week 3)
- [ ] Run on CAMDA 2026 Track 1 data (BRCA and/or COMBINED)
- [ ] Evaluate fidelity and biological plausibility metrics
- [ ] Sweep ε ∈ {1, 2, 5, 7, 10}
- [ ] Compare against last year's naive baseline (1-way + 2-way × label)

### Phase 3: Tuning + submission (weeks 4–5)
- [ ] Tune S, R, Q, P and budget allocation
- [ ] Write method description for CAMDA submission (~1 page)
- [ ] If time: begin Track 2 extension
- [ ] Draft workshop paper / extended abstract if results are strong

---

## Open Questions

1. **MI metric for 3/4-way**: Start with sum-of-pairwise proxy or total correlation?
   Need to benchmark speed vs. quality on a small dataset.

2. **Budget allocation**: Option C vs. D? Empirical tuning needed.

3. **Discretization bins K**: Start K=8. Test K=4 and K=16 for sensitivity.

4. **Private-PGM library**: Is `ryan112358/private-pgm` pip-installable and maintained?
   Check before implementing; may need to port.

5. **CAMDA privacy evaluation**: How exactly does CAMDA assess privacy in Track 1?
   Knowing this matters for tuning ε.

6. **Track 1 BRCA reference**: No public reference provided for BRCA subset. If
   needed for formal DP later, TCGA is the natural candidate.

7. **For scRNA-seq**: CellTypist evaluation uses `Immune_All_High` model — need to
   ensure synthetic data preserves enough cell-type signal to pass this.

8. **Formal DP path**: If we want to claim rigorous DP in a future paper, switch
   marginal selection to use TCGA or CZ CELLxGENE as public reference. Budget then
   fully captured by Private-PGM stage.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Higher-order marginals too noisy at reasonable ε | Medium | High | Reduce K, reduce P/Q, or raise ε |
| Private-PGM library unmaintained / hard to use | Low | Medium | Port core algorithm (~500 lines) |
| CAMDA data access delayed | Low | Medium | Request data ASAP |
| Time: competing with paper revision + red-team | High | High | Incremental; Track 2 is optional |
| MI estimation slow on high-dimensional data | Medium | Low | Use Spearman proxy; prune gene pool |
| Informal DP not accepted by competition | Low | Low | PGM noise still passes empirical privacy tests |

---

## Key References

- McKenna et al. (2019). "Graphical-model based estimation and inference for differential
  privacy." ICML.
- McKenna et al. (2021). "Winning the NIST Contest: A scalable and general approach to
  differentially private synthetic data." TPDP.
- Dwork et al. (2014). "The Algorithmic Foundations of Differential Privacy."
- Sun et al. (2023). scDesign3. Nature Methods.
- CAMDA 2025 Track 1 winning submission (to be obtained).
- Korsunsky et al. (2019). LISI. Nature Methods.
- McInnes et al. (2018). UMAP. JOSS.
- Dominguez Conde et al. (2022). CellTypist. Science.
