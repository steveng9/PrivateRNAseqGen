# Why private-pgm on bulk RNA-seq suddenly works: what was broken, how it was fixed, why it is valid

*2026-09-24.  For Steven's group, for Daniil, and for any agent porting these
fixes to another private-pgm pipeline.  Code: `steveng9/PrivateRNAseqGen`
commits `d370956`, `fa3e161`, `46fb379` and `cd5d1d5`, plus the forest commit;
full derivation of the accounting in `docs/ZCDP_ACCOUNTING.md`.*

## TL;DR

Last year's DP-PGM pipeline added **~50× more noise than its ε required**,
because it composed ~2,000 Gaussian measurements with the wrong theorem.  At
genomic scale that error grows like √(number of tables) and wipes out both
data quality and any attack.  Fixing the accounting and four smaller defects
makes the same model, at the same (ε, δ), produce usable data:

| BRCA, 978 genes, ε=10, δ=1e-5 | before | after |
|---|---|---|
| noise σ per cell of each 1-way table (871 rows) | **1,436** | **~26** |
| noise σ per cell of each gene×label table | 707 | ~26 |
| classifier utility (TSTR macro-F1 / real) | 0.11 (0.092/0.811) | **0.69** |
| best attack AUC ceiling (analytic, marginals) | 0.514 | 0.89 |

Every number in the "after" column is DP end to end at the stated (ε, δ).  An
empirical audit of the privacy loss (below) matches the theory to 3 decimals,
and no attack on a DP-valid release has exceeded the DP bound AUC ≤ Φ(√ρ)
(0.89 at ε=10); the strongest, MAMA-MIA v2, reaches 0.77 on BRCA at ε=10.

## The five defects, in order of damage

### 1. Basic composition instead of zCDP (the big one)

**What was wrong.** `pgm_fitter.py` split ε *linearly* across the k tables
(`eps_per = frac * eps / k`) and calibrated each Gaussian with the classical
bound `σ = √(2 ln(1.25/δ)) / eps_per`.  So σ grew **linearly** in k.

**Why that is wasteful.**  A Gaussian mechanism's privacy loss is exactly a
normal random variable, N(ρ, 2ρ) with ρ = Δ²/(2σ²).  Independent losses add,
and a sum of normals is normal, so **ρ simply adds** across tables (zCDP;
Bun & Steinke 2016).  Converting the total back to (ε, δ) gives
ε = ρ + 2√(ρ ln(1/δ)).  The required σ grows like **√k**, not k.  Basic
composition adds up worst cases; the sum actually concentrates.

| σ_basic / σ_zCDP | k=5 | k=50 | k=978 | k=1956 |
|---|---|---|---|---|
| | 2.0× | 6.5× | 28.6× | 40.5× |

On a 5-column tabular benchmark nobody notices.  At 1,956 tables it is fatal.
This framing is the transferable result: **omics feature counts turn a
harmless textbook choice into a 30–60× noise penalty.**

**Fix.** `composition="zcdp"` (default): ρ = ρ(ε, δ) from OpenDP's `cdp_rho`
(the same call MST and AIM make); each of k tables sharing a fraction w of ρ
gets σ = Δ₂·√(k / (2wρ)).  `"basic"` is kept only to reproduce old numbers.

**Why it is valid.** It is the accounting McKenna et al. use in MST and AIM
(`snsynth/mst/mst.py`: `sigma / weight` with weights L2-normalised is σ√k for k
equal weights).  `tests/test_composition.py` (22 tests) checks σ against MST's
formula and **audits the mechanism empirically**: over 20,000 releases the
measured privacy-loss variable is N(0.22407, 0.44814) against a predicted
N(0.22425, 0.44850).

### 2. The neighbouring relation and the row count disagreed

**What was wrong.** σ assumed L2 sensitivity 1, which is true under
*add/remove-one* neighbours.  But the fitter handed the exact training row count
to `FactoredInference`, and under add/remove the row count itself differs
between neighbours, so releasing it exactly breaks DP outright.  (Stratified
mode also sized each class's sample from exact class counts.)

**Fix.** `neighboring="add_remove"` (default) passes `total=None`; mbi then
estimates n from the noisy 1-way tables.  That estimate is free (post-processing)
and accurate to 0.2% of n; it is exactly why MST never passes a total.
`neighboring="replace"` is also available: n becomes public but every σ grows
by √2.

**Why it is valid.** Under add/remove each table's count vector changes by one
in one cell, so Δ₂ = 1, and nothing else touches the data.

### 3. Bin edges were private percentiles, released for free

**What was wrong.** The discretiser took each gene's bin edges from
percentiles of the training data and spent no budget on them.  Synthetic values
are drawn between those edges, so the release leaked the training data's
quantiles directly, outside ε.  (On our ND generator the same kind of leak alone
gave AUC 0.9996.)

**Fix.** Three DP-valid binnings:
- `uniform`: equal width over a public range (0, 24) log2 units;
- `dp_uniform`: DP per-gene bounds, then equal width (what smartnoise-synth's
  `BinTransformer` does for MST/AIM);
- `dp_quantile`: DP equal-depth edges, read off a noisy histogram on a public
  48-cell grid, costing 10% of ρ.

**Why it is valid.** A noisy histogram is a Gaussian mechanism (Δ₂ = 1 under
add/remove), composed in ρ with everything else; edges are post-processing of
it.

### 4. The DP edge estimator was biased (found 2026-09-23)

**What was wrong.** The first `dp_*` implementation clipped the noisy histogram
at zero (`max(h + noise, 0)`), so every empty grid cell kept its positive noise.
Empty cells dominate a grid over (0, 24), so even at ε=1000 the per-gene bounds
landed near 2 and 22 instead of the true ~9.5 and ~13.5, and 4 bins put 85% of
each gene into one cell.  Every `dp_*` result built with it is stamped
`BROKEN_DP_EDGES` (`results/BROKEN.md`).

**Fix.** `edge_estimator="threshold"`: zero every cell below
τ = σ·Φ⁻¹(1 − 0.05/cells), smartnoise's `approx_bounds` rule, then read
quantiles off the CDF with linear interpolation inside cells.  At ε=1000 the
bounds are now within 0.1 log2 units.

**Why it is valid.** It is post-processing of the same noisy histogram; privacy
cost is unchanged.

### 5. A column-order bug in our wrapper (ours, not upstream)

The upstream generator returns genes in variance order; our adapter treated
column j as gene j.  Global moments survive a permutation, so nothing that
checked totals could see it.  Per-gene W1 was 2.9 training SDs and the
discriminator was at 1.000.  Fixed by re-indexing by gene name.

## What else changed, and why the numbers are now honest

- **Joint model, not one model per class.** Stratified mode fitted a separate
  PGM per subtype (BRCA "Normal" has 32 rows).  Joint mode fits one model with
  the label as a variable plus 978 (gene, label) tables.
- **The table set is public or DP-selected.** With all 978 genes and
  `n_2way=0` the tables are fixed by config.  The tree and forest structures
  choose gene pairs with MST's exponential-mechanism selection, paid for out of
  the same ρ.  The old Spearman-ranked 2/3/4-way selection
  (`n_2way/n_3way/n_4way > 0`) ran on the private data with no budget, so it is
  **not DP** and should only be used as a labelled reference.
- **Budget is asserted.** Every fit records ρ spent per stage
  (binning / selection / measurement) and raises if the total exceeds ρ(ε, δ).
- **No attack exceeds the bound.** Across 270 models (9 ε values × 3 DP
  binnings × 2 cohorts × 5 splits) the best attack AUC stays under Φ(√ρ) (max
  +0.4 SE).  Those models used the biased edge estimator of defect 4, which was
  inaccurate but still DP, so the bound check stands.  With the leaky legacy
  binning the attack exceeds the bound by up to 14 SE, which is what a real
  leak looks like.

## What is still not covered by ε

- The expression values are VST-normalised by the challenge organisers over the
  whole cohort before we see them.  That preprocessing is outside our
  mechanism, as it would be for any participant.
- The public value range (0, 24) and the bin count are fixed a priori.

## Caveats on "works"

- Per-gene distributions and subtype classification are now close to CVAE/ND
  (COMBINED ε=10 utility 0.92 vs CVAE 0.96; per-gene W1 0.10 vs 0.19).
- **Gene–gene correlation is still poor** (correlation MAE 0.13 vs CVAE 0.07),
  and a discriminator separates real from synthetic at AUC 1.00, even at
  ε=1000.  A star around the label cannot express co-expression.  The tree and
  forest structures are the attempts to fix that; see FINDINGS §10i–10j.
- BRCA at ε=10 keeps less (utility 0.69): 871 rows spread over 16 bins × 5
  classes is small against σ≈26.

## Checklist for porting to another private-pgm pipeline

1. **Accounting.** Replace per-table ε splitting with ρ = `cdp_rho(eps, delta)`
   and σ = Δ₂·√(k / (2·w·ρ)) for k tables sharing fraction w.  Never split δ or
   ε linearly across Gaussian measurements.
2. **Neighbouring relation.** Pick one and make everything agree.
   - Add/remove: Δ₂ = 1 and call `engine.estimate(measurements, total=None)`.
   - Replace: Δ₂ = √2 and the true n may be passed.
   - Never size the synthetic sample from exact private counts under add/remove.
3. **Discretisation.** Edges must be public or DP.  If DP, estimate them from
   a noisy histogram with a noise threshold, not by clipping at zero.
4. **Table selection.** Any data-dependent choice of which tables to measure
   (top-variance genes when not all genes are kept, top correlated pairs) must
   be DP, e.g. the exponential mechanism at ε_round = √(8ρ_sel/rounds), since an
   ε-DP exponential mechanism is ε²/8-zCDP.
5. **Assert the budget.** Sum ρ over binning + selection + measurements, and
   fail if it exceeds ρ(ε, δ).
6. **Test it.** Compare σ to MST's formula, and audit empirically.  Release a
   table from neighbouring datasets many times and check that the log
   likelihood ratio is N(ρ, 2ρ).
7. **Check the plumbing.** Verify that output columns match input genes by
   name, and compare per-gene W1 against a permuted control.
