# StratHiM-PGM: Stratified Hierarchical Marginal-selection Private-PGM

**Status**: Prototype working — BRCA smoke-tested end-to-end  
**Competition**: CAMDA 2026, Track 1 (bulk RNA-seq) | Track 2 (scRNA-seq) stretch goal  
**Deadline**: May 4, 2026

---

## Overview

Differentially private synthetic bulk RNA-seq generator. Two innovations over last year's
CAMDA Track 1 winner (which used a single PGM with 1-way gene + 2-way gene×label marginals):

1. **Stratified fitting** — one graphical model per cancer class, eliminating the label as a
   PGM hub node. This avoids the treewidth explosion caused by dense gene×label edges while
   giving exact label conditioning.
2. **Hierarchical gene–gene marginal selection** — variance → pairwise Spearman → clique
   expansion, capturing gene–gene correlation structure instead of only gene×label structure.

**Architecture**: StratHiM-PGM (Stratified Hierarchical Marginal-selection PGM). Marginal
selection runs once on the full dataset so the selected gene set is shared across all class
models.

### Joint mode (comparison baseline)

Set `joint_mode: true` in the config to reproduce last year's structural approach: a single
PGM over all classes with the label as a node, all gene×label 2-way marginals preserved, plus
hierarchical gene–gene marginals on top. Useful for apples-to-apples comparison.

---

## Repo Structure

```
configs/
  BRCA.yaml              ← parameters for TCGA-BRCA (Track 1, bulk)
  COMBINED.yaml          ← parameters for TCGA-COMBINED (Track 1, bulk)
  onek1k.yaml            ← parameters for OneK1K scRNA-seq (Track 2)
src/
  discretization.py      ← quantile-bins genes into K discrete levels; zero_inflated mode for scRNA
  marginal_selection.py  ← hierarchical S→R→Q→P clique selection (Spearman)
  pgm_fitter.py          ← adds Gaussian noise, fits mbi.FactoredInference
  generator.py           ← top-level: stratified fit + generate (shared by both tracks)
  camda_runner.py        ← Track 1 runner: reads/writes CAMDA CSV format
  scrna_runner.py        ← Track 2 runner: reads h5ad, writes synthetic h5ad
scripts/
  smoke_test.py          ← fast end-to-end test on synthetic toy data
  run_camda.sh           ← Track 1 entry point: run all 5 splits for one dataset
  run_scrna.sh           ← Track 2 entry point: fit + generate scRNA-seq synthetic data
docs/
  PLAN_private_pgm_generator.md
```

---

## Quick Start

### Prerequisites

```bash
conda activate recon_          # env with mbi (private-pgm) installed
```

Data splits must exist at `~/Health-Privacy-Challenge/data_splits/TCGA-{BRCA,COMBINED}/real/`.
If they don't:

```bash
cd ~/Health-Privacy-Challenge
python -c "
import sys, yaml, os; sys.path.insert(0,'src')
from generators.utils.prepare_data import RealDataLoader
for dataset in ['BRCA', 'COMBINED']:
    cfg = yaml.safe_load(open(f'experiments/track_i/blue_team/2_generation/config_private_pgm_{dataset}.yaml'))
    cfg['dir_list']['home'] = os.path.expanduser(cfg['dir_list']['home'])
    rl = RealDataLoader(cfg); rl.save_split_indices(); rl.save_split_data()
"
```

### Run (from this repo root)

```bash
# All 5 splits, BRCA
./scripts/run_camda.sh BRCA eps7_k8

# All 5 splits, COMBINED
./scripts/run_camda.sh COMBINED eps7_k8

# Single split (faster for testing)
python src/camda_runner.py configs/BRCA.yaml --split 1 --experiment eps7_k8
```

The experiment name (e.g. `eps7_k8`) is just a label for the output folder — it does not
set any parameters. Change parameters in the config YAML, then re-run with a descriptive
experiment name.

### Smoke test (no real data needed)

```bash
python scripts/smoke_test.py
```

---

## Tuning Parameters

All parameters live in `configs/BRCA.yaml` or `configs/COMBINED.yaml`.

| Parameter | Config key | Meaning | Speed impact |
|---|---|---|---|
| ε (epsilon) | `epsilon` | Privacy budget (higher = less noise = better quality) | none |
| k | `n_bins` | Discrete bins per gene | large (smaller = much faster) |
| S | `n_1way` | Top genes by variance → 1-way marginals | large |
| R | `n_2way` | Top gene–gene pairs by \|Spearman\| → 2-way marginals | moderate |
| Q | `n_3way` | Top gene triples from genes in top R pairs | **see note** |
| P | `n_4way` | Top gene quads from genes in top Q triples | **see note** |
| — | `pgm_iters` | PGM optimiser iterations | large |
| — | `n_synth_samples` | Synthetic samples per split (-1 = match train size) | small |
| — | `joint_mode` | `false` = stratified (default); `true` = joint PGM with label node (last year's approach) | — |

> **Note on Q and P (3-way / 4-way marginals)**: Currently set to 0 in both configs.
> Re-enabling them risks a **treewidth explosion** in mbi's junction tree, causing OOM errors.
> When re-enabling, start small (Q=5, P=2) and monitor memory carefully. The 3-way/4-way
> marginals should not overlap heavily with 2-way marginals — overlapping cliques cause the
> JT to create enormous merged factors. This is a known open problem in the implementation.

**Example: faster run for parameter sweeping**

```yaml
generator:
  n_1way: 200
  n_2way: 50
  n_bins: 4
  pgm_iters: 300
```

**Example: sweep epsilon**

```yaml
generator:
  epsilon: 1.0    # then run: ./scripts/run_camda.sh BRCA eps1_k8
```

---

## Evaluating Results

From `~/Health-Privacy-Challenge/src/evaluation/`:

```bash
cd ~/Health-Privacy-Challenge
# Link the evaluation config (edit generator_name / experiment_name inside first)
cp experiments/track_i/blue_team/3_evaluation/config.yaml ./config.yaml
# edit /tmp/eval_config.yaml: generator_config.name = "private_pgm", experiment_name = "eps7_k8"

python src/evaluation/evaluate.py run-evaluator 1  # repeat for splits 2-5
python src/evaluation/evaluate.py combine-results
```

Metrics reported: `accuracy_synthetic`, `avg_pr_macro_synthetic`, `MMD_score`,
`discriminative_score`, `distance_to_closest`. See
`experiments/track_i/blue_team/3_evaluation/README.md` for baseline comparisons.

---

## Porting to the CAMDA Repo for Submission

The CAMDA repo (`~/Health-Privacy-Challenge/`) is assumed **unmodified** here. When preparing
a submission:

1. **Copy algorithm modules** into the CAMDA repo:
   ```
   cp src/discretization.py    ~/Health-Privacy-Challenge/src/generators/models/
   cp src/marginal_selection.py ~/Health-Privacy-Challenge/src/generators/models/
   cp src/pgm_fitter.py        ~/Health-Privacy-Challenge/src/generators/models/
   cp src/generator.py         ~/Health-Privacy-Challenge/src/generators/models/private_pgm_generator.py
   ```

2. **Write the CAMDA wrapper** `~/Health-Privacy-Challenge/src/generators/models/private_pgm.py`
   — a thin class that inherits `BaseDataGenerator`, reads the CAMDA config format, and
   delegates to `PrivatePGMRNASeqGenerator`. A draft of this wrapper was written earlier at
   that path but was removed to keep this repo standalone. Re-create it from `src/camda_runner.py`
   (the `run_split()` function contains equivalent logic). The wrapper must:
   - Inherit `BaseDataGenerator` from `generators.models.base`
   - Import modules with `from generators.models.{module} import ...` (absolute, not relative)
   - Implement `train()`, `generate()`, `generate_for_type()`, `load_from_checkpoint()`
   - Return `(pd.DataFrame, pd.DataFrame)` from `generate()`

3. **Register** in `src/generators/blue_team.py`:
   ```python
   'private_pgm': ('models.private_pgm', 'PrivatePGMDataGenerator'),
   ```

4. **Add config block** to the CAMDA `2_generation/config.yaml` under key `private_pgm_config`.

5. **Set a unique random seed** (required by CAMDA rules — each team must differ from 42).

---

## Track 2: scRNA-seq (OneK1K dataset)

**Status**: Implemented — `src/scrna_runner.py` + `configs/onek1k.yaml`.

### Quick start

```bash
# Default experiment (ε=7, K=4, 1118 HVGs, 100K-cell subsample)
./scripts/run_scrna.sh

# Custom experiment label (edit configs/onek1k.yaml first)
./scripts/run_scrna.sh eps10_k4_n200

# Or directly:
python src/scrna_runner.py configs/onek1k.yaml --experiment eps7_k4
```

Output: `results/scrna/synthetic_{experiment}.h5ad`

### Approach: Zero-inflated binning (Option B)

scRNA-seq data (~98% zeros) requires a different discretization strategy than bulk RNA-seq:

| Bin | Captures | Inverse decode |
|---|---|---|
| 0 | All zero counts (structural zeros) | → 0 (exact) |
| 1..K-1 | Equal-depth quantile bins over non-zero counts | → dithered uniform in bin range → rounded to int |

Non-HVG genes (all genes outside the 1,118 HVGs) are set to 0 in the output,
matching the scDesign2 convention.

### Key scRNA differences vs. bulk

| | Bulk (Track 1) | scRNA (Track 2) |
|---|---|---|
| Input data | VST-normalised log-counts | Raw integer counts |
| Sparsity | ~0% | ~98% |
| Label col | `Subtype` | `cell_label` (14 cell types) |
| `zero_inflated` | `False` | `True` |
| Default K | 8 | 4 |
| Runner | `src/camda_runner.py` | `src/scrna_runner.py` |
| Config | `configs/BRCA.yaml` | `configs/onek1k.yaml` |
| Output | CSV | h5ad |

### Memory and runtime

The full dataset is 1,267,733 cells × 25,834 genes. To keep fitting tractable:

- HVG columns are extracted from the backed h5ad in 50K-row chunks (~200 MB peak per chunk)
- A **stratified subsample** of up to `max_cells_subsample` cells (default: 100K) is
  densified and used for marginal selection + PGM fitting
- At ε=7 with 1,268 marginals, ε_per_marginal ≈ 0.006 and σ ≈ 850; with 100K cells
  the signal-to-noise ratio is ~118× — DP noise dominates, so subsampling is lossless
- Generating `n_synth_samples=-1` (full 1.27M cells) requires ~5.7 GB dense intermediate;
  set `n_synth_samples: 200000` if RAM is limited

### Config parameters (configs/onek1k.yaml)

| Key | Default | Meaning |
|---|---|---|
| `n_bins` | 4 | K bins (bin 0 = zeros; bins 1..K-1 = nonzero quantiles) |
| `n_1way` | 1118 | All HVG genes as 1-way marginals |
| `n_2way` | 150 | Top gene–gene pairs by \|Spearman\| |
| `max_cells_subsample` | 100000 | Max cells for marginal selection + fitting |
| `n_synth_samples` | -1 | −1 = match full dataset size (1.27M) |
| `epsilon` | 7.0 | Privacy budget ε |

### Future: ZINB→Gaussian encoding (Option A)

An alternative encoding that may improve fidelity:
1. Per-gene, fit a ZINB distribution (π, μ, r) on the nonzero counts
2. Map each count through the ZINB CDF → uniform [0,1] → inverse-normal → Gaussian
3. Run the existing bulk pipeline on the Gaussian-transformed data
4. Decode: inverse-normal → ZINB quantile → rounded integer count

This would be a novel contribution. Not yet implemented.

---

## DP Posture

Marginal selection uses the private training data without spending formal DP budget (selection
is informal). All ε goes to measuring the marginals with the Gaussian mechanism. The pipeline
provides empirical DP-like protection expected to pass CAMDA's MIA-based evaluation.

A fully formal end-to-end guarantee would require using a public reference dataset for
marginal selection (e.g. GTEx or TCGA held-out cohort) — noted as future work.

---

## Key References

- McKenna et al. (2019). "Graphical-model based estimation and inference for differential privacy." ICML.
- McKenna et al. (2021). "Winning the NIST Contest." TPDP.
- CAMDA 2025 Track 1 winner (naive Private-PGM: 1-way genes + 2-way gene×label).
