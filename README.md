# DP Synthetic RNA-seq via Hierarchical Marginal Selection + Private-PGM

**Status**: Prototype working — BRCA smoke-tested end-to-end  
**Competition**: CAMDA 2026, Track 1 (bulk RNA-seq) | Track 2 (scRNA-seq) stretch goal  
**Deadline**: May 15, 2026

---

## Overview

Differentially private synthetic bulk RNA-seq generator. The core innovation over last year's
CAMDA Track 1 winner: hierarchical selection of gene–gene marginals (not just gene×label),
giving the PGM richer correlation structure at the same ε.

**Architecture**: Stratified Private-PGM — one graphical model per cancer class, fitted on
that class's training samples. Marginal selection runs once on the full dataset to pick which
genes/pairs to measure. The same gene set is used across all class models.

---

## Repo Structure

```
configs/
  BRCA.yaml              ← all tunable parameters for TCGA-BRCA
  COMBINED.yaml          ← all tunable parameters for TCGA-COMBINED
src/
  discretization.py      ← quantile-bins genes into K discrete levels
  marginal_selection.py  ← hierarchical S→R→Q→P clique selection (Spearman)
  pgm_fitter.py          ← adds Gaussian noise, fits mbi.FactoredInference
  generator.py           ← top-level: stratified fit + generate
  camda_runner.py        ← standalone runner (reads/writes CAMDA CSV format)
scripts/
  smoke_test.py          ← fast end-to-end test on synthetic toy data
  run_camda.sh           ← entry point: run all 5 splits for one dataset
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
cp experiments/track_i/blue_team/3_evaluation/config.yaml /tmp/eval_config.yaml
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

## Track 2 Hook: scRNA-seq Adaptation

The discretizer already has `zero_inflated=True` mode (dedicated bin 0 for zeros).
The broader idea (not yet implemented):

1. **Encode** each gene's zero-inflated distribution into a pseudo-continuous space (e.g.
   CDF-transform: map zero→0, non-zero values→quantile rank in [0,1], producing a uniform
   marginal that is easier to discretize finely).
2. **Run the same pipeline** (marginal selection + Private-PGM) on the transformed data.
3. **Decode** synthetic values back through the inverse CDF to recover zero-inflated counts.

This would be a novel contribution. See `memory/project_scrna_future.md` for context.

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
