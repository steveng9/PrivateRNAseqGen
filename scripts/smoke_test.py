"""
End-to-end smoke test for the Private-PGM RNA-seq generator.

Uses small synthetic data so it runs quickly. Verifies:
  - Discretization round-trips without errors
  - Marginal selection returns the right structure
  - Private-PGM fitting and sampling complete without errors
  - Output shapes are correct

Run from the project root:
    python scripts/smoke_test.py
"""

import sys
import os
import numpy as np

# Add src/ to path so modules import each other without package prefix
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))

from discretization import Discretizer
from marginal_selection import HierarchicalMarginalSelector
from generator import PrivatePGMRNASeqGenerator


# -------------------------------------------------------------------
# Tiny synthetic dataset
# -------------------------------------------------------------------
N_SAMPLES = 200
N_GENES = 50
N_CLASSES = 3
SEED = 42

rng = np.random.default_rng(SEED)
X = rng.lognormal(mean=5.0, sigma=1.5, size=(N_SAMPLES, N_GENES)).astype(np.float32)
y = rng.integers(0, N_CLASSES, size=N_SAMPLES)
y_str = np.array([f"class_{i}" for i in y])
gene_names = [f"GENE_{i:04d}" for i in range(N_GENES)]

print("=" * 60)
print("Smoke test: Private-PGM RNA-seq generator")
print(f"  Data: {N_SAMPLES} samples × {N_GENES} genes, {N_CLASSES} classes")
print("=" * 60)

# -------------------------------------------------------------------
# 1. Discretization
# -------------------------------------------------------------------
print("\n[1] Discretization...")
disc = Discretizer(n_bins=4)
X_disc = disc.fit_transform(X)
assert X_disc.shape == X.shape
assert X_disc.min() >= 0
assert X_disc.max() <= 3
X_recon = disc.inverse_transform(X_disc)
assert X_recon.shape == X.shape
print(f"    X_disc range: [{X_disc.min()}, {X_disc.max()}]  OK")

# -------------------------------------------------------------------
# 2. Marginal selection
# -------------------------------------------------------------------
print("\n[2] Marginal selection...")
selector = HierarchicalMarginalSelector(
    n_1way=20,
    n_2way=10,
    n_3way=5,
    n_4way=2,
    include_label_marginals=True,
)
marginals = selector.select(X, gene_names, label_col="label")

assert len(marginals["1way"]) <= 20
assert len(marginals["2way"]) <= 10 + 20  # gene-gene + gene×label
assert len(marginals["3way"]) <= 5
assert len(marginals["4way"]) <= 2
for order, cliques in marginals.items():
    print(f"    {order}: {len(cliques)} cliques")

# -------------------------------------------------------------------
# 3. Full generator: fit + generate
# -------------------------------------------------------------------
print("\n[3] Generator fit + generate...")
gen = PrivatePGMRNASeqGenerator(
    epsilon=7.0,
    delta=1e-5,
    n_bins=4,
    n_1way=20,
    n_2way=10,
    n_3way=5,
    n_4way=2,
    include_label_marginals=True,
    pgm_iters=200,   # fewer iters for speed
    random_seed=SEED,
)
gen.fit(X, y_str, gene_names=gene_names)

N_SYNTH = 150
X_syn, y_syn = gen.generate(N_SYNTH)
assert X_syn.shape == (N_SYNTH, gen.n_selected_genes), (
    f"Expected ({N_SYNTH}, {gen.n_selected_genes}), got {X_syn.shape}"
)
assert y_syn.shape == (N_SYNTH,)
assert set(y_syn).issubset({f"class_{i}" for i in range(N_CLASSES)}), (
    f"Unexpected labels: {set(y_syn)}"
)

print(f"    X_syn shape:  {X_syn.shape}")
print(f"    y_syn shape:  {y_syn.shape}")
print(f"    Label distribution: { {lbl: (y_syn == lbl).sum() for lbl in sorted(set(y_syn))} }")
print(f"    X_syn range: [{X_syn.min():.2f}, {X_syn.max():.2f}]")

print("\n" + "=" * 60)
print("All smoke tests PASSED.")
print("=" * 60)
