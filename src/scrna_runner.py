"""
scRNA-seq runner for StratHiM-PGM (Track 2, CAMDA 2026).

Approach: Zero-inflated binning
  - Bin 0 captures all zero counts (structural zeros in scRNA-seq).
  - Bins 1..K-1 are equal-depth quantile bins over non-zero counts only.
  - Marginal selection and PGM fitting run on a stratified subsample
    (default: 100,000 cells) to keep memory bounded.
  - Synthetic outputs are rounded to non-negative integers (raw count space).
  - Non-HVG genes are filled with zeros in the output (matching scDesign2).

Memory notes:
  - HVG sparse matrix (all cells): ~100–200 MB
  - Dense subsample (100K × 1118 genes): ~450 MB
  - Generated output (n_synth × 1118 dense before sparse conversion):
    set n_synth_samples < 500_000 if RAM is limited.

Usage:
    python src/scrna_runner.py configs/onek1k.yaml --experiment eps7_k4
    # or via shell entry point:
    ./scripts/run_scrna.sh eps7_k4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

# Ensure sibling modules are importable when called as a script
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from generator import StratHiMPGMGenerator


def _expand(path: str) -> str:
    return os.path.expanduser(path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hvg_data(
    h5ad_path: str,
    hvg_csv_path: str,
    label_col: str = "cell_label",
    chunk_size: int = 50_000,
) -> tuple[sp.csr_matrix, np.ndarray, list[str], list[str]]:
    """
    Load h5ad and extract HVG columns as a compact sparse matrix.

    Reads the backed h5ad in row-chunks to avoid materialising the full
    dense matrix (~130 GB for this dataset).  Only HVG columns are
    retained, resulting in a ~100 MB sparse CSR matrix.

    Returns
    -------
    X_hvg        : sparse CSR float32, shape (n_cells, n_hvg)
    y            : cell-type label strings, shape (n_cells,)
    hvg_names    : list[str], HVG gene names in h5ad column order
    all_gene_names : list[str], all gene names in h5ad (for zero-padding output)
    """
    import anndata as ad

    hvg_df = pd.read_csv(_expand(hvg_csv_path), index_col=0)
    hvg_gene_set = set(hvg_df.index[hvg_df["highly_variable"]].tolist())

    print(f"[scrna] Loading h5ad (backed) from {h5ad_path} ...")
    adata = ad.read_h5ad(_expand(h5ad_path), backed="r")
    n_cells, n_genes = adata.shape
    all_gene_names = list(adata.var_names)
    print(f"[scrna]   h5ad shape: {n_cells:,} × {n_genes:,}")

    # HVG column indices in h5ad order
    hvg_names = [g for g in all_gene_names if g in hvg_gene_set]
    hvg_col_idx = np.array(
        [i for i, g in enumerate(all_gene_names) if g in hvg_gene_set],
        dtype=np.int32,
    )
    print(f"[scrna]   {len(hvg_names)} HVG columns selected")

    y = adata.obs[label_col].values.astype(str)

    # Read sparse rows in chunks, slice to HVG columns, stack
    print(f"[scrna] Extracting HVG columns in chunks of {chunk_size:,} ...")
    chunks: list[sp.csr_matrix] = []
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk = adata.X[start:end]
        if not sp.issparse(chunk):
            chunk = sp.csr_matrix(chunk)
        chunk_hvg = chunk[:, hvg_col_idx].tocsr().astype(np.float32)
        chunks.append(chunk_hvg)
        if (start // chunk_size) % 5 == 0:
            print(f"[scrna]   {end:>9,} / {n_cells:,}  ({100*end/n_cells:.0f}%)")

    X_hvg = sp.vstack(chunks, format="csr")
    density = X_hvg.nnz / (X_hvg.shape[0] * X_hvg.shape[1])
    print(f"[scrna] HVG matrix: {X_hvg.shape}, density={density:.4f}")

    return X_hvg, y, hvg_names, all_gene_names


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def stratified_subsample(
    X_sparse: sp.csr_matrix,
    y: np.ndarray,
    max_cells: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Stratified subsample up to max_cells cells and densify.

    The true full-dataset class counts are always returned so that
    proportional generation later reflects the real distribution.

    Returns
    -------
    X_dense          : float32 ndarray, shape (n_sub, n_hvg)
    y_sub            : label array for the subsample
    true_class_counts : full-dataset per-class counts
    """
    n_total = X_sparse.shape[0]
    classes, counts = np.unique(y, return_counts=True)
    true_class_counts: dict[str, int] = dict(zip(classes, counts.tolist()))

    if n_total <= max_cells:
        print(f"[scrna] Using all {n_total:,} cells (≤ max_cells={max_cells:,})")
        return X_sparse.toarray().astype(np.float32), y, true_class_counts

    print(f"[scrna] Stratified subsampling: {n_total:,} → ≤{max_cells:,} cells ...")
    selected: list[int] = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, round(max_cells * int(cnt) / n_total))
        n_take = min(n_take, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_take, replace=False)
        selected.extend(chosen.tolist())

    selected_arr = np.array(sorted(selected))
    print(f"[scrna]   Subsample size: {len(selected_arr):,}")
    X_dense = X_sparse[selected_arr].toarray().astype(np.float32)
    y_sub = y[selected_arr]
    return X_dense, y_sub, true_class_counts


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_synthetic_h5ad(
    X_syn_hvg: np.ndarray,
    y_syn: np.ndarray,
    selected_gene_names: list[str],
    all_gene_names: list[str],
    out_path: str,
) -> None:
    """
    Save synthetic data as h5ad with zeros for non-selected genes.

    The output h5ad has the same gene ordering as the input h5ad.
    Selected (HVG) genes carry synthetic counts; all other genes are 0.
    Stored as float32 sparse CSR.
    """
    import anndata as ad

    n_syn = X_syn_hvg.shape[0]
    n_all = len(all_gene_names)

    all_gene_idx = {g: i for i, g in enumerate(all_gene_names)}
    selected_col_in_all = np.array(
        [all_gene_idx[g] for g in selected_gene_names], dtype=np.int32
    )

    # Sparse COO assembly
    row_idx, col_local = np.nonzero(X_syn_hvg)
    if len(row_idx) > 0:
        col_global = selected_col_in_all[col_local]
        vals = X_syn_hvg[row_idx, col_local]
        X_full = sp.coo_matrix(
            (vals, (row_idx, col_global)), shape=(n_syn, n_all), dtype=np.float32
        ).tocsr()
    else:
        X_full = sp.csr_matrix((n_syn, n_all), dtype=np.float32)

    obs = pd.DataFrame(
        {"cell_label": y_syn},
        index=[f"syn_{i}" for i in range(n_syn)],
    )
    var = pd.DataFrame(index=all_gene_names)

    adata_syn = ad.AnnData(X=X_full, obs=obs, var=var)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    adata_syn.write_h5ad(out_path)
    print(f"[scrna] Synthetic h5ad saved → {out_path}")
    print(f"[scrna]   {n_syn:,} cells × {n_all:,} genes  "
          f"({len(selected_gene_names)} non-zero gene columns)")


def save_checkpoint(
    pgm: StratHiMPGMGenerator,
    model_dir: str,
    experiment_name: str,
) -> None:
    os.makedirs(_expand(model_dir), exist_ok=True)
    path = os.path.join(_expand(model_dir), f"{experiment_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pgm, f)
    print(f"[scrna] Checkpoint saved → {path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scrna(config: dict, experiment_name: str) -> None:
    data_cfg = config["data"]
    gen_cfg  = config["generator"]
    out_cfg  = config["output"]

    seed = gen_cfg.get("random_seed", 123)
    rng  = np.random.default_rng(seed)

    # --- Load HVG data ---
    X_sparse, y, hvg_names, all_gene_names = load_hvg_data(
        h5ad_path    = data_cfg["h5ad_path"],
        hvg_csv_path = data_cfg["hvg_csv_path"],
        label_col    = data_cfg.get("label_col", "cell_label"),
    )

    # --- Stratified subsample for fitting ---
    max_cells = gen_cfg.get("max_cells_subsample", 100_000)
    X_sub, y_sub, true_class_counts = stratified_subsample(
        X_sparse, y, max_cells, rng
    )
    print(f"[scrna] Class counts (full dataset):")
    for cls in sorted(true_class_counts):
        print(f"  {cls:<14s}: {true_class_counts[cls]:>7,}")

    # --- Build and fit generator ---
    budget_weights = tuple(gen_cfg.get("budget_weights", [0.33, 0.67, 0.0, 0.0]))
    pgm = StratHiMPGMGenerator(
        epsilon        = gen_cfg.get("epsilon", 7.0),
        delta          = gen_cfg.get("delta", 1e-5),
        n_bins         = gen_cfg.get("n_bins", 4),
        n_1way         = gen_cfg.get("n_1way", len(hvg_names)),
        n_2way         = gen_cfg.get("n_2way", 150),
        n_3way         = gen_cfg.get("n_3way", 0),
        n_4way         = gen_cfg.get("n_4way", 0),
        budget_weights = budget_weights,
        pgm_iters      = gen_cfg.get("pgm_iters", 500),
        joint_mode     = False,   # always stratified for scRNA-seq
        zero_inflated  = True,
        random_seed    = seed,
    )

    print(f"\n[scrna] Fitting StratHiM-PGM on "
          f"{X_sub.shape[0]:,} × {X_sub.shape[1]} subsample ...")
    pgm.fit(X_sub, y_sub, gene_names=hvg_names)

    # Override with true full-dataset class counts so proportional
    # generation reflects the real cell-type distribution, not the subsample
    pgm._class_counts = true_class_counts

    save_checkpoint(pgm, out_cfg["model_dir"], experiment_name)

    # --- Generate ---
    n_synth = gen_cfg.get("n_synth_samples", -1)
    if n_synth == -1:
        n_synth = X_sparse.shape[0]   # match full dataset size
    print(f"\n[scrna] Generating {n_synth:,} synthetic cells ...")
    X_syn, y_syn = pgm.generate(n_synth)

    # Round continuous dithered values back to non-negative integer counts
    X_syn = np.clip(np.round(X_syn), 0, None).astype(np.float32)

    label_dist = pd.Series(y_syn).value_counts().sort_index().to_dict()
    print(f"[scrna] Generated label distribution:")
    for cls, cnt in sorted(label_dist.items()):
        print(f"  {cls:<14s}: {cnt:>7,}")

    # --- Save ---
    out_dir = _expand(out_cfg["out_dir"])
    out_path = os.path.join(out_dir, f"synthetic_{experiment_name}.h5ad")
    save_synthetic_h5ad(
        X_syn, y_syn, pgm.selected_gene_names, all_gene_names, out_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run StratHiM-PGM for CAMDA Track II (scRNA-seq)"
    )
    parser.add_argument(
        "config", help="Path to config YAML (e.g. configs/onek1k.yaml)"
    )
    parser.add_argument(
        "--experiment", default="eps7_k4",
        help="Experiment label for output file name. Default: eps7_k4"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_scrna(config, args.experiment)
    print("\n[scrna] Done.")


if __name__ == "__main__":
    main()
