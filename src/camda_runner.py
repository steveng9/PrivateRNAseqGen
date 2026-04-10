"""
Standalone CAMDA runner for the Private-PGM RNA-seq generator.

Reads/writes data in the same CSV layout the CAMDA evaluation pipeline
expects, without depending on the Health-Privacy-Challenge codebase.

Layout (mirroring CAMDA conventions):
  {real_splits_dir}/{dataset}/real/
    X_train_real_split_{N}.csv
    y_train_real_split_{N}.csv
    column_names.csv

  {synthetic_dir}/{dataset}/synthetic/private_pgm/{experiment_name}/
    synthetic_data_split_{N}.csv
    synthetic_labels_split_{N}.csv

  {model_dir}/{dataset}/{experiment_name}/split_{N}.pkl

Usage (typically via scripts/run_camda.sh):
    python src/camda_runner.py configs/BRCA.yaml --split 1 --experiment eps7_k8
    python src/camda_runner.py configs/BRCA.yaml --split all --experiment eps7_k8
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# Ensure sibling modules are importable when called as a script
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from generator import StratHiMPGMGenerator


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

def _expand(path: str) -> str:
    return os.path.expanduser(path)


def load_split(real_splits_dir: str, dataset: str, split_no: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the pre-split training CSVs produced by the CAMDA data-split step."""
    real_dir = os.path.join(_expand(real_splits_dir), dataset, "real")
    X = pd.read_csv(os.path.join(real_dir, f"X_train_real_split_{split_no}.csv"))
    y = pd.read_csv(os.path.join(real_dir, f"y_train_real_split_{split_no}.csv"))
    cols = pd.read_csv(os.path.join(real_dir, "column_names.csv")).values.flatten().tolist()
    return X.values, y.values.ravel(), cols


def save_synthetic(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    gene_names: list[str],
    subtype_col: str,
    synthetic_dir: str,
    dataset: str,
    experiment_name: str,
    split_no: int,
) -> None:
    out_dir = os.path.join(
        _expand(synthetic_dir), dataset, "synthetic", "private_pgm", experiment_name
    )
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(X_syn, columns=gene_names).to_csv(
        os.path.join(out_dir, f"synthetic_data_split_{split_no}.csv"), index=False
    )
    pd.DataFrame({subtype_col: y_syn}).to_csv(
        os.path.join(out_dir, f"synthetic_labels_split_{split_no}.csv"), index=False
    )
    print(f"  Saved synthetic data → {out_dir}/synthetic_*_split_{split_no}.csv")


def save_checkpoint(pgm: StratHiMPGMGenerator, model_dir: str, dataset: str, experiment_name: str, split_no: int) -> None:
    ckpt_dir = os.path.join(_expand(model_dir), dataset, experiment_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"split_{split_no}.pkl")
    with open(path, "wb") as f:
        pickle.dump(pgm, f)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(model_dir: str, dataset: str, experiment_name: str, split_no: int) -> StratHiMPGMGenerator:
    path = os.path.join(_expand(model_dir), dataset, experiment_name, f"split_{split_no}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_split(config: dict, split_no: int, experiment_name: str) -> None:
    data_cfg = config["data"]
    gen_cfg  = config["generator"]
    out_cfg  = config["output"]

    dataset    = data_cfg["dataset"]
    subtype_col = data_cfg["subtype_col"]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  |  Split: {split_no}  |  Experiment: {experiment_name}")
    print(f"{'='*60}")

    # Load data
    X, y, gene_names = load_split(data_cfg["real_splits_dir"], dataset, split_no)
    n_train = len(X)
    print(f"  Training data: {n_train} samples × {len(gene_names)} genes, "
          f"{len(set(y))} classes")

    # Build generator from config
    budget_weights = tuple(gen_cfg.get("budget_weights", [0.50, 0.25, 0.15, 0.10]))
    pgm = StratHiMPGMGenerator(
        epsilon        = gen_cfg.get("epsilon", 7.0),
        delta          = gen_cfg.get("delta", 1e-5),
        n_bins         = gen_cfg.get("n_bins", 8),
        n_1way         = gen_cfg.get("n_1way", 1000),
        n_2way         = gen_cfg.get("n_2way", 150),
        n_3way         = gen_cfg.get("n_3way", 30),
        n_4way         = gen_cfg.get("n_4way", 7),
        budget_weights = budget_weights,
        pgm_iters      = gen_cfg.get("pgm_iters", 1000),
        joint_mode     = gen_cfg.get("joint_mode", False),
        random_seed    = gen_cfg.get("random_seed", 42),
    )

    # Fit
    pgm.fit(X, y, gene_names=gene_names)
    save_checkpoint(pgm, out_cfg["model_dir"], dataset, experiment_name, split_no)

    # Generate
    n_synth = gen_cfg.get("n_synth_samples", -1)
    if n_synth == -1:
        n_synth = n_train
    X_syn, y_syn = pgm.generate(n_synth)

    label_counts = {lbl: int((y_syn == lbl).sum()) for lbl in sorted(set(y_syn))}
    print(f"  Generated {n_synth} samples across {len(label_counts)} classes")

    # Save
    save_synthetic(
        X_syn, y_syn, pgm.selected_gene_names, subtype_col,
        out_cfg["synthetic_dir"], dataset, experiment_name, split_no
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Private-PGM for CAMDA Track I")
    parser.add_argument("config", help="Path to config YAML (e.g. configs/BRCA.yaml)")
    parser.add_argument("--split", default="all",
                        help="Split number (1–5) or 'all'. Default: all")
    parser.add_argument("--experiment", default="eps7_k8",
                        help="Experiment label for output folder. Default: eps7_k8")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    num_splits = config["data"].get("num_splits", 5)
    splits = list(range(1, num_splits + 1)) if args.split == "all" else [int(args.split)]

    for split_no in splits:
        run_split(config, split_no, args.experiment)

    print(f"\nAll splits complete.")


if __name__ == "__main__":
    main()
