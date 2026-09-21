#!/usr/bin/env python3
"""
Sweep script: generate synthetic data for a grid of experiments.

Varies epsilon, n_bins, joint_mode, and n_2way. Runs all 5 splits by default
(or a single split for fast iteration). Results land in the same CAMDA-format
directory that eval_label_free.py reads from.

Usage:
    python scripts/run_sweep.py                        # BRCA, all 5 splits
    python scripts/run_sweep.py --split 1              # BRCA, split 1 only
    python scripts/run_sweep.py --dataset COMBINED     # COMBINED, all splits
    python scripts/run_sweep.py --experiments joint    # only joint-mode experiments
"""

import argparse
import copy
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import camda_runner  # noqa: E402 — after sys.path insert

# ---------------------------------------------------------------------------
# Experiment grid
# Naming: k{bins}_{joint|strat}_eps{epsilon}
# joint_mode=True  → single PGM, label as a node, gene×label 2-way marginals
#                    (replicates last year's CAMDA winner structure)
# joint_mode=False → stratified PGM per class, gene×gene 2-way marginals
#                    (StratHiM-PGM, our contribution)
# ---------------------------------------------------------------------------
SWEEP = [
    # (experiment_name,       epsilon,   n_bins, joint_mode, n_1way, n_2way, max_degree)
    # --- Replicate last year: joint PGM, K=4, no gene-gene pairs ---
    #("k4_joint_eps10",        10.0,      4,      True,       978,    0,     None),
    #("k4_joint_eps1000",      1000.0,    4,      True,       978,    0,     None),
    # --- StratHiM-PGM: stratified, K=4, gene-gene pairs, degree-limited ---
    # max_degree=4: each gene participates in at most 4 pairs, bounding JT clique size
    ("k4_strat_eps10",        10.0,      4,      False,      978,    150,   4),
    ("k4_strat_eps1000",      1000.0,    4,      False,      978,    150,   4),
]

EXPERIMENT_GROUPS = {
    "joint":   [e for e in SWEEP if "joint" in e[0]],
    "strat":   [e for e in SWEEP if "strat" in e[0]],
    "eps10":   [e for e in SWEEP if "eps10" in e[0]],
    "eps1000": [e for e in SWEEP if "eps1000" in e[0]],
    "all":     SWEEP,
}


def load_base_config(dataset: str) -> dict:
    config_path = os.path.join(PROJECT_ROOT, "configs", f"{dataset}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(base: dict, epsilon, n_bins, joint_mode, n_1way, n_2way, max_degree) -> dict:
    cfg = copy.deepcopy(base)
    g = cfg["generator"]
    g["epsilon"]    = epsilon
    g["n_bins"]     = n_bins
    g["joint_mode"] = joint_mode
    g["n_1way"]     = n_1way
    g["n_2way"]     = n_2way
    g["n_3way"]     = 0
    g["n_4way"]     = 0
    g["budget_weights"] = [0.33, 0.67, 0.0, 0.0]
    g["max_degree"] = max_degree
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run StratHiM-PGM experiment sweep")
    parser.add_argument("--dataset",     default="BRCA", choices=["BRCA", "COMBINED"])
    parser.add_argument("--split",       default="all",
                        help="Split number (1-5) or 'all'")
    parser.add_argument("--experiments", default="all",
                        choices=list(EXPERIMENT_GROUPS.keys()),
                        help="Which experiment group to run (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments whose output files already exist")
    args = parser.parse_args()

    base_config = load_base_config(args.dataset)
    num_splits  = base_config["data"].get("num_splits", 5)
    splits      = list(range(1, num_splits + 1)) if args.split == "all" else [int(args.split)]
    experiments = EXPERIMENT_GROUPS[args.experiments]
    dataset     = f"TCGA-{args.dataset}"

    print(f"Dataset:     {dataset}")
    print(f"Splits:      {splits}")
    print(f"Experiments: {[e[0] for e in experiments]}")
    print()

    for exp_name, epsilon, n_bins, joint_mode, n_1way, n_2way, max_degree in experiments:
        cfg = apply_overrides(base_config, epsilon, n_bins, joint_mode, n_1way, n_2way, max_degree)

        for split_no in splits:
            if args.skip_existing:
                syn_dir = os.path.expanduser(cfg["output"]["synthetic_dir"])
                out_path = os.path.join(
                    syn_dir, dataset, "synthetic", "private_pgm",
                    exp_name, f"synthetic_data_split_{split_no}.csv"
                )
                if os.path.exists(out_path):
                    print(f"  [SKIP] {exp_name} split {split_no} — output exists")
                    continue

            camda_runner.run_split(cfg, split_no, exp_name)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
