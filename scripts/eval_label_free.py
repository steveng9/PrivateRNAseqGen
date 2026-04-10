"""
Label-free evaluation of synthetic RNA-seq data.

Computes the fidelity and privacy metrics that do NOT require synthetic labels:
  - MMD_score          (lower = better fidelity)
  - discriminative_score (lower = better; 1.0 = trivially distinguishable)
  - distance_to_closest  (higher = better privacy proxy)
  - distance_to_closest_base (real-vs-real baseline, constant per split)
  - kl_mean_train
  - kl_mean_test

Usage
-----
# Single file
python scripts/eval_label_free.py \
    --syn  /path/to/synthetic_data.csv \
    --dataset TCGA-BRCA \
    --split 1 \
    --label my_method/experiment_name \
    --out   results/files/label_free_comparison.csv

# Batch mode (runs all --syn entries and appends to same --out)
python scripts/eval_label_free.py --batch --out results/files/label_free_comparison.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Reuse statistics utilities from Health-Privacy-Challenge
HPC_SRC = os.path.expanduser("~/Health-Privacy-Challenge/src")
sys.path.insert(0, HPC_SRC)
from evaluation.utils.stats import Statistics

HPC_HOME  = os.path.expanduser("~/Health-Privacy-Challenge")
SPLITS_DIR = os.path.join(HPC_HOME, "data_splits")
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Discriminative score (copied from evaluate.py so we don't need the class)
# ---------------------------------------------------------------------------
def discriminative_score(synthetic_data, X_train_real, X_test_real, seed=RANDOM_SEED):
    X_train = np.vstack([X_train_real, synthetic_data])
    y_train = np.array([1] * X_train_real.shape[0] + [0] * synthetic_data.shape[0])
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.3, random_state=seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    X_test_real_sc = scaler.transform(X_test_real)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    X_ext = np.vstack([X_test_real_sc, X_te])
    y_ext = np.concatenate([[1] * X_test_real.shape[0], y_te])
    X_ext, y_ext = shuffle(X_ext, y_ext, random_state=seed)
    return f1_score(y_ext, model.predict(X_ext))


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------
def evaluate_label_free(syn_csv: str, dataset: str, split: int) -> dict:
    """Return dict of label-free metrics for one synthetic dataset."""
    real_dir = os.path.join(SPLITS_DIR, dataset, "real")
    X_train_real = pd.read_csv(
        os.path.join(real_dir, f"X_train_real_split_{split}.csv")).values
    X_test_real  = pd.read_csv(
        os.path.join(real_dir, f"X_test_real_split_{split}.csv")).values

    syn_df = pd.read_csv(syn_csv)

    # Align columns if real data has more/fewer genes
    real_cols = pd.read_csv(
        os.path.join(real_dir, f"X_train_real_split_{split}.csv"),
        nrows=0).columns.tolist()
    if list(syn_df.columns) != real_cols:
        shared = [c for c in syn_df.columns if c in set(real_cols)]
        if not shared:
            raise ValueError(f"No shared gene columns between {syn_csv} and real data.")
        syn_df = syn_df[shared]
        train_df = pd.read_csv(
            os.path.join(real_dir, f"X_train_real_split_{split}.csv"))
        test_df  = pd.read_csv(
            os.path.join(real_dir, f"X_test_real_split_{split}.csv"))
        X_train_real = train_df[shared].values
        X_test_real  = test_df[shared].values

    syn = syn_df.values

    mmd       = Statistics.get_mmd_score(X_train_real, syn)
    disc      = discriminative_score(syn, X_train_real, X_test_real)
    dtc       = Statistics.distance_to_the_closest_neighbor(X_train_real, syn)
    dtc_base  = Statistics.distance_to_the_closest_neighbor(X_train_real, X_test_real)
    kl_train, _ = Statistics.compute_kl_divergences(syn, X_train_real)
    kl_test,  _ = Statistics.compute_kl_divergences(syn, X_test_real)

    return {
        "MMD_score":               round(float(mmd),  6),
        "discriminative_score":    round(float(disc), 6),
        "distance_to_closest":     float(dtc),
        "distance_to_closest_base":float(dtc_base),
        "kl_mean_train":           round(float(kl_train), 6),
        "kl_mean_test":            round(float(kl_test),  6),
        "n_syn_samples":           syn.shape[0],
        "n_genes":                 syn.shape[1],
    }


# ---------------------------------------------------------------------------
# Batch definitions
# ---------------------------------------------------------------------------
BRCA_REDTEAM = [
    # (label,                       syn_csv,                                                      dataset,     split)
    ("NoisyDiffusion/CAMDA25_winner",       "/home/golobs/data/CAMDA26/RED_TCGA-BRCA/synthetic_data_1.csv", "TCGA-BRCA", 1),
    ("PrivatePGM_eps10/CAMDA25_winner",     "/home/golobs/data/CAMDA26/RED_TCGA-BRCA/synthetic_data_2.csv", "TCGA-BRCA", 5),
    ("CVAE/CAMDA25_baseline",               "/home/golobs/data/CAMDA26/RED_TCGA-BRCA/synthetic_data_3.csv", "TCGA-BRCA", 1),
    ("MVN_noise07/CAMDA25_baseline",        "/home/golobs/data/CAMDA26/RED_TCGA-BRCA/synthetic_data_4.csv", "TCGA-BRCA", 1),
]

COMBINED_REDTEAM = [
    ("NoisyDiffusion/CAMDA25_winner",       "/home/golobs/data/CAMDA26/RED_TCGA-COMBINED/synthetic_data_1.csv", "TCGA-COMBINED", 4),
    ("PrivatePGM_eps10/CAMDA25_winner",     "/home/golobs/data/CAMDA26/RED_TCGA-COMBINED/synthetic_data_2.csv", "TCGA-COMBINED", 4),
    ("CVAE/CAMDA25_baseline",               "/home/golobs/data/CAMDA26/RED_TCGA-COMBINED/synthetic_data_3.csv", "TCGA-COMBINED", 1),
    ("MVN_noise07/CAMDA25_baseline",        "/home/golobs/data/CAMDA26/RED_TCGA-COMBINED/synthetic_data_4.csv", "TCGA-COMBINED", 1),
]

# New private-pgm experiments (BRCA, split 1)
BRCA_NEW_PGM_SYN_DIR = os.path.join(
    SPLITS_DIR, "TCGA-BRCA", "synthetic", "private_pgm")

BRCA_NEW_PGM = [
    ("private_pgm/eps100_k8_PQRS_978_0_0_0",
        os.path.join(BRCA_NEW_PGM_SYN_DIR, "eps100_k8_PQRS_978_0_0_0",   "synthetic_data_split_1.csv"), "TCGA-BRCA", 1),
    ("private_pgm/eps100_k8_PQRS_978_70_20_5",
        os.path.join(BRCA_NEW_PGM_SYN_DIR, "eps100_k8_PQRS_978_70_20_5", "synthetic_data_split_1.csv"), "TCGA-BRCA", 1),
    ("private_pgm/eps7_k8_PQRS_978_0_0_0",
        os.path.join(BRCA_NEW_PGM_SYN_DIR, "eps7_k8_PQRS_978_0_0_0",    "synthetic_data_split_1.csv"), "TCGA-BRCA", 1),
    ("private_pgm/eps7_k8_PQRS_978_20_10_3",
        os.path.join(BRCA_NEW_PGM_SYN_DIR, "eps7_k8_PQRS_978_20_10_3",  "synthetic_data_split_1.csv"), "TCGA-BRCA", 1),
]


def run_batch(entries, out_csv: str, dataset_tag: str):
    rows = []
    for label, syn_csv, dataset, split in entries:
        print(f"  [{dataset_tag}] {label} (split {split}) ...", flush=True)
        try:
            metrics = evaluate_label_free(syn_csv, dataset, split)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        row = {"method": label, "dataset": dataset_tag, "split": split, **metrics}
        rows.append(row)
        print(f"    MMD={metrics['MMD_score']:.4f}  disc={metrics['discriminative_score']:.4f}"
              f"  dtc={metrics['distance_to_closest']:.2f}  kl_train={metrics['kl_mean_train']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"  → saved to {out_csv}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Label-free synthetic data evaluation")
    parser.add_argument("--syn",     help="Path to synthetic data CSV")
    parser.add_argument("--dataset", default="TCGA-BRCA")
    parser.add_argument("--split",   type=int, default=1)
    parser.add_argument("--label",   default="unknown")
    parser.add_argument("--out",     default="results/label_free_comparison.csv")
    parser.add_argument("--batch",   action="store_true",
                        help="Run full batch: red-team methods + new private_pgm experiments")
    args = parser.parse_args()

    if args.batch:
        out = args.out
        print("\n=== TCGA-BRCA: Red Team (CAMDA 2025 blue team) methods ===")
        df_brca_red = run_batch(BRCA_REDTEAM, out, "TCGA-BRCA")

        print("\n=== TCGA-BRCA: New Private-PGM experiments ===")
        df_brca_new = run_batch(BRCA_NEW_PGM, out, "TCGA-BRCA")

        print("\n=== TCGA-COMBINED: Red Team (CAMDA 2025 blue team) methods ===")
        df_comb_red = run_batch(COMBINED_REDTEAM, out, "TCGA-COMBINED")

        print("\n=== Summary (TCGA-BRCA) ===")
        brca_all = pd.read_csv(out)
        brca_all = brca_all[brca_all["dataset"] == "TCGA-BRCA"]
        print(brca_all[["method","MMD_score","discriminative_score",
                          "distance_to_closest","kl_mean_train"]].to_string(index=False))
    else:
        if not args.syn:
            parser.error("Provide --syn or use --batch")
        print(f"Evaluating {args.label} on {args.dataset} split {args.split} ...")
        metrics = evaluate_label_free(args.syn, args.dataset, args.split)
        row = {"method": args.label, "dataset": args.dataset, "split": args.split, **metrics}
        df = pd.DataFrame([row])
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if os.path.exists(args.out):
            df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
        df.to_csv(args.out, index=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
