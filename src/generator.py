"""
Top-level StratHiM-PGM generator (Stratified Hierarchical Marginal-selection PGM).

Orchestrates:
  1. Hierarchical marginal selection on continuous data.
  2. Discretization of continuous gene expression values.
  3. Private-PGM fitting — either stratified (one PGM per class) or joint
     (one PGM over all data with label as a node).
  4. Sampling and decoding back to approximate continuous values.

Why stratified (one PGM per class)?  [default mode]
------------------------------------
Including the label as a node in a shared PGM creates edges between the label
and every selected gene. Combined with gene–gene edges from 2-way marginals,
this creates dense subgraphs whose junction tree can have exponential treewidth —
leading to out-of-memory errors.

Stratified fitting sidesteps this entirely: each per-class PGM has no label node,
only gene–gene structure. Label conditioning is exact because each model was
trained on a single class. From a DP standpoint this is valid: each person's data
appears in exactly one class's PGM, so ε is charged only once per person.

Joint mode  [joint_mode=True — for comparison with last year's approach]
----------
Fits a single PGM over all classes with the label as an explicit node.
All gene×label 2-way marginals are preserved (one per selected gene), plus
the top n_2way gene–gene pairs selected hierarchically.  This reproduces the
structure of last year's CAMDA winner while keeping hierarchical gene–gene
marginal selection.  Warning: label as a hub node increases treewidth — use
small n_bins / n_1way / n_2way when enabling this mode.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

import mbi

from discretization import Discretizer
from marginal_selection import HierarchicalMarginalSelector
from pgm_fitter import PrivatePGMFitter


_LABEL_COL = "__label__"   # internal column name for the label node in joint mode


class StratHiMPGMGenerator:
    """
    StratHiM-PGM: Stratified Hierarchical Marginal-selection Private-PGM generator.

    Default (stratified) mode: one Private-PGM per class label, no label node in
    the graph. Joint mode: single PGM with label as a node + all gene×label
    2-way marginals, for direct comparison with last year's approach.

    Parameters
    ----------
    epsilon : float
        Total privacy budget ε applied per-class PGM. Default: 7.0.
    delta : float
        δ for the Gaussian mechanism. Default: 1e-5.
    n_bins : int
        Discrete bins per gene (k in the design plan). Default: 8.
    n_1way : int
        Top-S genes by variance to include as 1-way marginals.
    n_2way : int
        Top-R gene–gene pairs by |Spearman| to include as 2-way marginals.
        In joint_mode, gene×label marginals for all n_1way genes are always
        included on top of these.
    n_3way : int
        Top-Q gene triples built from genes in top pairs. Default: 0 (off).
    n_4way : int
        Top-P gene quads built from genes in top triples. Default: 0 (off).
    budget_weights : tuple[float, ...]
        Fraction of ε for each marginal order. Must sum to 1.
        If n_3way=0 and n_4way=0, only the first two entries are used and
        they are automatically renormalised.
    pgm_iters : int
        FactoredInference optimisation iterations.
    joint_mode : bool
        If True, fit a single joint PGM over all classes with the label as a
        node and all gene×label 2-way marginals included.  This mirrors last
        year's CAMDA approach while retaining hierarchical gene–gene selection.
        WARNING: label as a hub increases treewidth — use small params.
        Default: False (stratified mode).
    zero_inflated : bool
        Use zero-dedicated bin 0 (for scRNA-seq sparsity). Default: False.
    random_seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        epsilon: float = 7.0,
        delta: float = 1e-5,
        n_bins: int = 8,
        n_1way: int = 1000,
        n_2way: int = 150,
        n_3way: int = 0,
        n_4way: int = 0,
        budget_weights: tuple[float, ...] = (0.50, 0.25, 0.15, 0.10),
        pgm_iters: int = 1000,
        joint_mode: bool = False,
        zero_inflated: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.n_bins = n_bins
        self.n_1way = n_1way
        self.n_2way = n_2way
        self.n_3way = n_3way
        self.n_4way = n_4way
        self.budget_weights = budget_weights
        self.pgm_iters = pgm_iters
        self.joint_mode = joint_mode
        self.zero_inflated = zero_inflated
        self.random_seed = random_seed

        # Set after fit()
        self._discretizer: Optional[Discretizer] = None
        self._class_fitters: dict[str, PrivatePGMFitter] = {}   # label → fitter (stratified)
        self._joint_fitter: Optional[PrivatePGMFitter] = None   # single fitter (joint)
        self._label_encoder: dict[str, int] = {}                 # label_str → int (joint)
        self._label_decoder: dict[int, str] = {}                 # int → label_str (joint)
        self._class_counts: dict[str, int] = {}                  # label → n_train
        self._gene_names: list[str] = []
        self._selected_gene_names: list[str] = []
        self._marginals: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gene_names: Optional[list[str]] = None,
    ) -> "StratHiMPGMGenerator":
        """
        Fit Private-PGM model(s) on continuous gene expression data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_genes)
            Continuous gene expression (e.g. VST-normalised bulk RNA-seq).
        y : np.ndarray, shape (n_samples,)
            Class labels (strings or ints).
        gene_names : list[str], optional
            Column identifiers. Defaults to "gene_0", "gene_1", ...
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        n_samples, n_genes = X.shape
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(n_genes)]
        self._gene_names = list(gene_names)
        y_str = np.array([str(lbl) for lbl in y])

        unique_classes = sorted(set(y_str))
        for cls in unique_classes:
            self._class_counts[cls] = int((y_str == cls).sum())

        # --- Step 1: Marginal selection on the full dataset ---
        mode_label = "joint" if self.joint_mode else "stratified"
        print(f"[generator] Step 1: Hierarchical marginal selection (full dataset, {mode_label} mode)...")
        n_1way_actual = min(self.n_1way, n_genes)
        selector = HierarchicalMarginalSelector(
            n_1way=n_1way_actual,
            n_2way=self.n_2way,
            n_3way=self.n_3way,
            n_4way=self.n_4way,
            include_label_marginals=self.joint_mode,
        )
        label_col = _LABEL_COL if self.joint_mode else None
        self._marginals = selector.select(X, self._gene_names, label_col=label_col)

        selected_genes = [clique[0] for clique in self._marginals["1way"]]
        self._selected_gene_names = selected_genes
        selected_idx = [self._gene_names.index(g) for g in selected_genes]
        X_selected = X[:, selected_idx]

        # Resolve budget weights, renormalising if higher orders are disabled
        weights = self._resolve_budget_weights()

        # --- Step 2: Discretize selected genes (fit on full dataset) ---
        print("[generator] Step 2: Fitting discretizer on full dataset...")
        self._discretizer = Discretizer(
            n_bins=self.n_bins, zero_inflated=self.zero_inflated
        )
        self._discretizer.fit(X_selected)

        if self.joint_mode:
            self._fit_joint(X_selected, y_str, selected_genes, unique_classes, weights)
        else:
            self._fit_stratified(X_selected, y_str, selected_genes, unique_classes, weights)

        print("[generator] Fitting complete.")
        return self

    def _fit_stratified(
        self,
        X_selected: np.ndarray,
        y_str: np.ndarray,
        selected_genes: list[str],
        unique_classes: list[str],
        weights: tuple[float, ...],
    ) -> None:
        """Fit one PGM per class (default mode)."""
        print(f"[generator] Step 3: Fitting per-class PGMs ({len(unique_classes)} classes)...")
        domain = mbi.Domain(selected_genes, [self.n_bins] * len(selected_genes))

        for cls in unique_classes:
            mask = y_str == cls
            print(f"  Class '{cls}': {mask.sum()} samples")
            X_disc = self._discretizer.transform(X_selected[mask])
            df_cls = pd.DataFrame(X_disc, columns=selected_genes)

            fitter = PrivatePGMFitter(
                epsilon=self.epsilon,
                delta=self.delta,
                budget_weights=weights,
                pgm_iters=self.pgm_iters,
            )
            fitter.fit(df_cls, domain, self._marginals)
            self._class_fitters[cls] = fitter

    def _fit_joint(
        self,
        X_selected: np.ndarray,
        y_str: np.ndarray,
        selected_genes: list[str],
        unique_classes: list[str],
        weights: tuple[float, ...],
    ) -> None:
        """Fit a single joint PGM over all classes with label as a node."""
        n_classes = len(unique_classes)
        self._label_encoder = {cls: i for i, cls in enumerate(unique_classes)}
        self._label_decoder = {i: cls for i, cls in enumerate(unique_classes)}

        n_label_marginals = len(self._marginals.get("2way", []))
        if n_label_marginals > 0 and weights[1] == 0.0:
            warnings.warn(
                f"joint_mode=True added {n_label_marginals} gene×label 2-way marginals, "
                "but budget_weights[1]=0.0 — these marginals will NOT be measured. "
                "Set a non-zero 2-way budget weight to actually capture label structure.",
                stacklevel=3,
            )

        print(f"[generator] Step 3: Fitting joint PGM ({n_classes} classes as label node)...")

        X_disc = self._discretizer.transform(X_selected)
        y_int = np.array([self._label_encoder[c] for c in y_str])

        df_all = pd.DataFrame(X_disc, columns=selected_genes)
        df_all[_LABEL_COL] = y_int

        domain = mbi.Domain(
            selected_genes + [_LABEL_COL],
            [self.n_bins] * len(selected_genes) + [n_classes],
        )

        fitter = PrivatePGMFitter(
            epsilon=self.epsilon,
            delta=self.delta,
            budget_weights=weights,
            pgm_iters=self.pgm_iters,
        )
        fitter.fit(df_all, domain, self._marginals)
        self._joint_fitter = fitter

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample synthetic data from the fitted model(s).

        Stratified mode: samples proportionally from each per-class PGM.
        Joint mode: samples from the single joint PGM (label proportions emerge
        naturally from the model's learned distribution).

        Returns
        -------
        X_synthetic : np.ndarray, shape (n_samples, n_selected_genes)
        y_synthetic : np.ndarray, shape (n_samples,)
        """
        if self.joint_mode:
            return self._generate_joint(n_samples)
        return self._generate_stratified(n_samples)

    def _generate_stratified(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._class_fitters:
            raise RuntimeError("Call fit() before generate().")

        total_train = sum(self._class_counts.values())
        rng = np.random.default_rng(self.random_seed)
        X_parts, y_parts = [], []

        for cls, fitter in self._class_fitters.items():
            # Proportional allocation, at least 1 sample per class
            n_cls = max(1, round(n_samples * self._class_counts[cls] / total_train))
            synth_df = fitter.sample(n_cls)

            X_disc = synth_df[self._selected_gene_names].values.astype(np.int32)
            X_disc = np.clip(X_disc, 0, self.n_bins - 1)
            # dither=True samples uniformly within each bin so the output is
            # continuous rather than quantised to n_bins fixed values per gene
            X_cont = self._discretizer.inverse_transform(X_disc, dither=True, rng=rng)

            X_parts.append(X_cont)
            y_parts.append(np.full(n_cls, cls))

        X_syn = np.vstack(X_parts)
        y_syn = np.concatenate(y_parts)

        # Shuffle so classes aren't contiguous
        perm = rng.permutation(len(y_syn))
        return X_syn[perm], y_syn[perm]

    def _generate_joint(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self._joint_fitter is None:
            raise RuntimeError("Call fit() before generate().")

        rng = np.random.default_rng(self.random_seed)
        synth_df = self._joint_fitter.sample(n_samples)

        X_disc = synth_df[self._selected_gene_names].values.astype(np.int32)
        X_disc = np.clip(X_disc, 0, self.n_bins - 1)
        X_cont = self._discretizer.inverse_transform(X_disc, dither=True, rng=rng)

        y_int = synth_df[_LABEL_COL].values.astype(np.int32)
        y_syn = np.array([self._label_decoder[int(i)] for i in y_int])

        return X_cont, y_syn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_budget_weights(self) -> tuple[float, ...]:
        """
        Return budget weights renormalised to the active marginal orders.
        Orders with zero marginals are dropped and the remaining weights
        are scaled so they sum to 1.
        """
        orders = ["1way", "2way", "3way", "4way"]
        active = [
            (i, w) for i, (order, w) in enumerate(zip(orders, self.budget_weights))
            if len(self._marginals.get(order, [])) > 0
        ]
        if not active:
            raise ValueError("No marginals selected.")

        total = sum(w for _, w in active)
        result = [0.0] * 4
        for i, w in active:
            result[i] = w / total
        return tuple(result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def selected_gene_names(self) -> list[str]:
        return list(self._selected_gene_names)

    @property
    def n_selected_genes(self) -> int:
        return len(self._selected_gene_names)


# Backwards-compatible alias
PrivatePGMRNASeqGenerator = StratHiMPGMGenerator
