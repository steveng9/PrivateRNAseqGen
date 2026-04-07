"""
Top-level PrivatePGMRNASeqGenerator.

Orchestrates:
  1. Hierarchical marginal selection on continuous data.
  2. Discretization of continuous gene expression values.
  3. Stratified Private-PGM fitting: one PGM per class label.
  4. Sampling and decoding back to approximate continuous values.

Why stratified (one PGM per class)?
------------------------------------
Including the label as a node in a shared PGM creates edges between the label
and every selected gene. Combined with gene–gene edges from 2-way marginals,
this creates dense subgraphs whose junction tree can have exponential treewidth —
leading to out-of-memory errors.

Stratified fitting sidesteps this entirely: each per-class PGM has no label node,
only gene–gene structure. Label conditioning is exact because each model was
trained on a single class. From a DP standpoint this is valid: each person's data
appears in exactly one class's PGM, so ε is charged only once per person.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import mbi

from discretization import Discretizer
from marginal_selection import HierarchicalMarginalSelector
from pgm_fitter import PrivatePGMFitter


class PrivatePGMRNASeqGenerator:
    """
    Stratified DP synthetic bulk RNA-seq generator.

    One Private-PGM is fitted per class label. Marginal selection is run once
    on the full dataset (all classes combined) so that the selected gene set is
    shared across classes — ensuring the synthetic data has a consistent feature
    space. Each class then uses those same selected genes for its own PGM.

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
        self.zero_inflated = zero_inflated
        self.random_seed = random_seed

        # Set after fit()
        self._discretizer: Optional[Discretizer] = None
        self._class_fitters: dict[str, PrivatePGMFitter] = {}   # label → fitter
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
    ) -> "PrivatePGMRNASeqGenerator":
        """
        Fit per-class Private-PGM models on continuous gene expression data.

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

        # --- Step 1: Marginal selection on the full dataset ---
        # Selection is class-agnostic (variance + pairwise Spearman on all
        # samples). The same selected gene set is used for every class PGM.
        print("[generator] Step 1: Hierarchical marginal selection (full dataset)...")
        n_1way_actual = min(self.n_1way, n_genes)
        selector = HierarchicalMarginalSelector(
            n_1way=n_1way_actual,
            n_2way=self.n_2way,
            n_3way=self.n_3way,
            n_4way=self.n_4way,
            include_label_marginals=False,   # no label node — stratified approach
        )
        self._marginals = selector.select(X, self._gene_names)

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

        # --- Step 3: Fit one PGM per class ---
        unique_classes = sorted(set(y_str))
        print(f"[generator] Step 3: Fitting per-class PGMs ({len(unique_classes)} classes)...")

        domain = mbi.Domain(selected_genes, [self.n_bins] * len(selected_genes))

        for cls in unique_classes:
            mask = y_str == cls
            n_cls = mask.sum()
            self._class_counts[cls] = int(n_cls)
            print(f"  Class '{cls}': {n_cls} samples")

            X_cls = X_selected[mask]
            X_disc = self._discretizer.transform(X_cls)
            df_cls = pd.DataFrame(X_disc, columns=selected_genes)

            fitter = PrivatePGMFitter(
                epsilon=self.epsilon,
                delta=self.delta,
                budget_weights=weights,
                pgm_iters=self.pgm_iters,
            )
            fitter.fit(df_cls, domain, self._marginals)
            self._class_fitters[cls] = fitter

        print("[generator] Fitting complete.")
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample synthetic data proportionally from each class's PGM.

        The class proportions match the training set frequencies.

        Returns
        -------
        X_synthetic : np.ndarray, shape (n_samples, n_selected_genes)
        y_synthetic : np.ndarray, shape (n_samples,)
        """
        if not self._class_fitters:
            raise RuntimeError("Call fit() before generate().")

        total_train = sum(self._class_counts.values())
        X_parts, y_parts = [], []

        for cls, fitter in self._class_fitters.items():
            # Proportional allocation, at least 1 sample per class
            n_cls = max(1, round(n_samples * self._class_counts[cls] / total_train))
            synth_df = fitter.sample(n_cls)

            X_disc = synth_df[self._selected_gene_names].values.astype(np.int32)
            X_disc = np.clip(X_disc, 0, self.n_bins - 1)
            X_cont = self._discretizer.inverse_transform(X_disc)

            X_parts.append(X_cont)
            y_parts.append(np.full(n_cls, cls))

        X_syn = np.vstack(X_parts)
        y_syn = np.concatenate(y_parts)

        # Shuffle so classes aren't contiguous
        rng = np.random.default_rng(self.random_seed)
        perm = rng.permutation(len(y_syn))
        return X_syn[perm], y_syn[perm]

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
