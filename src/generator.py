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
    binning : str
        How bin edges are chosen; see ``discretization`` for the privacy of
        each.  "quantile" (legacy: private percentiles, NOT covered by ε),
        "uniform" (equal width over the public ``bin_range``, free) or
        "dp_quantile" (equal depth from a noisy histogram, costs
        ``binning_budget`` of ρ; the marginals get the rest) or "dp_uniform"
        (the same noisy histogram, but only the two bounds are bought and the
        edges are spaced evenly between them -- what smartnoise-synth's
        BinTransformer does when given ``preprocessor_eps``).
    bin_range : tuple
        Public value range for "uniform"; the DP strategies grid on it.  Must be chosen
        without looking at the private data.  (0, 24) is an a-priori bound for
        log2-scale normalised expression (2^24 ≈ 1.7e7).
    binning_budget : float
        Fraction of total ρ spent on DP edges under the dp_* strategies.
    bin_grid : int
        Cells of the public histogram grid under the dp_* strategies.
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
        max_degree: Optional[int] = None,
        composition: str = "zcdp",
        neighboring: str = "add_remove",
        binning: str = "quantile",
        bin_range: tuple = (0.0, 24.0),
        binning_budget: float = 0.1,
        bin_grid: int = 48,
        edge_estimator: str = "clip",
        structure: str = "hierarchical",
        select_budget: float = 0.3,
        k_label: int = 978,
        l_pairs: int = 0,
        with_1way: bool = False,
        max_component: Optional[int] = None,
    ):
        if structure not in ("hierarchical", "tree", "tree_label", "forest", "hairy_star"):
            raise ValueError(f"unknown structure {structure!r}")
        if structure != "hierarchical":
            if not joint_mode:
                raise ValueError("tree structures need joint_mode=True")
            if composition != "zcdp":
                raise ValueError("tree structures need composition='zcdp'")
            if not 0.0 < select_budget < 1.0:
                raise ValueError("select_budget must be in (0, 1)")
        self.edge_estimator = edge_estimator
        self.structure = structure
        self.select_budget = select_budget
        self.k_label = int(k_label)
        self.l_pairs = int(l_pairs)
        self.with_1way = bool(with_1way)
        self.max_component = None if max_component is None else int(max_component)
        if structure == "forest" and (self.k_label < 0 or self.l_pairs < 0):
            raise ValueError("k_label and l_pairs must be >= 0")
        #: rho actually spent per stage, filled in by fit().
        self.rho_breakdown: dict = {}
        self.selection_diagnostics: dict = {}
        if binning.startswith("dp_") and composition != "zcdp":
            raise ValueError(f"binning={binning!r} needs composition='zcdp'")
        if binning.startswith("dp_") and not 0.0 < binning_budget < 1.0:
            raise ValueError("binning_budget must be in (0, 1)")
        self.binning = binning
        self.bin_range = tuple(bin_range)
        self.binning_budget = binning_budget
        self.bin_grid = bin_grid
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
        self.max_degree = max_degree
        self.composition = composition
        self.neighboring = neighboring

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
            max_degree=self.max_degree,
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
        print(f"[generator] Step 2: Fitting discretizer ({self.binning})...")
        self._discretizer = Discretizer(
            n_bins=self.n_bins, zero_inflated=self.zero_inflated,
            strategy=self.binning, value_range=self.bin_range,
            grid_cells=self.bin_grid,
            edge_estimator=getattr(self, "edge_estimator", "clip"),
        )
        # Edges and marginals compose sequentially in ρ, so a DP-edge stage
        # takes its share off the top and the marginals get what is left.
        from pgm_fitter import _SENSITIVITY, rho_from_eps_delta
        if self.binning.startswith("dp_"):
            rho_edges = self.binning_budget * rho_from_eps_delta(self.epsilon, self.delta)
            self._discretizer.fit(X_selected, rho=rho_edges,
                                  sensitivity=_SENSITIVITY[self.neighboring],
                                  rng=np.random.default_rng(self.random_seed))
            self._marginal_rho_fraction = 1.0 - self.binning_budget
            print(f"  [discretizer] {self.binning}: ρ={rho_edges:.5f}, "
                  f"σ={self._discretizer.noise_sigma:.2f} on a "
                  f"{self.bin_grid}-cell grid over {self.bin_range}")
        else:
            self._discretizer.fit(X_selected)
            self._marginal_rho_fraction = 1.0

        if getattr(self, "structure", "hierarchical") == "forest":
            self._fit_forest(X_selected, y_str, selected_genes, unique_classes)
        elif self.structure == "hairy_star":
            self._fit_hairy_star(X_selected, y_str, selected_genes, unique_classes)
        elif getattr(self, "structure", "hierarchical") != "hierarchical":
            self._fit_tree(X_selected, y_str, selected_genes, unique_classes)
        elif self.joint_mode:
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
                composition=self.composition,
                neighboring=self.neighboring,
                rho_fraction=self._marginal_rho_fraction,
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
            composition=self.composition,
            neighboring=self.neighboring,
            rho_fraction=self._marginal_rho_fraction,
        )
        fitter.fit(df_all, domain, self._marginals)
        self._joint_fitter = fitter

    def _fit_tree(
        self,
        X_selected: np.ndarray,
        y_str: np.ndarray,
        selected_genes: list[str],
        unique_classes: list[str],
    ) -> None:
        """Joint model: star (gene, gene x label) plus a DP-selected gene tree.

        MST's recipe on top of last year's star: measure the star, select a
        spanning tree of gene pairs with the exponential mechanism against
        what the star already explains, then measure the tree -- as (a, b)
        pairs (``structure="tree"``) or as (a, b, label) triples
        (``"tree_label"``, class-conditional co-expression).  The label hub
        plus a tree is chordal with treewidth 2, so the junction tree's
        largest clique is (a, b, label) and inference stays cheap.

        As in MST every measured clique gets the same sigma; ``budget_weights``
        is not used.  Budget: ``binning_budget`` (dp_* only) to the edges,
        ``select_budget`` to selection, the rest to measurements.
        """
        from marginal_selection import dp_select_tree
        from pgm_fitter import _SENSITIVITY, rho_from_eps_delta

        n_classes = len(unique_classes)
        self._label_encoder = {cls: i for i, cls in enumerate(unique_classes)}
        self._label_decoder = {i: cls for i, cls in enumerate(unique_classes)}
        X_disc = self._discretizer.transform(X_selected)
        y_int = np.array([self._label_encoder[c] for c in y_str])
        df_all = pd.DataFrame(X_disc, columns=selected_genes)
        df_all[_LABEL_COL] = y_int
        domain = mbi.Domain(selected_genes + [_LABEL_COL],
                            [self.n_bins] * len(selected_genes) + [n_classes])
        dataset = mbi.Dataset(df_all, domain)

        rho_total = rho_from_eps_delta(self.epsilon, self.delta)
        f_bin = self.binning_budget if self.binning.startswith("dp_") else 0.0
        rho_sel = self.select_budget * rho_total
        rho_meas = (1.0 - f_bin - self.select_budget) * rho_total
        if rho_meas <= 0:
            raise ValueError("binning_budget + select_budget must be < 1")
        G = len(selected_genes)
        one = [(g,) for g in selected_genes]
        gl = [(g, _LABEL_COL) for g in selected_genes]
        n_meas = len(one) + len(gl) + (G - 1)
        sens = _SENSITIVITY[self.neighboring]
        sigma = sens * np.sqrt(n_meas / (2.0 * rho_meas))
        print(f"[generator] Step 3: tree ({self.structure}): {n_meas} cliques, "
              f"σ={sigma:.3f}, ρ bin/select/measure = {f_bin * rho_total:.4f}/"
              f"{rho_sel:.4f}/{rho_meas:.4f}", flush=True)

        fitter = PrivatePGMFitter(epsilon=self.epsilon, delta=self.delta,
                                  budget_weights=(1.0, 0.0, 0.0, 0.0),
                                  pgm_iters=self.pgm_iters,
                                  composition=self.composition,
                                  neighboring=self.neighboring,
                                  rho_fraction=1.0 - f_bin - self.select_budget)
        meas = fitter.measure(dataset, one, sigma)
        meas_gl = fitter.measure(dataset, gl, sigma)
        noisy_gl = np.stack([m[1].reshape(self.n_bins, n_classes) for m in meas_gl])

        pairs, diag = dp_select_tree(
            X_disc, y_int, noisy_gl, rho_sel,
            rng=np.random.default_rng(self.random_seed),
            with_label=(self.structure == "tree_label"),
            sensitivity=2.0 if self.neighboring == "replace" else 1.0)
        self.selection_diagnostics = diag
        print(f"  [selection] eps/round={diag['eps_round']:.4f}, chosen tree "
              f"scores {diag['score_ratio']:.3f} of the best tree", flush=True)
        tree = [(selected_genes[a], selected_genes[b]) for a, b in pairs]
        if self.structure == "tree_label":
            tree = [(a, b, _LABEL_COL) for a, b in tree]
        meas_tree = fitter.measure(dataset, tree, sigma)
        self._marginals = {"1way": one, "2way": gl + (tree if self.structure == "tree" else []),
                           "3way": tree if self.structure == "tree_label" else [],
                           "4way": []}

        fitter.estimate_from(domain, meas + meas_gl + meas_tree, len(df_all))
        self._joint_fitter = fitter
        self.rho_breakdown = {
            "binning": float(getattr(self._discretizer, "rho_spent", 0.0)),
            "selection": float(rho_sel), "measurement": float(fitter.rho_spent),
            "total_budget": float(rho_total)}
        spent = sum(v for k, v in self.rho_breakdown.items() if k != "total_budget")
        if spent > rho_total * (1 + 1e-9):
            raise AssertionError(f"overspent: {spent} > {rho_total}")

    def _fit_forest(
        self,
        X_selected: np.ndarray,
        y_str: np.ndarray,
        selected_genes: list[str],
        unique_classes: list[str],
    ) -> None:
        """Joint model on a sparse, DP-selected set of 2-way tables.

        Steven's design (2026-09-24): keep fewer tables so each carries less
        noise.  Four stages, all under one rho-zCDP budget:

        1. measure the label marginal (the class shares the scores need);
        2. choose ``k_label`` genes by how label-dependent their (gene, label)
           table is (``dp_select_label_genes``) and measure those tables;
        3. choose ``l_pairs`` gene pairs as a forest (MST's Kruskal selection,
           truncated after l rounds, against the model stage 2 implies) and
           measure them;
        4. measure a 1-way table for every gene no chosen table covers.

        k_label = n_genes skips stage 2's selection (every gene gets a label
        table: the label-only star); l_pairs = 0 skips stage 3.  Label hub plus
        a forest has treewidth <= 2, so inference stays cheap.

        Stages 1-2 are measured at the sigma the worst case would need (every
        gene left uncovered after stage 3), and stage 3-4 share what is left,
        so sigma there is never larger; total spend is asserted.
        """
        from marginal_selection import dp_select_label_genes, dp_select_tree
        from pgm_fitter import _SENSITIVITY, rho_from_eps_delta

        C = len(unique_classes)
        K = self.n_bins
        self._label_encoder = {cls: i for i, cls in enumerate(unique_classes)}
        self._label_decoder = {i: cls for i, cls in enumerate(unique_classes)}
        X_disc = self._discretizer.transform(X_selected)
        y_int = np.array([self._label_encoder[c] for c in y_str])
        df_all = pd.DataFrame(X_disc, columns=selected_genes)
        df_all[_LABEL_COL] = y_int
        domain = mbi.Domain(selected_genes + [_LABEL_COL], [K] * len(selected_genes) + [C])
        dataset = mbi.Dataset(df_all, domain)

        G = len(selected_genes)
        k = min(self.k_label, G)
        l = min(self.l_pairs, G - 1)
        select_k = 0 < k < G
        rounds = (k if select_k else 0) + l
        rho_total = rho_from_eps_delta(self.epsilon, self.delta)
        f_bin = self.binning_budget if self.binning.startswith("dp_") else 0.0
        f_sel = self.select_budget if rounds > 0 else 0.0
        rho_sel = f_sel * rho_total
        rho_sel_k = rho_sel * (k if select_k else 0) / max(rounds, 1)
        rho_sel_l = rho_sel - rho_sel_k
        rho_meas = (1.0 - f_bin - f_sel) * rho_total
        if rho_meas <= 0:
            raise ValueError("binning_budget + select_budget must be < 1")
        sens = _SENSITIVITY[self.neighboring]
        sel_sens = 2.0 if self.neighboring == "replace" else 1.0
        rng = np.random.default_rng(self.random_seed)

        fitter = PrivatePGMFitter(epsilon=self.epsilon, delta=self.delta,
                                  budget_weights=(1.0, 0.0, 0.0, 0.0),
                                  pgm_iters=self.pgm_iters,
                                  composition=self.composition,
                                  neighboring=self.neighboring,
                                  rho_fraction=1.0 - f_bin - f_sel)
        # Stages 1-2 at the worst-case sigma: 1 + k + l + (G - k) tables.
        m_max = 1 + G + l
        sigma_a = sens * np.sqrt(m_max / (2.0 * rho_meas))
        meas_lab = fitter.measure(dataset, [(_LABEL_COL,)], sigma_a)
        noisy_lab = meas_lab[0][1]

        diag = {}
        if select_k:
            genes_k, d = dp_select_label_genes(X_disc, y_int, noisy_lab, k, K,
                                               rho_sel_k, rng, sensitivity=sel_sens)
            diag.update(d)
        else:
            genes_k = list(range(k))
        gl = [(selected_genes[g], _LABEL_COL) for g in genes_k]
        meas_gl = fitter.measure(dataset, gl, sigma_a)

        pairs = []
        if l > 0:
            # Reference for the pair scores: the chosen genes' noisy label
            # tables; genes without one get N_c / K per bin (equal-depth bins).
            ref = np.tile(np.maximum(noisy_lab, 0.0)[None, None, :] / K, (G, K, 1))
            for g, m in zip(genes_k, meas_gl):
                ref[g] = m[1].reshape(K, C)
            pairs, d = dp_select_tree(X_disc, y_int, ref, rho_sel_l, rng=rng,
                                      with_label=False, sensitivity=sel_sens,
                                      n_edges=l)
            diag.update(d)
        self.selection_diagnostics = diag
        covered = set(genes_k) | {g for p in pairs for g in p}
        uncovered = [g for g in range(G) if g not in covered]
        tree = [(selected_genes[a], selected_genes[b]) for a, b in pairs]
        one = [(selected_genes[g],) for g in uncovered]

        rho_left = rho_meas - fitter.rho_spent
        n_c = len(tree) + len(one)
        meas_c = []
        if n_c:
            sigma_c = sens * np.sqrt(n_c / (2.0 * rho_left))
            meas_c = fitter.measure(dataset, tree + one, sigma_c)
        else:
            sigma_c = float("nan")
        print(f"[generator] Step 3: forest k={k} l={l}: {len(gl)} gene×label, "
              f"{len(tree)} gene–gene, {len(one)} 1-way (+label); "
              f"σ label/gl={sigma_a:.3f}, σ pairs/1-way={sigma_c:.3f}", flush=True)

        self._marginals = {"1way": one + [(_LABEL_COL,)], "2way": gl + tree,
                           "3way": [], "4way": []}
        fitter.estimate_from(domain, meas_lab + meas_gl + meas_c, len(df_all))
        self._joint_fitter = fitter
        self.rho_breakdown = {
            "binning": float(getattr(self._discretizer, "rho_spent", 0.0)),
            "selection": float(rho_sel), "measurement": float(fitter.rho_spent),
            "total_budget": float(rho_total)}
        spent = sum(v for k_, v in self.rho_breakdown.items() if k_ != "total_budget")
        if spent > rho_total * (1 + 1e-9):
            raise AssertionError(f"overspent: {spent} > {rho_total}")

    def _fit_hairy_star(
        self,
        X_selected: np.ndarray,
        y_str: np.ndarray,
        selected_genes: list[str],
        unique_classes: list[str],
    ) -> None:
        """Joint model on a spanning tree over genes + label: a star with hairs.

        Steven's design (2026-09-24):

        1. measure the label marginal;
        2. choose ``l_pairs`` gene pairs as a forest (MST's Kruskal selection,
           truncated after l rounds; the reference is N_c / K per bin, i.e.
           genes independent and flat, so the score is plain gene-gene
           dependence);
        3. in every connected component of that forest, choose the one gene
           whose (gene, label) table is most label-dependent
           (``dp_choose_hubs``); singletons are their own hub;
        4. measure the l pair tables and the G - l (hub, label) tables, and
           with ``with_1way`` a 1-way table for every gene as well.

        A forest with l edges on G genes has G - l components, so this is
        l + (G - l) = G edges on G + 1 nodes: a spanning tree with the label
        as a hub, i.e. an MST whose label node is forced to reach every
        component.  l = 0 is the label-only star.  Every table count is known
        up front, so all tables share one sigma.  ``max_component`` caps a
        component's size (2: the hairs are disjoint gene pairs, Steven's
        original picture; None: Kruskal's components grow freely, and with
        l=400 on COMBINED form a handful of large trees).  Selection splits
        ``select_budget`` evenly between the pair rounds and the hub rounds.
        """
        from marginal_selection import dp_choose_hubs, dp_select_tree, forest_components
        from pgm_fitter import _SENSITIVITY, rho_from_eps_delta

        C = len(unique_classes)
        K = self.n_bins
        self._label_encoder = {cls: i for i, cls in enumerate(unique_classes)}
        self._label_decoder = {i: cls for i, cls in enumerate(unique_classes)}
        X_disc = self._discretizer.transform(X_selected)
        y_int = np.array([self._label_encoder[c] for c in y_str])
        df_all = pd.DataFrame(X_disc, columns=selected_genes)
        df_all[_LABEL_COL] = y_int
        domain = mbi.Domain(selected_genes + [_LABEL_COL], [K] * len(selected_genes) + [C])
        dataset = mbi.Dataset(df_all, domain)

        G = len(selected_genes)
        l = min(self.l_pairs, G - 1)
        mc = getattr(self, "max_component", None)
        if mc is not None:
            l = min(l, G - -(-G // mc))      # most edges a forest of <= mc-gene trees has
        rho_total = rho_from_eps_delta(self.epsilon, self.delta)
        f_bin = self.binning_budget if self.binning.startswith("dp_") else 0.0
        f_sel = self.select_budget if l > 0 else 0.0
        rho_sel = f_sel * rho_total
        rho_meas = (1.0 - f_bin - f_sel) * rho_total
        if rho_meas <= 0:
            raise ValueError("binning_budget + select_budget must be < 1")
        sens = _SENSITIVITY[self.neighboring]
        sel_sens = 2.0 if self.neighboring == "replace" else 1.0
        rng = np.random.default_rng(self.random_seed)

        n_tab = 1 + l + (G - l) + (G if self.with_1way else 0)
        sigma = sens * np.sqrt(n_tab / (2.0 * rho_meas))
        fitter = PrivatePGMFitter(epsilon=self.epsilon, delta=self.delta,
                                  budget_weights=(1.0, 0.0, 0.0, 0.0),
                                  pgm_iters=self.pgm_iters,
                                  composition=self.composition,
                                  neighboring=self.neighboring,
                                  rho_fraction=1.0 - f_bin - f_sel)
        meas_lab = fitter.measure(dataset, [(_LABEL_COL,)], sigma)
        noisy_lab = meas_lab[0][1]

        diag, pairs = {}, []
        if l > 0:
            ref = np.tile(np.maximum(noisy_lab, 0.0)[None, None, :] / K, (G, K, 1))
            pairs, d = dp_select_tree(X_disc, y_int, ref, rho_sel / 2.0, rng=rng,
                                      with_label=False, sensitivity=sel_sens, n_edges=l,
                                      max_component=mc)
            diag.update(d)
        comps = forest_components(G, pairs)
        hubs, d = dp_choose_hubs(X_disc, y_int, noisy_lab, comps, K,
                                 rho_sel / 2.0 if l > 0 else 0.0, rng,
                                 sensitivity=sel_sens)
        diag.update(d)
        self.selection_diagnostics = diag
        assert len(hubs) == G - l, (len(hubs), G, l)
        hubs = sorted(hubs)
        tree = [(selected_genes[a], selected_genes[b]) for a, b in pairs]
        gl = [(selected_genes[g], _LABEL_COL) for g in hubs]
        one = [(g,) for g in selected_genes] if self.with_1way else []
        meas = fitter.measure(dataset, tree + gl + one, sigma)
        print(f"[generator] Step 3: hairy star l={l}: {len(tree)} gene–gene, "
              f"{len(gl)} gene×label, {len(one)} 1-way (+label); σ={sigma:.3f}; "
              f"{diag.get('hub_rounds', 0)} hub choices", flush=True)

        self._marginals = {"1way": one + [(_LABEL_COL,)], "2way": gl + tree,
                           "3way": [], "4way": []}
        fitter.estimate_from(domain, meas_lab + meas, len(df_all))
        self._joint_fitter = fitter
        self.rho_breakdown = {
            "binning": float(getattr(self._discretizer, "rho_spent", 0.0)),
            "selection": float(rho_sel), "measurement": float(fitter.rho_spent),
            "total_budget": float(rho_total)}
        spent = sum(v for k_, v in self.rho_breakdown.items() if k_ != "total_budget")
        if spent > rho_total * (1 + 1e-9):
            raise AssertionError(f"overspent: {spent} > {rho_total}")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:  # noqa: D401
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
        if self.joint_mode or self._joint_fitter is not None:
            return self._generate_joint(n_samples)
        return self._generate_stratified(n_samples)

    def _generate_stratified(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._class_fitters:
            raise RuntimeError("Call fit() before generate().")

        # The exact per-class counts are as sensitive as any other statistic, so
        # allocate from each submodel's noisy total instead.  Under add/remove
        # that total is the estimate mbi derives from the noisy marginals; under
        # `replace` the class sizes are public and it is the true count.
        class_totals = {cls: max(1.0, float(f.estimated_total))
                        for cls, f in self._class_fitters.items()}
        total_train = sum(class_totals.values())
        rng = np.random.default_rng(self.random_seed)
        X_parts, y_parts = [], []

        for cls, fitter in self._class_fitters.items():
            # Proportional allocation, at least 1 sample per class
            n_cls = max(1, round(n_samples * class_totals[cls] / total_train))
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
