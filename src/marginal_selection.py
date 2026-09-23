"""
Hierarchical marginal selection for Private-PGM.

Selects marginal cliques (tuples of column names) in a pruned, bottom-up
fashion so that 3-way and 4-way computation stays tractable:

  1-way  → top S genes by variance
  2-way  → top R gene–gene pairs by |Spearman correlation| from gene_pool_size pool
  3-way  → top Q triples built from genes appearing in the top R pairs
  4-way  → top P quads   built from genes appearing in the top Q triples

Gene×label 2-way marginals are optionally appended (always included by default)
so the PGM captures label-conditioned gene distributions.

All cliques are returned as tuples of column name strings (matching the
pandas DataFrame / mbi.Dataset interface).
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.stats import spearmanr


class HierarchicalMarginalSelector:
    """
    Parameters
    ----------
    n_1way : int
        Number of genes to include as 1-way marginals (chosen by variance).
    n_2way : int
        Number of gene–gene pairs to include as 2-way marginals.
    n_3way : int
        Number of gene triples to include as 3-way marginals.
    n_4way : int
        Number of gene quads to include as 4-way marginals.
    gene_pool_size : int
        Number of top-variance genes used as the candidate pool for pairwise
        correlation computation. Must be >= n_1way for sensible results;
        defaults to n_1way (i.e., use all selected genes as the pool).
    include_label_marginals : bool
        If True, append a (gene, label) 2-way clique for each of the top
        n_1way genes.  Requires the label column name to be passed to select().
    """

    def __init__(
        self,
        n_1way: int = 1000,
        n_2way: int = 150,
        n_3way: int = 30,
        n_4way: int = 7,
        gene_pool_size: Optional[int] = None,
        include_label_marginals: bool = True,
        max_degree: Optional[int] = None,
    ):
        self.n_1way = n_1way
        self.n_2way = n_2way
        self.n_3way = n_3way
        self.n_4way = n_4way
        self.gene_pool_size = gene_pool_size  # resolved in select()
        self.include_label_marginals = include_label_marginals
        # Bounds how many pairs any single gene can participate in.
        # Without this, top-Spearman selection creates dense cliques among
        # highly co-expressed genes, causing treewidth explosion in the JT.
        # Rule of thumb: max_degree=4 limits max JT clique to ~5 genes (K^5 cells).
        self.max_degree = max_degree

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        X_discrete: np.ndarray,
        gene_names: list[str],
        label_col: Optional[str] = None,
    ) -> dict[str, list[tuple[str, ...]]]:
        """
        Run hierarchical selection on discretized data.

        Parameters
        ----------
        X_discrete : np.ndarray, shape (n_samples, n_genes)
            Integer-valued discretized gene expression matrix.
        gene_names : list[str]
            Column names corresponding to columns of X_discrete.
        label_col : str, optional
            Name of the label column.  Required if include_label_marginals=True.

        Returns
        -------
        dict with keys '1way', '2way', '3way', '4way', each holding a list of
        clique tuples (tuples of column name strings).
        """
        n_genes = X_discrete.shape[1]
        pool_size = min(
            self.gene_pool_size if self.gene_pool_size is not None else self.n_1way,
            n_genes,
        )
        n_1way = min(self.n_1way, n_genes)
        n_2way = self.n_2way
        n_3way = self.n_3way
        n_4way = self.n_4way

        gene_names = list(gene_names)

        # --- Step 1: 1-way — top genes by variance ---
        variances = X_discrete.var(axis=0)
        top_gene_idx = np.argsort(variances)[::-1][:n_1way]
        cliques_1way = [(gene_names[i],) for i in top_gene_idx]
        selected_genes_1way = set(top_gene_idx.tolist())

        # --- Step 2: 2-way gene–gene — pairwise Spearman on pool ---
        pool_idx = np.argsort(variances)[::-1][:pool_size]
        print(
            f"  [marginal_selection] Computing pairwise Spearman on "
            f"{len(pool_idx)} genes ({len(pool_idx)*(len(pool_idx)-1)//2} pairs)..."
        )
        corr_matrix = self._pairwise_spearman(X_discrete[:, pool_idx])
        # Upper triangle indices (i < j)
        rows, cols = np.triu_indices(len(pool_idx), k=1)
        pair_scores = np.abs(corr_matrix[rows, cols])
        top_pair_order = np.argsort(pair_scores)[::-1]

        if self.max_degree is not None:
            degree = np.zeros(len(pool_idx), dtype=np.int32)
            selected = []
            for p in top_pair_order:
                r, c = rows[p], cols[p]
                if degree[r] < self.max_degree and degree[c] < self.max_degree:
                    selected.append(p)
                    degree[r] += 1
                    degree[c] += 1
                    if len(selected) >= n_2way:
                        break
            top_pairs = selected
            print(
                f"  [marginal_selection] Degree-limited to max_degree={self.max_degree}: "
                f"{len(top_pairs)} pairs selected (max gene degree: {degree.max()})"
            )
        else:
            n_2way_actual = min(n_2way, len(rows))
            top_pairs = list(top_pair_order[:n_2way_actual])

        cliques_2way_genes = [
            (gene_names[pool_idx[rows[p]]], gene_names[pool_idx[cols[p]]])
            for p in top_pairs
        ]

        # --- Step 3: 3-way — combinations of genes from top pairs ---
        genes_from_pairs = list(
            {pool_idx[rows[p]] for p in top_pairs}
            | {pool_idx[cols[p]] for p in top_pairs}
        )
        print(
            f"  [marginal_selection] Building 3-way from {len(genes_from_pairs)} "
            f"genes ({len(list(combinations(genes_from_pairs, 3)))} triples)..."
        )
        cliques_3way = self._top_kway_by_pairwise_sum(
            genes_from_pairs, corr_matrix, pool_idx, gene_names, k=3, top_n=n_3way
        )

        # --- Step 4: 4-way — combinations of genes from top triples ---
        genes_from_triples = list(
            {i for clique in cliques_3way for name in clique
             for i in [gene_names.index(name)]}
        )
        n_4way_candidates = len(list(combinations(genes_from_triples, 4)))
        print(
            f"  [marginal_selection] Building 4-way from {len(genes_from_triples)} "
            f"genes ({n_4way_candidates} quads)..."
        )
        cliques_4way = self._top_kway_by_pairwise_sum(
            genes_from_triples, corr_matrix, pool_idx, gene_names, k=4, top_n=n_4way
        )

        # --- Step 5: optional gene×label 2-way ---
        cliques_2way_label: list[tuple[str, ...]] = []
        if self.include_label_marginals:
            if label_col is None:
                warnings.warn(
                    "include_label_marginals=True but label_col not provided; "
                    "skipping gene×label marginals."
                )
            else:
                cliques_2way_label = [
                    (gene_names[i], label_col) for i in top_gene_idx
                ]

        cliques_2way = cliques_2way_genes + cliques_2way_label

        summary = (
            f"  [marginal_selection] Selected: "
            f"{len(cliques_1way)} 1-way | "
            f"{len(cliques_2way_genes)} gene–gene 2-way | "
            f"{len(cliques_2way_label)} gene×label 2-way | "
            f"{len(cliques_3way)} 3-way | "
            f"{len(cliques_4way)} 4-way"
        )
        print(summary)

        return {
            "1way": cliques_1way,
            "2way": cliques_2way,
            "3way": cliques_3way,
            "4way": cliques_4way,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_spearman(X: np.ndarray) -> np.ndarray:
        """
        Compute the full pairwise Spearman correlation matrix for columns of X.
        Returns an (n_genes × n_genes) float32 matrix.
        """
        # Rank-transform each column, then Pearson on ranks ≡ Spearman
        ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float32)
        # Standardise ranks
        ranks -= ranks.mean(axis=0)
        norms = np.linalg.norm(ranks, axis=0)
        norms[norms == 0] = 1.0
        ranks /= norms
        corr = ranks.T @ ranks
        np.fill_diagonal(corr, 0.0)  # zero diagonal so it doesn't pollute scoring
        return corr

    def _top_kway_by_pairwise_sum(
        self,
        candidate_gene_indices: list[int],
        corr_matrix: np.ndarray,
        pool_idx: np.ndarray,
        gene_names: list[str],
        k: int,
        top_n: int,
    ) -> list[tuple[str, ...]]:
        """
        Rank all k-way combinations of candidate_gene_indices by the sum of
        pairwise |Spearman correlations| within the combo.  Returns the top_n
        cliques as tuples of gene name strings.

        corr_matrix is indexed by position within pool_idx, so we need to map
        gene_indices → pool positions first.
        """
        # Build a map from global gene index → position in pool (if available)
        pool_pos = {gi: pos for pos, gi in enumerate(pool_idx.tolist())}

        # Filter candidate_gene_indices to those in the pool (correlation is only
        # defined for pool members)
        in_pool = [gi for gi in candidate_gene_indices if gi in pool_pos]

        if len(in_pool) < k:
            return []

        top_n_actual = min(top_n, len(list(combinations(in_pool, k))))
        if top_n_actual == 0:
            return []

        # Score each combo
        best: list[tuple[float, tuple[int, ...]]] = []
        for combo in combinations(in_pool, k):
            positions = [pool_pos[gi] for gi in combo]
            score = sum(
                abs(corr_matrix[positions[a], positions[b]])
                for a, b in combinations(range(k), 2)
            )
            best.append((score, combo))

        best.sort(key=lambda x: x[0], reverse=True)
        top_combos = best[:top_n_actual]

        return [
            tuple(gene_names[gi] for gi in combo)
            for _, combo in top_combos
        ]


# ----------------------------------------------------------------------
# DP selection of a gene-gene spanning tree (MST-style)
# ----------------------------------------------------------------------

def dp_select_tree(
    X_disc: np.ndarray,
    y: np.ndarray,
    noisy_gene_label: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    with_label: bool,
    k_select: int = 4,
    sensitivity: float = 1.0,
) -> tuple[list[tuple[int, int]], dict]:
    """Choose G-1 gene pairs forming a spanning tree, under rho-zCDP.

    This is MST's selection step (McKenna, Miklau & Sheldon 2021,
    ``snsynth/mst/mst.py: select``) with two changes that keep it tractable at
    978 genes and make it fit a model that already has the label as a hub:

    * The reference model is the one the star marginals already imply --
      genes independent given the label, N_c * p(a|c) * p(b|c), read off the
      *noisy* gene x label measurements -- rather than an mbi fit of the
      1-way marginals.  It is post-processing of released data either way.
    * Each pair is scored on a coarsened table of ``k_select`` levels per gene
      (fine bins merged at the noisy marginal's quartiles, again
      post-processing), so all 477k pairs are scored with one matrix product
      per class rather than 477k K^2 tables.

    The score of pair (a, b) is the L1 distance between its true count table
    and the reference: pooled over classes (``with_label=False``) or kept per
    class (``with_label=True``, the table the (a, b, label) marginal will
    measure).  Adding or removing one row moves one cell of the true table by
    one and leaves the reference alone, so the score has L2 = L1 sensitivity
    1 (2 under replace-one, passed in as ``sensitivity``).

    Edges are then drawn as in MST: G-1 rounds of the exponential mechanism
    over pairs joining two current components, each round at
    eps_r = sqrt(8 rho / (G-1)), since an eps-DP exponential mechanism is
    eps^2/8-zCDP (Cesar & Rogers 2021) and the rounds compose additively.

    Returns (pairs as gene-index tuples, diagnostics).
    """
    n, G = X_disc.shape
    K = noisy_gene_label.shape[1]
    classes = np.unique(y)
    P = np.maximum(noisy_gene_label, 0.0)                     # (G, K, C)

    # Coarsen each gene's fine bins at the quartiles of its noisy marginal.
    pooled = P.sum(axis=2)
    pooled = pooled / np.maximum(pooled.sum(axis=1, keepdims=True), 1e-12)
    mid = np.cumsum(pooled, axis=1) - pooled / 2.0
    cmap = np.minimum((mid * k_select).astype(int), k_select - 1)   # (G, K)
    Xc = np.take_along_axis(cmap, X_disc.T.astype(int), axis=1).T   # (n, G)

    # Reference: N_c p(a|c) p(b|c) on the coarse levels.
    Pc = np.zeros((G, k_select, P.shape[2]))
    for k in range(k_select):
        Pc[:, k, :] = (P * (cmap == k)[:, :, None]).sum(axis=1)
    Nc = Pc.sum(axis=1).mean(axis=0)                            # (C,)
    pc = Pc / np.maximum(Pc.sum(axis=1, keepdims=True), 1e-12)  # (G, k, C)

    cols = (np.arange(G) * k_select)[None, :] + Xc              # (n, G)
    D = G * k_select
    score = np.zeros((G, G))
    diff_pooled = np.zeros((D, D), dtype=np.float64) if not with_label else None
    for ci, c in enumerate(classes):
        rows = cols[y == c]
        A = np.zeros((len(rows), D), dtype=np.float32)
        np.put_along_axis(A, rows, 1.0, axis=1)
        M = (A.T @ A).astype(np.float64)                        # true counts
        v = (pc[:, :, ci] * np.sqrt(max(Nc[ci], 0.0))).reshape(D)
        M -= np.outer(v, v)                                     # minus reference
        if with_label:
            score += np.abs(M).reshape(G, k_select, G, k_select).sum(axis=(1, 3))
        else:
            diff_pooled += M
    if not with_label:
        score = np.abs(diff_pooled).reshape(G, k_select, G, k_select).sum(axis=(1, 3))

    ia, ib = np.triu_indices(G, k=1)
    w = score[ia, ib]
    eps_r = np.sqrt(8.0 * rho / (G - 1))
    logits = eps_r * w / (2.0 * sensitivity)

    comp = np.arange(G)
    chosen = []
    for _ in range(G - 1):
        ok = comp[ia] != comp[ib]
        g = np.where(ok, logits + rng.gumbel(size=len(logits)), -np.inf)
        e = int(np.argmax(g))
        a, b = int(ia[e]), int(ib[e])
        chosen.append((a, b))
        comp[comp == comp[b]] = comp[a]

    # How good was the draw, measured against what a non-private MST would pick?
    order = np.argsort(w)[::-1]
    comp2 = np.arange(G)
    best = 0.0
    for e in order:
        a, b = ia[e], ib[e]
        if comp2[a] != comp2[b]:
            best += w[e]
            comp2[comp2 == comp2[b]] = comp2[a]
    got = float(sum(score[a, b] for a, b in chosen))
    diag = {"eps_round": float(eps_r), "score_chosen": got,
            "score_best_tree": float(best), "score_ratio": got / max(best, 1e-12),
            "rho_select": float(rho)}
    return chosen, diag
