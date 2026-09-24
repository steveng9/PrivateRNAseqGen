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
    n_edges: int | None = None,
    max_component: int | None = None,
) -> tuple[list[tuple[int, int]], dict]:
    """Choose G-1 gene pairs forming a spanning tree, under rho-zCDP.

    With ``max_component``, a pair is a candidate only if joining its two
    components keeps them at most that many genes (2: disjoint pairs, a
    matching).  The candidate set depends only on earlier draws, so each round
    is still the exponential mechanism over a public candidate set.

    With ``n_edges`` < G-1, stop after that many rounds: a forest of the
    ``n_edges`` best-scoring acyclic pairs (Kruskal truncated), at
    eps_r = sqrt(8 rho / n_edges) per round.

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
    R = G - 1 if n_edges is None else int(n_edges)
    eps_r = np.sqrt(8.0 * rho / R)
    logits = eps_r * w / (2.0 * sensitivity)

    comp = np.arange(G)
    size = np.ones(G, dtype=int)                                # by component id
    chosen = []
    for _ in range(R):
        ok = comp[ia] != comp[ib]
        if max_component is not None:
            ok &= size[comp[ia]] + size[comp[ib]] <= max_component
        if not ok.any():
            break
        g = np.where(ok, logits + rng.gumbel(size=len(logits)), -np.inf)
        e = int(np.argmax(g))
        a, b = int(ia[e]), int(ib[e])
        chosen.append((a, b))
        size[comp[a]] += size[comp[b]]
        comp[comp == comp[b]] = comp[a]

    # How good was the draw, measured against what a non-private MST would pick?
    order = np.argsort(w)[::-1]
    comp2 = np.arange(G)
    best = 0.0
    n_best = 0
    for e in order:
        if n_best == R:
            break
        a, b = ia[e], ib[e]
        if comp2[a] != comp2[b]:
            best += w[e]
            n_best += 1
            comp2[comp2 == comp2[b]] = comp2[a]
    got = float(sum(score[a, b] for a, b in chosen))
    diag = {"eps_round": float(eps_r), "score_chosen": got,
            "score_best_tree": float(best), "score_ratio": got / max(best, 1e-12),
            "rho_select": float(rho)}
    return chosen, diag


def _label_scores(X_disc: np.ndarray, y: np.ndarray, noisy_label_counts: np.ndarray,
                  n_bins: int) -> np.ndarray:
    """Per gene, sum over (bin b, class c) of |n_gbc - n_gb * q_c|: the L1
    distance of its gene x label table from independence, q_c the class share
    read off the noisy label marginal.  L1 sensitivity 2 (see
    ``dp_select_label_genes``)."""
    G = X_disc.shape[1]
    C = len(noisy_label_counts)
    q = np.maximum(np.asarray(noisy_label_counts, float), 0.0)
    q = q / max(q.sum(), 1e-12)
    codes = (np.arange(G) * n_bins * C)[None, :] + X_disc.astype(np.int64) * C + y[:, None]
    t = np.bincount(codes.ravel(), minlength=G * n_bins * C).reshape(G, n_bins, C)
    t = t.astype(np.float64)
    return np.abs(t - t.sum(axis=2, keepdims=True) * q[None, None, :]).sum(axis=(1, 2))


def forest_components(G: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components of the forest on genes 0..G-1, in gene order."""
    comp = np.arange(G)
    for a, b in pairs:
        comp[comp == comp[b]] = comp[a]
    out: dict[int, list[int]] = {}
    for g in range(G):
        out.setdefault(int(comp[g]), []).append(g)
    return list(out.values())


def dp_choose_hubs(
    X_disc: np.ndarray,
    y: np.ndarray,
    noisy_label_counts: np.ndarray,
    components: list[list[int]],
    n_bins: int,
    rho: float,
    rng: np.random.Generator,
    sensitivity: float = 1.0,
) -> tuple[list[int], dict]:
    """One gene per component to carry that component's (gene, label) table.

    Singletons need no choice.  For every component of two or more genes, the
    exponential mechanism picks the member whose (gene, label) table is most
    label-dependent (the ``dp_select_label_genes`` score, L1 sensitivity 2),
    one round each at eps_r = sqrt(8 rho / rounds) under rho-zCDP.  With no
    multi-gene component, nothing is spent.
    """
    multi = [c for c in components if len(c) > 1]
    hubs = [c[0] for c in components if len(c) == 1]
    if not multi:
        return hubs, {"hub_rounds": 0, "rho_select_hub": 0.0}
    score = _label_scores(X_disc, y, noisy_label_counts, n_bins)
    eps_r = np.sqrt(8.0 * rho / len(multi))
    delta = 2.0 * sensitivity
    hit = 0
    for c in multi:
        c = np.asarray(c)
        g = eps_r * score[c] / (2.0 * delta) + rng.gumbel(size=len(c))
        pick = int(c[np.argmax(g)])
        hubs.append(pick)
        hit += pick == int(c[np.argmax(score[c])])
    return hubs, {"hub_rounds": len(multi), "eps_round_hub": float(eps_r),
                  "rho_select_hub": float(rho), "hub_best_share": hit / len(multi)}


def dp_select_label_genes(
    X_disc: np.ndarray,
    y: np.ndarray,
    noisy_label_counts: np.ndarray,
    k: int,
    n_bins: int,
    rho: float,
    rng: np.random.Generator,
    sensitivity: float = 1.0,
) -> tuple[list[int], dict]:
    """Choose the k genes whose (gene, label) table is most label-dependent, under rho-zCDP.

    Score of gene g: sum over (bin b, class c) of |n_gbc - n_gb * q_c|, the L1
    distance of its gene x label table from independence, where q_c is the
    class share read off the *noisy* label marginal (post-processing).  Adding
    or removing one row moves n_gbc by 1 and n_gb by 1, which moves the terms
    of row b by at most 1 + sum_c q_c = 2, so the L1 sensitivity is 2 (times
    the neighbouring relation's own factor, passed in as ``sensitivity``).

    Selection is the one-shot Gumbel top-k, which has exactly the privacy of k
    rounds of the exponential mechanism without replacement (Durfee & Rogers
    2019); each round at eps_r = sqrt(8 rho / k), since an eps-DP exponential
    mechanism is eps^2/8-zCDP (Cesar & Rogers 2021).
    """
    G = X_disc.shape[1]
    score = _label_scores(X_disc, y, noisy_label_counts, n_bins)
    delta = 2.0 * sensitivity
    eps_r = np.sqrt(8.0 * rho / k)
    g = eps_r * score / (2.0 * delta) + rng.gumbel(size=G)
    chosen = [int(i) for i in np.argsort(-g)[:k]]
    top = np.sort(score)[::-1][:k].sum()
    diag = {"eps_round_label": float(eps_r), "rho_select_label": float(rho),
            "label_score_ratio": float(score[chosen].sum() / max(top, 1e-12))}
    return chosen, diag
