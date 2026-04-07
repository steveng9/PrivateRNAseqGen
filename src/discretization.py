"""
Discretization module for continuous RNA-seq data.

Bins each gene (column) into K discrete integer levels using quantile edges
learned from training data. Supports an optional zero-inflated mode for
scRNA-seq where zero is treated as a dedicated bin.
"""

import numpy as np


class Discretizer:
    """
    Quantile-based discretizer for continuous gene expression data.

    Parameters
    ----------
    n_bins : int
        Number of discrete levels per gene. Default: 8.
    zero_inflated : bool
        If True, zeros are placed in bin 0 and non-zero values are quantile-
        binned into bins 1..n_bins-1. Intended for sparse scRNA-seq data.
        Default: False (bulk RNA-seq mode).
    """

    def __init__(self, n_bins: int = 8, zero_inflated: bool = False):
        self.n_bins = n_bins
        self.zero_inflated = zero_inflated
        self._edges: list[np.ndarray] = []   # one edge array per feature
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "Discretizer":
        """
        Learn quantile bin edges from training data X (shape: n_samples × n_genes).
        """
        n_genes = X.shape[1]
        self._edges = []

        for j in range(n_genes):
            col = X[:, j]
            if self.zero_inflated:
                nonzero = col[col != 0]
                if len(nonzero) == 0:
                    # All zeros: only one bin possible
                    self._edges.append(np.array([]))
                else:
                    k_nonzero = self.n_bins - 1  # bins 1..n_bins-1
                    quantiles = np.linspace(0, 100, k_nonzero + 1)
                    edges = np.percentile(nonzero, quantiles)
                    # Remove duplicate edges so np.digitize behaves correctly
                    edges = np.unique(edges)
                    self._edges.append(edges)
            else:
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(col, quantiles)
                edges = np.unique(edges)
                self._edges.append(edges)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map continuous values to integer bin indices in [0, n_bins-1].
        Returns array of same shape as X with dtype int32.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        X_disc = np.empty(X.shape, dtype=np.int32)

        for j, edges in enumerate(self._edges):
            col = X[:, j]
            if self.zero_inflated:
                out = np.zeros(len(col), dtype=np.int32)  # bin 0 = zeros
                mask = col != 0
                if mask.any() and len(edges) > 0:
                    # Bins 1..n_bins-1 for nonzero values; clip to valid range
                    raw = np.digitize(col[mask], edges[1:], right=False)
                    out[mask] = np.clip(raw, 0, self.n_bins - 2) + 1
                elif mask.any():
                    out[mask] = 1
                X_disc[:, j] = out
            else:
                if len(edges) <= 1:
                    X_disc[:, j] = 0
                else:
                    # digitize against interior edges (drop first and last)
                    raw = np.digitize(col, edges[1:-1], right=False)
                    X_disc[:, j] = np.clip(raw, 0, self.n_bins - 1)

        return X_disc

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Approximate inverse (bin centres)
    # ------------------------------------------------------------------

    def bin_centers(self, j: int) -> np.ndarray:
        """
        Return the approximate centre value for each bin of feature j.
        Useful for mapping synthetic discrete data back to continuous space.
        """
        edges = self._edges[j]
        if len(edges) == 0:
            return np.zeros(self.n_bins)

        if self.zero_inflated:
            # Bin 0 → 0.0; bins 1..n_bins-1 → midpoints of nonzero quantile ranges
            centres = np.zeros(self.n_bins)
            for b in range(1, self.n_bins):
                lo = edges[b - 1] if b - 1 < len(edges) else edges[-1]
                hi = edges[b]     if b     < len(edges) else edges[-1]
                centres[b] = 0.5 * (lo + hi)
            return centres
        else:
            centres = np.zeros(self.n_bins)
            full_edges = np.concatenate([[edges[0]], edges, [edges[-1]]])
            for b in range(self.n_bins):
                if b < len(edges) - 1:
                    centres[b] = 0.5 * (edges[b] + edges[b + 1])
                else:
                    centres[b] = edges[-1]
            return centres

    def inverse_transform(self, X_disc: np.ndarray) -> np.ndarray:
        """
        Map discrete bin indices back to approximate continuous values (bin centres).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        X_cont = np.empty(X_disc.shape, dtype=np.float32)
        for j in range(X_disc.shape[1]):
            centres = self.bin_centers(j)
            idx = np.clip(X_disc[:, j], 0, self.n_bins - 1)
            X_cont[:, j] = centres[idx]
        return X_cont

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self._edges)
