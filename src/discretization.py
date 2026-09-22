"""
Discretization module for continuous RNA-seq data.

Bins each gene (column) into K discrete integer levels. Supports an optional
zero-inflated mode for scRNA-seq where zero is treated as a dedicated bin.

Edge strategies and privacy
---------------------------
The bin edges are released: decoding draws synthetic values uniformly inside
each bin, so the output's per-gene support and density jumps sit exactly on
the edges.  How the edges are chosen therefore decides whether the pipeline's
epsilon is end to end.

``"quantile"`` (legacy default)
    Percentiles of the training data, including its exact min and max.  Spends
    NO privacy budget, so it is not covered by epsilon.  Measured: at eps=0.3 an
    attack reaches AUC 0.546 against a DP-implied maximum of 0.523.

``"uniform"``
    Equal-width edges over a fixed, publicly chosen ``value_range``.  Reads no
    data, so costs nothing.  Values outside the range are clipped into the end
    bins.  The price is utility: one global range wastes bins on genes that
    live in a narrow band of it.

``"dp_quantile"``
    Equal-depth edges from a *noisy* per-gene histogram on a fixed public grid
    of ``grid_cells`` cells over ``value_range``.  Each row adds 1 to exactly
    one cell of each gene's histogram, so the release is a Gaussian mechanism
    with the same sensitivity as a 1-way marginal and composes with the
    marginals in rho: noise sigma = sensitivity * sqrt(n_genes / (2 * rho)).
    Edges are read off the clipped noisy CDF and snapped to grid boundaries
    (post-processing, free).  Outer edges are the ``tail`` and ``1 - tail``
    noisy quantiles, so decoding does not smear values across the whole range.
"""

import numpy as np

STRATEGIES = ("quantile", "uniform", "dp_quantile", "dp_uniform")


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

    def __init__(self, n_bins: int = 8, zero_inflated: bool = False,
                 strategy: str = "quantile", value_range: tuple = (0.0, 24.0),
                 grid_cells: int = 48, tail: float = 0.005):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
        if strategy != "quantile" and zero_inflated:
            raise NotImplementedError("zero_inflated needs strategy='quantile'")
        self.n_bins = n_bins
        self.zero_inflated = zero_inflated
        self.strategy = strategy
        self.value_range = (float(value_range[0]), float(value_range[1]))
        self.grid_cells = grid_cells
        self.tail = tail
        #: rho spent choosing edges (0 unless the strategy is a dp_* one).
        self.rho_spent = 0.0
        self._edges: list[np.ndarray] = []   # one edge array per feature
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, rho: float = 0.0, sensitivity: float = 1.0,
            rng=None) -> "Discretizer":
        """
        Learn bin edges for X (shape: n_samples × n_genes).

        ``rho`` and ``sensitivity`` are used only by ``"dp_quantile"``;
        ``"uniform"`` ignores X entirely.
        """
        n_genes = X.shape[1]
        self._edges = []
        if self.strategy == "uniform":
            lo, hi = self.value_range
            edges = np.linspace(lo, hi, self.n_bins + 1)
            self._edges = [edges.copy() for _ in range(n_genes)]
            self._fitted = True
            return self
        if self.strategy in ("dp_quantile", "dp_uniform"):
            return self._fit_dp_quantile(X, rho, sensitivity, rng)

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

    def _fit_dp_quantile(self, X, rho, sensitivity, rng) -> "Discretizer":
        if rho <= 0:
            raise ValueError("dp_quantile needs a positive rho")
        rng = rng if rng is not None else np.random.default_rng()
        lo, hi = self.value_range
        grid = np.linspace(lo, hi, self.grid_cells + 1)
        n_genes = X.shape[1]

        # One histogram per gene; a row lands in exactly one cell of each, so
        # this is n_genes releases of L2 sensitivity `sensitivity`.
        cell = np.clip(np.digitize(np.clip(X, lo, hi), grid[1:-1]), 0,
                       self.grid_cells - 1)
        sigma = sensitivity * np.sqrt(n_genes / (2.0 * rho))
        self.noise_sigma = float(sigma)
        self.rho_spent = float(n_genes * sensitivity ** 2 / (2.0 * sigma ** 2))

        # dp_uniform buys only the two bounds and spaces the edges evenly
        # between them, which is what smartnoise-synth's BinTransformer does
        # when it is given preprocessor_eps instead of public bounds.
        interior = (np.array([]) if self.strategy == "dp_uniform"
                    else np.arange(1, self.n_bins) / self.n_bins)
        targets = np.concatenate([[self.tail], interior, [1.0 - self.tail]])
        self._edges = []
        for j in range(n_genes):
            h = np.bincount(cell[:, j], minlength=self.grid_cells).astype(float)
            h = np.maximum(h + rng.normal(0.0, sigma, self.grid_cells), 0.0)
            if h.sum() <= 0:                       # pure noise wiped it out
                self._edges.append(np.linspace(lo, hi, self.n_bins + 1))
                continue
            cdf = np.cumsum(h) / h.sum()
            # Index of the first cell whose CDF reaches each target; the edge is
            # that cell's upper boundary.
            idx = np.searchsorted(cdf, targets, side="left")
            idx = np.clip(idx, 0, self.grid_cells - 1)
            edges = grid[idx + 1]
            edges[0] = grid[idx[0]]                # lower outer edge: cell start
            # Keep edges strictly increasing by at least one grid cell, so no
            # bin is empty by construction.
            step = grid[1] - grid[0]
            for b in range(1, len(edges)):
                edges[b] = max(edges[b], edges[b - 1] + step)
            if edges[-1] > hi:                      # pushed past the grid: shift back
                edges = edges - (edges[-1] - hi)
            if self.strategy == "dp_uniform":
                edges = np.linspace(edges[0], edges[-1], self.n_bins + 1)
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

    def inverse_transform(
        self,
        X_disc: np.ndarray,
        dither: bool = True,
        rng=None,  # np.random.Generator or None
    ) -> np.ndarray:
        """
        Map discrete bin indices back to continuous values.

        Parameters
        ----------
        X_disc : ndarray of int, shape (n_samples, n_genes)
        dither : bool, default True
            If True, sample uniformly from within each bin's continuous range
            instead of returning the bin centre.  This avoids the severe
            quantisation artefact (each gene taking only n_bins distinct values)
            that makes synthetic data trivially distinguishable from real data.
            Set to False to recover the old bin-centre behaviour.
        rng : np.random.Generator or None
            Random generator used when dither=True.  If None, a default_rng(0)
            is created.  Pass the generator from the caller to reproduce results.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before inverse_transform().")
        if dither and rng is None:
            rng = np.random.default_rng(0)

        X_cont = np.empty(X_disc.shape, dtype=np.float32)
        for j in range(X_disc.shape[1]):
            edges = self._edges[j]
            idx = np.clip(X_disc[:, j], 0, self.n_bins - 1)

            if not dither:
                centres = self.bin_centers(j)
                X_cont[:, j] = centres[idx]
                continue

            # Build per-bin [lo, hi) boundaries.
            # For quantile edges with n_bins+1 points (edges[0]..edges[n_bins]):
            #   bin b → [edges[b], edges[b+1])   (last bin is closed on the right)
            if len(edges) == 0:
                X_cont[:, j] = 0.0
                continue

            n_edges = len(edges)
            col = np.empty(len(idx), dtype=np.float32)
            for b in range(self.n_bins):
                mask = idx == b
                if not mask.any():
                    continue
                n_samples = int(mask.sum())
                if self.zero_inflated and b == 0:
                    col[mask] = 0.0
                else:
                    # lo and hi from the quantile edge array
                    lo_i = b if b < n_edges else n_edges - 1
                    hi_i = b + 1 if b + 1 < n_edges else n_edges - 1
                    lo = float(edges[lo_i])
                    hi = float(edges[hi_i])
                    if hi <= lo:
                        col[mask] = lo
                    else:
                        col[mask] = rng.uniform(lo, hi, n_samples).astype(np.float32)
            X_cont[:, j] = col

        return X_cont

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self._edges)
