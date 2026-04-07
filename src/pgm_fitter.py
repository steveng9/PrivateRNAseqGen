"""
Private-PGM fitting and sampling wrapper.

Takes a discretized dataset (as a pandas DataFrame) together with pre-selected
marginal cliques, adds calibrated Gaussian noise, and fits a graphical model
via mbi.FactoredInference.

Budget allocation (Option C from the design plan):
  fraction_1way  = 0.50 of total ε → shared equally among all 1-way cliques
  fraction_2way  = 0.25 of total ε → shared equally among all 2-way cliques
  fraction_3way  = 0.15 of total ε → shared equally among all 3-way cliques
  fraction_4way  = 0.10 of total ε → shared equally among all 4-way cliques

Within each order, the per-marginal ε is the order's fraction divided by the
number of marginals in that order.  The Gaussian mechanism noise scale is:

    σ = sqrt(2 * ln(1.25 / delta)) / ε_per_marginal

with L2 sensitivity 1 (one record changes one count cell by ±1).
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# mbi uses a deprecated pandas groupby pattern; suppress the noise.
warnings.filterwarnings("ignore", category=FutureWarning, module="mbi")

import mbi


# Fractions of total ε allocated per order (must sum to 1).
_DEFAULT_BUDGET_WEIGHTS = (0.50, 0.25, 0.15, 0.10)


class PrivatePGMFitter:
    """
    Fits a graphical model using Private-PGM with (ε, δ)-DP Gaussian noise.

    Parameters
    ----------
    epsilon : float
        Total privacy budget ε.
    delta : float
        δ parameter for (ε, δ)-DP Gaussian mechanism.
    budget_weights : tuple[float, float, float, float]
        Fractions of ε allocated to (1-way, 2-way, 3-way, 4-way) marginals.
        Must sum to 1.
    pgm_iters : int
        Number of optimisation iterations for FactoredInference.
    """

    def __init__(
        self,
        epsilon: float = 7.0,
        delta: float = 1e-5,
        budget_weights: tuple[float, float, float, float] = _DEFAULT_BUDGET_WEIGHTS,
        pgm_iters: int = 1000,
    ):
        if abs(sum(budget_weights) - 1.0) > 1e-6:
            raise ValueError("budget_weights must sum to 1.")
        self.epsilon = epsilon
        self.delta = delta
        self.budget_weights = budget_weights
        self.pgm_iters = pgm_iters
        self._model: Optional[mbi.GraphicalModel] = None
        self._domain: Optional[mbi.Domain] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        domain: mbi.Domain,
        marginals: dict[str, list[tuple[str, ...]]],
    ) -> "PrivatePGMFitter":
        """
        Fit the Private-PGM model from noisy marginal measurements.

        Parameters
        ----------
        df : pd.DataFrame
            Discretised data (integer-valued columns matching domain.attrs).
        domain : mbi.Domain
            Attribute names and cardinalities.
        marginals : dict
            Keys '1way', '2way', '3way', '4way'; values are lists of clique
            tuples of column name strings.
        """
        self._domain = domain
        dataset = mbi.Dataset(df, domain)
        n_total = len(df)

        cliques_by_order = [
            marginals.get("1way", []),
            marginals.get("2way", []),
            marginals.get("3way", []),
            marginals.get("4way", []),
        ]

        measurements = self._build_measurements(dataset, cliques_by_order)

        print(
            f"  [pgm_fitter] Fitting FactoredInference: "
            f"{len(measurements)} measurements, N={n_total}, "
            f"ε={self.epsilon}, δ={self.delta}, iters={self.pgm_iters}"
        )
        engine = mbi.FactoredInference(domain, iters=self.pgm_iters)
        self._model = engine.estimate(measurements, total=n_total)
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Draw synthetic samples from the fitted model.

        Returns a DataFrame with integer-valued columns matching domain.attrs.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before sample().")
        synth_dataset = self._model.synthetic_data(rows=n_samples)
        return synth_dataset.df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gaussian_sigma(self, epsilon_per_marginal: float) -> float:
        """
        Gaussian mechanism noise σ for L2 sensitivity=1 and given per-marginal ε.
        Uses the standard formula: σ = sqrt(2 ln(1.25/δ)) / ε.
        """
        return math.sqrt(2 * math.log(1.25 / self.delta)) / epsilon_per_marginal

    def _build_measurements(
        self,
        dataset: mbi.Dataset,
        cliques_by_order: list[list[tuple[str, ...]]],
    ) -> list[tuple]:
        """
        For each clique, compute the true marginal count vector, add Gaussian
        noise, and return a list of (Q, y, sigma, clique) tuples for mbi.
        """
        measurements = []

        for order_idx, cliques in enumerate(cliques_by_order):
            if not cliques:
                continue

            frac = self.budget_weights[order_idx]
            eps_per_marginal = (frac * self.epsilon) / len(cliques)
            sigma = self._gaussian_sigma(eps_per_marginal)

            order_label = f"{order_idx + 1}-way"
            print(
                f"  [pgm_fitter] {order_label}: {len(cliques)} marginals, "
                f"ε_per={eps_per_marginal:.4f}, σ={sigma:.4f}"
            )

            for clique in cliques:
                # mbi projects onto the clique and returns a flattened count vector
                true_marginal = dataset.project(clique).datavector()
                noise = np.random.normal(0, sigma, true_marginal.shape)
                y = true_marginal + noise
                measurements.append((None, y, sigma, clique))

        return measurements
