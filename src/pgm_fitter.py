"""
Private-PGM fitting and sampling wrapper.

Takes a discretized dataset (as a pandas DataFrame) together with pre-selected
marginal cliques, adds calibrated Gaussian noise, and fits a graphical model
via mbi.FactoredInference.

Budget allocation (Option C from the design plan) splits the budget across
marginal orders by ``budget_weights`` -- fractions of ρ under zCDP, fractions of
ε under the legacy basic composition -- and shares each order's slice equally
among the cliques of that order.

Composition (``composition="zcdp"``, the default)
------------------------------------------------
Gaussian mechanisms compose additively in zero-concentrated DP: a release with
L2 sensitivity 1 and noise scale σ is ρ-zCDP for ρ = 1/(2σ²), and k of them cost
Σ ρ_i.  Inverting that for a fixed budget gives

    σ_order = sqrt(k_order / (2 * ρ_order)),    ρ_order = weight_order * ρ_total

so σ grows as sqrt(k), not k.  ρ_total is the largest ρ whose zCDP-to-(ε, δ)
conversion still lands at ε.

This is the accounting McKenna's own mechanisms use.  In MST the same rule is
written as a weight vector normalised by its L2 norm --
``weights / np.linalg.norm(weights)``, then ``sigma / weight`` per clique --
which for k equal weights is exactly σ·sqrt(k).  ``tests/test_composition.py``
checks our σ against that formulation directly.

Composition (``composition="basic"``, legacy)
---------------------------------------------
The original accounting, kept so earlier results reproduce.  It divides ε
linearly over marginals and uses the classical Gaussian bound:

    ε_per = weight_order * ε / k_order
    σ     = sqrt(2 * ln(1.25 / delta)) / ε_per

σ then grows *linearly* in k.  Both satisfy (ε, δ)-DP; basic is simply far
looser, and the gap widens with the number of marginals.  At ε=10, δ=1e-5 and
k=1956 (978 genes, joint mode) it is σ=1436 against σ=23, on a cohort of 871
rows -- noise seven times the size of the counts being measured.

Neighbouring relation and sensitivity (``neighboring``)
-------------------------------------------------------
Every σ above is a *multiple of the query's L2 sensitivity*, and that
sensitivity depends on what "one person's data" means:

``"add_remove"`` (unbounded DP, the default, and what MST/AIM assume)
    Neighbours differ by inserting or deleting one row.  A marginal count
    vector then changes in exactly one cell by 1, so Δ₂ = 1.  The dataset size
    n is itself sensitive, so it must NOT be released exactly: we pass
    ``total=None`` and let mbi reconstruct n as the minimum-variance unbiased
    estimate from the noisy 1-way marginals.  That estimate is free (it reuses
    measurements already paid for) and accurate to ~0.2% of n at ε=10.

``"replace"`` (bounded DP)
    Neighbours differ by changing one row in place, so n is public by
    construction and may be released exactly.  But one cell drops by 1 while
    another rises by 1, so Δ₂ = sqrt(2), and every σ grows by sqrt(2) -- a
    2× cost in ρ for the same ε.

The two must not be mixed.  Before 2026-09-20 this module used the Δ₂ = 1
noise scale (an add/remove claim) while passing the exact row count to
``estimate`` (a bounded-DP assumption).  Neither guarantee held: under
add/remove, n distinguishes neighbours with certainty.  Stratified mode had
the same defect twice over, since it also sized each class's synthetic sample
from the exact per-class counts.  Both now read the model's noisy totals.

Note also that parallel composition across the per-class subsets in stratified
mode is only valid for ``add_remove``: under ``replace``, one changed row can
move between two classes and touch both submodels, so their budgets would have
to compose sequentially instead.

WHAT NEITHER SETTING FIXES
--------------------------
Marginal selection (``marginal_selection.select``, which ranks genes by the
variance of the private data and pairs by its Spearman correlations) and
discretisation (``discretization.fit``, whose bin edges are percentiles of the
private data) both read the training data and spend no budget.  MST spends a
full ρ/3 on selection via the exponential mechanism.  Until those are either
privatised or fitted on public data, the end-to-end ε reported here is the
budget spent on the *measurements only*, not a guarantee for the pipeline.
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

# L2 sensitivity of a marginal count vector under each neighbouring relation.
_SENSITIVITY = {"add_remove": 1.0, "replace": math.sqrt(2.0)}


def rho_from_eps_delta(epsilon: float, delta: float) -> float:
    """Largest ρ whose ρ-zCDP guarantee implies (ε, δ)-DP.

    Prefers OpenDP's numerically-inverted conversion, which is what snsynth's
    ``cdp_rho`` -- and therefore MST and AIM -- uses.  Falls back to the closed
    form ε = ρ + 2·sqrt(ρ·ln(1/δ)) (Bun & Steinke 2016, Prop. 1.3) when OpenDP
    is unavailable.

    The fallback is strictly *conservative*: it returns a smaller ρ, hence more
    noise, than the numerical inversion (ρ 1.550 against 1.783 at ε=10,
    δ=1e-5).  Erring toward more noise is the safe direction for a privacy
    parameter, so the fallback never weakens the guarantee.
    """
    if epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("epsilon must be > 0 and delta in (0, 1).")
    try:
        from snsynth.utils import cdp_rho
        return float(cdp_rho(epsilon, delta))
    except Exception:
        L = math.log(1.0 / delta)
        t = -math.sqrt(L) + math.sqrt(L + epsilon)   # t = sqrt(rho)
        return t * t


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
        composition: str = "zcdp",
        neighboring: str = "add_remove",
    ):
        if abs(sum(budget_weights) - 1.0) > 1e-6:
            raise ValueError("budget_weights must sum to 1.")
        if composition not in ("zcdp", "basic"):
            raise ValueError("composition must be 'zcdp' or 'basic'.")
        if neighboring not in _SENSITIVITY:
            raise ValueError("neighboring must be 'add_remove' or 'replace'.")
        self.epsilon = epsilon
        self.delta = delta
        self.budget_weights = budget_weights
        self.pgm_iters = pgm_iters
        self.composition = composition
        self.neighboring = neighboring
        #: L2 sensitivity of one marginal release under `neighboring`.
        self.sensitivity = _SENSITIVITY[neighboring]
        #: n as the fitted model sees it: estimated from noisy marginals
        #: under add/remove, the exact row count under replace.
        self.estimated_total: Optional[float] = None
        #: ρ actually spent, summed over measurements; filled in by fit().
        self.rho_spent: Optional[float] = None
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

        # Under add/remove neighbours the row count is itself sensitive, so we
        # must not hand mbi the true n.  Passing total=None makes it derive the
        # minimum-variance unbiased estimate from the noisy marginals we have
        # already paid for -- free, and what MST does.  Under `replace`, n is
        # public by construction and may be passed exactly.
        total = n_total if self.neighboring == "replace" else None
        print(
            f"  [pgm_fitter] Fitting FactoredInference: "
            f"{len(measurements)} measurements, "
            f"N={n_total if total is not None else 'estimated'}, "
            f"ε={self.epsilon}, δ={self.delta}, iters={self.pgm_iters} "
            f"[{self.neighboring}]"
        )
        engine = mbi.FactoredInference(domain, iters=self.pgm_iters)
        self._model = engine.estimate(measurements, total=total)
        self.estimated_total = float(self._model.total)
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
        """Legacy basic-composition σ: Δ₂·sqrt(2 ln(1.25/δ)) / ε."""
        return (self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta))
                / epsilon_per_marginal)

    def _sigma_zcdp(self, n_cliques: int, weight: float, rho_total: float) -> float:
        """σ for `n_cliques` sensitivity-1 Gaussians sharing `weight` of ρ.

        Each costs ρ_i = Δ₂²/(2σ²), so n_cliques of them cost n·Δ₂²/(2σ²);
        setting that equal to weight·ρ_total and solving gives the sqrt(k)
        growth, scaled by the sensitivity.
        """
        return self.sensitivity * math.sqrt(n_cliques / (2.0 * weight * rho_total))

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

        active = [i for i, c in enumerate(cliques_by_order)
                  if c and self.budget_weights[i] > 0.0]
        if not active:
            self.rho_spent = 0.0
            return measurements

        # Under zCDP, renormalise over the orders that actually have cliques, so
        # an order requested but not selected does not silently waste budget.
        # `basic` deliberately keeps the original behaviour -- an absent order's
        # share simply goes unspent -- so legacy results still reproduce exactly.
        total_weight = (sum(self.budget_weights[i] for i in active)
                        if self.composition == "zcdp" else 1.0)

        rho_total = (rho_from_eps_delta(self.epsilon, self.delta)
                     if self.composition == "zcdp" else None)
        rho_spent = 0.0

        for order_idx in active:
            cliques = cliques_by_order[order_idx]
            frac = self.budget_weights[order_idx] / total_weight

            if self.composition == "zcdp":
                sigma = self._sigma_zcdp(len(cliques), frac, rho_total)
                budget_label = f"ρ={frac * rho_total:.4f}"
            else:
                eps_per_marginal = (frac * self.epsilon) / len(cliques)
                sigma = self._gaussian_sigma(eps_per_marginal)
                budget_label = f"ε_per={eps_per_marginal:.4f}"

            print(
                f"  [pgm_fitter] {order_idx + 1}-way: {len(cliques)} marginals, "
                f"{budget_label}, σ={sigma:.4f} [{self.composition}]"
            )

            for clique in cliques:
                # mbi projects onto the clique and returns a flattened count vector
                true_marginal = dataset.project(clique).datavector()
                noise = np.random.normal(0, sigma, true_marginal.shape)
                y = true_marginal + noise
                measurements.append((None, y, sigma, clique))
                # Each release costs rho = Delta_2^2 / (2 sigma^2).
                rho_spent += self.sensitivity ** 2 / (2.0 * sigma ** 2)

        self.rho_spent = rho_spent
        if self.composition == "zcdp" and rho_spent > rho_total * (1 + 1e-9):
            raise AssertionError(
                f"zCDP accounting overspent: {rho_spent:.6f} > {rho_total:.6f}"
            )
        return measurements
