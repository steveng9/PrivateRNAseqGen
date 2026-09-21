"""The privacy accounting is the one place a plausible-looking edit is silent.

These check the zCDP arithmetic against three independent references: the
closed-form conversion, MST's own weight-vector formulation, and the invariant
that rho actually spent never exceeds the budget.
"""
import math
import sys
from pathlib import Path

import mbi
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pgm_fitter import PrivatePGMFitter, rho_from_eps_delta  # noqa: E402


def closed_form_rho(eps, delta):
    L = math.log(1.0 / delta)
    t = -math.sqrt(L) + math.sqrt(L + eps)
    return t * t


@pytest.mark.parametrize("eps", [0.5, 1.0, 3.0, 10.0])
def test_closed_form_conversion_round_trips(eps):
    """rho + 2*sqrt(rho*ln(1/delta)) must land back on epsilon."""
    delta = 1e-5
    rho = closed_form_rho(eps, delta)
    back = rho + 2 * math.sqrt(rho * math.log(1 / delta))
    assert back == pytest.approx(eps, rel=1e-9)


@pytest.mark.parametrize("eps", [0.5, 1.0, 3.0, 10.0])
def test_opendp_conversion_is_never_weaker_than_closed_form(eps):
    """The numerical inversion may allow a larger rho, never a smaller one.

    A smaller rho means more noise, so the closed-form fallback is the safe
    side; this pins the direction so a bad fallback cannot weaken the claim.
    """
    delta = 1e-5
    assert rho_from_eps_delta(eps, delta) >= closed_form_rho(eps, delta) - 1e-12


def test_sigma_matches_mst_weight_formulation():
    """MST writes the same rule as `weights / norm(weights)`, then sigma/weight.

    For k equal weights that is sigma*sqrt(k), where sigma = sqrt(1/(2*rho)).
    Our _sigma_zcdp must agree exactly.
    """
    rho_total = 1.5
    for k in (1, 10, 978, 1956):
        ours = PrivatePGMFitter(composition="zcdp")._sigma_zcdp(k, 1.0, rho_total)
        w = np.ones(k) / np.linalg.norm(np.ones(k))
        mst = math.sqrt(1.0 / (2.0 * rho_total)) / w[0]
        assert ours == pytest.approx(mst, rel=1e-12)


def test_sigma_grows_as_sqrt_k_not_k():
    f = PrivatePGMFitter(composition="zcdp")
    s1 = f._sigma_zcdp(100, 1.0, 1.5)
    s2 = f._sigma_zcdp(400, 1.0, 1.5)
    assert s2 / s1 == pytest.approx(2.0, rel=1e-12)   # sqrt(4), not 4


def test_zcdp_beats_basic_at_genomic_scale():
    """The whole point, at the configuration we actually shipped.

    BRCA joint mode: 978 one-way and 978 gene-by-label cliques, budget_weights
    (0.33, 0.67), epsilon 10, delta 1e-5.  The basic figures here are the ones
    the shipped fitter printed in logs/pgm_sweep.log.
    """
    f = PrivatePGMFitter(epsilon=10.0, delta=1e-5,
                         budget_weights=(0.33, 0.67, 0.0, 0.0))
    rho = rho_from_eps_delta(10.0, 1e-5)

    sigma_basic_1way = f._gaussian_sigma(0.33 * 10.0 / 978)
    sigma_basic_2way = f._gaussian_sigma(0.67 * 10.0 / 978)
    assert sigma_basic_1way == pytest.approx(1435.8, rel=1e-3)
    assert sigma_basic_2way == pytest.approx(707.2, rel=1e-3)

    sigma_zcdp_1way = f._sigma_zcdp(978, 0.33, rho)
    sigma_zcdp_2way = f._sigma_zcdp(978, 0.67, rho)
    assert sigma_zcdp_1way < 35 and sigma_zcdp_2way < 35
    assert sigma_basic_1way / sigma_zcdp_1way > 45


def test_small_k_is_where_basic_is_tolerable():
    """Below ~50 measurements basic composition is not catastrophic.

    This is the regime the original code was implicitly written for; it is the
    gene-scale k that breaks it.
    """
    f = PrivatePGMFitter(epsilon=10.0, delta=1e-5, composition="zcdp")
    rho = rho_from_eps_delta(10.0, 1e-5)
    ratio = f._gaussian_sigma(10.0 / 20) / f._sigma_zcdp(20, 1.0, rho)
    assert ratio < 5


def _fake_dataset(n_cliques, cells=4):
    class _Proj:
        def datavector(self):
            return np.zeros(cells)

    class _DS:
        def project(self, clique):
            return _Proj()

    return _DS(), [(f"g{i}",) for i in range(n_cliques)]


@pytest.mark.parametrize("weights", [(1.0, 0.0, 0.0, 0.0), (0.33, 0.67, 0.0, 0.0)])
def test_rho_spent_never_exceeds_budget(weights):
    f = PrivatePGMFitter(epsilon=10.0, delta=1e-5, budget_weights=weights,
                         composition="zcdp")
    ds, cliques = _fake_dataset(500)
    orders = [cliques if w > 0 else [] for w in weights]
    f._build_measurements(ds, orders)
    assert f.rho_spent == pytest.approx(rho_from_eps_delta(10.0, 1e-5), rel=1e-9)


def test_basic_mode_leaves_absent_orders_unspent():
    """Legacy behaviour is preserved exactly, so old results still reproduce."""
    f = PrivatePGMFitter(epsilon=10.0, delta=1e-5,
                         budget_weights=(0.33, 0.67, 0.0, 0.0),
                         composition="basic")
    ds, cliques = _fake_dataset(10)
    ms = f._build_measurements(ds, [cliques, [], [], []])
    sigma = ms[0][2]
    assert sigma == pytest.approx(f._gaussian_sigma(0.33 * 10.0 / 10), rel=1e-12)


def test_rejects_unknown_composition():
    with pytest.raises(ValueError):
        PrivatePGMFitter(composition="renyi")


# ----------------------------------------------------------------------
# Neighbouring relation / sensitivity
# ----------------------------------------------------------------------

def test_replace_costs_sqrt2_more_noise_than_add_remove():
    """Bounded DP has L2 sensitivity sqrt(2), so every sigma grows by sqrt(2)."""
    kw = dict(epsilon=10.0, delta=1e-5, budget_weights=(0.33, 0.67, 0.0, 0.0))
    a = PrivatePGMFitter(**kw, neighboring="add_remove")
    r = PrivatePGMFitter(**kw, neighboring="replace")
    rho = rho_from_eps_delta(10.0, 1e-5)
    for k, w in [(1, 0.33), (978, 0.33), (978, 0.67)]:
        assert r._sigma_zcdp(k, w, rho) == pytest.approx(
            math.sqrt(2) * a._sigma_zcdp(k, w, rho), rel=1e-12
        )


def test_both_neighboring_relations_spend_exactly_rho_total():
    """Accounting must close for either sensitivity, not just the default."""
    domain = mbi.Domain(["g0", "g1", "g2"], [4, 4, 4])
    df = pd.DataFrame(np.random.randint(0, 4, size=(60, 3)), columns=["g0", "g1", "g2"])
    marginals = {"1way": [("g0",), ("g1",), ("g2",)],
                 "2way": [("g0", "g1"), ("g1", "g2")]}
    rho = rho_from_eps_delta(4.0, 1e-5)
    for nb in ("add_remove", "replace"):
        f = PrivatePGMFitter(epsilon=4.0, delta=1e-5, pgm_iters=10,
                             budget_weights=(0.5, 0.5, 0.0, 0.0), neighboring=nb)
        f._build_measurements(mbi.Dataset(df, domain),
                              [marginals["1way"], marginals["2way"], [], []])
        assert f.rho_spent == pytest.approx(rho, rel=1e-9), nb


def test_add_remove_never_releases_the_exact_row_count():
    """Under unbounded DP, n is sensitive: the model's total must be estimated.

    This is the defect fixed on 2026-09-20 -- sensitivity-1 noise was being
    combined with an exact release of n, which is a guarantee for neither
    neighbouring relation.
    """
    domain = mbi.Domain(["g0", "g1"], [4, 4])
    n = 137
    df = pd.DataFrame(np.random.randint(0, 4, size=(n, 2)), columns=["g0", "g1"])
    marginals = {"1way": [("g0",), ("g1",)], "2way": [("g0", "g1")]}

    f = PrivatePGMFitter(epsilon=10.0, delta=1e-5, pgm_iters=25,
                         budget_weights=(0.5, 0.5, 0.0, 0.0),
                         neighboring="add_remove").fit(df, domain, marginals)
    # Estimated from noise, so it must be close to n but essentially never equal.
    assert f.estimated_total != n
    assert abs(f.estimated_total - n) < 0.25 * n

    g = PrivatePGMFitter(epsilon=10.0, delta=1e-5, pgm_iters=25,
                         budget_weights=(0.5, 0.5, 0.0, 0.0),
                         neighboring="replace").fit(df, domain, marginals)
    assert g.estimated_total == pytest.approx(n)


def test_rejects_unknown_neighboring():
    with pytest.raises(ValueError, match="neighboring"):
        PrivatePGMFitter(neighboring="bounded")


def _privacy_loss_samples(fitter, cliques, dsD, dsO, trials, rng):
    """Log-likelihood ratio of releases drawn from D, evaluated under D vs D'."""
    rho_total = rho_from_eps_delta(fitter.epsilon, fitter.delta)
    active = [i for i in (0, 1) if cliques[i]]
    denom = sum(fitter.budget_weights[i] for i in active)
    Z = np.zeros(trials)
    for order in active:
        sigma = fitter._sigma_zcdp(len(cliques[order]),
                                   fitter.budget_weights[order] / denom, rho_total)
        for clique in cliques[order]:
            xD = dsD.project(clique).datavector()
            xO = dsO.project(clique).datavector()
            y = xD + rng.normal(0, sigma, size=(trials, xD.size))
            Z += ((-((y - xD) ** 2) + ((y - xO) ** 2)) / (2 * sigma ** 2)).sum(axis=1)
    return Z


@pytest.mark.parametrize("neighboring", ["add_remove", "replace"])
def test_measured_privacy_loss_matches_the_predicted_normal(neighboring):
    """Audit the guarantee empirically rather than trusting the sigma formula.

    The theory says the privacy loss of the whole release is exactly
    Z ~ N(rho, 2rho).  Measure it: build the real measurements on D and on a
    worst-case neighbour D', then score actual releases under both.  This
    exercises the sigma, the true marginal difference (so the real sensitivity,
    not an assumed one) and the composition across every clique at once.

    A wrong sensitivity shows up here as a mean that misses rho_spent, which no
    amount of internally-consistent arithmetic elsewhere would catch.
    """
    rng = np.random.default_rng(0)
    n_genes, n_bins, n_rows, trials = 5, 4, 300, 6000
    cols = [f"g{i}" for i in range(n_genes)]
    domain = mbi.Domain(cols, [n_bins] * n_genes)
    base = pd.DataFrame(rng.integers(0, n_bins, size=(n_rows, n_genes)), columns=cols)

    if neighboring == "add_remove":
        other = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    else:
        # Worst case: every gene moves, so every clique sees the full sqrt(2).
        other = base.copy()
        for c in cols:
            other.loc[0, c] = (other.loc[0, c] + 1) % n_bins

    cliques = [[(c,) for c in cols],
               [(cols[i], cols[i + 1]) for i in range(n_genes - 1)], [], []]
    f = PrivatePGMFitter(epsilon=3.0, delta=1e-5, neighboring=neighboring,
                         budget_weights=(0.4, 0.6, 0.0, 0.0))
    f._build_measurements(mbi.Dataset(base, domain), cliques)
    rho = f.rho_spent

    Z = _privacy_loss_samples(f, cliques, mbi.Dataset(base, domain),
                              mbi.Dataset(other, domain), trials, rng)

    # Mean is rho, variance is 2*rho -- both, since one number controls both.
    se = Z.std() / math.sqrt(trials)
    assert abs(Z.mean() - rho) < 4 * se, (
        f"{neighboring}: measured loss {Z.mean():.5f} vs budgeted {rho:.5f}")
    assert Z.var() == pytest.approx(2 * rho, rel=0.08)
    # And the guarantee itself: exceeding epsilon must be a delta-rare event.
    assert (Z > f.epsilon).mean() <= f.delta
