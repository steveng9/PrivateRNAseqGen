"""Bin edges are released, so how they are chosen is part of the privacy story.

`quantile` edges are private percentiles and spend nothing -- that is the leak
these strategies exist to close.  `uniform` must not read the data at all, and
`dp_quantile` must spend exactly its share of rho, leaving the marginals the
rest, so the pipeline as a whole still spends exactly rho_total.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from discretization import Discretizer  # noqa: E402
from generator import StratHiMPGMGenerator  # noqa: E402
from pgm_fitter import rho_from_eps_delta  # noqa: E402


def _data(n=400, g=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(10.0, 2.0, size=(n, g)) + np.arange(g)
    y = rng.integers(0, 3, size=n).astype(str)
    return X, y


def test_uniform_edges_do_not_depend_on_the_data():
    X, _ = _data()
    a = Discretizer(n_bins=4, strategy="uniform", value_range=(0, 24)).fit(X)
    b = Discretizer(n_bins=4, strategy="uniform", value_range=(0, 24)).fit(X * 0 + 5)
    for ea, eb in zip(a._edges, b._edges):
        np.testing.assert_array_equal(ea, eb)
    np.testing.assert_array_equal(a._edges[0], [0, 6, 12, 18, 24])
    assert a.rho_spent == 0.0


def test_uniform_clips_out_of_range_values_into_end_bins():
    d = Discretizer(n_bins=4, strategy="uniform", value_range=(0, 24)).fit(np.zeros((1, 1)))
    out = d.transform(np.array([[-5.0], [30.0]]))
    assert out.ravel().tolist() == [0, 3]


def test_dp_quantile_spends_exactly_its_rho():
    X, _ = _data(g=10)
    d = Discretizer(n_bins=4, strategy="dp_quantile").fit(
        X, rho=0.05, rng=np.random.default_rng(1))
    assert d.rho_spent == pytest.approx(0.05, rel=1e-12)


def test_dp_quantile_edges_are_strictly_increasing_even_under_heavy_noise():
    X, _ = _data(n=50, g=20)
    d = Discretizer(n_bins=4, strategy="dp_quantile").fit(
        X, rho=1e-6, rng=np.random.default_rng(2))
    for e in d._edges:
        assert len(e) == 5 and np.all(np.diff(e) > 0)


def test_dp_quantile_approaches_true_quartiles_with_a_large_budget():
    X, _ = _data(n=20000, g=3)
    d = Discretizer(n_bins=4, strategy="dp_quantile", grid_cells=480).fit(
        X, rho=1e6, rng=np.random.default_rng(3))
    for j, e in enumerate(d._edges):
        true = np.percentile(X[:, j], [25, 50, 75])
        np.testing.assert_allclose(e[1:4], true, atol=0.1)


@pytest.mark.parametrize("binning", ["uniform", "dp_quantile"])
def test_whole_pipeline_spends_exactly_rho_total(binning):
    X, y = _data(g=6)
    eps = 3.0
    gen = StratHiMPGMGenerator(epsilon=eps, n_bins=4, n_1way=6, n_2way=0,
                               budget_weights=(0.33, 0.67, 0.0, 0.0),
                               joint_mode=True, pgm_iters=50, random_seed=0,
                               binning=binning, binning_budget=0.1)
    gen.fit(X, y)
    spent = gen._discretizer.rho_spent + gen._joint_fitter.rho_spent
    assert spent == pytest.approx(rho_from_eps_delta(eps, 1e-5), rel=1e-9)


def test_dp_quantile_rejects_basic_composition():
    with pytest.raises(ValueError):
        StratHiMPGMGenerator(binning="dp_quantile", composition="basic")
