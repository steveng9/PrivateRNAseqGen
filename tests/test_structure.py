"""The threshold edge estimator and the DP spanning-tree structures.

Edges: with little noise the threshold estimator must recover per-gene bounds
that the legacy clip estimator misses (the phantom mass of empty cells), and
spend exactly as much as clip does -- it is post-processing of the same release.

Tree: selection + measurement + edges must spend exactly rho_total, the tree
must span the genes, and with a big budget selection must find planted pairs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from discretization import Discretizer  # noqa: E402
from generator import StratHiMPGMGenerator  # noqa: E402
from marginal_selection import dp_select_tree  # noqa: E402
from pgm_fitter import rho_from_eps_delta  # noqa: E402


def _genes(n=800, g=30, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(8, 14, size=g)
    return rng.normal(mu, 0.7, size=(n, g))


@pytest.mark.parametrize("strategy", ["dp_uniform", "dp_quantile"])
def test_threshold_recovers_bounds_that_clip_misses(strategy):
    X = _genes()
    lo, hi = np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0)
    err = {}
    for est in ("clip", "threshold"):
        d = Discretizer(n_bins=8, strategy=strategy, edge_estimator=est).fit(
            X, rho=50.0, rng=np.random.default_rng(1))
        E = np.array(d._edges)
        err[est] = np.median(np.abs(E[:, 0] - lo) + np.abs(E[:, -1] - hi))
    assert err["threshold"] < 0.5
    assert err["clip"] > 3 * err["threshold"]


def test_threshold_costs_the_same_as_clip():
    X = _genes()
    a = Discretizer(n_bins=8, strategy="dp_quantile", edge_estimator="clip").fit(
        X, rho=0.3, rng=np.random.default_rng(0))
    b = Discretizer(n_bins=8, strategy="dp_quantile", edge_estimator="threshold").fit(
        X, rho=0.3, rng=np.random.default_rng(0))
    assert a.rho_spent == pytest.approx(b.rho_spent, rel=1e-12)


def test_threshold_edges_increase_even_when_nothing_clears_the_noise():
    X = _genes(n=30)
    d = Discretizer(n_bins=16, strategy="dp_quantile", edge_estimator="threshold").fit(
        X, rho=1e-6, rng=np.random.default_rng(2))
    for e in d._edges:
        assert len(e) == 17 and np.all(np.diff(e) > 0)


def test_selection_spans_and_finds_planted_pairs():
    rng = np.random.default_rng(0)
    n, G, K, C = 2000, 12, 4, 2
    y = rng.integers(0, C, size=n)
    Z = rng.normal(size=(n, G))
    Z[:, 1] = Z[:, 0] + 0.2 * rng.normal(size=n)      # planted: (0,1), (2,3)
    Z[:, 3] = Z[:, 2] + 0.2 * rng.normal(size=n)
    Xd = np.stack([np.digitize(Z[:, j], np.quantile(Z[:, j], [.25, .5, .75]))
                   for j in range(G)], axis=1)
    gl = np.stack([[np.bincount(Xd[y == c, j], minlength=K) for c in range(C)]
                   for j in range(G)]).transpose(0, 2, 1).astype(float)
    pairs, diag = dp_select_tree(Xd, y, gl, rho=100.0, rng=rng, with_label=True)
    assert len(pairs) == G - 1
    parent = list(range(G))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for a, b in pairs:
        assert find(a) != find(b)                     # no cycles
        parent[find(a)] = find(b)
    s = {tuple(sorted(p)) for p in pairs}
    assert (0, 1) in s and (2, 3) in s
    assert diag["score_ratio"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("structure", ["tree", "tree_label"])
@pytest.mark.parametrize("binning", ["uniform", "dp_quantile"])
def test_tree_pipeline_spends_exactly_rho_total(structure, binning):
    X = _genes(n=300, g=8)
    y = np.random.default_rng(0).integers(0, 3, size=300).astype(str)
    g = StratHiMPGMGenerator(epsilon=5.0, n_bins=4, n_1way=8, n_2way=0,
                             joint_mode=True, pgm_iters=20, random_seed=0,
                             binning=binning, edge_estimator="threshold",
                             structure=structure)
    g.fit(X, y)
    b = g.rho_breakdown
    spent = b["binning"] + b["selection"] + b["measurement"]
    assert spent == pytest.approx(rho_from_eps_delta(5.0, 1e-5), rel=1e-9)
    order = "2way" if structure == "tree" else "3way"
    tree = [c for c in g._marginals[order] if "__label__" not in c or len(c) == 3]
    assert len(tree) == 7
    Xs, ys = g.generate(50)
    assert Xs.shape == (50, 8) and len(ys) == 50


def test_tree_needs_joint_mode_and_zcdp():
    with pytest.raises(ValueError):
        StratHiMPGMGenerator(structure="tree", joint_mode=False)
    with pytest.raises(ValueError):
        StratHiMPGMGenerator(structure="tree", joint_mode=True, composition="basic")


@pytest.mark.parametrize("k,l", [(0, 3), (3, 0), (3, 3), (8, 0), (8, 5)])
def test_forest_spends_exactly_rho_total_and_covers_every_gene(k, l):
    X = _genes(n=300, g=8)
    y = np.random.default_rng(0).integers(0, 3, size=300).astype(str)
    g = StratHiMPGMGenerator(epsilon=5.0, n_bins=4, n_1way=8, n_2way=0,
                             joint_mode=True, pgm_iters=20, random_seed=0,
                             binning="dp_quantile", edge_estimator="threshold",
                             structure="forest", k_label=k, l_pairs=l)
    g.fit(X, y)
    b = g.rho_breakdown
    assert b["binning"] + b["selection"] + b["measurement"] == pytest.approx(
        rho_from_eps_delta(5.0, 1e-5), rel=1e-9)
    m = g._marginals
    gl = [c for c in m["2way"] if "__label__" in c]
    pairs = [c for c in m["2way"] if "__label__" not in c]
    assert len(gl) == k and len(pairs) == l
    covered = {x for c in m["2way"] + m["1way"] for x in c}
    assert set(g._selected_gene_names) <= covered
    # a gene with its own table in a chosen 2-way gets no 1-way
    ones = {c[0] for c in m["1way"] if c[0] != "__label__"}
    assert not ones & {x for c in m["2way"] for x in c}
    Xs, ys = g.generate(40)
    assert Xs.shape == (40, 8)


def test_label_gene_selection_finds_planted_label_genes():
    from marginal_selection import dp_select_label_genes
    rng = np.random.default_rng(0)
    n, G, K = 2000, 40, 4
    y = rng.integers(0, 3, n)
    Xd = rng.integers(0, K, (n, G))
    for gi in (5, 17, 33):                    # these genes follow the label
        Xd[:, gi] = np.where(rng.random(n) < 0.8, y, Xd[:, gi])
    counts = np.bincount(y, minlength=3).astype(float)
    chosen, _ = dp_select_label_genes(Xd, y, counts, 3, K, rho=1.0, rng=rng)
    assert set(chosen) == {5, 17, 33}
