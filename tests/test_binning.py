import numpy as np
import pandas as pd

from credit_risk.binning import MISSING, assign_bins, fit_edges, monotonic_edges, quantile_edges, tree_edges


def data(n=6000, seed=0, noise=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 0.1 + 0.15 * (x > 0) + (rng.uniform(-0.05, 0.05, n) if noise else 0)
    return x, (rng.random(n) < p).astype(int)


def test_quantile_bins_are_equal_sized():
    x, _ = data()
    idx = np.searchsorted(quantile_edges(x, 10), x, side="right")
    assert np.bincount(idx).min() > 0.09 * len(x)


def test_monotonic_edges_give_a_one_directional_pattern_and_big_enough_bins():
    x, y = data(noise=True)
    edges = monotonic_edges(x, y)
    idx = np.searchsorted(edges, x, side="right")
    rate = np.array([y[idx == i].mean() for i in range(len(edges) + 1)])
    size = np.bincount(idx)
    assert (size >= 0.05 * len(x)).all()
    assert (np.diff(rate) >= -1e-12).all() or (np.diff(rate) <= 1e-12).all()


def test_tree_finds_the_obvious_threshold():
    x, y = data(seed=3)
    assert any(abs(e) < 0.15 for e in tree_edges(x, y, max_bins=3))


def test_missing_values_get_their_own_bin_and_edges_are_reused_unchanged():
    x, y = data()
    edges = fit_edges(pd.Series(x), y, "monotonic")
    labels = assign_bins([np.nan, -9.0, 9.0], edges)
    assert labels.iloc[0] == MISSING and labels.iloc[1] == "bin_00" and labels.iloc[2] != "bin_00"
