"""Cut a numeric field into bins for WoE. Missing values form their own bin.

Methods:
  quantile        equal-sized bins, no merging;
  monotonic       start from equal-sized bins, then merge neighbours until every bin holds at least
                  `min_share` of the loans and the default rate moves in one direction;
  tree            thresholds from a small decision tree (leaves of at least `min_share` of the loans).
Edges are learned on the fit window and reused unchanged when scoring.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

MISSING = "missing"


def _stats(x, y, edges):
    idx = np.searchsorted(edges, x, side="right")
    n = np.bincount(idx, minlength=len(edges) + 1)
    d = np.bincount(idx, weights=y, minlength=len(edges) + 1)
    return n, d


def quantile_edges(x, n_bins=10):
    return list(np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])))


def monotonic_edges(x, y, n_start=20, min_share=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    edges = quantile_edges(x, n_start)
    n, d = _stats(x, y, edges)
    # A field with many values at its minimum (counts, zeros) gets a first edge equal to that minimum, which
    # leaves an empty bin below it. Its "rate" of 0 is not data, so drop it before judging the direction.
    while len(edges) > 0 and n[0] == 0:
        edges.pop(0)
        n, d = _stats(x, y, edges)
    rate = d / np.maximum(n, 1)
    sign = np.sign(np.polyfit(np.arange(len(rate)), rate, 1)[0]) or 1.0
    while len(edges) > 0:
        n, d = _stats(x, y, edges)
        rate = d / np.maximum(n, 1)
        small = np.where(n < min_share * len(x))[0]
        if len(small):
            i = small[np.argmin(n[small])]
            left, right = i - 1, i + 1
            if left < 0:
                drop = i  # merge with the right neighbour: remove the edge between them
            elif right >= len(n):
                drop = i - 1
            else:
                drop = i - 1 if abs(rate[i] - rate[left]) <= abs(rate[i] - rate[right]) else i
            edges.pop(drop)
            continue
        bad = np.where(sign * np.diff(rate) < 0)[0]
        if len(bad):
            # merge the violating pair with the smallest gap in default rate
            drop = bad[np.argmin(np.abs(np.diff(rate))[bad])]
            edges.pop(drop)
            continue
        break
    return edges


def tree_edges(x, y, max_bins=6, min_share=0.05):
    x = np.asarray(x, float).reshape(-1, 1)
    tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, min_samples_leaf=min_share, random_state=0).fit(x, y)
    t = tree.tree_
    return sorted(set(float(v) for v in t.threshold[t.feature >= 0]))


def fit_edges(x, y, method, **kw):
    x, y = pd.Series(x), pd.Series(y)
    ok = x.notna()
    xv, yv = x[ok].to_numpy(), y[ok].to_numpy()
    if method == "quantile":
        return quantile_edges(xv, kw.get("n_bins", 10))
    if method == "monotonic":
        return monotonic_edges(xv, yv, min_share=kw.get("min_share", 0.05))
    if method == "tree":
        return tree_edges(xv, yv, max_bins=kw.get("max_bins", 6), min_share=kw.get("min_share", 0.05))
    raise ValueError(method)


def assign_bins(x, edges):
    """Bin number (0, 1, 2, ...) as a label like 'bin_03'; missing values get 'missing'."""
    x = pd.Series(x)
    idx = np.searchsorted(edges, x.to_numpy(float), side="right")
    labels = pd.Series([f"bin_{i:02d}" for i in idx], index=x.index)
    return labels.where(x.notna(), MISSING)
