"""Population Stability Index: how much a field's distribution moved between two groups of loans.

PSI = sum over bins of (share_new - share_old) x ln(share_new / share_old), with bins set on the old group.
Usual reading: under 0.1 stable, 0.1 to 0.25 moderate shift, 0.25 and above major shift.
"""
import numpy as np
import pandas as pd

FLOOR = 1e-4


def psi_from_shares(old, new):
    old, new = np.clip(np.asarray(old, float), FLOOR, None), np.clip(np.asarray(new, float), FLOOR, None)
    return float(((new - old) * np.log(new / old)).sum())


def numeric_psi(old, new, n_bins=10):
    """PSI of a numeric field, with bins at the deciles of the old values (missing values are their own bin)."""
    old, new = pd.Series(old), pd.Series(new)
    edges = np.unique(np.quantile(old.dropna(), np.linspace(0, 1, n_bins + 1)[1:-1]))
    def shares(s):
        idx = pd.Series(np.searchsorted(edges, s.to_numpy(float), side="right"), index=s.index).where(s.notna(), -1)
        return idx.value_counts(normalize=True)
    a, b = shares(old), shares(new)
    levels = a.index.union(b.index)
    return psi_from_shares(a.reindex(levels, fill_value=0), b.reindex(levels, fill_value=0))


def categorical_psi(old, new):
    a, b = pd.Series(old).fillna("missing").value_counts(normalize=True), pd.Series(new).fillna("missing").value_counts(normalize=True)
    levels = a.index.union(b.index)
    return psi_from_shares(a.reindex(levels, fill_value=0), b.reindex(levels, fill_value=0))


def label(psi):
    return "stable" if psi < 0.1 else "moderate shift" if psi < 0.25 else "major shift"
