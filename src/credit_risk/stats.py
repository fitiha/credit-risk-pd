"""Small statistics helpers shared by the analysis and the model code."""
import numpy as np
import pandas as pd


def wilson_interval(successes, n, z=1.96):
    """95% Wilson confidence interval for a rate. Stays inside 0-1 for small groups."""
    successes, n = np.asarray(successes, float), np.asarray(n, float)
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def rate_table(df, by, target="defaulted"):
    """Default rate with 95% CI for each group in `by` (column name or list)."""
    g = df.groupby(by, observed=True)[target].agg(loans="size", defaults="sum")
    g["rate"] = g.defaults / g.loans
    g["ci_low"], g["ci_high"] = wilson_interval(g.defaults, g.loans)
    return g.reset_index()
