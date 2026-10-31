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


PERCENTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def describe_field(x):
    """Distribution summary of one numeric column (missing values ignored)."""
    x = pd.Series(x).dropna()
    q = x.quantile(PERCENTILES)
    return {
        "n": len(x), "mean": x.mean(), "median": x.median(), "std": x.std(),
        "iqr": q[0.75] - q[0.25], "skew": x.skew(),
        **{f"p{int(p * 100)}": q[p] for p in PERCENTILES},
        "min": x.min(), "max": x.max(), "zeros_pct": 100 * (x == 0).mean(),
    }


def outlier_counts(x):
    """How many values three common rules flag as outliers (low, high)."""
    x = pd.Series(x).dropna()
    m, s = x.mean(), x.std()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    rules = {
        "mean_3std": (m - 3 * s, m + 3 * s),
        "iqr_fence": (q1 - 1.5 * iqr, q3 + 1.5 * iqr),
        "robust_z": (med - 3 * mad, med + 3 * mad),
    }
    out = {}
    for name, (lo, hi) in rules.items():
        out[f"{name}_low"] = int((x < lo).sum())
        out[f"{name}_high"] = int((x > hi).sum())
        out[f"{name}_upper_limit"] = hi
        out[f"{name}_lower_limit"] = lo
    return out


def cochran_armitage_trend(scores, defaults, totals):
    """Test whether the default rate rises (or falls) steadily across ordered groups. Returns (z, two-sided p)."""
    from scipy.stats import norm

    x, d, n = np.asarray(scores, float), np.asarray(defaults, float), np.asarray(totals, float)
    p = d.sum() / n.sum()
    xbar = (n * x).sum() / n.sum()
    z = ((d - n * p) * (x - xbar)).sum() / np.sqrt(p * (1 - p) * (n * (x - xbar) ** 2).sum())
    return float(z), float(2 * norm.sf(abs(z)))


def level_vs_rest(defaults, totals):
    """For each level of a segmentation: p-value of its default rate against all other loans (chi-square, Fisher if small)."""
    from scipy.stats import chi2_contingency, fisher_exact

    d, n = np.asarray(defaults, int), np.asarray(totals, int)
    out = []
    for i in range(len(n)):
        table = [[d[i], n[i] - d[i]], [d.sum() - d[i], (n.sum() - n[i]) - (d.sum() - d[i])]]
        small = min(min(r) for r in table) < 5
        out.append(fisher_exact(table)[1] if small else chi2_contingency(table, correction=False)[1])
    return np.array(out)
