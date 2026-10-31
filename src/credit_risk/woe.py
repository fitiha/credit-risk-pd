"""Weight of Evidence (WoE) and Information Value (IV).

WoE of a group = ln(share of all repaid loans in the group / share of all defaulted loans in the group).
Positive means safer than average, negative means riskier. A small constant is added to the counts so a
group with no defaults or no repayments still gets a finite value.
IV = sum over groups of (share repaid - share defaulted) x WoE: how well the field separates the two outcomes.
Usual reading: under 0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, over 0.5 suspicious.
"""
import numpy as np
import pandas as pd

SMOOTHING = 0.5


def woe_table(groups, y, smoothing=SMOOTHING):
    """One row per group: loans, defaults, default rate, WoE and its share of the IV."""
    df = pd.DataFrame({"group": pd.Series(groups).astype(object).to_numpy(), "bad": np.asarray(y)})
    g = df.groupby("group", dropna=False, observed=True)["bad"].agg(loans="size", defaults="sum")
    g["repaid"] = g.loans - g.defaults
    good = (g.repaid + smoothing) / (g.repaid.sum() + smoothing * len(g))
    bad = (g.defaults + smoothing) / (g.defaults.sum() + smoothing * len(g))
    g["default_rate"] = g.defaults / g.loans
    g["woe"] = np.log(good / bad)
    g["iv_part"] = (good - bad) * g.woe
    return g.reset_index()


def information_value(table):
    return float(table.iv_part.sum())


def apply_woe(groups, table, unseen=0.0):
    """Map each group to its WoE; groups never seen when fitting get `unseen` (neutral by default)."""
    lookup = dict(zip(table.group, table.woe))
    return pd.Series(groups).map(lookup).fillna(unseen).astype(float).to_numpy()


def iv_on_new_data(table, groups, y, smoothing=SMOOTHING):
    """IV of a field on later loans, using WoE values learned earlier (honest, out-of-time IV)."""
    woe = apply_woe(groups, table)
    y = np.asarray(y)
    new = pd.DataFrame({"woe": woe, "bad": y})
    n_good, n_bad = (y == 0).sum(), (y == 1).sum()
    iv = 0.0
    for w, grp in new.groupby("woe"):
        good = (grp.bad == 0).sum() / n_good
        bad = (grp.bad == 1).sum() / n_bad
        iv += (good - bad) * w
    return float(iv)


def strength(iv):
    for limit, label in [(0.02, "useless"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong")]:
        if iv < limit:
            return label
    return "suspicious"
