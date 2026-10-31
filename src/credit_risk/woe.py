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


from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.binning import assign_bins, fit_edges
from credit_risk.io import FIT_CUTOFF


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Adds `<field>_woe` columns. Bins and WoE values are learned on loans up to the fit cutoff only.

    numeric fields are binned with monotonic merge (missing gets its own bin); categorical fields use
    their levels as groups. Values never seen when fitting get WoE 0 (neutral).
    """

    def __init__(self, numeric=None, categorical=None, target="defaulted", min_share=0.05):
        self.numeric = numeric
        self.categorical = categorical
        self.target = target
        self.min_share = min_share

    def fit(self, X, y=None):
        pre = X[X["issue_d"] <= FIT_CUTOFF]
        y = pre[self.target] if y is None else pd.Series(y, index=X.index).loc[pre.index]
        self.edges_, self.tables_ = {}, {}
        for f in self.numeric or []:
            self.edges_[f] = fit_edges(pre[f], y, "monotonic", min_share=self.min_share)
            self.tables_[f] = woe_table(assign_bins(pre[f], self.edges_[f]), y)
        for f in self.categorical or []:
            self.tables_[f] = woe_table(pre[f].fillna("missing"), y)
        return self

    def _groups(self, X, f):
        if f in self.edges_:
            return assign_bins(X[f], self.edges_[f])
        return X[f].fillna("missing")

    def transform(self, X):
        d = X.copy()
        for f, t in self.tables_.items():
            d[f"{f}_woe"] = apply_woe(self._groups(d, f), t)
        return d

    def iv_table(self):
        rows = [{"field": f, "kind": "numeric" if f in self.edges_ else "categorical", "bins_or_groups": len(t),
                 "iv": information_value(t), "strength": strength(information_value(t))} for f, t in self.tables_.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def spec_entries(self):
        out = {}
        for f, t in self.tables_.items():
            out[f] = {"method": "monotonic_merge" if f in self.edges_ else "category_levels",
                      "edges": [float(e) for e in self.edges_.get(f, [])],
                      "bins": [{"label": str(r.group), "loans": int(r.loans), "default_rate": round(float(r.default_rate), 4),
                                "woe": round(float(r.woe), 4)} for r in t.itertuples()],
                      "iv": round(information_value(t), 4), "strength": strength(information_value(t)),
                      "woe_convention": "ln(share repaid / share defaulted); positive = safer",
                      "unseen_value_woe": 0.0}
        return out
