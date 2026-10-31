"""Experiment E2: how to handle missing ratio values (revol_util, bc_util).

Part 1: is a missing value related to default?
Part 2: compare fill rules by how well a simple model predicts later loans.
Uses loans up to the fit cutoff only. Writes reports/a2_missing_amounts_experiment/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.outliers import AMOUNT_FIELDS, AmountTreater
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval

FIG, TAB = fresh_dirs("a2_missing_amounts_experiment")
RATIOS = ["dti", "revol_util", "bc_util"]

raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)

# Part 1: missingness vs default
rows = []
groups = {
    "revol_util missing": fit.revol_util.isna(),
    "bc_util missing in the bureau file": fit.bc_util.isna() & ~fit.bc_util_invalid,
    "bc_util set to missing (negative sign)": fit.bc_util_invalid,
}
base = fit.defaulted.mean()
for name, m in groups.items():
    n, k = int(m.sum()), int(fit.defaulted[m].sum())
    lo, hi = wilson_interval(k, n)
    p = fisher_exact([[k, n - k], [fit.defaulted[~m].sum(), (~m).sum() - fit.defaulted[~m].sum()]])[1]
    rows.append({"group": name, "loans": n, "default_rate": k / n, "ci_low": lo, "ci_high": hi,
                 "default_rate_when_present": fit.defaulted[~m].mean(), "p_value_fisher": p})
miss = pd.DataFrame(rows)
miss.round(4).to_csv(TAB / "missingness_vs_default.csv", index=False)

# Why is revol_util missing? (ratio is undefined when there is no revolving limit)
rv = fit[fit.revol_util.isna()]
why = pd.Series({"missing revol_util (fit window)": len(rv),
                 "  of which total_rev_hi_lim = 0": int((rv.total_rev_hi_lim == 0).sum()),
                 "  of which revol_bal = 0": int((rv.revol_bal == 0).sum())}, name="loans")
why.to_csv(TAB / "revol_util_missing_reasons.csv", index_label="item")

# Part 2: fill rules
folds = [("train 2013 -> test 2014 H1", fit.issue_d < "2014-01-01", fit.issue_d.between("2014-01-01", "2014-06-30")),
         ("train to 2014 H1 -> test 2014 H2", fit.issue_d <= "2014-06-30", fit.issue_d >= "2014-07-01")]
OPTIONS = ["median", "median_plus_missing_flag", "bc_util_from_revol_util_plus_flag"]


def cap(tr, te, c):
    lo, hi = tr[c].quantile(0.005), tr[c].quantile(0.995)
    return tr[c].clip(lo, hi), te[c].clip(lo, hi)


def design(option, tr, te):
    treater = AmountTreater().fit(tr)
    xtr = treater.transform(tr)[[f"{f}_log" for f in AMOUNT_FIELDS]].copy()
    xte = treater.transform(te)[[f"{f}_log" for f in AMOUNT_FIELDS]].copy()
    ctr, cte = {}, {}
    for c in RATIOS:
        ctr[c], cte[c] = cap(tr, te, c)
    for c in RATIOS:
        med = ctr[c].median()
        if option == "bc_util_from_revol_util_plus_flag" and c == "bc_util":
            ok = ctr["bc_util"].notna() & ctr["revol_util"].notna()
            reg = LinearRegression().fit(ctr["revol_util"][ok].to_frame(), ctr["bc_util"][ok])
            fill_tr = pd.Series(reg.predict(ctr["revol_util"].fillna(ctr["revol_util"].median()).to_frame()), index=tr.index)
            fill_te = pd.Series(reg.predict(cte["revol_util"].fillna(ctr["revol_util"].median()).to_frame()), index=te.index)
        else:
            fill_tr = fill_te = med
        if option != "median" and c in ("revol_util", "bc_util"):
            xtr[f"{c}_missing"] = ctr[c].isna().astype(int); xte[f"{c}_missing"] = cte[c].isna().astype(int)
        xtr[c] = ctr[c].fillna(fill_tr); xte[c] = cte[c].fillna(fill_te)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return (xtr - mu) / sd, (xte - mu) / sd


rng = np.random.default_rng(0)
out = []
for label, mtr, mte in folds:
    tr, te = fit[mtr], fit[mte]
    y = te.defaulted.to_numpy()
    for opt in OPTIONS:
        xtr, xte = design(opt, tr, te)
        p = LogisticRegression(max_iter=2000).fit(xtr, tr.defaulted).predict_proba(xte)[:, 1]
        boot = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
        out.append({"fold": label, "fill_rule": opt, "auc": roc_auc_score(y, p),
                    "auc_ci_low": np.percentile(boot, 2.5), "auc_ci_high": np.percentile(boot, 97.5)})
auc = pd.DataFrame(out)
summary = auc.groupby("fill_rule", sort=False)[["auc", "auc_ci_low", "auc_ci_high"]].mean()
summary.round(4).to_csv(TAB / "auc_by_fill_rule.csv")

fig, ax = plt.subplots(figsize=(7, 3.8))
y_pos = range(len(miss))
ax.errorbar(miss.default_rate * 100, y_pos, xerr=[(miss.default_rate - miss.ci_low) * 100, (miss.ci_high - miss.default_rate) * 100],
            fmt="o", color="#c1440e", capsize=3)
ax.axvline(base * 100, color="grey", ls="--", lw=1)
ax.set_yticks(list(y_pos), [f"{g}\n(n={n})" for g, n in zip(miss.group, miss.loans)], fontsize=8)
ax.set_xlabel("Default rate (%)  (dashed = overall rate; bars = 95% interval)")
ax.set_title("Is a missing ratio linked to default?", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_when_ratio_missing.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 220)
print(miss.round(3).to_string(index=False)); print("\n", why.to_string()); print("\nAUC (mean of 2 folds):\n", summary.round(4).to_string())
