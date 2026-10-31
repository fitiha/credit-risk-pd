"""Experiment E1: compare outlier treatments for the heavy-tailed money fields.

Every statistic (mean, std, percentiles, MAD) is computed on training data only.
Only loans issued up to the fit cutoff are used; 2015 is never touched.
Writes reports/a2_outlier_method_comparison/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs

AMOUNTS = ["annual_inc", "loan_amnt", "installment", "revol_bal", "tot_cur_bal",
           "avg_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim"]
RATIOS = ["dti", "revol_util", "bc_util"]
FIG, TAB = fresh_dirs("a2_outlier_method_comparison")


def clipper(lo, hi):
    return lambda x: x.clip(lo, hi)


def fit_method(name, tr):
    """Return a function that treats values using statistics learned from `tr`."""
    q = lambda p: tr.quantile(p)
    mad = 1.4826 * (tr - tr.median()).abs().median()
    iqr = q(.75) - q(.25)
    if name == "none":
        return lambda x: x
    if name == "clip_mean_3std":
        return clipper(tr.mean() - 3 * tr.std(), tr.mean() + 3 * tr.std())
    if name == "clip_iqr_fences":
        return clipper(q(.25) - 1.5 * iqr, q(.75) + 1.5 * iqr)
    if name == "clip_robust_z":
        return clipper(tr.median() - 3 * mad, tr.median() + 3 * mad)
    if name == "winsor_p1_p99":
        return clipper(q(.01), q(.99))
    if name == "winsor_p0.5_p99.5":
        return clipper(q(.005), q(.995))
    if name == "log1p":
        return lambda x: np.log1p(x)
    if name == "log1p_winsor_p0.5_p99.5":
        lo, hi = q(.005), q(.995)
        return lambda x: np.log1p(x.clip(lo, hi))
    if name == "log1p_clip_mean_3std":
        lt = np.log1p(tr)
        lo, hi = lt.mean() - 3 * lt.std(), lt.mean() + 3 * lt.std()
        return lambda x: np.log1p(x).clip(lo, hi)
    raise ValueError(name)


METHODS = ["none", "clip_mean_3std", "clip_iqr_fences", "clip_robust_z", "winsor_p1_p99",
           "winsor_p0.5_p99.5", "log1p", "log1p_winsor_p0.5_p99.5", "log1p_clip_mean_3std"]

raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
print(f"{len(fit):,} loans up to {FIT_CUTOFF.date()}")

# 1) shape and rows touched, fitted on the whole fit window
skew, touched = {}, {}
for m in METHODS:
    for c in AMOUNTS:
        tr = fit[c].dropna()
        out = fit_method(m, tr)(tr)
        skew[(c, m)] = out.skew()
        touched[(c, m)] = np.nan if m == "log1p" else float((out != tr).mean() if not m.startswith("log1p") else (np.log1p(tr) != out).mean())
skew = pd.Series(skew).unstack().loc[AMOUNTS, METHODS]
touched = pd.Series(touched).unstack().loc[AMOUNTS, METHODS] * 100
skew.round(2).to_csv(TAB / "skewness_after_treatment.csv", index_label="field")
touched.round(2).to_csv(TAB / "rows_changed_pct.csv", index_label="field")

# 2) usefulness: logistic regression on treated fields, time-ordered folds before 2015
folds = [("train 2013 -> test 2014 H1", fit.issue_d < "2014-01-01", fit.issue_d.between("2014-01-01", "2014-06-30")),
         ("train to 2014 H1 -> test 2014 H2", fit.issue_d <= "2014-06-30", fit.issue_d >= "2014-07-01")]


def design(method, tr, te):
    cols_tr, cols_te = [], []
    for c in AMOUNTS if method else []:
        f = fit_method(method, tr[c])
        cols_tr.append(f(tr[c]).rename(c)); cols_te.append(f(te[c]).rename(c))
    for c in RATIOS:
        f = fit_method("winsor_p0.5_p99.5", tr[c].dropna())
        med = f(tr[c].dropna()).median()
        cols_tr.append(f(tr[c]).fillna(med).rename(c)); cols_te.append(f(te[c]).fillna(med).rename(c))
    xtr, xte = pd.concat(cols_tr, axis=1), pd.concat(cols_te, axis=1)
    mu, sd = xtr.mean(), xtr.std().replace(0, 1)
    return (xtr - mu) / sd, (xte - mu) / sd


rng = np.random.default_rng(0)
rows = []
for label, mtr, mte in folds:
    tr, te = fit[mtr], fit[mte]
    for method in [None] + METHODS:
        xtr, xte = design(method, tr, te)
        p = LogisticRegression(max_iter=2000).fit(xtr, tr.defaulted).predict_proba(xte)[:, 1]
        y = te.defaulted.to_numpy()
        boot = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
        rows.append({"fold": label, "method": method or "(amount fields left out)",
                     "auc": roc_auc_score(y, p), "auc_ci_low": np.percentile(boot, 2.5), "auc_ci_high": np.percentile(boot, 97.5)})
auc = pd.DataFrame(rows)
summary = auc.groupby("method", sort=False)[["auc", "auc_ci_low", "auc_ci_high"]].mean().round(4)
auc.round(4).to_csv(TAB / "auc_by_method_and_fold.csv", index=False)
summary.to_csv(TAB / "auc_by_method.csv")

# charts
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(skew.abs().to_numpy(), cmap="viridis_r", aspect="auto", vmax=8)
ax.set_xticks(range(len(METHODS)), METHODS, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(AMOUNTS)), AMOUNTS, fontsize=8)
for i in range(len(AMOUNTS)):
    for j in range(len(METHODS)):
        ax.text(j, i, f"{skew.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white")
ax.set_title("Skewness after treatment (closer to 0 is more symmetric; colour capped at 8)", fontsize=10)
fig.colorbar(im, label="|skewness|")
fig.tight_layout(); fig.savefig(FIG / "skewness_by_method_and_field.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
s = summary.iloc[::-1]
ax.errorbar(s.auc, range(len(s)), xerr=[s.auc - s.auc_ci_low, s.auc_ci_high - s.auc], fmt="o", color="#2a6f97", capsize=3)
ax.set_yticks(range(len(s)), s.index, fontsize=8)
ax.set_xlabel("AUC on later, unseen loans (average of two folds; bars = 95% interval)")
ax.set_title("Predictive power of a simple model by outlier treatment", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "auc_by_method.png", dpi=150); plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 20)
print("\nSKEWNESS after treatment:\n", skew.round(1).to_string())
print("\n% ROWS CHANGED:\n", touched.round(1).to_string())
print("\nAUC (mean of 2 folds):\n", summary.to_string())
