"""A7, part 3: what do the factors add compared with the platform's own grade?

(1) Predictive power on later loans (time-ordered folds before 2015): grade alone, factors alone, grade + factors.
(2) Within the same grade band, do the factors still separate good from bad loans?
Factor scores are fitted without the label, so using them across folds leaks nothing about default.
Writes reports/a6_factors_vs_grade/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval
from credit_risk.transform import ApplicationTransformer

FIG, TAB = fresh_dirs("a6_factors_vs_grade")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
FACTORS = [c for c in fit.columns if c.startswith("factor_")]
fit["sub_grade_rank"] = fit.sub_grade.map({g: i for i, g in enumerate(sorted(fit.sub_grade.unique()))})
folds = [(fit.issue_d < "2014-01-01", fit.issue_d.between("2014-01-01", "2014-06-30")), (fit.issue_d <= "2014-06-30", fit.issue_d >= "2014-07-01")]
SETS = {"sub-grade alone": ["sub_grade_rank"], "factors alone": FACTORS, "sub-grade + factors": ["sub_grade_rank"] + FACTORS}

rng = np.random.default_rng(0)
rows, preds = [], {}
for name, cols in SETS.items():
    aucs, los, his = [], [], []
    for k, (mtr, mte) in enumerate(folds):
        tr, te = fit[mtr], fit[mte]
        sc = StandardScaler().fit(tr[cols])
        m = LogisticRegression(max_iter=3000).fit(sc.transform(tr[cols]), tr.defaulted)
        p = m.predict_proba(sc.transform(te[cols]))[:, 1]
        preds[(name, k)] = (te.index.to_numpy(), p, m, sc, cols)
        yv = te.defaulted.to_numpy()
        boot = [roc_auc_score(yv[i], p[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(200))]
        aucs.append(roc_auc_score(yv, p)); los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
    rows.append({"inputs": name, "auc": np.mean(aucs), "auc_ci_low": np.mean(los), "auc_ci_high": np.mean(his)})
auc = pd.DataFrame(rows)
auc["gain_over_sub_grade"] = auc.auc - auc.auc[0]
auc.round(4).to_csv(TAB / "auc_grade_vs_factors.csv", index=False)

# paired check: does adding the factors improve the AUC on the same loans?
gain = []
for k, (mtr, mte) in enumerate(folds):
    idx, p_g, *_ = preds[("sub-grade alone", k)]
    _, p_gf, *_ = preds[("sub-grade + factors", k)]
    yv = fit.loc[idx, "defaulted"].to_numpy()
    diffs = [roc_auc_score(yv[i], p_gf[i]) - roc_auc_score(yv[i], p_g[i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(300))]
    gain.append({"fold": f"fold {k + 1}", "auc_gain": roc_auc_score(yv, p_gf) - roc_auc_score(yv, p_g), "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5)})
pd.DataFrame(gain).round(4).to_csv(TAB / "paired_gain_from_factors.csv", index=False)

# within grade: the factor-based risk score from the earlier fold, applied to the later fold
idx, p_f, *_ = preds[("factors alone", 1)]
te = fit.loc[idx].assign(factor_risk=p_f)
te["grade_band"] = te.grade.map({"A": "A-B", "B": "A-B", "C": "C", "D": "D-G", "E": "D-G", "F": "D-G", "G": "D-G"})
wrows = []
for band, g in te.groupby("grade_band"):
    lo_cut, hi_cut = g.factor_risk.quantile([1 / 3, 2 / 3])
    low, high = g[g.factor_risk <= lo_cut], g[g.factor_risk > hi_cut]
    dl, dh = int(low.defaulted.sum()), int(high.defaulted.sum())
    p = chi2_contingency(np.array([[dl, len(low) - dl], [dh, len(high) - dh]]), correction=False)[1]
    l1, h1 = wilson_interval(dl, len(low)); l2, h2 = wilson_interval(dh, len(high))
    wrows.append({"grade_band": band, "loans_per_group": len(low), "rate_lowest_third": dl / len(low), "ci_low_1": l1, "ci_high_1": h1,
                  "rate_highest_third": dh / len(high), "ci_low_2": l2, "ci_high_2": h2, "p_value": p})
within = pd.DataFrame(wrows)
within.round(4).to_csv(TAB / "within_grade_separation.csv", index=False)

corr = pd.DataFrame({"factor": [c.replace("factor_", "") for c in FACTORS], "spearman_with_sub_grade": [spearmanr(fit[c], fit.sub_grade_rank)[0] for c in FACTORS]})
corr.round(3).to_csv(TAB / "factor_correlation_with_grade.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(within))
for off, col, lab, cl, ch in [(-0.17, "rate_lowest_third", "lowest third of factor risk", "ci_low_1", "ci_high_1"), (0.17, "rate_highest_third", "highest third", "ci_low_2", "ci_high_2")]:
    ax.errorbar(x + off, within[col] * 100, yerr=[(within[col] - within[cl]) * 100, (within[ch] - within[col]) * 100], fmt="o", capsize=3, label=lab, color="#2a6f97" if off < 0 else "#c1440e")
ax.set_xticks(x, [f"grade {b}" for b in within.grade_band]); ax.set_ylabel("Default rate (%) on 2014 H2 loans")
ax.set_title("Within the same grade, factor-based risk still separates loans (bars = 95% interval)", fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "within_grade_separation.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 220)
print(auc.round(4).to_string(index=False)); print(pd.DataFrame(gain).round(4).to_string(index=False)); print(within.round(3).to_string(index=False)); print(corr.round(2).to_string(index=False))
