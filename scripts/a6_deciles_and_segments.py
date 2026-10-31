"""A7, part 2: default rate by factor-score decile and by key segment, with 95% intervals and significance tests.

A segment is called a real difference only if it has at least 300 loans and its test survives Holm correction
for the number of levels compared. Uses loans up to the fit cutoff. Writes reports/a6_deciles_and_segments/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import cochran_armitage_trend, level_vs_rest, wilson_interval
from credit_risk.transform import ApplicationTransformer

FIG, TAB = fresh_dirs("a6_deciles_and_segments")
MIN_LOANS = 300
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
y = fit.defaulted
base = y.mean()
FACTORS = [c for c in fit.columns if c.startswith("factor_")]

# factor-score deciles
dec_rows, test_rows = [], []
for c in FACTORS:
    g = pd.qcut(fit[c].rank(method="first"), 10, labels=range(1, 11))
    t = y.groupby(g, observed=True).agg(loans="size", defaults="sum").reset_index().rename(columns={"index": "decile"})
    t.columns = ["decile", "loans", "defaults"]
    t["rate"] = t.defaults / t.loans
    t["ci_low"], t["ci_high"] = wilson_interval(t.defaults, t.loans)
    t.insert(0, "factor", c.replace("factor_", ""))
    dec_rows.append(t)
    chi_p = chi2_contingency(np.array([t.defaults, t.loans - t.defaults]), correction=False)[1]
    z, trend_p = cochran_armitage_trend(t.decile.astype(float), t.defaults, t.loans)
    mw_p = mannwhitneyu(fit[c][y == 1], fit[c][y == 0]).pvalue
    a = roc_auc_score(y, fit[c])
    test_rows.append({"factor": c.replace("factor_", ""), "default_rate_decile_1": t.rate.iloc[0], "default_rate_decile_10": t.rate.iloc[-1],
                      "chi_square_p": chi_p, "trend_z": z, "trend_p": trend_p, "mann_whitney_p": mw_p, "auc_alone": max(a, 1 - a)})
dec = pd.concat(dec_rows)
tests = pd.DataFrame(test_rows)
tests["holm_adjusted_trend_p"] = multipletests(tests.trend_p, method="holm")[1]
tests["real_trend"] = tests.holm_adjusted_trend_p < 0.05
dec.round(4).to_csv(TAB / "default_rate_by_factor_decile.csv", index=False)
tests.to_csv(TAB / "factor_decile_tests.csv", index=False, float_format="%.4g")

# segments
fit["emp_length_band"] = pd.cut(fit.emp_length_years, [-1, 1, 9, 10], labels=["under 2 years", "2 to 9 years", "10+ years"]).astype(object).fillna("missing")
fit["income_band"] = pd.qcut(fit.annual_inc, 5, labels=["income Q1 (lowest)", "income Q2", "income Q3", "income Q4", "income Q5 (highest)"]).astype(str)
fit["loan_band"] = pd.qcut(fit.loan_amnt, 5, labels=["loan Q1 (smallest)", "loan Q2", "loan Q3", "loan Q4", "loan Q5 (largest)"]).astype(str)
fit["year"] = fit.issue_d.dt.year.astype(str)
SEGMENTS = ["term", "grade", "purpose_group", "home_ownership_group", "verification_status", "emp_length_band", "income_band", "loan_band", "year", "job_group", "addr_state_group"]
seg_rows, seg_tests = [], []
for s in SEGMENTS:
    t = y.groupby(fit[s]).agg(loans="size", defaults="sum").reset_index()
    t.columns = ["level", "loans", "defaults"]
    t["rate"] = t.defaults / t.loans
    t["ci_low"], t["ci_high"] = wilson_interval(t.defaults, t.loans)
    t["p_vs_rest"] = level_vs_rest(t.defaults, t.loans)
    t["holm_p"] = multipletests(t.p_vs_rest, method="holm")[1]
    t["enough_loans"] = t.loans >= MIN_LOANS
    t["verdict"] = np.where(~t.enough_loans, "too few loans to say",
                            np.where(t.holm_p < 0.05, np.where(t.rate > base, "real: riskier than the rest", "real: safer than the rest"), "no clear difference"))
    t.insert(0, "segment", s)
    seg_rows.append(t)
    big = t[t.enough_loans]
    chi = chi2_contingency(np.array([big.defaults, big.loans - big.defaults]), correction=False)
    seg_tests.append({"segment": s, "levels": len(t), "levels_with_enough_loans": int(t.enough_loans.sum()), "chi_square_p_across_levels": chi[1],
                      "highest_rate": big.rate.max(), "lowest_rate": big.rate.min()})
seg = pd.concat(seg_rows)
seg.round(4).to_csv(TAB / "default_rate_by_segment.csv", index=False)
pd.DataFrame(seg_tests).to_csv(TAB / "segment_overall_tests.csv", index=False, float_format="%.4g")

fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True)
for ax, c in zip(axes.ravel(), FACTORS):
    t = dec[dec.factor == c.replace("factor_", "")]
    ax.errorbar(t.decile, t.rate * 100, yerr=[(t.rate - t.ci_low) * 100, (t.ci_high - t.rate) * 100], fmt="o-", color="#2a6f97", capsize=2, ms=4)
    ax.axhline(base * 100, color="grey", ls="--", lw=1); ax.set_title(c.replace("factor_", ""), fontsize=9); ax.set_xlabel("decile of factor score", fontsize=8)
for r in range(2):
    axes[r, 0].set_ylabel("Default rate (%)")
fig.suptitle("Default rate by factor-score decile (bars = 95% interval, dashed = overall rate)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_by_factor_decile.png", dpi=130); plt.close(fig)

plot = seg[seg.segment.isin(["term", "grade", "purpose_group", "verification_status", "emp_length_band", "income_band", "loan_band", "home_ownership_group"]) & seg.enough_loans].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 9))
yy = np.arange(len(plot))[::-1]
colour = np.where(plot.verdict.str.startswith("real: riskier"), "#c1440e", np.where(plot.verdict.str.startswith("real: safer"), "#2a6f97", "#888888"))
for i in range(len(plot)):
    ax.errorbar(plot.rate[i] * 100, yy[i], xerr=[[(plot.rate[i] - plot.ci_low[i]) * 100], [(plot.ci_high[i] - plot.rate[i]) * 100]], fmt="o", color=colour[i], capsize=2, ms=4)
ax.set_yticks(yy, [f"{a}: {b} (n={n:,})" for a, b, n in zip(plot.segment, plot.level, plot.loans)], fontsize=7)
ax.axvline(base * 100, color="grey", ls="--", lw=1)
ax.set_xlabel("Default rate (%). Red = real and riskier, blue = real and safer, grey = no clear difference (Holm-corrected)")
ax.set_title("Default rate by segment (segments with at least 300 loans)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_by_segment.png", dpi=130); plt.close(fig)

pd.set_option("display.width", 230, "display.max_columns", 20)
print(tests.round(4).to_string(index=False))
print(pd.DataFrame(seg_tests).round(4).to_string(index=False))
print(seg[(seg.verdict.str.startswith("real")) | (~seg.enough_loans)][["segment", "level", "loans", "rate", "holm_p", "verdict"]].round(4).to_string(index=False))
