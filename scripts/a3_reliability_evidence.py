"""A4 evidence: income vs verification, income consistency checks, and missingness.

Uses cleaned loans up to the fit cutoff. Writes reports/a3_reliability_evidence/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications, load_servicing
from sklearn.metrics import roc_auc_score
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval

FIG, TAB = fresh_dirs("a3_reliability_evidence")
raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
GROUPS = ["Not Verified", "Source Verified", "Verified"]

# 1. income by verification status
fit["income_to_credit_limit"] = fit.tot_hi_cred_lim / fit.annual_inc
rows = []
for g in GROUPS:
    s = fit[fit.verification_status == g]
    ok = s[["annual_inc", "tot_hi_cred_lim"]].notna().all(axis=1)
    rows.append({"verification_status": g, "loans": len(s), "median_income": s.annual_inc.median(),
                 "p10_income": s.annual_inc.quantile(.10), "p90_income": s.annual_inc.quantile(.90),
                 "pct_typed_in_thousands": 100 * s.annual_inc_fixed.mean(),
                 "pct_income_above_200k": 100 * (s.annual_inc > 2e5).mean(),
                 "spearman_income_vs_credit_limit": spearmanr(s.annual_inc[ok], s.tot_hi_cred_lim[ok])[0],
                 "default_rate": s.defaulted.mean()})
inc = pd.DataFrame(rows)
kw = kruskal(*[fit.annual_inc[fit.verification_status == g] for g in GROUPS])
inc["kruskal_p_value_income_differs"] = kw.pvalue
inc.round(4).to_csv(TAB / "income_by_verification.csv", index=False)

# 2. consistency of installment and dti with income
fit["pmt_to_income"] = fit.installment * 12 / fit.annual_inc
fit["revol_bal_to_income"] = fit.revol_bal / fit.annual_inc
rows = []
for g in GROUPS:
    s = fit[fit.verification_status == g]
    ok = s[["dti", "revol_bal_to_income"]].notna().all(axis=1)
    rows.append({"verification_status": g,
                 "median_installment_to_income_pct": 100 * s.pmt_to_income.median(),
                 "pct_installment_above_40pct_of_income": 100 * (s.pmt_to_income > .4).mean(),
                 "pct_installment_above_100pct_of_income": 100 * (s.pmt_to_income > 1).mean(),
                 "spearman_dti_vs_revolving_balance_to_income": spearmanr(s.dti[ok], s.revol_bal_to_income[ok])[0],
                 "pct_dti_under_5_with_balance_over_half_income": 100 * ((s.dti < 5) & (s.revol_bal_to_income > .5)).mean()})
cons = pd.DataFrame(rows)
cons.round(4).to_csv(TAB / "consistency_with_income.csv", index=False)

# 3. missingness: related to default? random or structural?
fit["title_missing"] = fit.title.isna()
fit["bc_util_gap"] = fit.bc_util.isna() & ~fit.bc_util_invalid
FIELDS = {"emp_title": fit.emp_title.isna(), "emp_length": fit.emp_length.isna(), "title": fit.title.isna(),
          "mths_since_last_delinq": fit.mths_since_last_delinq.isna(), "bc_util": fit.bc_util_gap,
          "revol_util": fit.revol_util.isna(), "earliest_cr_line (corrupted date)": fit.earliest_cr_line_invalid}
num = fit[["annual_inc", "loan_amnt", "fico_range_low", "total_acc", "open_acc", "delinq_2yrs", "pub_rec",
           "inq_last_6mths", "revol_bal", "tot_cur_bal", "dti", "int_rate", "term"]].copy()
num[["annual_inc", "loan_amnt", "revol_bal", "tot_cur_bal"]] = np.log1p(num[["annual_inc", "loan_amnt", "revol_bal", "tot_cur_bal"]])
num = num.fillna(num.median())
cat = pd.get_dummies(fit[["verification_status", "home_ownership", "purpose", "grade"]], dtype=float)
X = pd.concat([num, cat, (fit.issue_d.dt.year - 2013).rename("year")], axis=1)

dr, mech = [], []
for name, m in FIELDS.items():
    n, k = int(m.sum()), int(fit.defaulted[m].sum())
    lo, hi = wilson_interval(k, n)
    other = fit.defaulted[~m]
    p = fisher_exact([[k, n - k], [other.sum(), len(other) - other.sum()]])[1]
    dr.append({"field": name, "loans_missing": n, "pct_missing": 100 * n / len(fit), "default_rate_when_missing": k / n,
               "ci_low": lo, "ci_high": hi, "default_rate_when_present": other.mean(), "p_value_fisher": p})
    if n >= 100:
        auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, m.astype(int), cv=5, scoring="roc_auc").mean()
        by_year = m.groupby(fit.issue_d.dt.year).mean() * 100
        mech.append({"field": name, "auc_predicting_missingness_from_other_fields": auc,
                     "pct_missing_2013": by_year.get(2013, np.nan), "pct_missing_2014": by_year.get(2014, np.nan)})
drivers = []
for name, m in FIELDS.items():
    if int(m.sum()) >= 100:
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, m.astype(int))
        coef = pd.Series(lr[-1].coef_[0], index=X.columns)
        top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(5)
        drivers += [{"field": name, "rank": i + 1, "predictor": k, "standardised_coefficient": v} for i, (k, v) in enumerate(top.items())]
pd.DataFrame(drivers).round(3).to_csv(TAB / "missingness_top_predictors.csv", index=False)
dr, mech = pd.DataFrame(dr), pd.DataFrame(mech)
dr.round(4).to_csv(TAB / "missingness_vs_default.csv", index=False)
mech.round(3).to_csv(TAB / "missingness_mechanism.csv", index=False)
overlap = pd.Series({"emp_title missing and emp_length missing": int((FIELDS["emp_title"] & FIELDS["emp_length"]).sum()),
                     "emp_length missing": int(FIELDS["emp_length"].sum()), "emp_title missing": int(FIELDS["emp_title"].sum())}, name="loans")
overlap.to_csv(TAB / "emp_fields_overlap.csv", index_label="item")
ms = fit[fit.mths_since_last_delinq.isna()]
print("mths_since_last_delinq missing: share with delinq_2yrs=0:", round((ms.delinq_2yrs == 0).mean(), 3), "| share with pub_rec=0:", round((ms.pub_rec == 0).mean(), 3),
      "| present-group share with delinq_2yrs=0:", round((fit.mths_since_last_delinq.notna() & (fit.delinq_2yrs == 0)).sum() / fit.mths_since_last_delinq.notna().sum(), 3))

# 4. rule checks behind the "high" ratings (all loans)
checks = {
    "open_acc above total_acc": d.open_acc > d.total_acc,
    "avg_cur_bal above tot_cur_bal": d.avg_cur_bal > d.tot_cur_bal + 1,
    "total_rev_hi_lim above tot_hi_cred_lim": d.total_rev_hi_lim > d.tot_hi_cred_lim,
    "pub_rec_bankruptcies above pub_rec": d.pub_rec_bankruptcies > d.pub_rec,
    "acc_open_past_24mths above total_acc": d.acc_open_past_24mths > d.total_acc,
    "num_tl_op_past_12m above acc_open_past_24mths": d.num_tl_op_past_12m > d.acc_open_past_24mths,
    "FICO outside 300-850": (d.fico_range_low < 300) | (d.fico_range_high > 850),
    "FICO range width not 4": (d.fico_range_high - d.fico_range_low) != 4,
    "negative value in a bureau count": (d[["inq_last_6mths", "delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "open_acc", "total_acc",
                                            "mort_acc", "acc_open_past_24mths", "num_tl_op_past_12m"]] < 0).any(axis=1),
    "non-whole bureau count": (d[["inq_last_6mths", "delinq_2yrs", "pub_rec", "open_acc", "total_acc", "mort_acc"]] % 1 != 0).any(axis=1),
    "installment above 40% of income": d.installment * 12 / d.annual_inc > .4,
    "revolving balance above 1.5x the total current balance (after repair)": d.revol_bal > 1.5 * d.tot_cur_bal,
}
pd.Series({k: int(v.sum()) for k, v in checks.items()}, name="loans_violating").to_csv(TAB / "bureau_consistency_checks.csv", index_label="check")

# 5. leakage scan: how well does ONE field alone predict default? (servicing used for audit only)
srv = load_servicing().set_index("id")
inputs = ["loan_amnt", "annual_inc", "dti", "int_rate", "fico_range_low", "revol_util", "delinq_2yrs", "inq_last_6mths", "open_acc", "tot_hi_cred_lim"]
scan = []
for c in inputs:
    x = fit[c].fillna(fit[c].median())
    a = roc_auc_score(fit.defaulted, x); scan.append({"field": c, "known_at_application": True, "single_field_auc": max(a, 1 - a)})
sv = srv.reindex(fit.id)
for c in ["total_rec_prncp", "total_pymnt", "recoveries", "last_fico_range_high", "last_pymnt_amnt"]:
    x = sv[c].fillna(sv[c].median()).to_numpy()
    a = roc_auc_score(fit.defaulted, x); scan.append({"field": c, "known_at_application": False, "single_field_auc": max(a, 1 - a)})
flag = raw.set_index("id").bureau_recovery_flag.reindex(fit.id).to_numpy()
scan.append({"field": "bureau_recovery_flag", "known_at_application": False, "single_field_auc": roc_auc_score(fit.defaulted, flag)})
scan = pd.DataFrame(scan)
scan.round(3).to_csv(TAB / "leakage_scan_single_field_auc.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4))
y = range(len(dr))
ax.errorbar(dr.default_rate_when_missing * 100, y, xerr=[(dr.default_rate_when_missing - dr.ci_low) * 100, (dr.ci_high - dr.default_rate_when_missing) * 100],
            fmt="o", color="#c1440e", capsize=3, label="when the value is missing")
ax.plot(dr.default_rate_when_present * 100, y, "s", color="#2a6f97", label="when the value is present")
ax.set_yticks(list(y), [f"{f} (n={n:,})" for f, n in zip(dr.field, dr.loans_missing)], fontsize=8)
ax.set_xlabel("Default rate (%) (bars = 95% interval)")
ax.set_title("Is a missing value linked to default?", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "default_rate_when_missing.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
data = [np.log10(fit.annual_inc[fit.verification_status == g]) for g in GROUPS]
ax.boxplot(data, tick_labels=[f"{g}\n(n={len(x):,})" for g, x in zip(GROUPS, data)], showfliers=False)
ax.set_ylabel("log10(annual income)"); ax.set_title("Stated income by verification status", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "income_by_verification.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 20)
print("\nINCOME BY VERIFICATION\n", inc.round(3).T.to_string(header=False))
print("\nCONSISTENCY WITH INCOME\n", cons.round(3).T.to_string(header=False))
print("\nMISSINGNESS vs DEFAULT\n", dr.round(3).to_string(index=False))
print("\nMISSINGNESS MECHANISM (AUC 0.5 = unpredictable from other fields)\n", mech.round(3).to_string(index=False))
print("\n", overlap.to_string())
print("\nLEAKAGE SCAN (single-field AUC)\n", scan.round(3).to_string(index=False))
print("\nTOP PREDICTORS OF MISSINGNESS\n", pd.DataFrame(drivers).round(2).to_string(index=False))
