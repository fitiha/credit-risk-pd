"""A3: distribution profile of every financial field on loans up to the fit cutoff.

Writes reports/financial_values_profile/tables/{field_profile,outlier_rule_counts}.csv
and one chart per field (linear and log scale) in .../figures/distribution_<field>.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import describe_field, outlier_counts

AMOUNTS = ["annual_inc", "loan_amnt", "installment", "revol_bal", "tot_cur_bal",
           "avg_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim"]
RATIOS = ["dti", "revol_util", "bc_util"]
COUNTS = ["delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "inq_last_6mths", "mort_acc"]
FIG, TAB = fresh_dirs("financial_values_profile")

raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF]
print(f"{len(fit):,} loans issued up to {FIT_CUTOFF.date()}")

profile = pd.DataFrame({c: describe_field(fit[c]) for c in AMOUNTS + RATIOS + COUNTS}).T
profile.insert(0, "group", ["amount"] * len(AMOUNTS) + ["ratio"] * len(RATIOS) + ["count"] * len(COUNTS))
profile.to_csv(TAB / "field_profile.csv", index_label="field")

rules = pd.DataFrame({c: outlier_counts(fit[c]) for c in AMOUNTS + RATIOS + COUNTS}).T
rules.to_csv(TAB / "outlier_rule_counts.csv", index_label="field")

for c in AMOUNTS + RATIOS:
    x = fit[c].dropna()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(x, bins=60, color="#2a6f97")
    ax[0].set_title(f"{c}: linear scale")
    ax[0].set_xlabel(c); ax[0].set_ylabel("loans")
    ax[1].hist(np.log10(x + 1), bins=60, color="#c1440e")
    ax[1].set_title(f"{c}: log10(value + 1)")
    ax[1].set_xlabel(f"log10({c} + 1)"); ax[1].set_ylabel("loans")
    for a in ax:
        a.axvline(x.mean() if a is ax[0] else np.log10(x.mean() + 1), color="black", ls="--", lw=1)
    ax[0].text(0.97, 0.95, "dashed = mean", transform=ax[0].transAxes, ha="right", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"distribution_{c}.png", dpi=110)
    plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 30)
show = ["n", "mean", "median", "std", "iqr", "skew", "p1", "p5", "p25", "p75", "p95", "p99", "max"]
print(profile[show].astype(float).round(1).to_string())
cols = ["mean_3std_low", "mean_3std_high", "iqr_fence_low", "iqr_fence_high", "robust_z_low", "robust_z_high"]
print("\nvalues flagged as outliers by each rule (low / high):")
print(rules[cols].to_string())
print("\nmean+3std upper limit vs 99th percentile vs max:")
print(pd.DataFrame({"mean+3std": rules.mean_3std_upper_limit, "p99": profile.p99, "max": profile["max"],
                    "mean-3std": rules.mean_3std_lower_limit}).astype(float).round(0).to_string())
