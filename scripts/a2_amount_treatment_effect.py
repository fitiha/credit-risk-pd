"""Effect of the chosen money-field treatment (cap at p0.5/p99.5, then log).

Shows each field before and after, and the default rate by decile before and after.
Uses loans up to the fit cutoff. Writes reports/a2_amount_treatment_effect/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.outliers import AMOUNT_FIELDS, AmountTreater
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval

FIG, TAB = fresh_dirs("a2_amount_treatment_effect")

raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
t = AmountTreater().fit(d)
fit = t.transform(d[d.issue_d <= FIT_CUTOFF])


def decile_rates(values, y):
    """Default rate in ten equal-sized groups ordered by `values`."""
    group = pd.qcut(values.rank(method="first"), 10, labels=range(1, 11))
    g = y.groupby(group, observed=True).agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]
    return g


rows = []
for f in AMOUNT_FIELDS:
    before = decile_rates(fit[f], fit.defaulted)
    after = decile_rates(fit[f"{f}_log"], fit.defaulted)
    for k in before.index:
        rows.append({"field": f, "decile": k, "loans": before.loc[k, "size"],
                     "default_rate_before": before.loc[k, "rate"], "default_rate_after": after.loc[k, "rate"]})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(fit[f], bins=60, color="#2a6f97")
    ax[0].set_title(f"{f}: before (cleaned, dollars)")
    ax[1].hist(fit[f"{f}_log"], bins=60, color="#c1440e")
    lo, hi = t.caps_[f]
    ax[1].set_title(f"{f}: after (capped at {lo:,.0f} / {hi:,.0f}, then log)", fontsize=9)
    ax[1].set_xlabel(f"log(1 + {f})")
    ax[0].set_xlabel(f); ax[0].set_ylabel("loans"); ax[1].set_ylabel("loans")
    fig.tight_layout(); fig.savefig(FIG / f"before_after_{f}.png", dpi=110); plt.close(fig)

tab = pd.DataFrame(rows)
tab["same_before_and_after"] = np.isclose(tab.default_rate_before, tab.default_rate_after)
tab.round(4).to_csv(TAB / "default_rate_by_decile.csv", index=False)

fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
base = fit.defaulted.mean()
for ax, f in zip(axes.ravel(), AMOUNT_FIELDS):
    s = tab[tab.field == f]
    lo, hi = wilson_interval(s.default_rate_after * s.loans, s.loans)
    ax.errorbar(s.decile, s.default_rate_after * 100, yerr=[(s.default_rate_after - lo) * 100, (hi - s.default_rate_after) * 100],
                fmt="o-", color="#2a6f97", capsize=2, ms=4)
    ax.axhline(base * 100, color="grey", ls="--", lw=1)
    ax.set_title(f, fontsize=9); ax.set_xlabel("decile (1 = lowest values)", fontsize=8)
axes[0, 0].set_ylabel("Default rate (%)"); axes[1, 0].set_ylabel("Default rate (%)")
fig.suptitle("Default rate by decile of each treated money field (bars = 95% interval, dashed = overall rate)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_by_decile.png", dpi=130); plt.close(fig)

print("deciles identical before and after treatment:", bool(tab.same_before_and_after.all()))
sp = tab.groupby("field").apply(lambda s: s.decile.astype(int).corr(s.default_rate_after, method="spearman"), include_groups=False)
print("\nrank correlation between decile and default rate (1 = always rising, -1 = always falling):\n", sp.round(2).to_string())
print("\nlowest vs highest decile default rate (%):")
print(tab[tab.decile.isin([1, 10])].pivot(index="field", columns="decile", values="default_rate_after").mul(100).round(1).to_string())
