"""Effect of the chosen treatments on money fields, ratios and counts.

Money fields: cap at p0.5/p99.5, then log. Ratios: cap at p0.5/p99.5, fill missing with the
median. Counts: cap at p99.5. Shows each field before and after, and the default rate by
decile before and after. Uses loans up to the fit cutoff.
Writes reports/a2_treatment_effect/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.outliers import (AMOUNT_FIELDS, COUNT_FIELDS, RATIO_FIELDS, AmountTreater,
                                  CountTreater, RatioTreater)
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval

FIG, TAB = fresh_dirs("a2_treatment_effect")

raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
amounts, ratios, counts = AmountTreater().fit(d), RatioTreater().fit(d), CountTreater().fit(d)
fit = counts.transform(ratios.transform(amounts.transform(d[d.issue_d <= FIT_CUTOFF])))

AFTER = {**{f: f"{f}_log" for f in AMOUNT_FIELDS}, **{f: f"{f}_treated" for f in RATIO_FIELDS}}


def decile_rates(values, y):
    """Default rate in ten equal-sized groups ordered by `values` (missing values excluded)."""
    ok = values.notna()
    group = pd.qcut(values[ok].rank(method="first"), 10, labels=range(1, 11))
    g = y[ok].groupby(group, observed=True).agg(["size", "sum"])
    g["rate"] = g["sum"] / g["size"]
    return g


rows = []
for f, after_col in AFTER.items():
    before, after = decile_rates(fit[f], fit.defaulted), decile_rates(fit[after_col], fit.defaulted)
    for k in after.index:
        rows.append({"field": f, "decile": k, "loans": after.loc[k, "size"],
                     "default_rate_before": before.loc[k, "rate"], "default_rate_after": after.loc[k, "rate"]})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(fit[f].dropna(), bins=60, color="#2a6f97")
    ax[0].set_title(f"{f}: before (cleaned)")
    ax[1].hist(fit[after_col], bins=60, color="#c1440e")
    if f in amounts.caps_:
        lo, hi = amounts.caps_[f]
        ax[1].set_title(f"{f}: after (capped at {lo:,.0f} / {hi:,.0f}, then log)", fontsize=9)
        ax[1].set_xlabel(f"log(1 + {f})")
    else:
        lo, hi = ratios.caps_[f]
        ax[1].set_title(f"{f}: after (capped at {lo:,.1f} / {hi:,.1f}, missing filled)", fontsize=9)
        ax[1].set_xlabel(f)
    ax[0].set_xlabel(f); ax[0].set_ylabel("loans"); ax[1].set_ylabel("loans")
    fig.tight_layout(); fig.savefig(FIG / f"before_after_{f}.png", dpi=110); plt.close(fig)

tab = pd.DataFrame(rows)
tab["same_before_and_after"] = np.isclose(tab.default_rate_before, tab.default_rate_after)
tab.round(4).to_csv(TAB / "default_rate_by_decile.csv", index=False)

counts_tab = pd.DataFrame({
    "cap": counts.caps_,
    "max_before": {f: fit[f].max() for f in COUNT_FIELDS},
    "rows_capped": {f: int((fit[f] > counts.caps_[f]).sum()) for f in COUNT_FIELDS},
}).rename_axis("field")
counts_tab["rows_capped_pct"] = 100 * counts_tab.rows_capped / len(fit)
counts_tab.round(2).to_csv(TAB / "count_caps.csv")

fields = list(AFTER)
fig, axes = plt.subplots(3, 4, figsize=(13, 8.5), sharey=True)
base = fit.defaulted.mean()
for ax, f in zip(axes.ravel(), fields):
    s = tab[tab.field == f]
    lo, hi = wilson_interval(s.default_rate_after * s.loans, s.loans)
    ax.errorbar(s.decile, s.default_rate_after * 100, yerr=[(s.default_rate_after - lo) * 100, (hi - s.default_rate_after) * 100],
                fmt="o-", color="#2a6f97", capsize=2, ms=4)
    ax.axhline(base * 100, color="grey", ls="--", lw=1)
    ax.set_title(f, fontsize=9); ax.set_xlabel("decile (1 = lowest values)", fontsize=8)
for ax in axes.ravel()[len(fields):]:
    ax.axis("off")
for r in range(3):
    axes[r, 0].set_ylabel("Default rate (%)")
fig.suptitle("Default rate by decile of each treated field (bars = 95% interval, dashed = overall rate)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_by_decile.png", dpi=130); plt.close(fig)

print("fields whose deciles are identical before and after:", sorted(tab.groupby("field").same_before_and_after.all().pipe(lambda s: s[s].index)))
print("fields that differ (missing values were filled):", sorted(tab.groupby("field").same_before_and_after.all().pipe(lambda s: s[~s].index)))
print("\nratio caps (floor, cap) and fill values:", {f: (round(ratios.caps_[f][0], 1), round(ratios.caps_[f][1], 1), round(ratios.fill_[f], 1)) for f in RATIO_FIELDS})
print("ratio fields that get a missing flag:", ratios.flag_fields_)
print("\n", counts_tab.round(2).to_string())
print("\nratio deciles 1 vs 10 default rate (%):")
print(tab[tab.field.isin(RATIO_FIELDS) & tab.decile.isin([1, 10])].pivot(index="field", columns="decile", values="default_rate_after").mul(100).round(1).to_string())
