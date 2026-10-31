"""A2: default rate over time and by term, and the in-progress-loan bias.

Writes reports/target_analysis/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.cleaning import Cleaner
from credit_risk.io import load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import rate_table

SNAPSHOT = pd.Timestamp("2019-03-01")  # servicing data snapshot date (brief)
FIG, TAB = fresh_dirs("target_analysis")

raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
d["half"] = d.issue_d.dt.year.astype(str) + " H" + ((d.issue_d.dt.month - 1) // 6 + 1).astype(str)
end = [i + pd.DateOffset(months=int(t)) for i, t in zip(d.issue_d, d.term)]
d["matured"] = pd.Series(end, index=d.index) <= SNAPSHOT

by_half = rate_table(d, ["half", "term"])
by_half["share_matured"] = by_half.apply(
    lambda r: d[(d.half == r.half) & (d.term == r.term)].matured.mean(), axis=1)
by_half.to_csv(TAB / "default_by_half_year_and_term.csv", index=False)
d["month"] = d.issue_d.dt.to_period("M").astype(str)
rate_table(d, "month").to_csv(TAB / "default_by_month.csv", index=False)
rate_table(d, "term").to_csv(TAB / "default_by_term.csv", index=False)
rate_table(d.assign(year=d.issue_d.dt.year), "year").to_csv(TAB / "default_by_year.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
halves = sorted(d.half.unique())
x = {h: i for i, h in enumerate(halves)}
for term, color in [(36, "#2a6f97"), (60, "#c1440e")]:
    t = by_half[by_half.term == term]
    xs = t.half.map(x)
    ax.plot(xs, t.rate * 100, marker="o", color=color, label=f"{term}-month loans")
    ax.fill_between(xs, t.ci_low * 100, t.ci_high * 100, color=color, alpha=0.15)
first_unmatured = x["2014 H2"] - 0.5
ax.axvspan(first_unmatured, len(halves) - 0.5, color="grey", alpha=0.12)
ax.text(first_unmatured + 0.1, 41, "60-month loans here had not\nfinished by Mar 2019:\nonly early finishers are kept", fontsize=8, va="top")
ax.set_xticks(range(len(halves)), halves)
ax.set_ylabel("Default rate (%)")
ax.set_xlabel("Loan issue date (half-year)")
ax.set_title("Default rate by issue date and loan term (bands = 95% confidence interval)", fontsize=10)
ax.set_ylim(0, 42)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "default_rate_by_term_and_half_year.png", dpi=150)
print(by_half.round(3).to_string(index=False))
