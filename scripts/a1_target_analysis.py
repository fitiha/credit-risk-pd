"""A2: default rate over time and by term, and the in-progress-loan bias.

Writes reports/a1_target_analysis/{figures,tables}.
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
FIG, TAB = fresh_dirs("a1_target_analysis")

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

s60 = d[d.term == 60].groupby("matured").agg(loans=("id", "size"), defaults=("defaulted", "sum"))
s60["rate"] = s60.defaults / s60.loans
s60.rename(index={True: "scheduled end before Mar 2019", False: "scheduled end after Mar 2019"}) \
    .to_csv(TAB / "default_60m_finished_vs_unfinished.csv", index_label="group")

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
# monthly view: default rate per term with 95% CI, and how many loans each month rests on
by_month = rate_table(d, ["month", "term"])
months = sorted(d.month.unique())
mx = {m: i for i, m in enumerate(months)}
fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
for term, color in [(36, "#2a6f97"), (60, "#c1440e")]:
    t_ = by_month[by_month.term == term]
    xs = t_.month.map(mx)
    top.plot(xs, t_.rate * 100, marker="o", ms=3, color=color, label=f"{term}-month loans")
    top.fill_between(xs, t_.ci_low * 100, t_.ci_high * 100, color=color, alpha=0.15)
start = mx["2014-07"] - 0.5
top.axvspan(start, len(months) - 0.5, color="grey", alpha=0.12)
top.set_ylabel("Default rate (%)")
top.set_title("Default rate by issue month and loan term (bands = 95% confidence interval)", fontsize=10)
top.legend(loc="upper left")
counts = d.groupby("month").size()
bottom.bar([mx[m] for m in counts.index], counts.values, color="#888888")
bottom.set_ylabel("Loans issued")
bottom.set_xlabel("Loan issue month")
bottom.set_xticks(range(0, len(months), 3), months[::3], rotation=45, ha="right")
fig.tight_layout()
fig.savefig(FIG / "default_rate_by_month_and_term.png", dpi=150)
plt.close(fig)

print(by_half.round(3).to_string(index=False))
