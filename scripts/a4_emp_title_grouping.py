"""Experiment E3: how to turn job titles into a few usable groups.

Options: keyword groups, the N most common titles, and a simple missing flag.
IV is shown on the data the groups were learned from and on later loans (2014 H2) never seen
when learning the WoE, because more groups always look better in-sample.
Writes reports/a4_emp_title_grouping/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.categorical import group_job_titles
from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval
from credit_risk.woe import information_value, iv_on_new_data, woe_table

FIG, TAB = fresh_dirs("a4_emp_title_grouping")
raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
train, test = fit[fit.issue_d <= "2014-06-30"], fit[fit.issue_d > "2014-06-30"]


def top_n(n):
    keep = set(train.emp_title.value_counts().head(n).index)
    return lambda s: s.map(lambda t: "missing" if pd.isna(t) else (t if t in keep else "other_title"))


OPTIONS = {
    "keyword_groups": group_job_titles,
    "top_20_titles": top_n(20),
    "top_50_titles": top_n(50),
    "top_200_titles": top_n(200),
    "missing_flag_only": lambda s: s.isna().map({True: "missing", False: "present"}),
}
rows = []
for name, f in OPTIONS.items():
    gtr, gte = f(train.emp_title), f(test.emp_title)
    t = woe_table(gtr, train.defaulted)
    rows.append({"option": name, "groups": len(t), "smallest_group_loans": int(t.loans.min()),
                 "share_in_other_pct": 100 * gtr.isin(["other_job", "other_title"]).mean(),
                 "iv_on_training_loans": information_value(t),
                 "iv_on_later_loans": iv_on_new_data(t, gte, test.defaulted)})
exp = pd.DataFrame(rows)
exp.round(4).to_csv(TAB / "grouping_options.csv", index=False)

g = woe_table(group_job_titles(fit.emp_title), fit.defaulted)
lo, hi = wilson_interval(g.defaults, g.loans)
g["ci_low"], g["ci_high"] = lo, hi
g.round(4).sort_values("woe").to_csv(TAB / "job_group_woe.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.2))
s = g.sort_values("default_rate")
ax.errorbar(s.default_rate * 100, range(len(s)), xerr=[(s.default_rate - s.ci_low) * 100, (s.ci_high - s.default_rate) * 100], fmt="o", color="#2a6f97", capsize=3)
ax.axvline(fit.defaulted.mean() * 100, color="grey", ls="--", lw=1)
ax.set_yticks(range(len(s)), [f"{a} (n={n:,})" for a, n in zip(s.group, s.loans)], fontsize=8)
ax.set_xlabel("Default rate (%) (bars = 95% interval, dashed = overall)")
ax.set_title("Default rate by job group (loans to 2014)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_by_job_group.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 220)
print(exp.round(4).to_string(index=False))
print("\n", g[["group", "loans", "defaults", "default_rate", "woe", "iv_part"]].round(3).sort_values("woe").to_string(index=False))
other = fit[group_job_titles(fit.emp_title) == "other_job"].emp_title.value_counts()
print("\nmost common titles left in other_job:", other.head(25).to_dict())
