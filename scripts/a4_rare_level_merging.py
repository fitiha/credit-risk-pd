"""Experiment: minimum group size for merging rare levels of purpose, home_ownership and addr_state.

For each threshold (share of loans), levels below it become one `other` level. IV is shown on the
loans the levels were learned from (up to mid-2014) and on later loans (second half of 2014).
Also shows default rate by employment length. Writes reports/a4_rare_level_merging/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.categorical import RareLevelMerger, order_emp_length
from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval
from credit_risk.woe import information_value, iv_on_new_data, woe_table

FIG, TAB = fresh_dirs("a4_rare_level_merging")
raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
train, test = fit[fit.issue_d <= "2014-06-30"], fit[fit.issue_d > "2014-06-30"]
SHARES = [0.0, 0.0025, 0.005, 0.01, 0.02]

rows = []
for field in ["purpose", "home_ownership", "addr_state"]:
    for share in SHARES:
        m = RareLevelMerger(field, share).fit(train[field])
        t = woe_table(m.transform(train[field]), train.defaulted)
        rows.append({"field": field, "min_share_pct": 100 * share, "min_loans_in_training": m.min_loans_ if share else 0,
                     "levels_before": train[field].nunique(), "groups_after": len(t),
                     "levels_merged": len(m.merged_), "loans_in_other_pct": 100 * (m.transform(train[field]) == m.other_label).mean(),
                     "smallest_group_loans": int(t.loans.min()), "iv_on_training_loans": information_value(t),
                     "iv_on_later_loans": iv_on_new_data(t, m.transform(test[field]), test.defaulted)})
exp = pd.DataFrame(rows)
exp.round(4).to_csv(TAB / "threshold_comparison.csv", index=False)

# employment length, ordered
el = fit.assign(emp_length_years=order_emp_length(fit.emp_length))
el["level"] = el.emp_length_years.map(lambda v: "missing" if pd.isna(v) else int(v))
g = woe_table(el.level, el.defaulted)
lo, hi = wilson_interval(g.defaults, g.loans)
g["ci_low"], g["ci_high"] = lo, hi
order = {**{i: i for i in range(11)}, "missing": 11}
g = g.sort_values("group", key=lambda s: s.map(order))
g.round(4).to_csv(TAB / "emp_length_woe.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
x = range(len(g))
ax.errorbar(x, g.default_rate * 100, yerr=[(g.default_rate - g.ci_low) * 100, (g.ci_high - g.default_rate) * 100], fmt="o-", color="#2a6f97", capsize=3)
ax.axhline(fit.defaulted.mean() * 100, color="grey", ls="--", lw=1)
ax.set_xticks(list(x), [("<1" if k == 0 else "10+" if k == 10 else str(k)) for k in g.group], fontsize=8)
ax.set_xlabel("Years of employment (last point = missing)"); ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by employment length (loans to 2014; bars = 95% interval)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_by_emp_length.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 20)
print(exp.round(4).to_string(index=False))
print("\nEMP_LENGTH\n", g[["group", "loans", "default_rate", "woe"]].round(3).to_string(index=False), "\n IV:", round(information_value(g), 4))
for f in ["purpose", "home_ownership"]:
    print("\n", f, fit[f].value_counts().to_dict())
print("\naddr_state rarest levels (fit window):", fit.addr_state.value_counts().tail(14).to_dict())
