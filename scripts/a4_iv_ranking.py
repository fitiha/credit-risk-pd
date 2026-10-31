"""Rank every field by Information Value on loans up to the fit cutoff, and question the suspicious ones.

Contrast: the same calculation on post-issue servicing fields, which are known leaks.
Writes reports/a4_iv_ranking/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.binning import assign_bins, fit_edges
from credit_risk.io import FIT_CUTOFF, load_applications, load_servicing
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer
from credit_risk.woe import WoEEncoder, information_value, strength, woe_table
from credit_risk.woe_fields import WOE_CATEGORICAL, WOE_NUMERIC

FIG, TAB = fresh_dirs("a4_iv_ranking")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)

NUMERIC, CATEGORICAL = WOE_NUMERIC, WOE_CATEGORICAL
enc = WoEEncoder(numeric=NUMERIC, categorical=CATEGORICAL).fit(d)
iv = enc.iv_table()
# sub_grade (35 levels) is a platform score like grade; it is ranked here but gets no contract column
fit_rows = d[d.issue_d <= FIT_CUTOFF]
sub = information_value(woe_table(fit_rows.sub_grade.astype(str), fit_rows.defaulted))
iv = pd.concat([iv, pd.DataFrame([{"field": "sub_grade", "kind": "categorical", "bins_or_groups": fit_rows.sub_grade.nunique(),
                                   "iv": sub, "strength": strength(sub)}])], ignore_index=True).sort_values("iv", ascending=False)
role = {"int_rate": "platform score", "grade": "platform score", "sub_grade": "platform score", "job_group": "restricted", "addr_state_group": "restricted"}
iv["role"] = iv.field.map(role).fillna("input")
iv.round(4).to_csv(TAB / "iv_ranking.csv", index=False)

# contrast: known post-issue fields
fit = d[d.issue_d <= FIT_CUTOFF]
srv = load_servicing().set_index("id").reindex(fit.id)
rows = []
for f in ["last_fico_range_high", "recoveries", "total_rec_prncp", "total_pymnt", "last_pymnt_amnt"]:
    x = srv[f].reset_index(drop=True)
    e = fit_edges(x, fit.defaulted.reset_index(drop=True), "monotonic")
    v = information_value(woe_table(assign_bins(x, e), fit.defaulted.reset_index(drop=True)))
    rows.append({"field": f, "known_at_application": False, "iv": v, "strength": strength(v)})
flag = raw.set_index("id").bureau_recovery_flag.reindex(fit.id).reset_index(drop=True).map({0: "no", 1: "yes"})
v = information_value(woe_table(flag, fit.defaulted.reset_index(drop=True)))
rows.append({"field": "bureau_recovery_flag", "known_at_application": False, "iv": v, "strength": strength(v)})
leak = pd.DataFrame(rows).sort_values("iv", ascending=False)
leak.round(3).to_csv(TAB / "leak_contrast.csv", index=False)

colors = {"useless": "#bbbbbb", "weak": "#9ecae1", "medium": "#2a6f97", "strong": "#08306b", "suspicious": "#c1440e"}
top = iv.head(22).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top.field, top.iv, color=[colors[s] for s in top.strength])
for y, (v, f) in enumerate(zip(top.iv, top.field)):
    ax.text(v + 0.005, y, f"{v:.3f}" + ("  (platform score)" if role.get(f) == "platform score" else "  (restricted)" if role.get(f) == "restricted" else ""), va="center", fontsize=7)
for x in (0.02, 0.1, 0.3, 0.5):
    ax.axvline(x, color="grey", ls=":", lw=0.8)
ax.set_xlabel("Information value (dotted: 0.02, 0.1, 0.3, 0.5 = weak / medium / strong / suspicious)")
ax.set_title("Information value of each field (loans to 2014)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "iv_ranking.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 220)
print(iv.round(3).head(20).to_string()); print("\nfields by strength:", iv.strength.value_counts().to_dict())
print("\nLEAK CONTRAST\n", leak.round(3).to_string(index=False))
print("\nWoE of int_rate bins:\n", enc.tables_["int_rate"][["group", "loans", "default_rate", "woe"]].round(3).to_string(index=False))
