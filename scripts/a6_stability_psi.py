"""A7, part 1: Population Stability Index between 2013 and 2014 loans (and 2014 vs 2015 for context).

Bins are set on the earlier year. Writes reports/a6_stability_psi/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.factor_fields import FACTOR_FIELDS
from credit_risk.features import FEATURE_FIELDS
from credit_risk.io import load_applications
from credit_risk.psi import categorical_psi, label, numeric_psi
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer

FIG, TAB = fresh_dirs("a6_stability_psi")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
yr = d.issue_d.dt.year
y13, y14, y15 = d[yr == 2013], d[yr == 2014], d[yr == 2015]
FACTORS = [c for c in d.columns if c.startswith("factor_")]
NUMERIC = (FACTOR_FIELDS + [f"{f}_treated" for f in FEATURE_FIELDS if f not in FACTOR_FIELDS and f"{f}_treated" not in FACTOR_FIELDS]
           + ["mths_since_last_delinq", "emp_length_years"] + FACTORS + ["int_rate"])
NUMERIC = list(dict.fromkeys(NUMERIC))
CATEGORICAL = ["term", "purpose_group", "home_ownership_group", "verification_status", "grade", "job_group", "addr_state_group"]
kind = {**{c: "factor score" for c in FACTORS}, "int_rate": "platform score", **{c: "category" for c in CATEGORICAL}}
rows = []
for c in NUMERIC:
    rows.append({"field": c, "kind": kind.get(c, "numeric"), "psi_2013_vs_2014": numeric_psi(y13[c], y14[c]), "psi_2014_vs_2015": numeric_psi(y14[c], y15[c])})
for c in CATEGORICAL:
    rows.append({"field": c, "kind": "category", "psi_2013_vs_2014": categorical_psi(y13[c], y14[c]), "psi_2014_vs_2015": categorical_psi(y14[c], y15[c])})
res = pd.DataFrame(rows).sort_values("psi_2013_vs_2014", ascending=False).reset_index(drop=True)
res["reading_2013_vs_2014"] = res.psi_2013_vs_2014.map(label)
res["reading_2014_vs_2015"] = res.psi_2014_vs_2015.map(label)
res.round(4).to_csv(TAB / "psi_by_field.csv", index=False)

top = res.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top.field, top.psi_2013_vs_2014, color=["#c1440e" if v >= .25 else "#e6a26a" if v >= .1 else "#2a6f97" for v in top.psi_2013_vs_2014])
ax.axvline(0.1, color="grey", ls=":", lw=1); ax.axvline(0.25, color="grey", ls=":", lw=1)
ax.set_xlabel("PSI, 2013 vs 2014 (dotted: 0.1 moderate, 0.25 major)"); ax.tick_params(axis="y", labelsize=8)
ax.set_title("Which fields moved most between 2013 and 2014 loans", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "psi_2013_vs_2014.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 220)
print(f"loans: 2013 {len(y13):,} | 2014 {len(y14):,} | 2015 {len(y15):,}")
print(res.round(3).head(14).to_string(index=False))
print("\ncount by reading 2013 vs 2014:", res.reading_2013_vs_2014.value_counts().to_dict(), "| 2014 vs 2015:", res.reading_2014_vs_2015.value_counts().to_dict())
print("factor scores:\n", res[res.kind == "factor score"][["field", "psi_2013_vs_2014", "psi_2014_vs_2015"]].round(3).to_string(index=False))
