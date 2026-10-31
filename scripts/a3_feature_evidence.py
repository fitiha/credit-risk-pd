"""Evidence for the application-time features: distribution, treatment and link to default.

Uses cleaned loans up to the fit cutoff. Writes reports/a3_feature_evidence/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.cleaning import Cleaner
from credit_risk.features import FEATURE_FIELDS, FeatureBuilder, FeatureTreater
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval

FIG, TAB = fresh_dirs("a3_feature_evidence")
raw = load_applications()
d = FeatureBuilder().transform(Cleaner().fit(raw).transform(raw))
t = FeatureTreater().fit(d)
fit = t.transform(d[d.issue_d <= FIT_CUTOFF])

rows, dec_rows = [], []
for f in FEATURE_FIELDS:
    x = fit[f]
    cap = x.clip(*t.caps_[f])
    treated = fit[f"{f}_treated"]
    ok = x.notna()
    auc = roc_auc_score(fit.defaulted[ok], treated[ok])
    rows.append({"feature": f, "definition": t.spec_entries()[f]["definition"], "missing_in_fit_window": int((~ok).sum()),
                 "median": x.median(), "min": x.min(), "max": x.max(), "cap_low": t.caps_[f][0], "cap_high": t.caps_[f][1],
                 "fill_value": t.fill_[f], "skew_raw": x[ok].skew(), "skew_capped": cap[ok].skew(),
                 "log_added": f in t.log_fields_, "auc_alone": max(auc, 1 - auc)})
    group = pd.qcut(treated[ok].rank(method="first"), 10, labels=range(1, 11))
    g = fit.defaulted[ok].groupby(group, observed=True).agg(["size", "sum"])
    for k, r in g.iterrows():
        dec_rows.append({"feature": f, "decile": k, "loans": r["size"], "default_rate": r["sum"] / r["size"]})
feat = pd.DataFrame(rows)
feat.round(3).to_csv(TAB / "feature_summary.csv", index=False)
dec = pd.DataFrame(dec_rows)
dec.round(4).to_csv(TAB / "default_rate_by_decile.csv", index=False)

fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey=True)
base = fit.defaulted.mean()
for ax, f in zip(axes.ravel(), FEATURE_FIELDS):
    s = dec[dec.feature == f]
    lo, hi = wilson_interval(s.default_rate * s.loans, s.loans)
    ax.errorbar(s.decile, s.default_rate * 100, yerr=[(s.default_rate - lo) * 100, (hi - s.default_rate) * 100],
                fmt="o-", color="#2a6f97", capsize=2, ms=4)
    ax.axhline(base * 100, color="grey", ls="--", lw=1)
    ax.set_title(f, fontsize=9); ax.set_xlabel("decile (1 = lowest values)", fontsize=8)
axes.ravel()[-1].axis("off")
axes[0, 0].set_ylabel("Default rate (%)"); axes[1, 0].set_ylabel("Default rate (%)")
fig.suptitle("Default rate by decile of each application-time feature (bars = 95% interval, dashed = overall rate)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "default_rate_by_decile.png", dpi=130); plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 20)
print(feat.drop(columns="definition").round(2).to_string(index=False))
print("\nlog added to:", t.log_fields_, "| missing flags:", t.flag_fields_)
print("\ndecile 1 vs 10 default rate (%):")
print(dec[dec.decile.isin([1, 10])].pivot(index="feature", columns="decile", values="default_rate").mul(100).round(1).to_string())
print("\nfinal feature columns:", [c for c in fit.columns if c.endswith(("_treated", "_log")) and any(c.startswith(f) for f in FEATURE_FIELDS)])
