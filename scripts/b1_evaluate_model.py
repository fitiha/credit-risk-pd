"""Part B evaluation on the 2015 out-of-time loans: ranking, calibration, stability by quarter and term, comparison with grade alone,
the decision policy and its profit, outcomes by region and income band, and the ablation.

Thresholds are fixed from the profit assumptions and checked on 2015 H1; results are reported on 2015 H2.
Writes reports/b1_model_evaluation/{figures,tables}.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.model import metrics, policy
from credit_risk.model.data import load_config, split, training_frames
from credit_risk.model.models import add_grade_rank
from credit_risk.model.regions import STATE_TO_REGION
from credit_risk.reporting import fresh_dirs
from credit_risk.stats import wilson_interval

FIG, TAB = fresh_dirs("b1_model_evaluation")
cfg = load_config()
transformer, spec, frame = training_frames(cfg)
frame = add_grade_rank(frame)
bundles = {n: joblib.load(f"{cfg['model_dir']}/{n}.joblib") for n in ["grade_baseline", "woe_scorecard", "lightgbm", "scorecard_plus_grade", "lightgbm_plus_grade", "scorecard_boosted_average"]}
selected = joblib.load(f"{cfg['model_dir']}/pd_model.joblib").metadata["model"]
v15 = split(frame, "2015-01-01", "2015-12-31").copy()
h1, h2 = split(v15, **{"start": cfg["policy_window"]["start"], "end": cfg["policy_window"]["end"]}), split(v15, cfg["report_window"]["start"], cfg["report_window"]["end"])
for n, b in bundles.items():
    v15[f"pd_{n}"] = b.score(v15)
v15["quarter"] = v15.issue_d.dt.to_period("Q").astype(str)
y = v15.defaulted

# 1. overall and by quarter and term, every model
rows = []
for n in bundles:
    for label, sub in [("2015 all", v15), ("2015 H1", v15[v15.index.isin(h1.index)]), ("2015 H2", v15[v15.index.isin(h2.index)])] + \
                      [(q, g) for q, g in v15.groupby("quarter")] + [(f"{t} months", g) for t, g in v15.groupby("term")]:
        rows.append({"model": n, "group": label, **metrics.summary(sub.defaulted, sub[f"pd_{n}"])})
res = pd.DataFrame(rows)
res.round(4).to_csv(TAB / "metrics_by_model_and_group.csv", index=False)

pd.DataFrame([{"group": g, "auc_of_the_raw_sub_grade_rank_no_calibration": metrics.auc(s.defaulted, s.sub_grade_rank)} for g, s in [("2015 all", v15)] + [(f"{k} months", s) for k, s in v15.groupby("term")]]).round(4).to_csv(TAB / "raw_sub_grade_auc_2015.csv", index=False)
# 2. paired comparison against grade alone (bootstrap on the same 2015 loans)
rng = np.random.default_rng(0)
gap = []
yy = y.to_numpy()
for n in ["woe_scorecard", "lightgbm", "scorecard_plus_grade", "lightgbm_plus_grade", "scorecard_boosted_average"]:
    a, g = v15[f"pd_{n}"].to_numpy(), v15["pd_grade_baseline"].to_numpy()
    d = [metrics.auc(yy[i], a[i]) - metrics.auc(yy[i], g[i]) for i in (rng.integers(0, len(yy), len(yy)) for _ in range(300))]
    gap.append({"model": n, "auc_model": metrics.auc(yy, a), "auc_grade_alone": metrics.auc(yy, g), "gain": metrics.auc(yy, a) - metrics.auc(yy, g),
                "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
pd.DataFrame(gap).round(4).to_csv(TAB / "gain_over_grade_alone_2015.csv", index=False)

# 3. calibration of the selected model: overall, by term, and the reliability plot
sel = f"pd_{selected}"
cal = metrics.calibration_table(y, v15[sel]); cal.round(4).to_csv(TAB / "calibration_deciles_2015.csv", index=False)
slope = {t: metrics.calibration_slope_intercept(g.defaulted, g[sel]) for t, g in [("all", v15)] + [(f"{k} months", g) for k, g in v15.groupby("term")]}
pd.DataFrame([{"group": k, "slope": v[0], "intercept": v[1]} for k, v in slope.items()]).round(3).to_csv(TAB / "calibration_slope_intercept.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for term, color in [(36, "#2a6f97"), (60, "#c1440e")]:
    c = metrics.calibration_table(v15[v15.term == term].defaulted, v15[v15.term == term][sel], bins=8)
    ax[0].plot(c.mean_pd * 100, c.default_rate * 100, "o-", color=color, label=f"{term}-month loans")
ax[0].plot([0, 60], [0, 60], color="grey", ls="--", lw=1); ax[0].set_xlabel("Predicted default rate (%)"); ax[0].set_ylabel("Observed default rate (%)"); ax[0].legend(fontsize=8)
ax[0].set_title(f"Calibration on 2015 loans ({selected})", fontsize=10)
by = v15.groupby("quarter").apply(lambda g: pd.Series({"predicted": g[sel].mean() * 100, "observed": g.defaulted.mean() * 100}), include_groups=False)
by.plot.bar(ax=ax[1], color=["#2a6f97", "#c1440e"], rot=0); ax[1].set_ylabel("Default rate (%)"); ax[1].set_title("Predicted against observed by quarter", fontsize=10); ax[1].set_ylim(0, by.to_numpy().max() * 1.3); ax[1].legend(loc="upper center", ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "calibration_2015.png", dpi=140); plt.close(fig)

# 4. decision policy: thresholds from the profit assumptions; empirical check on H1; results on H2
th = policy.thresholds(cfg)
d1, d2 = v15[v15.index.isin(h1.index)].assign(pd=lambda t: t[sel]), v15[v15.index.isin(h2.index)].assign(pd=lambda t: t[sel])
curve = policy.best_decline_threshold(d1, cfg); curve.round(4).to_csv(TAB / "profit_by_decline_threshold_2015H1.csv", index=False)
best_t = curve.loc[curve.profit.idxmax()]
bands = {"H1 (thresholds checked here)": policy.band_table(d1, th, cfg), "H2 (reported)": policy.band_table(d2, th, cfg)}
pd.concat({k: v for k, v in bands.items()}, names=["window"]).reset_index(level=0).round(4).to_csv(TAB / "decision_bands.csv", index=False)
# the same policy driven by grade alone, for comparison
d2g = d2.assign(pd=d2.pd_grade_baseline)
cmp_rows = []
for name, dd in [("model", d2), ("grade alone", d2g)]:
    dec = pd.Series(policy.decide(dd.pd, th), index=dd.index)
    funded = dd[dec != "decline"]
    appr = dd[dec == "approve"]
    cmp_rows.append({"policy_driven_by": name, "approval_rate": (dec == "approve").mean(), "review_rate": (dec == "review").mean(), "decline_rate": (dec == "decline").mean(),
                     "default_rate_of_approved": appr.defaulted.mean(), "profit_if_approve_and_review_funded": float(policy.realised_profit(funded.loan_amnt, funded.defaulted, cfg).sum()),
                     "profit_if_only_approved_funded": float(policy.realised_profit(appr.loan_amnt, appr.defaulted, cfg).sum()), "profit_fund_everyone": float(policy.realised_profit(dd.loan_amnt, dd.defaulted, cfg).sum())})
pd.DataFrame(cmp_rows).round(4).to_csv(TAB / "policy_model_vs_grade_2015H2.csv", index=False)
# is the profit gap between the two policies larger than chance? Resample the same 2015 H2 loans (profit of each loan counts only where exactly one policy funds it)
loan_profit = np.asarray(policy.realised_profit(d2.loan_amnt, d2.defaulted, cfg), float)
funded_model = np.array(policy.decide(d2.pd, th)) != "decline"
funded_grade = np.array(policy.decide(d2g.pd, th)) != "decline"
per_loan_gap = loan_profit * (funded_model.astype(float) - funded_grade.astype(float))
rng_p = np.random.default_rng(2)
boot = [per_loan_gap[rng_p.integers(0, len(per_loan_gap), len(per_loan_gap))].sum() for _ in range(2000)]
pd.DataFrame([{"profit_difference_model_minus_grade": per_loan_gap.sum(), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
               "loans_funded_by_only_one_policy": int((funded_model != funded_grade).sum())}]).round(0).to_csv(TAB / "profit_difference_model_vs_grade_2015H2.csv", index=False)

# the comparison above funds different numbers of loans under the two scores (the same PD threshold funds 45% of loans under the model and 35% under the grade).
# To isolate ranking from calibration, fund the safest X% of the same loans by each score. Ties (the grade has about 70 distinct values) are broken at random.
policy_volume = round(float(funded_model.mean()), 3)          # the share of loans the model's own thresholds fund: fixed by the policy, not chosen from these results
def equal_volume_table(frame, grade_pd, seed):
    rng_v = np.random.default_rng(seed)
    sm = frame.pd.to_numpy(float) + rng_v.random(len(frame)) * 1e-9
    sg = np.asarray(grade_pd, float) + rng_v.random(len(frame)) * 1e-9
    gain_ = np.asarray(policy.realised_profit(frame.loan_amnt, frame.defaulted, cfg), float)
    def funded_profit(score, share, idx):
        k = int(round(share * len(idx)))
        return gain_[idx[np.argsort(score[idx], kind="stable")[:k]]].sum()
    everyone = np.arange(len(frame))
    resamples = [rng_v.integers(0, len(frame), len(frame)) for _ in range(1000)]
    rows_ = []
    for share in sorted({0.20, 0.30, 0.35, 0.40, 0.45, 0.50, policy_volume}):
        diffs = [funded_profit(sm, share, ix) - funded_profit(sg, share, ix) for ix in resamples]
        pm_, pg_ = funded_profit(sm, share, everyone), funded_profit(sg, share, everyone)
        rows_.append({"share_of_loans_funded": share, "profit_model": round(pm_), "profit_grade": round(pg_), "difference": round(pm_ - pg_),
                      "ci_low": round(np.percentile(diffs, 2.5)), "ci_high": round(np.percentile(diffs, 97.5)), "is_policy_volume": share == policy_volume, "loans": len(frame)})
    return pd.DataFrame(rows_)
eq_h2 = equal_volume_table(d2, d2g.pd, 3)
eq_all = equal_volume_table(v15.assign(pd=v15[sel]), v15.pd_grade_baseline, 4)
eq_h2.to_csv(TAB / "profit_at_equal_volume_2015H2.csv", index=False)
eq_all.to_csv(TAB / "profit_at_equal_volume_2015_all.csv", index=False)
print("\nprofit at equal funded volume, model minus grade (2015 H2):\n", eq_h2.to_string(index=False))
print("\nthe same on all of 2015 (a ranking question: no threshold is involved):\n", eq_all.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve.threshold * 100, curve.profit / 1e6, color="#2a6f97"); ax.axvline(th["decline_at_or_above"] * 100, color="#c1440e", ls="--", label="break-even PD (12.5%)")
ax.axvline(best_t.threshold * 100, color="grey", ls=":", label=f"best on 2015 H1 ({best_t.threshold * 100:.1f}%)")
ax.set_xlabel("Decline loans with a PD at or above (%)"); ax.set_ylabel("Realised profit if the rest are funded ($ millions)"); ax.legend(fontsize=8)
ax.set_title("Profit against the decline threshold, 2015 H1", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "profit_by_decline_threshold.png", dpi=140); plt.close(fig)

# 5. regions and income bands (monitoring only; neither is a model input)
d2 = d2.assign(decision=policy.decide(d2.pd, th), region=d2.addr_state.map(STATE_TO_REGION).fillna("other"),
               income_band=pd.cut(d2.annual_inc, [0, 40000, 60000, 90000, 1e9], labels=["under 40k", "40-60k", "60-90k", "90k+"]).astype(str))
fair = []
for col in ["region", "income_band"]:
    for lvl, g in d2.groupby(col):
        lo, hi = wilson_interval(int(g.defaulted.sum()), len(g))
        a = g[g.decision == "approve"]
        fair.append({"segment": col, "level": lvl, "loans": len(g), "approval_rate": (g.decision == "approve").mean(), "review_rate": (g.decision == "review").mean(),
                     "decline_rate": (g.decision == "decline").mean(), "observed_default_rate": g.defaulted.mean(), "ci_low": lo, "ci_high": hi,
                     "mean_predicted_pd": g.pd.mean(), "calibration_gap": g.defaulted.mean() - g.pd.mean(), "default_rate_of_approved": a.defaulted.mean() if len(a) else np.nan})
pd.DataFrame(fair).round(4).to_csv(TAB / "outcomes_by_region_and_income.csv", index=False)

# 6. ablation (from the tuning folds) and the model-selection record
sel_meta = json.load(open(f"{cfg['model_dir']}/metadata.json"))["selection"]
pd.DataFrame([{"family": f, "input_set": k, "mean_auc_time_folds": v} for f, d in [("lightgbm", sel_meta["ablation_lightgbm"]), ("logistic", sel_meta["ablation_logistic"])] for k, v in d.items()]).to_csv(TAB / "ablation_time_folds.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 20)
print(res[res.group.isin(["2015 all", "2015 H2", "36 months", "60 months"])][["model", "group", "loans", "default_rate", "mean_pd", "auc", "gini", "ks", "brier"]].round(3).to_string(index=False))
print("\nquarterly AUC:", res[res.model == selected][res.group.str.startswith("2015Q")][["group", "auc", "mean_pd", "default_rate"]].round(3).values.tolist())
print("\ngain over grade alone:\n", pd.DataFrame(gap).round(4).to_string(index=False)); print("\ncalibration slope/intercept:", {k: (round(v[0], 2), round(v[1], 2)) for k, v in slope.items()})
print("\nbest decline threshold on H1:", round(float(best_t.threshold), 3), "| break-even 0.125\n", pd.DataFrame(cmp_rows).round(3).to_string(index=False)); print(bands["H2 (reported)"].round(3).to_string(index=False))
print(pd.DataFrame(fair)[["segment", "level", "loans", "approval_rate", "decline_rate", "observed_default_rate", "mean_predicted_pd"]].round(3).to_string(index=False))
