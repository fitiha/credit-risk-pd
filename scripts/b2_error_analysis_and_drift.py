"""Part B: where does the model fail, how does it behave on bad or missing inputs, and how far has 2016 drifted from 2015?

Scores the 2016 holdout (outputs/holdout_scores.csv: id, pd). Reports only aggregates and made-up applicants, never individual real rows.
Writes reports/b2_error_analysis_and_drift/{figures,tables}.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from credit_risk.model import metrics, policy
from credit_risk.model.data import holdout_frame, load_config, split, training_frames
from credit_risk.model.inputs import REASONS
from credit_risk.model.models import add_grade_rank
from credit_risk.model.predict import Predictor
from credit_risk.model.regions import STATE_TO_REGION
from credit_risk.psi import categorical_psi, label, numeric_psi
from credit_risk.reporting import fresh_dirs
from credit_risk.spec import ContractViolation

warnings.filterwarnings("ignore")
FIG, TAB = fresh_dirs("b2_error_analysis_and_drift")
cfg = load_config()
transformer, spec, frame = training_frames(cfg)
frame = add_grade_rank(frame)
bundle = joblib.load(f"{cfg['model_dir']}/pd_model.joblib")
th = bundle.thresholds
v15 = split(frame, "2015-01-01", "2015-12-31").copy()
v15["pd"] = bundle.score(v15)
v15["decision"] = policy.decide(v15.pd, th)

# 1. segments where calibration is poor (observed against predicted default rate; binomial test, Holm-corrected; at least 300 loans)
v15["region"] = v15.addr_state.map(STATE_TO_REGION).fillna("other")
v15["income_band"] = pd.cut(v15.annual_inc, [0, 40000, 60000, 90000, 1e9], labels=["under 40k", "40-60k", "60-90k", "90k+"]).astype(str)
v15["loan_band"] = pd.qcut(v15.loan_amnt, 4, labels=["loan Q1", "loan Q2", "loan Q3", "loan Q4"]).astype(str)
v15["fico_band"] = pd.cut(v15.fico_mid_treated, [0, 680, 700, 730, 900], labels=["under 680", "680-700", "700-730", "730+"]).astype(str)
rows = []
for seg in ["term", "grade", "purpose_group", "home_ownership_group", "verification_status", "region", "income_band", "loan_band", "fico_band", "job_group"]:
    for lvl, g in v15.groupby(seg):
        if len(g) < 300:
            continue
        k, n, p = int(g.defaulted.sum()), len(g), float(g.pd.mean())
        rows.append({"segment": seg, "level": lvl, "loans": n, "predicted": p, "observed": k / n, "gap": k / n - p, "p_value": binomtest(k, n, p).pvalue})
seg = pd.DataFrame(rows)
seg["holm_p"] = multipletests(seg.p_value, method="holm")[1]
seg["verdict"] = np.where(seg.holm_p < 0.05, np.where(seg.gap > 0, "under-predicts risk", "over-predicts risk"), "calibrated")
seg.round(4).sort_values("gap", key=abs, ascending=False).to_csv(TAB / "calibration_by_segment_2015.csv", index=False)

# 2. the most confident wrong predictions (aggregated: no individual rows are listed)
conf_bad = v15[(v15.defaulted == 1) & (v15.pd < 0.06)]
conf_bad_good = v15[(v15.defaulted == 0) & (v15.pd > 0.40)]
profile_cols = ["term", "annual_inc", "dti_treated", "fico_mid_treated", "revol_util_treated", "credit_history_months_treated", "loan_amnt", "inq_last_6mths_capped"]
prof = pd.DataFrame({"all 2015 loans": v15[profile_cols].median(), "defaulted although PD below 6%": conf_bad[profile_cols].median(),
                     "repaid although PD above 40%": conf_bad_good[profile_cols].median()})
prof.loc["number of loans"] = [len(v15), len(conf_bad), len(conf_bad_good)]
prof.round(2).to_csv(TAB / "confident_errors_profile.csv", index_label="median_of")
miss_by_pd = v15.groupby(pd.cut(v15.pd, [0, .04, .0625, .10, .125, .20, .30, 1], include_lowest=True)).agg(loans=("defaulted", "size"), observed=("defaulted", "mean"), predicted=("pd", "mean")).reset_index()
miss_by_pd["pd_band"] = miss_by_pd.pd.astype(str); miss_by_pd.drop(columns="pd").round(4).to_csv(TAB / "observed_vs_predicted_by_pd_band_2015.csv", index=False)

# 3. behaviour on missing and out-of-range inputs, using a made-up typical applicant
predictor = Predictor(f"{cfg['model_dir']}/pd_model.joblib", cfg["contract_dir"])
base = json.loads(Path("examples/borderline.json").read_text())
base_pd = predictor.predict(base)["pd"]
tests = [("nothing changed", {}), ("employer title missing", {"emp_title": None}), ("employment length missing", {"emp_length": None}), ("no months since last delinquency", {"mths_since_last_delinq": None}),
         ("bank-card utilisation missing", {"bc_util": None}), ("revolving utilisation missing", {"revol_util": None}), ("credit start date missing", {"earliest_cr_line": None}),
         ("income in thousands (58)", {"annual_inc": 58}), ("income $5 million", {"annual_inc": 5_000_000}), ("debt-to-income 120", {"dti": 120}), ("revolving utilisation 400%", {"revol_util": 400}),
         ("FICO low above high", {"fico_range_low": 694, "fico_range_high": 690}), ("state written in full", {"addr_state": "Texas"}), ("date as 2015-06-01", {"issue_d": "2015-06-01"}),
         ("zip code of an unseen area", {"zip_code": "999xx"}), ("loan amount 10x too large", {"loan_amnt": 150000}),
         ("term 48 months (not offered)", {"term": 48}), ("loan amount $500 (below product minimum)", {"loan_amnt": 500}), ("income negative", {"annual_inc": -1}), ("utilisation sent as a fraction (0.55)", {"revol_util": 0.55}),
         ("required field removed (annual_inc)", {"annual_inc": "__drop__"}), ("loan amount sent as text", {"loan_amnt": "abc"})]
out = []
for name, change in tests:
    rec = dict(base)
    for k, v in change.items():
        rec.pop(k) if v == "__drop__" else rec.__setitem__(k, v)
    try:
        r = predictor.predict(rec)
        out.append({"scenario": name, "result": "scored", "pd": r["pd"], "change_vs_base": round(r["pd"] - base_pd, 4), "decision": r["decision"], "message": ""})
    except ContractViolation as e:
        out.append({"scenario": name, "result": "rejected with a clear message", "pd": np.nan, "change_vs_base": np.nan, "decision": "", "message": str(e).splitlines()[1].strip(" -")[:90]})
    except Exception as e:
        out.append({"scenario": name, "result": f"CRASHED: {type(e).__name__}", "pd": np.nan, "change_vs_base": np.nan, "decision": "", "message": str(e)[:80]})
stress = pd.DataFrame(out)
stress.to_csv(TAB / "behaviour_on_missing_and_bad_inputs.csv", index=False)

# 4. holdout: scores and drift against 2015
hold = holdout_frame(cfg, transformer)
hold = add_grade_rank(hold)
hold["pd"] = bundle.score(hold)
Path("outputs").mkdir(exist_ok=True)
hold[["id", "pd"]].assign(pd=lambda t: t.pd.round(6)).to_csv("outputs/holdout_scores.csv", index=False)
hold["decision"] = policy.decide(hold.pd, th)
used = [c.removesuffix("_woe") for c in bundle.model.columns]
# importance = the average size of each input's contribution on 2015 loans (works for any model: scorecard, boosting or their average)
imp = pd.Series(np.abs(bundle.model.contributions(v15.sample(min(3000, len(v15)), random_state=0))).mean(axis=0), index=used).sort_values(ascending=False)
top10 = list(imp.index[:10])
watch = top10 + (["sub_grade_rank"] if "sub_grade_rank" in used and "sub_grade_rank" not in top10 else [])  # the platform score is always monitored (DESIGN.md)
rows = [{"item": "PD distribution", "kind": "model output", "psi": numeric_psi(v15.pd, hold.pd)}]
for c in watch:
    f = numeric_psi if pd.api.types.is_numeric_dtype(hold[c]) else categorical_psi
    rows.append({"item": c, "kind": "top-10 input" if c in top10 else "platform score (always monitored)", "psi": f(v15[c], hold[c])})
drift = pd.DataFrame(rows)
drift["alert"] = drift.psi.map(lambda v: "act (above 0.25)" if v >= .25 else "investigate (0.10 to 0.25)" if v >= .1 else "stable (below 0.10)")
drift.round(4).to_csv(TAB / "holdout_drift_psi.csv", index=False)
cmp_dec = pd.DataFrame({"2015 validation": v15.decision.value_counts(normalize=True), "2016 holdout": hold.decision.value_counts(normalize=True)}).reindex(["approve", "review", "decline"])
cmp_dec.loc["mean PD"] = [v15.pd.mean(), hold.pd.mean()]; cmp_dec.round(4).to_csv(TAB / "decision_mix_2015_vs_2016.csv", index_label="item")
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 0.6, 31)
ax.hist(v15.pd, bins=bins, alpha=.6, density=True, label="2015 validation", color="#2a6f97"); ax.hist(hold.pd, bins=bins, alpha=.6, density=True, label="2016 holdout", color="#c1440e")
ax.axvline(th["approve_below"], color="grey", ls=":"); ax.axvline(th["decline_at_or_above"], color="grey", ls="--"); ax.set_xlabel("Predicted PD"); ax.legend(fontsize=8)
ax.set_title("PD distribution: 2016 holdout against 2015 (dotted: approve, dashed: decline thresholds)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "pd_distribution_2015_vs_2016.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 70)
print(seg[seg.verdict != "calibrated"].round(3)[["segment", "level", "loans", "predicted", "observed", "gap", "holm_p", "verdict"]].to_string(index=False))
print("\nconfident errors:", len(conf_bad), "defaulted below 6% PD;", len(conf_bad_good), "repaid above 40% PD\n", prof.round(1).to_string())
print("\n", stress[["scenario", "result", "pd", "change_vs_base", "message"]].to_string(index=False))
print("\ndrift 2016 vs 2015:\n", drift.round(3).to_string(index=False)); print("\n", cmp_dec.round(3).to_string()); print("\nholdout rows scored:", len(hold), "| mean PD", round(hold.pd.mean(), 3), "| any missing:", int(hold.pd.isna().sum()))
