"""A7, part 4: could zip_code or addr_state act as a proxy for protected characteristics, and what do they add?

The data holds no race, sex or age, so a proxy cannot be measured directly. What can be measured: how much location
tells us about income and risk, how sparse it is, and what it adds to a model. Writes reports/a6_location_proxy/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from credit_risk.factor_fields import FACTOR_FIELDS
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer
from credit_risk.woe import WoEEncoder

FIG, TAB = fresh_dirs("a6_location_proxy")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
fit["zip3"] = fit.zip_code.str[:3]

# 1. how sparse is location?
zc = fit.zip3.value_counts()
sparse = pd.Series({"loans": len(fit), "distinct states": fit.addr_state.nunique(), "distinct 3-digit zip prefixes": len(zc),
                    "median loans per zip prefix": zc.median(), "zip prefixes with under 30 loans (%)": 100 * (zc < 30).mean(),
                    "loans in zip prefixes with under 30 loans (%)": 100 * zc[zc < 30].sum() / zc.sum()}, name="value")
sparse.round(1).to_csv(TAB / "location_sparsity.csv", index_label="measure")

# 2. does location carry socio-economic information?
big = fit.addr_state.map(fit.addr_state.value_counts()).ge(300)
S = fit[big]
st = S.groupby("addr_state").agg(loans=("id", "size"), median_income=("annual_inc", "median"), median_dti=("dti", "median"),
                                  median_fico=("fico_mid", "median"), default_rate=("defaulted", "mean")).sort_values("median_income", ascending=False)
st.round(3).to_csv(TAB / "state_profile_states_with_300_loans.csv", index_label="state")
groups = [g.annual_inc.to_numpy() for _, g in S.groupby("addr_state")]
eta2 = np.nan
grand = S.annual_inc.mean()
eta2 = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / ((S.annual_inc - grand) ** 2).sum()
summary = pd.Series({"states with at least 300 loans": S.addr_state.nunique(),
                     "share of income variation explained by state (eta squared)": eta2,
                     "Kruskal-Wallis p, income differs by state": kruskal(*groups).pvalue,
                     "median income, highest state": st.median_income.max(), "median income, lowest state": st.median_income.min(),
                     "default rate, highest state": st.default_rate.max(), "default rate, lowest state": st.default_rate.min()}, name="value")
summary.to_csv(TAB / "state_income_and_risk_summary.csv", index_label="measure", float_format="%.4g")

# 3. what location adds to a model, on later loans (time-ordered folds before 2015)
folds = [(fit.issue_d < "2014-01-01", fit.issue_d.between("2014-01-01", "2014-06-30")), (fit.issue_d <= "2014-06-30", fit.issue_d >= "2014-07-01")]
rng = np.random.default_rng(0)
rows = []
for k, (mtr, mte) in enumerate(folds):
    tr, te = fit[mtr].copy(), fit[mte].copy()
    enc = WoEEncoder(categorical=["addr_state_group"]).fit(tr)
    tr, te = enc.transform(tr), enc.transform(te)
    res = {}
    for name, cols in {"without location": FACTOR_FIELDS + ["term_60"], "with state": FACTOR_FIELDS + ["term_60", "addr_state_group_woe"]}.items():
        for df in (tr, te):
            df["term_60"] = (df.term == 60).astype(float)
        sc = StandardScaler().fit(tr[cols])
        p = LogisticRegression(max_iter=3000).fit(sc.transform(tr[cols]), tr.defaulted).predict_proba(sc.transform(te[cols]))[:, 1]
        res[name] = p
    yv = te.defaulted.to_numpy()
    diffs = [roc_auc_score(yv[i], res["with state"][i]) - roc_auc_score(yv[i], res["without location"][i]) for i in (rng.integers(0, len(yv), len(yv)) for _ in range(300))]
    rows.append({"fold": f"fold {k + 1}", "auc_without_location": roc_auc_score(yv, res["without location"]), "auc_with_state": roc_auc_score(yv, res["with state"]),
                 "gain": roc_auc_score(yv, res["with state"]) - roc_auc_score(yv, res["without location"]), "gain_ci_low": np.percentile(diffs, 2.5), "gain_ci_high": np.percentile(diffs, 97.5)})
add = pd.DataFrame(rows)
add.round(4).to_csv(TAB / "what_state_adds_to_a_model.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(st.median_income / 1000, st.default_rate * 100, s=st.loans / 8, color="#2a6f97", alpha=0.7)
for s, r in st.iterrows():
    ax.annotate(s, (r.median_income / 1000, r.default_rate * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.axhline(fit.defaulted.mean() * 100, color="grey", ls="--", lw=1)
ax.set_xlabel("Median stated income of the state's borrowers ($ thousands)"); ax.set_ylabel("Default rate (%)")
ax.set_title("States with at least 300 loans (dot size = loans)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "state_income_vs_default.png", dpi=140); plt.close(fig)
pd.set_option("display.width", 200)
print(sparse.round(1).to_string()); print(summary.to_string()); print(add.round(4).to_string(index=False))
