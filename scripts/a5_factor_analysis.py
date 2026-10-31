"""A6, part 3: factor analysis of the treated, standardized fields.

Suitability (KMO, Bartlett), number of factors (parallel analysis on the reduced correlation matrix), rotation
(varimax against promax, decided by the factor correlations), loadings, factor scores and stability.
Uses loans up to the fit cutoff. Writes reports/a5_factor_analysis/{figures,tables}.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from credit_risk.factor_fields import FACTOR_FIELDS
from credit_risk.factors import name_factors
from credit_risk.factor_tools import bartlett_sphericity, kmo, parallel_analysis, promax, tucker_congruence
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer

warnings.filterwarnings("ignore")
FIG, TAB = fresh_dirs("a5_factor_analysis")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
n, p = len(fit), len(FACTOR_FIELDS)
Z = pd.DataFrame(StandardScaler().fit_transform(fit[FACTOR_FIELDS]), columns=FACTOR_FIELDS)

# 1. suitability
chi2, bart_p = bartlett_sphericity(Z.to_numpy())
kmo_each, kmo_all = kmo(Z.to_numpy())
pd.DataFrame({"field": FACTOR_FIELDS, "kmo": kmo_each}).round(3).to_csv(TAB / "kmo_by_field.csv", index=False)
pd.DataFrame({"test": ["Bartlett chi-square", "Bartlett p-value", "KMO overall"], "value": [chi2, bart_p, kmo_all]}).round(4).to_csv(TAB / "suitability.csv", index=False)

# 2. number of factors: parallel analysis on the reduced correlation matrix (communalities on the diagonal)
k_pa, obs, noise95 = parallel_analysis(Z.to_numpy())
k_kaiser = int((np.linalg.eigvalsh(np.corrcoef(Z.to_numpy(), rowvar=False)) > 1).sum())
pd.DataFrame({"factor": range(1, p + 1), "reduced_eigenvalue": obs, "noise_95th_percentile": noise95}).round(4).to_csv(TAB / "parallel_analysis.csv", index=False)
# choice rule: the largest K for which every factor is a real, readable and stable factor:
# at least two fields with |loading| >= 0.4, at least 3% of the variance, and congruence >= 0.95 between 2013 and 2014 fits
def fit_fa_k(z, k):
    return FactorAnalysis(n_components=k, rotation="varimax", random_state=0, max_iter=2000).fit(z)

a13, a14 = (fit.issue_d.dt.year == 2013).to_numpy(), (fit.issue_d.dt.year == 2014).to_numpy()
rows = []
for k in range(4, k_pa + 1):
    Lk = fit_fa_k(Z, k).components_.T
    salient = (np.abs(Lk) >= 0.4).sum(axis=0)
    share = (Lk ** 2).sum(axis=0) / p
    cong = tucker_congruence(fit_fa_k(Z[a13], k).components_.T, fit_fa_k(Z[a14], k).components_.T)
    rows.append({"factors": k, "min_salient_loadings_per_factor": int(salient.min()), "min_variance_share": share.min(),
                 "total_variance": share.sum(), "min_congruence_2013_vs_2014": cong.min(),
                 "passes_rule": bool(salient.min() >= 2 and share.min() >= 0.03 and cong.min() >= 0.95)})
sweep_tbl = pd.DataFrame(rows)
sweep_tbl.round(3).to_csv(TAB / "number_of_factors_rule.csv", index=False)
K = int(sweep_tbl[sweep_tbl.passes_rule].factors.max())

# 3. rotation: ML factor analysis with varimax; then an oblique (promax) rotation to see whether the factors are correlated
def fit_fa(z):
    return FactorAnalysis(n_components=K, rotation="varimax", random_state=0, max_iter=2000).fit(z)

fa = fit_fa(Z)
loadings = fa.components_.T
_, phi_arr = promax(loadings)
phi = pd.DataFrame(phi_arr, columns=[f"F{i + 1}" for i in range(K)], index=[f"F{i + 1}" for i in range(K)])
phi.round(3).to_csv(TAB / "factor_correlations_promax.csv")
off = np.abs(phi_arr[np.triu_indices(K, 1)])
L = pd.DataFrame(loadings, index=FACTOR_FIELDS, columns=[f"F{i + 1}" for i in range(K)])
sign = np.array([1.0 if L[c].iloc[L[c].abs().argmax()] > 0 else -1.0 for c in L.columns])  # largest loading of each factor is positive
L = L * sign
comm = (L ** 2).sum(axis=1)
ss = (L ** 2).sum()
var = pd.DataFrame({"factor": L.columns, "variance_explained": ss.to_numpy() / p, "cumulative": np.cumsum(ss.to_numpy() / p)})
L.round(3).assign(communality=comm.round(3)).to_csv(TAB / "loadings_varimax.csv", index_label="field")
var.round(4).to_csv(TAB / "variance_explained.csv", index=False)
cross = ((L.abs() >= 0.4).sum(axis=1) > 1).sum()

# 4. scores and stability
scores = pd.DataFrame(fa.transform(Z) * sign, columns=L.columns)
y = fit.defaulted
aucs = pd.DataFrame({"factor": L.columns, "auc_alone": [max(roc_auc_score(y, scores[c]), 1 - roc_auc_score(y, scores[c])) for c in L.columns],
                     "correlation_with_default": [scores[c].corr(y) for c in L.columns]})
aucs.round(3).to_csv(TAB / "factor_scores_vs_default.csv", index=False)
stab = pd.DataFrame({"factor": L.columns, "congruence_2013_vs_2014": tucker_congruence(fit_fa(Z[a13]).components_.T, fit_fa(Z[a14]).components_.T)})
stab.round(3).to_csv(TAB / "stability_2013_vs_2014.csv", index=False)

# charts
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(1, p + 1)
ax[0].plot(x, obs, "o-", color="#2a6f97", label="observed (reduced)"); ax[0].plot(x, noise95, "s--", color="#c1440e", ms=3, label="noise (95th percentile)")
ax[0].axhline(0, color="grey", lw=0.8); ax[0].axvline(K + 0.5, color="black", ls=":", lw=1)
ax[0].set_xlabel("Factor"); ax[0].set_ylabel("Eigenvalue"); ax[0].legend(fontsize=8); ax[0].set_title(f"Parallel analysis: keep {K} factors", fontsize=10)
ax[1].barh(FACTOR_FIELDS[::-1], kmo_each[::-1], color="#2a6f97")
ax[1].axvline(0.6, color="grey", ls=":", lw=1); ax[1].tick_params(axis="y", labelsize=7); ax[1].set_xlabel("KMO (0.6 is the usual minimum)")
ax[1].set_title(f"Suitability: overall KMO {kmo_all:.2f}", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "parallel_analysis_and_kmo.png", dpi=140); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 7.5))
factor_names = name_factors(L)
im = ax.imshow(L, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(K), [f"{c}: {factor_names[c] or 'unnamed'}" for c in L.columns], rotation=40, ha="right", fontsize=7); ax.set_yticks(range(p), L.index, fontsize=7)
for i in range(p):
    for j in range(K):
        if abs(L.iloc[i, j]) >= 0.3:
            ax.text(j, i, f"{L.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, label="loading"); ax.set_title(f"Factor loadings after varimax rotation (labels where |loading| >= 0.3)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "factor_loadings_varimax.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 250)
print(f"Bartlett chi2={chi2:,.0f} p={bart_p:.3g} | KMO overall={kmo_all:.3f} | fields with KMO<0.6: {[f for f, k in zip(FACTOR_FIELDS, kmo_each) if k < 0.6]}")
print(f"factors: parallel analysis {k_pa}, Kaiser {k_kaiser}, chosen by rule {K}\n", sweep_tbl.round(3).to_string(index=False))
print("KMO by field:", dict(zip(FACTOR_FIELDS, np.round(kmo_each, 2))))
print("promax factor correlations: max |r| =", round(off.max(), 3), "| mean |r| =", round(off.mean(), 3))
print("variance explained:", var.round(3).to_dict("list")["variance_explained"], "cumulative:", round(var.cumulative.iloc[-1], 3), "| fields loading >=0.4 on more than one factor:", int(cross))
for c in L.columns:
    top = L[c].reindex(L[c].abs().sort_values(ascending=False).index).head(5)
    print(c, {k: round(v, 2) for k, v in top.items()})
print("\ncommunalities min/max:", round(comm.min(), 2), round(comm.max(), 2), "| lowest:", comm.sort_values().head(4).round(2).to_dict())
print("\nscores vs default:\n", aucs.round(3).to_string(index=False)); print("\nstability 2013 vs 2014 (Tucker congruence, >0.95 = same factor):\n", stab.round(3).to_string(index=False))
