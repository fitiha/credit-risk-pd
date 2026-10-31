"""A6, part 2: PCA on the treated, standardized fields, and the effect of the outlier treatment.

Uses loans up to the fit cutoff. Writes reports/a5_pca/{figures,tables}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from credit_risk.factor_fields import FACTOR_FIELDS
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer

FIG, TAB = fresh_dirs("a5_pca")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
n, p = len(fit), len(FACTOR_FIELDS)
X = fit[FACTOR_FIELDS]
Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
eig = pca.explained_variance_
cum = np.cumsum(pca.explained_variance_ratio_)

# parallel analysis: eigenvalues of independent noise of the same shape
rng = np.random.default_rng(0)
sim = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1] for _ in range(200)])
noise95 = np.percentile(sim, 95, axis=0)
k_parallel = int(np.argmax(eig <= noise95)) if (eig <= noise95).any() else p
k_kaiser = int((eig > 1).sum())
k_80 = int(np.searchsorted(cum, 0.80) + 1)
rules = pd.DataFrame({"rule": ["Kaiser (eigenvalue above 1)", "Parallel analysis (above 95% of noise)", "Cumulative variance at least 80%", "Cumulative variance at least 70%"],
                      "components": [k_kaiser, k_parallel, k_80, int(np.searchsorted(cum, 0.70) + 1)]})
rules.to_csv(TAB / "number_of_components_rules.csv", index=False)
pd.DataFrame({"component": range(1, p + 1), "eigenvalue": eig, "noise_95th_percentile": noise95,
              "variance_share": pca.explained_variance_ratio_, "cumulative_variance": cum}).round(4).to_csv(TAB / "explained_variance.csv", index=False)

K = k_parallel
load = pd.DataFrame(pca.components_[:K].T * np.sqrt(eig[:K]), index=FACTOR_FIELDS, columns=[f"PC{i + 1}" for i in range(K)])
load.round(3).to_csv(TAB / "loadings.csv", index_label="field")

# effect of the outlier treatment: same fields, cleaned but not capped, logged or filled-by-rule
UNTREATED = {"annual_inc_log": "annual_inc", "loan_amnt_log": "loan_amnt", "installment_log": "installment", "revol_bal_log": "revol_bal",
             "tot_cur_bal_log": "tot_cur_bal", "avg_cur_bal_log": "avg_cur_bal", "total_rev_hi_lim_log": "total_rev_hi_lim",
             "tot_hi_cred_lim_log": "tot_hi_cred_lim", "dti_treated": "dti", "revol_util_treated": "revol_util", "bc_util_treated": "bc_util",
             "delinq_2yrs_capped": "delinq_2yrs", "pub_rec_capped": "pub_rec", "pub_rec_bankruptcies_capped": "pub_rec_bankruptcies",
             "inq_last_6mths_capped": "inq_last_6mths", "mort_acc_capped": "mort_acc", "fico_mid_treated": "fico_mid",
             "credit_history_months_treated": "credit_history_months"}
U = fit[[UNTREATED.get(f, f) for f in FACTOR_FIELDS]].copy()
U = U.fillna(U.median())  # same median fill as the treated version, so only capping and log differ
Zu = StandardScaler().fit_transform(U)
pca_u = PCA().fit(Zu)
sc_t, sc_u = pca.transform(Z)[:, :3], pca_u.transform(Zu)[:, :3]
rows = []
for i in range(3):
    j = int(np.argmax([abs(np.dot(pca.components_[i], pca_u.components_[m])) for m in range(p)]))  # best-matching untreated component
    rows.append({"component": f"PC{i + 1}", "variance_share_treated": pca.explained_variance_ratio_[i], "variance_share_untreated": pca_u.explained_variance_ratio_[j],
                 "loading_similarity (1 = identical)": abs(np.dot(pca.components_[i], pca_u.components_[j])),
                 "score_rank_correlation": abs(spearmanr(sc_t[:, i], sc_u[:, j])[0]),
                 "score_kurtosis_treated": kurtosis(sc_t[:, i]), "score_kurtosis_untreated": kurtosis(sc_u[:, j]),
                 "largest_single_score_in_std_units_treated": np.abs(sc_t[:, i]).max() / sc_t[:, i].std(),
                 "largest_single_score_in_std_units_untreated": np.abs(sc_u[:, j]).max() / sc_u[:, j].std()})
effect = pd.DataFrame(rows)
effect.round(3).to_csv(TAB / "outlier_treatment_effect.csv", index=False)
summary = pd.DataFrame({"version": ["treated and standardized", "untreated, standardized"],
                        "components_by_kaiser": [k_kaiser, int((pca_u.explained_variance_ > 1).sum())],
                        "variance_in_first_3": [cum[2], np.cumsum(pca_u.explained_variance_ratio_)[2]]})
summary.round(3).to_csv(TAB / "treated_vs_untreated_summary.csv", index=False)

# why standardise: PCA on raw scale is dominated by the largest-unit field
raw_scale = PCA().fit(X.to_numpy())
dom = pd.Series(raw_scale.components_[0], index=FACTOR_FIELDS).abs().sort_values(ascending=False)
pd.DataFrame({"first_component_variance_share_without_standardising": [raw_scale.explained_variance_ratio_[0]],
              "field_dominating_it": [dom.index[0]], "its_weight": [dom.iloc[0]]}).round(3).to_csv(TAB / "raw_scale_pca_for_contrast.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(1, p + 1)
ax[0].plot(x, eig, "o-", color="#2a6f97", label="observed"); ax[0].plot(x, noise95, "s--", color="#c1440e", ms=3, label="noise (95th percentile)")
ax[0].axhline(1, color="grey", ls=":", lw=1); ax[0].axvline(K + 0.5, color="black", ls=":", lw=1)
ax[0].set_xlabel("Component"); ax[0].set_ylabel("Eigenvalue"); ax[0].legend(fontsize=8)
ax[0].set_title(f"Scree plot with parallel analysis: keep {K}", fontsize=10)
ax[1].plot(x, cum * 100, "o-", color="#2a6f97"); ax[1].axhline(80, color="grey", ls=":", lw=1); ax[1].axvline(K + 0.5, color="black", ls=":", lw=1)
ax[1].set_xlabel("Number of components"); ax[1].set_ylabel("Cumulative variance explained (%)"); ax[1].set_title("Cumulative variance", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "scree_and_cumulative_variance.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7.5))
im = ax.imshow(load, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(K), load.columns); ax.set_yticks(range(p), load.index, fontsize=7)
for i in range(p):
    for j in range(K):
        if abs(load.iloc[i, j]) >= 0.4:
            ax.text(j, i, f"{load.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, label="loading"); ax.set_title(f"PCA loadings (first {K} components; labels where |loading| >= 0.4)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "pca_loadings.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 20)
print(rules.to_string(index=False)); print("\nvariance by component (first", K + 2, "):", (pca.explained_variance_ratio_[:K + 2] * 100).round(1).tolist(), "| cumulative at K:", round(cum[K - 1] * 100, 1))
for c in load.columns:
    top = load[c].reindex(load[c].abs().sort_values(ascending=False).index).head(5)
    print(c, {k: round(v, 2) for k, v in top.items()})
print("\nOUTLIER TREATMENT EFFECT\n", effect.round(2).T.to_string(header=False)); print(summary.round(3).to_string(index=False))
print("\nraw-scale PCA:", pd.read_csv(TAB / "raw_scale_pca_for_contrast.csv").to_dict("records")[0])
