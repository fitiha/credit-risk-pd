"""A6 robustness: does the factor solution depend on a few fields, and what do the excluded fields add?

1. Refit the factor model without or with single fields and compare with the final model.
2. How much of each left-out ratio feature is already explained by the included fields (R-squared).
3. How closely do factor scores follow the PCA components (they are different constructs).
Uses loans up to the fit cutoff. Writes reports/a5_factor_robustness/tables.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from credit_risk.factor_fields import FACTOR_FIELDS
from credit_risk.factor_tools import kmo, tucker_congruence
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer

warnings.filterwarnings("ignore")
_, TAB = fresh_dirs("a5_factor_robustness")
raw = load_applications()
tr = ApplicationTransformer().fit(raw)
d = tr.transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
K = tr.step("factors").n_factors


def fit_fa(cols):
    z = StandardScaler().fit_transform(fit[cols])
    f = FactorAnalysis(n_components=K, rotation="varimax", random_state=0, max_iter=2000).fit(z)
    return pd.DataFrame(f.components_.T, index=cols), f.noise_variance_, kmo(z)[1]


base_L, base_u, base_kmo = fit_fa(FACTOR_FIELDS)
rows = [{"variant": "final model (21 fields)", "fields": len(FACTOR_FIELDS), "smallest_unique_variance": base_u.min(), "kmo_overall": base_kmo, "min_agreement_with_final": 1.0}]
variants = {"add back tot_cur_bal_log (22 fields)": FACTOR_FIELDS + ["tot_cur_bal_log"],
            "drop delinq_2yrs_capped (KMO below 0.5)": [c for c in FACTOR_FIELDS if c != "delinq_2yrs_capped"],
            "drop installment_log": [c for c in FACTOR_FIELDS if c != "installment_log"],
            "drop tot_hi_cred_lim_log": [c for c in FACTOR_FIELDS if c != "tot_hi_cred_lim_log"]}
for name, cols in variants.items():
    L, u, k = fit_fa(cols)
    common = [c for c in FACTOR_FIELDS if c in cols]
    rows.append({"variant": name, "fields": len(cols), "smallest_unique_variance": u.min(), "kmo_overall": k,
                 "min_agreement_with_final": tucker_congruence(base_L.loc[common].to_numpy(), L.loc[common].to_numpy()).min()})
pd.DataFrame(rows).round(3).to_csv(TAB / "field_set_variants.csv", index=False)

Z = StandardScaler().fit_transform(fit[FACTOR_FIELDS])
LEFT_OUT = ["loan_to_income_treated", "installment_to_income_treated", "revol_bal_to_income_log", "open_acc_share_treated",
            "recent_accounts_share_treated", "int_rate", "tot_cur_bal_log"]
r2 = pd.DataFrame({"field_left_out": LEFT_OUT, "r_squared_on_included_fields": [LinearRegression().fit(Z, fit[f]).score(Z, fit[f]) for f in LEFT_OUT]})
r2.round(3).to_csv(TAB / "left_out_fields_explained_by_included.csv", index=False)

F = fit[[c for c in fit.columns if c.startswith("factor_")]]
P = fit[[c for c in fit.columns if c.startswith("pc_")]]
agree = pd.DataFrame({"factor": [c.replace("factor_", "") for c in F], "best_abs_correlation_with_a_pca_component": [P.corrwith(F[c]).abs().max() for c in F],
                      "best_matching_component": [P.corrwith(F[c]).abs().idxmax() for c in F]})
agree.round(3).to_csv(TAB / "factor_vs_pca_agreement.csv", index=False)
off = np.abs(np.corrcoef(F.to_numpy().T) - np.eye(F.shape[1])).max()
print(pd.DataFrame(rows).round(3).to_string(index=False)); print("\n", r2.round(2).to_string(index=False)); print("\n", agree.round(2).to_string(index=False))
print("\nlargest correlation between two factor scores:", round(float(off), 2))
