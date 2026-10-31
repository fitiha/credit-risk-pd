"""PCA components and factor scores, fitted on loans up to the fit cutoff.

Fields are standardized first (never raw scale). Factor analysis uses maximum likelihood with a varimax rotation.
Each factor is named from its loadings: it takes the name whose signature fields it loads on most, and stays
unnamed if no signature fits.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

from credit_risk.factor_fields import FACTOR_FIELDS
from credit_risk.io import FIT_CUTOFF

# What each named factor is made of. A fitted factor receives a name only if it loads strongly on these fields.
SIGNATURES = {
    "credit_capacity": ["avg_cur_bal_log", "tot_cur_bal_log", "tot_hi_cred_lim_log"],
    "loan_size": ["installment_log", "loan_amnt_log"],
    "revolving_credit_size": ["total_rev_hi_lim_log", "revol_bal_log"],
    "account_count": ["open_acc", "total_acc"],
    "revolving_utilization": ["revol_util_treated", "bc_util_treated"],
    "public_records": ["pub_rec_capped", "pub_rec_bankruptcies_capped"],
    "recent_account_opening": ["num_tl_op_past_12m", "acc_open_past_24mths"],
}
MIN_MEAN_LOADING = 0.5


def name_factors(loadings, signatures=SIGNATURES, min_mean_loading=MIN_MEAN_LOADING):
    """Assign each factor (column of a loadings DataFrame) the signature it matches best; None if it matches none."""
    names = list(signatures)
    fit_matrix = np.array([[loadings.loc[signatures[n], c].abs().mean() for n in names] for c in loadings.columns])
    rows, cols = linear_sum_assignment(-fit_matrix)
    out = {}
    for r, c in zip(rows, cols):
        out[loadings.columns[r]] = names[c] if fit_matrix[r, c] >= min_mean_loading else None
    return out


class FactorModel(BaseEstimator, TransformerMixin):
    """Adds `pc_1..pc_k` (PCA components) and `factor_<name>` (factor scores)."""

    def __init__(self, fields=None, n_components=7, n_factors=7, signatures=None):
        self.fields = fields
        self.n_components = n_components
        self.n_factors = n_factors
        self.signatures = signatures

    def fit(self, X, y=None):
        fields = list(self.fields or FACTOR_FIELDS)
        sig = self.signatures or SIGNATURES
        pre = X[X["issue_d"] <= FIT_CUTOFF]
        self.fields_ = fields
        self.scaler_ = StandardScaler().fit(pre[fields])
        Z = self.scaler_.transform(pre[fields])
        self.pca_ = PCA(n_components=self.n_components, random_state=0).fit(Z)
        self.pc_sign_ = np.array([1.0 if c[np.abs(c).argmax()] > 0 else -1.0 for c in self.pca_.components_])
        self.fa_ = FactorAnalysis(n_components=self.n_factors, rotation="varimax", random_state=0, max_iter=2000).fit(Z)
        L = pd.DataFrame(self.fa_.components_.T, index=fields, columns=range(self.n_factors))
        self.factor_sign_ = np.array([1.0 if L[c].iloc[L[c].abs().argmax()] > 0 else -1.0 for c in L.columns])
        L = L * self.factor_sign_
        names = name_factors(L, sig)
        order = sorted(range(self.n_factors), key=lambda c: (names[c] is None, list(sig).index(names[c]) if names[c] else c))
        self.factor_order_ = order
        self.factor_names_ = [names[c] or f"unnamed_{c + 1}" for c in order]
        self.loadings_ = L[order].set_axis(self.factor_names_, axis=1)
        self.factor_variance_ = ((self.loadings_ ** 2).sum() / len(fields)).to_dict()
        return self

    def transform(self, X):
        d = X.copy()
        Z = self.scaler_.transform(d[self.fields_])
        pcs = self.pca_.transform(Z) * self.pc_sign_
        scores = (self.fa_.transform(Z) * self.factor_sign_)[:, self.factor_order_]
        for i in range(pcs.shape[1]):
            d[f"pc_{i + 1}"] = pcs[:, i]
        for j, name in enumerate(self.factor_names_):
            d[f"factor_{name}"] = scores[:, j]
        return d

    def spec_entries(self):
        top = lambda s, k=3: {f: round(float(v), 3) for f, v in s.reindex(s.abs().sort_values(ascending=False).index).head(k).items()}
        pc_load = pd.DataFrame(self.pca_.components_.T * np.sqrt(self.pca_.explained_variance_) * self.pc_sign_, index=self.fields_)
        return {
            "fields": self.fields_,
            "pca": [{"name": f"pc_{i + 1}", "method": "pca_standardized", "variance_explained": round(float(v), 4),
                     "top_loadings": top(pc_load[i])} for i, v in enumerate(self.pca_.explained_variance_ratio_)],
            "factors": [{"name": n, "method": "factor_analysis_varimax", "top_loadings": top(self.loadings_[n]),
                         "variance_explained": round(float(self.factor_variance_[n]), 4)} for n in self.factor_names_],
        }
