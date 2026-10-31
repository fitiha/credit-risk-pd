"""Application-time features built from cleaned fields.

Every input is known when the applicant applies; nothing from the servicing file is used.
FeatureBuilder creates the raw features. FeatureTreater applies the same rules as the other
ratio fields (cap at p0.5/p99.5, fill missing with the fit-window median) and adds a log
version when a feature is still heavily skewed after capping.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.io import FIT_CUTOFF
from credit_risk.outliers import RatioTreater

FEATURE_FIELDS = ["credit_history_months", "loan_to_income", "installment_to_income",
                  "revol_bal_to_income", "fico_mid", "open_acc_share", "recent_accounts_share"]

DEFINITIONS = {
    "credit_history_months": "months from the first credit line to the loan issue date",
    "loan_to_income": "loan amount divided by annual income",
    "installment_to_income": "monthly installment divided by monthly income",
    "revol_bal_to_income": "revolving balance divided by annual income",
    "fico_mid": "midpoint of the FICO range",
    "open_acc_share": "open accounts divided by total accounts",
    "recent_accounts_share": "accounts opened in the past 24 months divided by total accounts",
}


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds the raw application-time features. Nothing is learned from data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        months = ((d["issue_d"].dt.year - d["earliest_cr_line"].dt.year) * 12
                  + (d["issue_d"].dt.month - d["earliest_cr_line"].dt.month))
        d["credit_history_months"] = months  # missing where the credit date was corrupted
        d["loan_to_income"] = d["loan_amnt"] / d["annual_inc"]
        d["installment_to_income"] = d["installment"] / (d["annual_inc"] / 12)
        d["revol_bal_to_income"] = d["revol_bal"] / d["annual_inc"]
        d["fico_mid"] = (d["fico_range_low"] + d["fico_range_high"]) / 2
        d["open_acc_share"] = d["open_acc"] / d["total_acc"]
        d["recent_accounts_share"] = d["acc_open_past_24mths"] / d["total_acc"]
        return d


class FeatureTreater(RatioTreater):
    """Cap and fill the features; add `<feature>_log` when the capped skewness exceeds `log_skew_above`."""

    def __init__(self, fields=None, lower_q=0.005, upper_q=0.995, min_missing_share=0.01, log_skew_above=2.0):
        super().__init__(fields=fields or FEATURE_FIELDS, lower_q=lower_q, upper_q=upper_q,
                         min_missing_share=min_missing_share)
        self.log_skew_above = log_skew_above

    def fit(self, X, y=None):
        super().fit(X)
        pre = X[X["issue_d"] <= FIT_CUTOFF]
        self.log_fields_ = [f for f in self.fields_
                            if pre[f].clip(*self.caps_[f]).dropna().skew() > self.log_skew_above]
        return self

    def transform(self, X):
        d = super().transform(X)
        for f in self.log_fields_:
            d[f"{f}_log"] = np.log1p(d[f"{f}_treated"])
        return d

    def spec_entries(self):
        out = super().spec_entries()
        for f in out:
            out[f]["definition"] = DEFINITIONS.get(f, "")
            out[f]["log1p"] = f in self.log_fields_
        return out
