"""Application-time features built from cleaned fields.

Every input is known when the applicant applies; nothing from the servicing file is used.
FeatureBuilder creates the raw features. FeatureTreater applies the same rules as the other
ratio fields (cap at p0.5/p99.5, fill missing with the fit-window median) and adds a log
version when a feature is still heavily skewed after capping.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.frames import add_columns
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
        months = ((X["issue_d"].dt.year - X["earliest_cr_line"].dt.year) * 12
                  + (X["issue_d"].dt.month - X["earliest_cr_line"].dt.month))
        return add_columns(X, {
            "credit_history_months": months,  # missing where the credit date was corrupted
            "loan_to_income": X["loan_amnt"] / X["annual_inc"],
            "installment_to_income": X["installment"] / (X["annual_inc"] / 12),
            "revol_bal_to_income": X["revol_bal"] / X["annual_inc"],
            "fico_mid": (X["fico_range_low"] + X["fico_range_high"]) / 2,
            "open_acc_share": X["open_acc"] / X["total_acc"],
            "recent_accounts_share": X["acc_open_past_24mths"] / X["total_acc"],
        })


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
        return add_columns(d, {f"{f}_log": np.log1p(d[f"{f}_treated"]) for f in self.log_fields_})

    def spec_entries(self):
        out = super().spec_entries()
        for f in out:
            out[f]["definition"] = DEFINITIONS.get(f, "")
            out[f]["log1p"] = f in self.log_fields_
        return out
