"""Cap and log-transform the heavy-tailed money fields.

Decision (docs/decisions.md): cap each field at its 0.5th and 99.5th percentile,
then take log(1 + value). Limits are learned only from loans issued up to the
fit cutoff, so 2015 loans cannot influence them, and they are stored in the
transformer so the same rule applies at scoring time.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.io import FIT_CUTOFF

AMOUNT_FIELDS = ["annual_inc", "loan_amnt", "installment", "revol_bal", "tot_cur_bal",
                 "avg_cur_bal", "total_rev_hi_lim", "tot_hi_cred_lim"]


class AmountTreater(BaseEstimator, TransformerMixin):
    """Adds `<field>_capped` (dollars) and `<field>_log` for each money field.

    The lower cap lifts a genuine zero to the floor value, so every field that has
    zeros in the fit window also gets `<field>_is_zero` to keep that fact. A missing
    value stays missing: it is never turned into zero, and how to fill it is
    decided in a later step.
    """

    def __init__(self, fields=None, lower_q=0.005, upper_q=0.995):
        self.fields = fields
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        fields = list(self.fields or AMOUNT_FIELDS)
        pre = X[X["issue_d"] <= FIT_CUTOFF]
        self.fields_ = fields
        self.caps_ = {
            f: (max(float(pre[f].quantile(self.lower_q)), 0.0), float(pre[f].quantile(self.upper_q)))
            for f in fields
        }
        self.zero_fields_ = [f for f in fields if (pre[f] == 0).any()]
        return self

    def transform(self, X):
        d = X.copy()
        for f, (lo, hi) in self.caps_.items():
            d[f"{f}_capped"] = d[f].clip(lo, hi)
            d[f"{f}_log"] = np.log1p(d[f"{f}_capped"])
        for f in self.zero_fields_:
            d[f"{f}_is_zero"] = d[f] == 0  # NaN == 0 is False: missing is not zero
        return d
