"""Cap and log-transform the heavy-tailed money fields.

Decision (docs/decisions.md): cap each field at its 0.5th and 99.5th percentile,
then take log(1 + value). Limits are learned only from loans issued up to the
fit cutoff, so 2015 loans cannot influence them, and they are stored in the
transformer so the same rule applies at scoring time.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.frames import add_columns
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
        new = {}
        for f, (lo, hi) in self.caps_.items():
            capped = X[f].clip(lo, hi)
            new[f"{f}_capped"] = capped
            new[f"{f}_log"] = np.log1p(capped)
        for f in self.zero_fields_:
            new[f"{f}_is_zero"] = X[f] == 0  # NaN == 0 is False: missing is not zero
        return add_columns(X, new)

    def spec_entries(self):
        """Per-field rule for data_spec.json, so scoring applies the same limits."""
        return {
            f: {"cap_low": lo, "cap_high": hi, "log1p": True, "zero_indicator": f in self.zero_fields_,
                "cleaning": f"capped at p{self.lower_q * 100:g} = {lo:,.0f} and p{self.upper_q * 100:g} = {hi:,.0f}, then log(1 + x)",
                "missing_strategy": "left missing (none in the fit window)"}
            for f, (lo, hi) in self.caps_.items()
        }


RATIO_FIELDS = ["dti", "revol_util", "bc_util"]
COUNT_FIELDS = ["delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "inq_last_6mths", "mort_acc"]


class RatioTreater(BaseEstimator, TransformerMixin):
    """Cap ratio fields at their 0.5th and 99.5th percentile and fill missing values.

    Adds `<field>_treated` (capped, then missing filled with the fit-window median).
    A ratio above 100% after a limit cut is a genuine extreme, so it is capped, not
    removed. Fields with at least `min_missing_share` of values missing in the fit
    window also get `<field>_missing`, because missingness may carry information.
    """

    def __init__(self, fields=None, lower_q=0.005, upper_q=0.995, min_missing_share=0.01):
        self.fields = fields
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.min_missing_share = min_missing_share

    def fit(self, X, y=None):
        fields = list(self.fields or RATIO_FIELDS)
        pre = X[X["issue_d"] <= FIT_CUTOFF]
        self.fields_ = fields
        self.caps_ = {f: (float(pre[f].quantile(self.lower_q)), float(pre[f].quantile(self.upper_q)))
                      for f in fields}
        self.fill_ = {f: float(pre[f].clip(*self.caps_[f]).median()) for f in fields}
        self.flag_fields_ = [f for f in fields if pre[f].isna().mean() >= self.min_missing_share]
        return self

    def transform(self, X):
        new = {f"{f}_treated": X[f].clip(lo, hi).fillna(self.fill_[f]) for f, (lo, hi) in self.caps_.items()}
        for f in self.flag_fields_:
            new[f"{f}_missing"] = X[f].isna()
        return add_columns(X, new)

    def spec_entries(self):
        return {
            f: {"cap_low": lo, "cap_high": hi, "fill_value": self.fill_[f], "missing_indicator": f in self.flag_fields_,
                "cleaning": f"capped at p{self.lower_q * 100:g} = {lo:,.1f} and p{self.upper_q * 100:g} = {hi:,.1f}",
                "missing_strategy": f"median of the fit window ({self.fill_[f]:,.1f})"
                                    + (" plus a missing indicator" if f in self.flag_fields_ else "")}
            for f, (lo, hi) in self.caps_.items()
        }


class CountTreater(BaseEstimator, TransformerMixin):
    """Cap long-tailed count fields at their 99.5th percentile (zeros stay zero)."""

    def __init__(self, fields=None, upper_q=0.995):
        self.fields = fields
        self.upper_q = upper_q

    def fit(self, X, y=None):
        pre = X[X["issue_d"] <= FIT_CUTOFF]
        self.fields_ = list(self.fields or COUNT_FIELDS)
        self.caps_ = {f: float(pre[f].quantile(self.upper_q)) for f in self.fields_}
        return self

    def transform(self, X):
        return add_columns(X, {f"{f}_capped": X[f].clip(upper=hi) for f, hi in self.caps_.items()})

    def spec_entries(self):
        return {f: {"cap_high": hi, "cleaning": f"capped at p{self.upper_q * 100:g} = {hi:g}; zeros stay zero"}
                for f, hi in self.caps_.items()}
