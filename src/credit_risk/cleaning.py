"""Fix the data problems listed in docs/issue_log.md.

Each step below carries the issue number it implements. Only the fitted
values (the sub-grade median rates) depend on data; they are learned from
loans issued up to the fit cutoff so 2015 loans cannot influence them.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.io import FIT_CUTOFF, parse_dates

STATE_NAMES = {
    "california": "CA", "florida": "FL", "georgia": "GA", "illinois": "IL",
    "michigan": "MI", "new jersey": "NJ", "new york": "NY",
    "north carolina": "NC", "ohio": "OH", "pennsylvania": "PA",
    "texas": "TX", "virginia": "VA",
}
LEAKY_COLUMNS = ["bureau_recovery_flag"]  # issue 7: copy of a servicing column


class Cleaner(BaseEstimator, TransformerMixin):
    """Turn a raw application+bureau table into a cleaned one."""

    def __init__(self, income_thousands_below=1000, placeholder_rate=6.0,
                 amount_ratio=10.0, amount_ratio_tol=0.5, balance_ratio_range=(8.0, 20.0), balance_min_util=1.0, balance_over_total=1.5):
        self.income_thousands_below = income_thousands_below
        self.placeholder_rate = placeholder_rate
        self.amount_ratio = amount_ratio
        self.amount_ratio_tol = amount_ratio_tol
        self.balance_ratio_range = balance_ratio_range
        self.balance_over_total = balance_over_total
        self.balance_min_util = balance_min_util

    def fit(self, X, y=None):
        """Learn median interest rate per sub_grade from pre-cutoff loans."""
        issue = parse_dates(X["issue_d"])
        pre = X[(issue <= FIT_CUTOFF) & (X["int_rate"] != self.placeholder_rate)]
        self.rate_by_subgrade_ = pre.groupby("sub_grade")["int_rate"].median().to_dict()
        self.rate_by_grade_ = pre.groupby("grade")["int_rate"].median().to_dict()
        return self

    def transform(self, X):
        d = X.copy()
        d = d.drop(columns=[c for c in LEAKY_COLUMNS if c in d])

        # issues 3, 4: mixed date formats
        d["issue_d"] = parse_dates(d["issue_d"])
        d["earliest_cr_line"] = parse_dates(d["earliest_cr_line"])
        # issue 6: " 36 months" -> 36
        d["term"] = d["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        # issue 5: state codes
        state = d["addr_state"].astype(str).str.strip()
        d["addr_state"] = state.str.lower().map(STATE_NAMES).fillna(state.str.upper())
        # issue 2: job title case and spaces (grouping happens in A5)
        title = d["emp_title"].astype("object")  # an all-empty column arrives as float and has no text methods
        d["emp_title"] = title.where(title.isna(), title.astype(str).str.strip().str.lower())

        # issue 8: loan amount entered 10x too large. The installment is right,
        # so the amortised payment implied by the amount is ~10x the installment.
        r = d["int_rate"] / 1200
        implied = d["loan_amnt"] * r / (1 - (1 + r) ** -d["term"])
        ratio = implied / d["installment"]
        shifted = (ratio - self.amount_ratio).abs() < self.amount_ratio_tol
        d["loan_amnt_fixed"] = shifted
        d.loc[shifted, "loan_amnt"] = d.loc[shifted, "loan_amnt"] / 10

        # issue 38: revolving balance 10x too large. The utilisation and the limit are right, so the utilisation implied
        # by balance / limit is a multiple of the reported one: 9-11x for most, up to about 16x because the bureau reports
        # utilisation at a slightly different moment. Genuine loans sit at 0.8-1.25x. Needs a utilisation above 1% and a
        # nonzero limit to be checkable.
        limit = d["total_rev_hi_lim"].where(d["total_rev_hi_lim"] > 0)
        implied_ratio = (100 * d["revol_bal"] / limit) / d["revol_util"].abs().where(d["revol_util"].abs() > self.balance_min_util)
        low, high = self.balance_ratio_range
        big_balance = implied_ratio.between(low, high)
        # second, independent check: a revolving balance cannot exceed the total current balance (which includes it)
        total = d["tot_cur_bal"].where(d["tot_cur_bal"] > 0)
        big_balance |= (d["revol_bal"] > self.balance_over_total * total) & (d["revol_bal"] / 10 <= self.balance_over_total * total)
        d["revol_bal_fixed"] = big_balance
        d.loc[big_balance, "revol_bal"] = d.loc[big_balance, "revol_bal"] / 10

        # issue 10: income typed in thousands
        small = d["annual_inc"] < self.income_thousands_below
        d["annual_inc_fixed"] = small
        d.loc[small, "annual_inc"] = d.loc[small, "annual_inc"] * 1000

        # issue 12: FICO low and high swapped
        swap = d["fico_range_low"] > d["fico_range_high"]
        lo, hi = d["fico_range_low"].copy(), d["fico_range_high"].copy()
        d.loc[swap, "fico_range_low"], d.loc[swap, "fico_range_high"] = hi[swap], lo[swap]

        # issue 13: credit history cannot start after the loan -> missing + flag
        bad_date = d["earliest_cr_line"] > d["issue_d"]
        d["earliest_cr_line_invalid"] = bad_date
        d.loc[bad_date, "earliest_cr_line"] = pd.NaT

        # issue 14: revol_util sign flip is provable (fix); bc_util is not (missing)
        d["revol_util"] = d["revol_util"].abs()
        bad_bc = d["bc_util"] < 0
        d["bc_util_invalid"] = bad_bc
        d.loc[bad_bc, "bc_util"] = np.nan

        # issue 39: "0 months since the last delinquency" with no delinquency in the last two years is how the historical files
        # code "none on record" (the 2016 holdout leaves it blank), so it becomes missing; a real zero (a delinquency this month,
        # delinq_2yrs > 0) is kept
        if {"mths_since_last_delinq", "delinq_2yrs"} <= set(d.columns):
            zero_code = (d["mths_since_last_delinq"] == 0) & (d["delinq_2yrs"] == 0)
            d["mths_since_last_delinq_recoded"] = zero_code
            d.loc[zero_code, "mths_since_last_delinq"] = np.nan

        # issue 19: placeholder interest rate -> median of the sub_grade
        ph = d["int_rate"] == self.placeholder_rate
        fill = d["sub_grade"].map(self.rate_by_subgrade_).fillna(d["grade"].map(self.rate_by_grade_))
        d["int_rate_imputed"] = ph
        d.loc[ph, "int_rate"] = fill[ph]
        return d

    def spec_rules(self):
        """Thresholds and fitted values for data_spec.json, so scoring repeats these fixes."""
        return {
            "annual_inc": f"values below {self.income_thousands_below:,} treated as thousands (x1000)",
            "loan_amnt": f"if the payment implied by amount, rate and term is {self.amount_ratio:g}x the installment "
                         f"(within {self.amount_ratio_tol:g}), the amount is divided by 10",
            "int_rate": f"exactly {self.placeholder_rate:g} is a placeholder, replaced by the median rate of the sub_grade",
            "revol_bal": f"if balance / limit is {self.balance_ratio_range[0]:g} to {self.balance_ratio_range[1]:g} times the reported utilisation, or the balance exceeds {self.balance_over_total:g}x the total current balance and a tenth of it fits, the balance is divided by 10",
            "revol_util": "negative values: sign removed",
            "bc_util": "negative values: set to missing, indicator added",
            "fico_range": "low and high swapped when low > high",
            "earliest_cr_line": "later than issue_d: set to missing, indicator added",
            "mths_since_last_delinq": "0 with no delinquency in the last two years is the historical coding of 'none on record': set to missing, indicator added",
            "median_int_rate_by_sub_grade": self.rate_by_subgrade_,
        }

