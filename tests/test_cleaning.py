import numpy as np
import pandas as pd
import pytest

from credit_risk.cleaning import Cleaner
from credit_risk.io import parse_dates


def row(**kw):
    base = dict(
        id="x", issue_d="Mar-2014", loan_amnt=10000.0, term=" 36 months",
        int_rate=12.0, installment=332.14, grade="B", sub_grade="B3",
        emp_title="  Teacher ", annual_inc=60000.0, addr_state="CA",
        earliest_cr_line="Jan-2000", fico_range_low=680.0, fico_range_high=684.0,
        revol_util=40.0, bc_util=50.0, bureau_recovery_flag=0,
    )
    base.update(kw)
    return base


def clean(*rows):
    df = pd.DataFrame(rows)
    return Cleaner().fit(df).transform(df)


def test_parse_three_date_formats():
    s = pd.Series(["Feb-2015", "2015-02-01", "02/2015"])
    assert (parse_dates(s) == pd.Timestamp("2015-02-01")).all()


def test_term_and_state_and_title():
    out = clean(row(addr_state="california"), row(addr_state=" tx "))
    assert out["term"].tolist() == [36, 36]
    assert out["addr_state"].tolist() == ["CA", "TX"]
    assert out["emp_title"].iloc[0] == "teacher"


def test_loan_amount_times_ten_is_fixed_using_installment():
    out = clean(row(loan_amnt=100000.0))  # installment still matches 10,000
    assert out["loan_amnt"].iloc[0] == pytest.approx(10000.0)
    assert out["loan_amnt_fixed"].iloc[0]
    assert not clean(row())["loan_amnt_fixed"].iloc[0]


def test_income_in_thousands():
    out = clean(row(annual_inc=63.5), row(annual_inc=63500.0))
    assert out["annual_inc"].tolist() == [63500.0, 63500.0]
    assert out["annual_inc_fixed"].tolist() == [True, False]


def test_fico_swap():
    out = clean(row(fico_range_low=684.0, fico_range_high=680.0))
    assert (out["fico_range_low"].iloc[0], out["fico_range_high"].iloc[0]) == (680.0, 684.0)


def test_credit_history_after_issue_becomes_missing_with_flag():
    out = clean(row(earliest_cr_line="Dec-2014"))
    assert pd.isna(out["earliest_cr_line"].iloc[0])
    assert out["earliest_cr_line_invalid"].iloc[0]


def test_utilisation_rules():
    out = clean(row(revol_util=-58.0, bc_util=-68.0))
    assert out["revol_util"].iloc[0] == 58.0          # sign flip repaired
    assert np.isnan(out["bc_util"].iloc[0])           # not provable -> missing
    assert out["bc_util_invalid"].iloc[0]


def test_placeholder_rate_uses_pre_cutoff_subgrade_median():
    train = [row(int_rate=10.0), row(int_rate=12.0), row(int_rate=14.0),
             row(int_rate=99.0, issue_d="Jun-2015")]  # 2015 must not influence the fit
    df = pd.DataFrame(train + [row(int_rate=6.0)])
    out = Cleaner().fit(df).transform(df)
    assert out["int_rate"].iloc[-1] == 12.0
    assert out["int_rate_imputed"].iloc[-1]


def test_leaky_column_removed():
    assert "bureau_recovery_flag" not in clean(row()).columns
