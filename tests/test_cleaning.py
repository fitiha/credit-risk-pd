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
        revol_util=40.0, bc_util=50.0, bureau_recovery_flag=0, revol_bal=4000.0, total_rev_hi_lim=10000.0, tot_cur_bal=50000.0,
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


def test_spec_rules_state_thresholds_and_fitted_rates():
    df = pd.DataFrame([row(int_rate=10.0), row(int_rate=14.0)])
    rules = Cleaner().fit(df).spec_rules()
    assert "1,000" in rules["annual_inc"]
    assert rules["median_int_rate_by_sub_grade"] == {"B3": 12.0}


def test_revolving_balance_times_ten_is_fixed_using_utilisation_and_limit():
    ok = row(revol_bal=6000.0, revol_util=50.0, total_rev_hi_lim=12000.0)
    bad = row(revol_bal=60000.0, revol_util=50.0, total_rev_hi_lim=12000.0)  # balance 10x too large
    out = clean(ok, bad)
    assert out["revol_bal"].tolist() == [6000.0, 6000.0] and out["revol_bal_fixed"].tolist() == [False, True]


def test_balance_check_needs_a_usable_utilisation_and_limit():
    rows = [row(revol_bal=60000.0, revol_util=0.5, total_rev_hi_lim=12000.0),    # utilisation under 1%: not checkable
            row(revol_bal=60000.0, revol_util=50.0, total_rev_hi_lim=0.0),       # no limit
            row(revol_bal=60000.0, revol_util=np.nan, total_rev_hi_lim=12000.0)]  # no utilisation
    out = clean(*rows)
    assert out["revol_bal"].tolist() == [60000.0] * 3 and not out["revol_bal_fixed"].any()


def test_a_balance_larger_than_the_total_balance_is_fixed_even_when_utilisation_cannot_confirm_it():
    bad = row(revol_bal=600000.0, revol_util=np.nan, total_rev_hi_lim=12000.0, tot_cur_bal=100000.0)
    fine = row(revol_bal=6000.0, revol_util=np.nan, total_rev_hi_lim=12000.0, tot_cur_bal=100000.0)
    slightly_over = row(revol_bal=105000.0, revol_util=np.nan, total_rev_hi_lim=120000.0, tot_cur_bal=100000.0)  # timing noise, not 10x
    out = clean(bad, fine, slightly_over)
    assert out["revol_bal"].tolist() == [60000.0, 6000.0, 105000.0]


import pytest
from pathlib import Path


@pytest.mark.skipif(not Path("data/applications.csv").exists(), reason="data files not present")
def test_cleaned_training_data_satisfies_the_accounting_identities():
    from credit_risk.io import load_applications

    raw = load_applications("data")
    c = Cleaner().fit(raw).transform(raw)
    assert (c.revol_bal > 1.5 * c.tot_cur_bal).sum() == 0                   # a balance cannot exceed the total that contains it
    assert (c.open_acc > c.total_acc).sum() == 0 and (c.total_rev_hi_lim > c.tot_hi_cred_lim).sum() == 0
    util = (100 * c.revol_bal / c.total_rev_hi_lim.where(c.total_rev_hi_lim > 0)) / c.revol_util.abs().where(c.revol_util.abs() > 1)
    assert util.between(8, 20).sum() == 0                                   # no loan left with a tenfold balance
    assert util.between(0.8, 1.25).mean() > 0.95                            # and the great majority agree with the reported utilisation


def test_zero_months_since_delinquency_with_no_delinquency_means_none_on_record():
    out = clean(row(mths_since_last_delinq=0.0, delinq_2yrs=0.0),     # coded zero, no delinquency in two years: becomes missing
                row(mths_since_last_delinq=0.0, delinq_2yrs=2.0),     # a delinquency this month: a real zero, kept
                row(mths_since_last_delinq=14.0, delinq_2yrs=1.0),    # ordinary value: kept
                row(mths_since_last_delinq=np.nan, delinq_2yrs=0.0))  # already missing: stays missing
    assert out["mths_since_last_delinq"].isna().tolist() == [True, False, False, True]
    assert out["mths_since_last_delinq"].iloc[1] == 0.0 and out["mths_since_last_delinq"].iloc[2] == 14.0
    assert out["mths_since_last_delinq_recoded"].tolist() == [True, False, False, False]
