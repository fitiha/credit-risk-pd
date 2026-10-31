import numpy as np
import pandas as pd
import pytest

from credit_risk.outliers import AmountTreater


def frame(values, dates=None):
    n = len(values)
    return pd.DataFrame({"issue_d": pd.to_datetime(dates or ["2014-03-01"] * n), "annual_inc": values})


def test_caps_are_percentiles_of_pre_cutoff_loans():
    pre = frame(list(range(1, 1001)))
    t = AmountTreater(fields=["annual_inc"]).fit(pre)
    lo, hi = t.caps_["annual_inc"]
    assert lo == pytest.approx(np.quantile(range(1, 1001), 0.005))
    assert hi == pytest.approx(np.quantile(range(1, 1001), 0.995))


def test_2015_loans_do_not_change_the_fitted_caps():
    base = frame(list(range(1, 1001)))
    with_2015 = pd.concat([base, frame([10**9] * 500, ["2015-06-01"] * 500)], ignore_index=True)
    a = AmountTreater(fields=["annual_inc"]).fit(base).caps_
    b = AmountTreater(fields=["annual_inc"]).fit(with_2015).caps_
    assert a == b


def test_extreme_values_are_capped_not_dropped_and_log_applied():
    train = frame(list(range(1, 1001)))
    t = AmountTreater(fields=["annual_inc"]).fit(train)
    out = t.transform(frame([7_500_000.0, 500.0]))
    assert len(out) == 2
    hi = t.caps_["annual_inc"][1]
    assert out["annual_inc_capped"].iloc[0] == hi
    assert out["annual_inc_log"].iloc[0] == pytest.approx(np.log1p(hi))
    assert out["annual_inc_capped"].iloc[1] == 500.0


def test_zero_is_flagged_and_missing_is_not_zero():
    t = AmountTreater(fields=["annual_inc"]).fit(frame(list(range(0, 1000))))
    assert t.zero_fields_ == ["annual_inc"]
    out = t.transform(frame([0.0, np.nan, 50.0]))
    assert out["annual_inc_is_zero"].tolist() == [True, False, False]
    assert np.isnan(out["annual_inc_log"].iloc[1])  # missing stays missing
    assert out["annual_inc_capped"].iloc[0] == t.caps_["annual_inc"][0]  # zero lifted to the floor, flag keeps the fact


def test_no_zero_indicator_when_field_has_no_zeros():
    t = AmountTreater(fields=["annual_inc"]).fit(frame(list(range(1, 1001))))
    assert "annual_inc_is_zero" not in t.transform(frame([5.0])).columns
