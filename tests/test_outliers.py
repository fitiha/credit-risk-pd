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


from credit_risk.outliers import CountTreater, RatioTreater


def ratio_frame(values, dates=None):
    n = len(values)
    return pd.DataFrame({"issue_d": pd.to_datetime(dates or ["2014-03-01"] * n), "bc_util": values})


def test_ratio_is_capped_and_missing_filled_with_fit_median():
    train = ratio_frame(list(range(0, 101)) * 10 + [np.nan] * 20)  # 20 of 1,030 missing (about 2%)
    t = RatioTreater(fields=["bc_util"]).fit(train)
    out = t.transform(ratio_frame([366.0, np.nan, 40.0]))
    assert out["bc_util_treated"].iloc[0] == t.caps_["bc_util"][1] <= 100
    assert out["bc_util_treated"].iloc[1] == t.fill_["bc_util"] == 50
    assert out["bc_util_treated"].iloc[2] == 40.0


def test_missing_flag_only_when_enough_values_are_missing():
    few = ratio_frame(list(range(0, 1000)) + [np.nan])  # 0.1% missing
    many = ratio_frame(list(range(0, 900)) + [np.nan] * 100)  # 10% missing
    assert RatioTreater(fields=["bc_util"]).fit(few).flag_fields_ == []
    t = RatioTreater(fields=["bc_util"]).fit(many)
    assert t.flag_fields_ == ["bc_util"]
    assert t.transform(ratio_frame([np.nan, 5.0]))["bc_util_missing"].tolist() == [True, False]


def test_ratio_limits_ignore_2015_loans():
    base = ratio_frame(list(range(0, 101)) * 10)
    later = pd.concat([base, ratio_frame([5000.0] * 300, ["2015-06-01"] * 300)], ignore_index=True)
    assert RatioTreater(fields=["bc_util"]).fit(base).caps_ == RatioTreater(fields=["bc_util"]).fit(later).caps_


def test_count_cap_keeps_zeros():
    df = pd.DataFrame({"issue_d": pd.to_datetime(["2014-01-01"] * 1000), "pub_rec": [0] * 990 + [1] * 8 + [10, 12]})
    t = CountTreater(fields=["pub_rec"]).fit(df)
    out = t.transform(df)
    assert out["pub_rec_capped"].max() == t.caps_["pub_rec"] < 10
    assert (out["pub_rec_capped"][:990] == 0).all()


def test_spec_entries_carry_the_fitted_limits_and_are_json_ready():
    import json

    t = AmountTreater(fields=["annual_inc"]).fit(frame(list(range(0, 1000))))
    entry = t.spec_entries()["annual_inc"]
    assert (entry["cap_low"], entry["cap_high"]) == t.caps_["annual_inc"]
    assert entry["zero_indicator"] is True and "log(1 + x)" in entry["cleaning"]
    r = RatioTreater(fields=["bc_util"]).fit(ratio_frame(list(range(0, 900)) + [np.nan] * 100))
    assert r.spec_entries()["bc_util"]["missing_indicator"] is True
    c = CountTreater(fields=["pub_rec"]).fit(pd.DataFrame({"issue_d": pd.to_datetime(["2014-01-01"] * 10), "pub_rec": range(10)}))
    json.dumps([t.spec_entries(), r.spec_entries(), c.spec_entries()])  # must serialise


def test_count_caps_ignore_2015_loans():
    base = pd.DataFrame({"issue_d": pd.to_datetime(["2014-01-01"] * 1000), "pub_rec": [0] * 950 + [1] * 40 + [2] * 10})
    later = pd.concat([base, pd.DataFrame({"issue_d": pd.to_datetime(["2015-06-01"] * 500), "pub_rec": [9] * 500})], ignore_index=True)
    assert CountTreater(fields=["pub_rec"]).fit(base).caps_ == CountTreater(fields=["pub_rec"]).fit(later).caps_
