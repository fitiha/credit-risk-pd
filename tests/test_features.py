import numpy as np
import pandas as pd
import pytest

from credit_risk.features import FeatureBuilder, FeatureTreater


def loan(**kw):
    base = dict(issue_d=pd.Timestamp("2014-03-01"), earliest_cr_line=pd.Timestamp("2000-01-01"), loan_amnt=10000.0,
                annual_inc=60000.0, installment=300.0, revol_bal=6000.0, fico_range_low=680.0, fico_range_high=684.0,
                open_acc=8.0, total_acc=20.0, acc_open_past_24mths=4.0)
    base.update(kw)
    return base


def test_feature_values():
    out = FeatureBuilder().transform(pd.DataFrame([loan()])).iloc[0]
    assert out.credit_history_months == 14 * 12 + 2
    assert out.loan_to_income == pytest.approx(1 / 6)
    assert out.installment_to_income == pytest.approx(300 / 5000)
    assert out.revol_bal_to_income == pytest.approx(0.1)
    assert out.fico_mid == 682
    assert out.open_acc_share == pytest.approx(0.4)
    assert out.recent_accounts_share == pytest.approx(0.2)


def test_corrupted_credit_date_gives_missing_history_which_is_then_filled():
    rows = [loan(annual_inc=50000.0 + i * 100) for i in range(300)]
    rows.append(loan(earliest_cr_line=pd.NaT))
    df = FeatureBuilder().transform(pd.DataFrame(rows))
    assert np.isnan(df.credit_history_months.iloc[-1])
    out = FeatureTreater().fit(df).transform(df)
    assert out.credit_history_months_treated.iloc[-1] == 14 * 12 + 2  # the fit-window median


def test_log_added_only_for_heavily_skewed_features():
    rng = np.random.default_rng(0)
    rows = [loan(annual_inc=float(max(rng.lognormal(10.5, 1.6), 1000)), revol_bal=float(rng.lognormal(9, 1.2))) for _ in range(2000)]
    df = FeatureBuilder().transform(pd.DataFrame(rows))
    t = FeatureTreater().fit(df)
    out = t.transform(df)
    assert "fico_mid" not in t.log_fields_
    for f in t.log_fields_:
        assert f"{f}_log" in out.columns


def test_features_ignore_2015_loans_when_fitted():
    base = pd.DataFrame([loan(annual_inc=40000.0 + 100 * i) for i in range(500)])
    later = pd.concat([base, pd.DataFrame([loan(issue_d=pd.Timestamp("2015-06-01"), annual_inc=1000.0) for _ in range(500)])], ignore_index=True)
    a = FeatureTreater().fit(FeatureBuilder().transform(base))
    b = FeatureTreater().fit(FeatureBuilder().transform(later))
    assert a.caps_ == b.caps_ and a.fill_ == b.fill_
