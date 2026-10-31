import numpy as np
import pandas as pd
import pytest

from credit_risk.model import metrics, policy

CFG = {"loss_given_default": 0.70, "margin_if_repaid": 0.10, "min_approval_margin": 0.05}


def test_breakeven_and_thresholds_follow_the_profit_assumptions():
    assert policy.breakeven_pd(CFG) == pytest.approx(0.125)
    th = policy.thresholds(CFG)
    assert th["decline_at_or_above"] == pytest.approx(0.125) and th["approve_below"] == pytest.approx(0.0625)
    assert policy.margin_per_dollar(0.125, CFG) == pytest.approx(0.0)
    assert policy.margin_per_dollar(0.0625, CFG) == pytest.approx(0.05)  # exactly the minimum margin at the approve threshold


def test_decision_logic_at_the_boundaries():
    th = policy.thresholds(CFG)
    out = policy.decide([0.0, 0.0624, 0.0625, 0.1249, 0.125, 0.9], th)
    assert out.tolist() == ["approve", "approve", "review", "review", "decline", "decline"]


def test_realised_profit_uses_what_happened():
    assert policy.realised_profit([1000, 1000], [0, 1], CFG).tolist() == [100.0, -700.0]


def test_band_table_adds_up():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pd": rng.uniform(0, 0.4, 2000), "loan_amnt": 10000.0})
    df["defaulted"] = (rng.random(2000) < df.pd).astype(int)
    t = policy.band_table(df, policy.thresholds(CFG), CFG)
    assert t.loans.sum() == 2000 and t.share_of_loans.sum() == pytest.approx(1.0)
    assert t.set_index("decision").loc["approve", "default_rate"] < t.set_index("decision").loc["decline", "default_rate"]


def test_metrics_on_a_perfect_and_a_random_ranking():
    y = np.array([0] * 50 + [1] * 50)
    assert metrics.auc(y, np.r_[np.zeros(50), np.ones(50)]) == 1 and metrics.ks(y, np.r_[np.zeros(50), np.ones(50)]) == 1
    assert metrics.gini(y, np.random.default_rng(0).random(100)) == pytest.approx(0, abs=0.35)
    assert metrics.brier(y, np.full(100, 0.5)) == pytest.approx(0.25)


def test_calibration_slope_is_about_one_for_honest_probabilities():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.5, 20000)
    y = (rng.random(20000) < p).astype(int)
    slope, intercept = metrics.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1, abs=0.1) and intercept == pytest.approx(0, abs=0.1)
    t = metrics.calibration_table(y, p)
    assert t.gap.abs().max() < 0.03
