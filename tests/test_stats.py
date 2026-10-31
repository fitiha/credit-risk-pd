import pandas as pd
import pytest

from credit_risk.stats import rate_table, wilson_interval


def test_wilson_matches_known_value():
    lo, hi = wilson_interval(10, 100)  # textbook: 0.0552 to 0.1744
    assert lo == pytest.approx(0.0552, abs=1e-3)
    assert hi == pytest.approx(0.1744, abs=1e-3)


def test_wilson_stays_in_bounds_for_zero_defaults():
    lo, hi = wilson_interval(0, 8)
    assert lo >= 0 and 0 < hi < 1


def test_rate_table():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "defaulted": [1, 0, 0, 0]})
    t = rate_table(df, "g")
    assert t.set_index("g").rate.to_dict() == {"a": 0.5, "b": 0.0}


from credit_risk.stats import cochran_armitage_trend, level_vs_rest


def test_trend_test_detects_a_steady_rise_and_ignores_a_flat_rate():
    n = [1000] * 10
    z_up, p_up = cochran_armitage_trend(range(10), [100 + 10 * i for i in range(10)], n)
    z_flat, p_flat = cochran_armitage_trend(range(10), [150] * 10, n)
    assert z_up > 5 and p_up < 1e-6 and abs(z_flat) < 1e-9 and p_flat == 1


def test_level_vs_rest_flags_the_odd_group_out():
    p = level_vs_rest([300, 150, 150, 150, 150, 150], [1000] * 6)
    assert p[0] < 1e-6 and p[0] < p[1:].min() / 100  # the odd group is far more significant than any of the ordinary ones
