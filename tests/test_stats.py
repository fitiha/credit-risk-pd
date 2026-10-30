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
