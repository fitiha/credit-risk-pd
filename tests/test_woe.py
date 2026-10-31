import numpy as np
import pandas as pd
import pytest

from credit_risk.woe import apply_woe, information_value, iv_on_new_data, strength, woe_table


def test_woe_sign_and_iv_on_a_simple_case():
    groups = ["safe"] * 100 + ["risky"] * 100
    y = [0] * 95 + [1] * 5 + [0] * 70 + [1] * 30
    t = woe_table(groups, y).set_index("group")
    assert t.loc["safe", "woe"] > 0 > t.loc["risky", "woe"]  # positive = safer than average
    assert information_value(woe_table(groups, y)) > 0.3


def test_a_field_with_no_information_has_near_zero_iv():
    rng = np.random.default_rng(1)
    groups = rng.choice(list("abcd"), 4000)
    y = rng.random(4000) < 0.2
    assert information_value(woe_table(groups, y)) < 0.01


def test_group_with_no_defaults_is_finite_thanks_to_smoothing():
    t = woe_table(["a"] * 50 + ["b"] * 50, [0] * 50 + [0] * 25 + [1] * 25)
    assert np.isfinite(t.woe).all()


def test_unseen_group_gets_neutral_woe_and_missing_is_its_own_group():
    t = woe_table(["a", "b", None, "a", "b", None], [0, 1, 1, 0, 0, 1])
    out = apply_woe(["a", "zzz", None], t)
    assert out[1] == 0.0
    assert out[2] == t[t.group.isna()].woe.iloc[0]


def test_out_of_time_iv_is_lower_than_in_sample_for_noise_groups():
    rng = np.random.default_rng(2)
    g_tr, g_te = rng.choice(list("abcdefghij"), 600), rng.choice(list("abcdefghij"), 600)
    y_tr, y_te = rng.random(600) < 0.2, rng.random(600) < 0.2
    t = woe_table(g_tr, y_tr)
    assert iv_on_new_data(t, g_te, y_te) < information_value(t) + 0.05


@pytest.mark.parametrize("iv,label", [(0.01, "useless"), (0.05, "weak"), (0.2, "medium"), (0.4, "strong"), (0.9, "suspicious")])
def test_strength_labels(iv, label):
    assert strength(iv) == label
