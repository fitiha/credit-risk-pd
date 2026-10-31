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


from credit_risk.woe import WoEEncoder


def _frame(n=4000, seed=0, with_2015=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n)
    p = 0.12 + 0.1 * (x > 0) + 0.05 * (cat == "c")
    df = pd.DataFrame({"issue_d": pd.Timestamp("2014-03-01"), "x": x, "cat": cat, "defaulted": (rng.random(n) < p).astype(int)})
    df.loc[::50, "x"] = np.nan
    if with_2015:
        late = df.sample(1000, random_state=1).assign(issue_d=pd.Timestamp("2015-06-01"), defaulted=lambda t: 1 - t.defaulted)
        df = pd.concat([df, late], ignore_index=True)
    return df


def test_encoder_adds_woe_columns_and_handles_missing_and_unseen():
    df = _frame()
    enc = WoEEncoder(numeric=["x"], categorical=["cat"]).fit(df)
    out = enc.transform(pd.concat([df.head(5), pd.DataFrame({"issue_d": [pd.Timestamp("2014-05-01")], "x": [np.nan], "cat": ["zzz"], "defaulted": [0]})]))
    assert {"x_woe", "cat_woe"} <= set(out.columns)
    assert out["cat_woe"].iloc[-1] == 0.0  # unseen group is neutral
    assert np.isfinite(out["x_woe"]).all()  # missing x has its own bin


def test_2015_loans_do_not_change_the_fitted_bins_or_woe():
    a = WoEEncoder(numeric=["x"], categorical=["cat"]).fit(_frame())
    b = WoEEncoder(numeric=["x"], categorical=["cat"]).fit(_frame(with_2015=True)[lambda t: t.index.isin(range(10**6))])
    assert a.edges_ == b.edges_
    assert a.tables_["cat"].woe.round(8).tolist() == b.tables_["cat"].woe.round(8).tolist()


def test_spec_entries_are_json_ready_and_iv_is_ranked():
    import json

    enc = WoEEncoder(numeric=["x"], categorical=["cat"]).fit(_frame())
    json.dumps(enc.spec_entries())
    iv = enc.iv_table()
    assert iv.iv.is_monotonic_decreasing and set(iv.field) == {"x", "cat"}
