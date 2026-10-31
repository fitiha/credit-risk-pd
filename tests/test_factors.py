import json

import numpy as np
import pandas as pd

from credit_risk.factors import FactorModel, name_factors

SIGS = {"size": ["a", "b"], "risk": ["c", "d"]}


def data(n=3000, seed=0, with_2015=False):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.standard_normal(n), rng.standard_normal(n)
    df = pd.DataFrame({"a": f1 + .4 * rng.standard_normal(n), "b": f1 + .4 * rng.standard_normal(n),
                       "c": f2 + .4 * rng.standard_normal(n), "d": f2 + .4 * rng.standard_normal(n),
                       "issue_d": pd.Timestamp("2014-03-01")})
    if with_2015:
        late = pd.DataFrame(rng.standard_normal((800, 4)) * [5, 1, 1, 5], columns=list("abcd")).assign(issue_d=pd.Timestamp("2015-06-01"))
        df = pd.concat([df, late], ignore_index=True)
    return df


def model(**kw):
    return FactorModel(fields=list("abcd"), n_components=2, n_factors=2, signatures=SIGS, **kw)


def test_factors_are_named_from_their_loadings_with_a_stable_sign():
    m = model().fit(data())
    assert m.factor_names_ == ["size", "risk"]
    assert (m.loadings_.loc[["a", "b"], "size"] > 0.8).all() and (m.loadings_.loc[["c", "d"], "risk"] > 0.8).all()
    out = m.transform(data().head(5))
    assert {"pc_1", "pc_2", "factor_size", "factor_risk"} <= set(out.columns)


def test_a_factor_that_matches_no_signature_stays_unnamed():
    L = pd.DataFrame({0: [.9, .9, .0, .0], 1: [.1, .0, .1, .1]}, index=list("abcd"))
    assert name_factors(L, SIGS) == {0: "size", 1: None}


def test_fit_ignores_2015_loans_and_scores_are_repeatable():
    a, b = model().fit(data()), model().fit(data(with_2015=True))
    assert np.allclose(a.scaler_.mean_, b.scaler_.mean_) and np.allclose(a.loadings_.abs(), b.loadings_.abs(), atol=1e-6)
    assert np.allclose(a.transform(data().head(20))["factor_size"], a.transform(data().head(20))["factor_size"])


def test_spec_entries_are_json_ready():
    spec = model().fit(data()).spec_entries()
    json.dumps(spec)
    assert [f["name"] for f in spec["factors"]] == ["size", "risk"] and set(spec["factors"][0]["top_loadings"]) <= set("abcd")
