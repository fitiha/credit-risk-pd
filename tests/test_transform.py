from pathlib import Path

import pytest

from credit_risk.io import load_applications, load_holdout
from credit_risk.transform import ApplicationTransformer

DATA = Path("data")
pytestmark = pytest.mark.skipif(not (DATA / "applications.csv").exists(), reason="data files not present")


def test_transformer_handles_holdout_and_single_rows_without_gaps():
    tr = ApplicationTransformer().fit(load_applications(DATA))
    holdout = load_holdout(DATA)
    out = tr.transform(holdout)
    assert len(out) == len(holdout)
    treated = out.filter(regex="_(treated|log|capped)$")
    assert not treated.isna().any().any()
    one = tr.transform(holdout.iloc[[0]])
    assert len(one) == 1 and one.filter(regex="_(treated|log|capped)$").iloc[0].equals(treated.iloc[0])  # same result alone


def test_applying_the_transform_does_not_change_what_was_learned():
    train = load_applications(DATA)
    tr = ApplicationTransformer().fit(train)
    before = tr.step("amounts").caps_.copy()
    tr.transform(load_holdout(DATA))
    assert tr.step("amounts").caps_ == before
