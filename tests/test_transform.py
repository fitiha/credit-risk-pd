from pathlib import Path

import pytest

from credit_risk.io import load_applications, load_holdout
from credit_risk.transform import ApplicationTransformer


@pytest.fixture(scope="module")
def fitted():
    return ApplicationTransformer().fit(load_applications(DATA))

DATA = Path("data")
pytestmark = pytest.mark.skipif(not (DATA / "applications.csv").exists(), reason="data files not present")


def test_transformer_handles_holdout_and_single_rows_without_gaps(fitted):
    tr = fitted
    holdout = load_holdout(DATA)
    out = tr.transform(holdout)
    assert len(out) == len(holdout)
    treated = out.filter(regex="_(treated|log|capped)$")
    assert not treated.isna().any().any()
    one = tr.transform(holdout.iloc[[0]])
    assert len(one) == 1 and one.filter(regex="_(treated|log|capped)$").iloc[0].equals(treated.iloc[0])  # same result alone


def test_applying_the_transform_does_not_change_what_was_learned(fitted):
    tr = fitted
    before = tr.step("amounts").caps_.copy()
    tr.transform(load_holdout(DATA))
    assert tr.step("amounts").caps_ == before


def test_factor_scores_are_named_finite_and_repeatable(fitted):
    import numpy as np

    tr = fitted
    out = tr.transform(load_holdout(DATA))
    factors = [c for c in out.columns if c.startswith("factor_")]
    assert factors == ["factor_credit_capacity", "factor_loan_size", "factor_revolving_credit_size", "factor_account_count",
                       "factor_revolving_utilization", "factor_public_records", "factor_recent_account_opening", "factor_debt_burden"]
    assert len([c for c in out.columns if c.startswith("pc_")]) == 7
    assert np.isfinite(out[factors].to_numpy()).all()
    assert np.allclose(out[factors], tr.transform(load_holdout(DATA))[factors])


def test_a_batch_with_percentages_stored_as_fractions_is_rejected_not_scored(fitted):
    bad = load_applications(DATA).head(300).copy()
    bad["revol_util"] = bad["revol_util"] / 100
    with pytest.raises(ValueError, match="revol_util"):
        fitted.transform(bad)


def test_a_single_low_utilisation_is_not_mistaken_for_a_fraction(fitted):
    one = load_applications(DATA).head(1).copy()
    one["revol_util"] = 0.9  # a genuine 0.9% utilisation
    assert len(fitted.transform(one)) == 1


def test_the_holdout_loader_converts_its_fraction_column_and_says_so():
    with pytest.warns(UserWarning, match="revol_util"):
        h = load_holdout(DATA)
    assert h.revol_util.median() > 20 and h.revol_util.max() < 200
