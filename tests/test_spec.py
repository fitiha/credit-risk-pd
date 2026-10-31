import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.io import load_applications, load_holdout
from credit_risk.spec import ContractViolation, build_spec, validate_cleaned, validate_raw
from credit_risk.transform import ApplicationTransformer

DATA = Path("data")
pytestmark = pytest.mark.skipif(not (DATA / "applications.csv").exists(), reason="data files not present")


@pytest.fixture(scope="module")
def fitted():
    return ApplicationTransformer().fit(load_applications(DATA))


def test_spec_has_the_required_shape_and_is_json(fitted):
    spec = build_spec(fitted)
    json.dumps(spec)
    assert spec["contract_version"] == "1.0.0" and spec["fit_cutoff"] == "2014-12-31"
    f = next(x for x in spec["fields"] if x["name"] == "annual_inc")
    assert {"name", "type", "min", "max", "cleaning", "missing_strategy", "reliability", "reliability_reason"} <= set(f)
    assert "1,000" in f["cleaning"] and "capped" in f["cleaning"]
    fac = spec["factors"][0]
    assert {"name", "method", "top_loadings", "variance_explained"} <= set(fac) and fac["method"] == "factor_analysis_varimax"
    assert len(spec["factors"]) == 8 and len(spec["woe"]) == 37 and spec["woe"]["term"]["bins"]


def test_every_training_and_holdout_row_passes_validation_after_cleaning(fitted):
    for raw in (load_applications(DATA), load_holdout(DATA)):
        validate_cleaned(fitted.transform(raw))


def test_bad_inputs_fail_fast_with_a_clear_message(fitted):
    good = fitted.transform(load_applications(DATA).head(5))
    for column, bad, text in [("loan_amnt", 500.0, "loan_amnt"), ("term", 48, "term"), ("addr_state", "ZZ", "addr_state"), ("fico_range_low", 100.0, "fico_range_low")]:
        broken = good.copy()
        broken.loc[broken.index[0], column] = bad
        with pytest.raises(ContractViolation, match=text):
            validate_cleaned(broken)


def test_missing_required_field_is_rejected_before_cleaning():
    raw = load_applications(DATA).head(3).drop(columns=["annual_inc"])
    with pytest.raises(ContractViolation, match="annual_inc"):
        validate_raw(raw)
