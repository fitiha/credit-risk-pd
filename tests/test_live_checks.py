"""Live-scoring checks: impossible inputs are rejected, doubtful ones are scored with a warning, and the score breakdown matches the reasons."""
import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.model.predict import IMPOSSIBLE, Predictor, impossible_combinations
from credit_risk.spec import ContractViolation

MODEL, CONTRACT, EXAMPLES, DATA = Path("models/pd_model.joblib"), Path("contract"), Path("examples"), Path("data")
pytestmark = pytest.mark.skipif(not MODEL.exists() or not (CONTRACT / "transform.joblib").exists(), reason="model or contract not built")


@pytest.fixture(scope="module")
def predictor():
    return Predictor(MODEL, CONTRACT)


def example(name, **change):
    return {**json.loads((EXAMPLES / f"{name}.json").read_text()), **change}


@pytest.mark.parametrize("change, words", [
    (dict(open_acc=30, total_acc=10), "open_acc"),
    (dict(num_tl_op_past_12m=8, acc_open_past_24mths=2), "num_tl_op_past_12m"),
    (dict(pub_rec_bankruptcies=3, pub_rec=0), "bankruptcy"),
    (dict(total_rev_hi_lim=200000, tot_hi_cred_lim=50000), "total_rev_hi_lim"),
    (dict(grade="A"), "does not match sub_grade"),
    (dict(earliest_cr_line="yesterday"), "could not be read as a date"),
    (dict(issue_d="hello"), "could not be read as a date"),
    (dict(annual_inc=-5000), "annual_inc cannot be negative"),
    (dict(revol_util=-20), "revol_util cannot be negative"),
])
def test_an_impossible_or_unreadable_application_is_rejected_with_a_plain_message(predictor, change, words):
    with pytest.raises(ContractViolation, match=words):
        predictor.predict(example("borderline", **change))


def test_repair_mode_used_for_historical_data_is_not_affected_by_the_live_checks(predictor):
    assert 0 < predictor.predict(example("borderline", open_acc=30, total_acc=10), strict=False)["pd"] < 1


def test_the_example_applicants_raise_no_warnings(predictor):
    for name in ("low_risk", "borderline", "high_risk"):
        assert predictor.input_warnings(example(name)) == []


def test_doubtful_values_are_scored_with_a_warning(predictor):
    record = example("borderline", revol_util=0.48, installment=999, fico_range_low=600, fico_range_high=604, dti=120)
    assert 0 < predictor.predict(record)["pd"] < 1
    text = " ".join(predictor.input_warnings(record))
    assert "fraction" in text and "installment" in text and "FICO" in text and "Debt-to-income" in text


def test_the_breakdown_is_ordered_and_contains_every_reason(predictor):
    record = example("high_risk")
    rows = predictor.breakdown(predictor.frame(record, strict=True))
    assert len(rows) == 6 and [r["strength"] for r in rows] == sorted((r["strength"] for r in rows), reverse=True)
    factors_up = {r["factor"] for r in rows if r["direction"] > 0}
    for reason in predictor.predict(record)["reason_codes"]:
        assert reason.split(":")[0] in factors_up


@pytest.mark.skipif(not (DATA / "applications.csv").exists(), reason="training data not downloaded")
def test_no_historical_loan_breaks_an_impossible_combination():
    apps = pd.read_csv(DATA / "applications.csv").drop_duplicates("id").merge(pd.read_csv(DATA / "credit_bureau.csv"), on="id")
    assert len(apps) == 30000 and impossible_combinations(apps) == [] and len(IMPOSSIBLE) == 4
