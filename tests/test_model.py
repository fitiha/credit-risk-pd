"""Inference and model tests. They need the trained model and the contract (both are committed) but not the training data."""
import copy
import json
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pytest

from credit_risk.model.inputs import REASONS
from credit_risk.model.predict import ContractVersionMismatch, Predictor
from credit_risk.spec import ContractViolation

MODEL, CONTRACT, EXAMPLES = Path("models/pd_model.joblib"), Path("contract"), Path("examples")
pytestmark = pytest.mark.skipif(not MODEL.exists() or not (CONTRACT / "transform.joblib").exists(), reason="model or contract not built")


@pytest.fixture(scope="module")
def predictor():
    return Predictor(MODEL, CONTRACT)


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


def test_the_three_example_applicants_get_ordered_pds_and_decisions(predictor):
    low, mid, high = (predictor.predict(example(n)) for n in ("low_risk", "borderline", "high_risk"))
    assert low["pd"] < mid["pd"] < high["pd"]
    assert low["decision"] == "approve" and high["decision"] == "decline"
    assert {"pd", "decision", "model_version", "reason_codes"} == set(low)
    assert low["model_version"] == high["model_version"] and "contract1.0.0" in low["model_version"]


def test_reason_codes_are_few_plain_and_only_for_factors_that_raise_risk(predictor):
    for name in ("low_risk", "borderline", "high_risk"):
        r = predictor.predict(example(name))["reason_codes"]
        assert len(r) <= 4 and len(set(r)) == len(r)
        assert all(isinstance(x, str) and ":" in x for x in r)
    assert len(predictor.predict(example("high_risk"))["reason_codes"]) >= 3


def test_the_first_reason_is_the_input_with_the_largest_upward_contribution(predictor):
    frame = predictor.frame(example("high_risk"))
    model = predictor.bundle.model
    contrib = model.contributions(frame)[0]
    top = model.columns[int(np.argmax(contrib))]
    label = REASONS[predictor.bundle.reason_map.get(top, top)][0]
    assert predictor.predict(example("high_risk"))["reason_codes"][0].startswith(label)
    assert contrib.max() > 0.1


def test_a_single_application_scores_the_same_as_in_a_batch(predictor):
    recs = [example(n) for n in ("low_risk", "borderline", "high_risk")]
    batch = predictor.bundle.score(predictor.frame(recs))
    for i, rec in enumerate(recs):
        assert predictor.predict(rec)["pd"] == pytest.approx(round(float(batch[i]), 4), abs=1e-9)


def test_bad_input_is_rejected_before_scoring(predictor):
    missing = example("low_risk")
    del missing["annual_inc"]
    with pytest.raises(ContractViolation, match="annual_inc"):
        predictor.predict(missing)
    empty = example("low_risk")
    empty["annual_inc"] = None
    with pytest.raises(ContractViolation, match="annual_inc"):
        predictor.predict(empty)
    for field, value in [("term", 48), ("loan_amnt", 500), ("addr_state", "ZZ"), ("fico_range_low", 100), ("purpose", "holiday"), ("loan_amnt", "abc"), ("dti", -3)]:
        bad = example("low_risk")
        bad[field] = value
        with pytest.raises(ContractViolation, match=field):
            predictor.predict(bad)


def test_optional_fields_may_be_left_out(predictor):
    rec = example("borderline")
    for f in ("emp_title", "emp_length", "mths_since_last_delinq", "bc_util", "title"):
        rec.pop(f, None)
    assert 0 < predictor.predict(rec)["pd"] < 1


def test_the_decision_follows_the_thresholds_saved_with_the_model(predictor):
    th = predictor.bundle.thresholds
    assert th["approve_below"] == pytest.approx(0.0625) and th["decline_at_or_above"] == pytest.approx(0.125)
    for name in ("low_risk", "borderline", "high_risk"):
        r = predictor.predict(example(name))
        expected = "approve" if r["pd"] < th["approve_below"] else "decline" if r["pd"] >= th["decline_at_or_above"] else "review"
        assert r["decision"] == expected


def test_a_model_built_on_another_contract_version_is_refused(tmp_path):
    bundle = joblib.load(MODEL)
    other = copy.deepcopy(bundle)
    other.metadata["contract_version"] = "9.9.9"
    joblib.dump(other, tmp_path / "m.joblib")
    with pytest.raises(ContractVersionMismatch, match="9.9.9"):
        Predictor(tmp_path / "m.joblib", CONTRACT)


def test_nothing_forbidden_reaches_the_model(predictor):
    used = {c.removesuffix("_woe") for c in predictor.bundle.model.columns}
    forbidden = {"int_rate", "grade", "sub_grade", "sub_grade_rank", "job_group", "addr_state", "addr_state_group", "zip_code", "defaulted",
                 "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries", "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d",
                 "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag", "bureau_recovery_flag"}
    assert not used & forbidden, used & forbidden


def test_the_model_carries_its_lineage(predictor):
    m = predictor.bundle.metadata
    for key in ("git_commit", "training_data_hash", "training_date", "contract_version", "thresholds", "seed", "calibration"):
        assert key in m and m[key]
    assert len(m["training_data_hash"]) == 64 and m["fit_cutoff"] == "2014-12-31"


def test_scoring_one_application_is_fast_enough(predictor):
    rec = example("borderline")
    predictor.predict(rec)
    times = []
    for _ in range(20):
        t = time.perf_counter()
        predictor.predict(rec)
        times.append((time.perf_counter() - t) * 1000)
    assert statistics.median(times) < 100


def test_numbers_sent_as_text_are_accepted(predictor):
    rec = example("borderline")
    as_text = {k: (str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in rec.items()}
    assert predictor.predict(as_text)["pd"] == predictor.predict(rec)["pd"]
