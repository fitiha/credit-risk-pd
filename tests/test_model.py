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
    # the platform's sub-grade is the one platform score that may be an input (DESIGN.md section 11); the rate, the grade letter, restricted fields
    # and everything recorded after the loan never are
    forbidden = {"int_rate", "grade", "sub_grade", "job_group", "addr_state", "addr_state_group", "zip_code", "defaulted",
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


@pytest.mark.parametrize("field,value,message", [("loan_amnt", 80000, "10 times too large"), ("annual_inc", 58, "entered in thousands"),
                                                   ("fico_range_low", 780, "fico_range_low is above"), ("revol_util", -20, "negative"),
                                                   ("revol_bal", 90000, "revol_bal looks 10 times too large")])
def test_a_live_application_that_looks_mis_entered_is_stopped_but_historical_repair_still_works(predictor, field, value, message):
    rec = example("low_risk")
    rec[field] = value
    with pytest.raises(ContractViolation, match=message):
        predictor.predict(rec)
    assert 0 < predictor.predict(rec, strict=False)["pd"] < 1  # the contract's repair rules still apply when asked for


def test_the_example_applicants_need_no_repair(predictor):
    for name in ("low_risk", "borderline", "high_risk"):
        predictor.predict(example(name))  # strict is the default and must not complain


def test_a_declined_applicant_is_told_why_and_an_approved_one_needs_few_reasons(predictor):
    assert predictor.predict(example("borderline"))["decision"] == "review"
    declined = predictor.predict(example("high_risk"))
    assert declined["decision"] == "decline" and len(declined["reason_codes"]) >= 3
    assert all("typical approved applicant" in r or r.split(":")[0] in ("Loan purpose", "Home ownership", "Income verification") for r in declined["reason_codes"])
    assert len(predictor.predict(example("low_risk"))["reason_codes"]) <= len(declined["reason_codes"])
    assert predictor.predict(example("borderline"))["reason_codes"]  # a loan sent to review is also told what to look at


def test_the_platform_sub_grade_is_the_only_platform_score_in_the_model(predictor):
    used = {c.removesuffix("_woe") for c in predictor.bundle.model.columns}
    assert used & {"sub_grade_rank"} == ({"sub_grade_rank"} if predictor.bundle.metadata["selection"]["platform_score"] == "sub_grade" else set())
    assert not used & {"int_rate", "grade"}


def test_the_sub_grade_pushes_risk_in_the_safe_direction_and_wrong_signed_weights_are_dropped():
    import numpy as np, pandas as pd
    from credit_risk.model.models import LogitPD

    rng = np.random.default_rng(0)
    n = 6000
    grade = rng.integers(0, 35, n).astype(float)
    good = rng.normal(size=n)       # a WoE-like column: higher means safer (its weight should be negative)
    bad = rng.normal(size=n)        # pure noise tied to the label the wrong way round below
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 0.06 * grade - 0.5 * good)))).astype(int)
    X = pd.DataFrame({"sub_grade_rank": grade, "good_woe": good, "bad_woe": bad + 0.9 * y})  # bad_woe is raised by default: a positive weight
    m = LogitPD(["sub_grade_rank", "good_woe", "bad_woe"], 1.0, {"sub_grade_rank"}, 1, nonpositive=True, positive=["sub_grade_rank"]).fit(X, y)
    coef = dict(zip(m.columns, m.model_.coef_[0]))
    assert coef["sub_grade_rank"] > 0 and coef["good_woe"] < 0
    assert "bad_woe" in m.dropped_ and "bad_woe" not in m.columns


def test_a_required_positive_column_with_the_wrong_sign_is_dropped_and_scaling_is_per_column():
    import numpy as np, pandas as pd
    from credit_risk.model.models import LogitPD

    rng = np.random.default_rng(1)
    n = 4000
    grade = rng.integers(0, 35, n).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 - 0.08 * grade)))).astype(int)   # a "grade" that lowers risk, which must not be allowed
    X = pd.DataFrame({"sub_grade_rank": grade, "a_woe": rng.normal(size=n)})
    m = LogitPD(["sub_grade_rank", "a_woe"], 1.0, {"sub_grade_rank"}, 1, nonpositive=True, positive=["sub_grade_rank"]).fit(X, y)
    assert "sub_grade_rank" in m.dropped_
    m2 = LogitPD(["sub_grade_rank", "a_woe"], 1.0, {"sub_grade_rank"}, 1).fit(X, y)
    assert m2.scale_[1] == 1.0 and abs(m2.scale_[0] - grade.std()) < 1e-6      # only the named column is standardised


def test_the_boosted_model_replaces_the_scorecard_only_if_ahead_on_average_and_in_every_fold():
    from credit_risk.model.train import boosted_model_wins

    assert boosted_model_wins([0.71, 0.73], [0.70, 0.72])[0] is True                 # +0.01 in both folds
    assert boosted_model_wins([0.710, 0.713], [0.700, 0.720])[0] is False            # ahead in fold 1, behind in fold 2
    assert boosted_model_wins([0.720, 0.719], [0.700, 0.720])[0] is False            # +0.0095 on average but behind in one fold
    assert boosted_model_wins([0.703, 0.724], [0.700, 0.720])[0] is False            # ahead in both, but only +0.0035 on average
    wins, gains = boosted_model_wins([0.7007, 0.7175], [0.6901, 0.7204])             # the tuning-fold figures from DESIGN.md section 11
    assert wins is False and abs(gains[0] - 0.0106) < 1e-4


def test_the_averaged_model_explains_exactly_what_it_predicts():
    import numpy as np, pandas as pd
    from credit_risk.model.models import LgbmPD, LogOddsAverage, LogitPD

    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    X["a_woe"], X["b_woe"] = -X.a, -X.b                      # WoE columns of fields a and b (higher WoE = safer)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.4 + 0.8 * X.a + 0.5 * X.b + 0.3 * X.c)))).astype(int)
    lg = LgbmPD(["a", "b", "c"], dict(num_leaves=7, n_estimators=60, learning_rate=0.1, min_child_samples=50), None, False, 0)
    avg = LogOddsAverage([LogitPD(["a_woe", "b_woe"], 1.0, False, 0, nonpositive=True), lg], {"a_woe": "a", "b_woe": "b"}).fit(X, y)
    assert avg.columns == ["a", "b", "c"]
    approved = X[avg.predict(X) < np.quantile(avg.predict(X), 0.3)]
    avg.set_reference(approved)
    z = avg._logit(avg.predict(X.iloc[:200]))
    z_ref = avg._logit(np.vstack([m.predict(approved) for m in avg.members]).T).mean(axis=1).mean()   # mean averaged log-odds of the approved
    contrib = avg.contributions(X.iloc[:200])
    assert np.allclose(contrib.sum(axis=1), z - z_ref, atol=1e-6)    # the reasons add up to the score, exactly
    assert contrib[:, 0].std() > 0 and contrib[:, 2].std() > 0        # field c comes only from the boosted model and still gets its share
