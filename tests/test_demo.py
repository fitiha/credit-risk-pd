import json
from pathlib import Path

import pytest

pytest.importorskip("gradio")
from credit_risk.demo import EXAMPLE_LABELS, build_demo, clean_value, score_form  # noqa: E402

pytestmark = pytest.mark.skipif(not Path("models/pd_model.joblib").exists() or not Path("contract/transform.joblib").exists(), reason="model or contract not built")


@pytest.fixture(scope="module")
def demo():
    return build_demo()


def values(demo, name, **change):
    rec = json.loads(Path(f"examples/{name}.json").read_text())
    rec.update(change)
    return [rec.get(n) for n in demo.field_names]


def test_the_form_has_one_input_for_every_field_the_spec_says_is_known_at_application(demo):
    spec_fields = [f["name"] for f in demo.predictor.spec["fields"] if f["known_at_application"]]
    assert demo.field_names == spec_fields and len(spec_fields) == 38
    assert not {"defaulted", "bureau_recovery_flag", "total_pymnt", "recoveries"} & set(demo.field_names)


def test_the_demo_gives_the_same_pd_as_the_predictor_the_tests_use(demo):
    for name in EXAMPLE_LABELS:
        pd_text, decision_html, reasons_md, meta = score_form(demo.predictor, demo.field_names, *values(demo, name))
        direct = demo.predictor.predict(json.loads(Path(f"examples/{name}.json").read_text()))
        assert pd_text == f"{direct['pd']:.1%}" and direct["decision"].upper() in decision_html
        assert meta["model_version"] == direct["model_version"] and meta["contract_version"] == "1.0.0"


def test_bad_input_is_explained_and_nothing_is_scored(demo):
    pd_text, decision_html, reasons_md, meta = score_form(demo.predictor, demo.field_names, *values(demo, "low_risk", annual_inc=58))
    assert pd_text == "-" and "NOT SCORED" in decision_html and "thousands" in reasons_md and meta == {}
    _, html, md, _ = score_form(demo.predictor, demo.field_names, *values(demo, "low_risk", term=48))
    assert "NOT SCORED" in html and "term" in md


def test_the_explanation_says_what_the_pd_means_and_what_moved_it(demo):
    _, _, md, _ = score_form(demo.predictor, demo.field_names, *values(demo, "borderline"))
    assert "About 9 in 100" in md and "manual check" not in md and "reviews the application" in md
    assert "What moved the score" in md and "▲ raises" in md and "Platform risk sub-grade" in md


def test_a_value_outside_the_valid_range_is_explained_instead_of_blocked_by_the_form(demo):
    pd_text, html, md, meta = score_form(demo.predictor, demo.field_names, *values(demo, "borderline", loan_amnt=90000))
    assert pd_text == "-" and "NOT SCORED" in html and "loan_amnt" in md and meta == {}


def test_an_unexpected_error_is_shown_as_not_scored_never_as_a_stack_trace(demo, monkeypatch):
    def broken(record, strict=True):
        raise ValueError("boom")
    monkeypatch.setattr(demo.predictor, "predict", broken)
    pd_text, html, md, meta = score_form(demo.predictor, demo.field_names, *values(demo, "borderline"))
    assert pd_text == "-" and "NOT SCORED" in html and "could not be scored" in md and meta == {}


def test_empty_optional_fields_become_missing_not_empty_text():
    assert clean_value("") is None and clean_value("  ") is None and clean_value(float("nan")) is None and clean_value(0) == 0
    assert clean_value("RENT") == "RENT"
