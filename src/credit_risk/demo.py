"""The Gradio demo. The form is built from data_spec.json and scoring calls the same Predictor as the tests and the command line.

Only made-up applicants (examples/*.json) are used; no real loan rows are ever loaded or shown.
"""
import datetime as dt
import json
from pathlib import Path

import gradio as gr
import numpy as np

from credit_risk.model.predict import Predictor
from credit_risk.spec import ContractViolation

SECTIONS = {
    "The loan": ["issue_d", "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "purpose", "title", "application_type"],
    "The applicant": ["emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status", "dti", "zip_code", "addr_state"],
}
EXAMPLE_LABELS = {"low_risk": "Low-risk applicant", "borderline": "Borderline applicant", "high_risk": "High-risk applicant"}
HELP = {"issue_d": "Application date, month and year (e.g. Jun-2015). Credit history length is measured from it.",
        "revol_util": "In percent (55 means 55%).", "bc_util": "In percent (55 means 55%).", "zip_code": "First three digits and 'xx' (e.g. 100xx).",
        "emp_title": "Free text; not used by the model.", "title": "Free text; not used by the model.", "mths_since_last_delinq": "Leave empty if there is no delinquency on record."}


def _component(f, default):
    n = f["name"]
    info = HELP.get(n, "")
    if f["type"] in ("float", "int"):
        rng = f"valid {f['min']:,} to {f['max']:,}" + ("" if not f["nullable"] else "; may be left empty")
        return gr.Number(label=n, value=default, minimum=f["min"], maximum=f["max"], precision=0 if f["type"] == "int" else None, info=(info + " " + rng).strip())
    if f["type"] == "category":
        choices = list(f["allowed"]) + ([None] if f["nullable"] else [])
        return gr.Dropdown(label=n, choices=choices, value=default, info=info or ("may be left empty" if f["nullable"] else ""))
    note = {"date": "date as Mon-YYYY, e.g. Jun-2015", "str": "text"}[f["type"]]
    return gr.Textbox(label=n, value=default, info=(info or note) + ("; may be left empty" if f["nullable"] else ""))


def clean_value(v):
    if v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def format_result(result):
    p = result["pd"]
    colour = {"approve": "#1b7f3b", "review": "#b26b00", "decline": "#b3261e"}[result["decision"]]
    decision = f"<div style='font-size:1.6em;font-weight:700;color:{colour}'>{result['decision'].upper()}</div>"
    reasons = "\n".join(f"- {r}" for r in result["reason_codes"]) or "No single factor raises the risk noticeably."
    return f"{p:.1%}", decision, "**Main reasons the risk is higher than for a typical approved applicant**\n\n" + reasons


def score_form(predictor, names, *values):
    """Form values -> (pd text, decision html, reasons markdown, model metadata). Bad input is explained, never a stack trace."""
    record = {n: clean_value(v) for n, v in zip(names, values)}
    try:
        result = predictor.predict(record)
    except ContractViolation as e:
        lines = [l.strip(" -") for l in str(e).splitlines()[1:]] or [str(e)]
        return "-", "<div style='font-size:1.4em;font-weight:700;color:#b3261e'>NOT SCORED</div>", "**Please correct the input:**\n\n" + "\n".join(f"- {l}" for l in lines), {}
    pd_text, decision, reasons = format_result(result)
    m = predictor.bundle.metadata
    meta = {"model_version": result["model_version"], "model": m["model"], "training_date": m["training_date"], "git_commit": m["git_commit"][:12],
            "contract_version": m["contract_version"], "trained_on": f"{m['training_loans']:,} loans issued {m['training_years'][0]} to {m['training_years'][1]}",
            "decision_thresholds": m["thresholds"]}
    return pd_text, decision, reasons, meta


def build_demo(model_path="models/pd_model.joblib", contract_dir="contract", examples_dir="examples"):
    predictor = Predictor(model_path, contract_dir)
    fields = {f["name"]: f for f in predictor.spec["fields"] if f["known_at_application"]}
    examples = {n: json.loads(Path(examples_dir, f"{n}.json").read_text()) for n in EXAMPLE_LABELS}
    base = examples["borderline"]
    names = list(fields)
    with gr.Blocks(title="Personal-loan default risk (demo)") as demo:
        gr.Markdown("# Personal-loan default risk\nEnter an application and its credit-bureau record, or press an example button. The model returns the "
                    "probability of default (PD), a decision and the main reasons. Examples are made-up applicants; no real loan data is stored here.\n\n"
                    f"*Model: {predictor.model_version}. Training window: 2013 to 2014 loans. See the model card below for limits.*")
        with gr.Row():
            buttons = {n: gr.Button(label) for n, label in EXAMPLE_LABELS.items()}
        comps = {}
        sections = dict(SECTIONS)
        sections["Credit bureau record"] = [n for n in names if n not in sum(SECTIONS.values(), [])]
        for title, members in sections.items():
            with gr.Accordion(title, open=(title != "Credit bureau record")):
                for i in range(0, len(members), 3):
                    with gr.Row():
                        for n in members[i:i + 3]:
                            default = base.get(n)
                            comps[n] = _component(fields[n], default)
        go = gr.Button("Score this application", variant="primary")
        with gr.Row():
            pd_out = gr.Textbox(label="Probability of default (PD)", interactive=False)
            decision_out = gr.HTML(label="Decision")
        reasons_out = gr.Markdown()
        meta_out = gr.JSON(label="Model information")
        ordered = [comps[n] for n in names]
        go.click(lambda *v: score_form(predictor, names, *v), inputs=ordered, outputs=[pd_out, decision_out, reasons_out, meta_out])
        for n, b in buttons.items():
            b.click(lambda n=n: [examples[n].get(k) for k in names], outputs=ordered).then(lambda *v: score_form(predictor, names, *v), inputs=ordered,
                                                                                         outputs=[pd_out, decision_out, reasons_out, meta_out])
        card = (Path(model_path).resolve().parent.parent / "model_card.md")
        with gr.Accordion("Model card", open=False):
            gr.Markdown(card.read_text() if card.exists() else "See model_card.md.")
    demo.predictor, demo.field_names = predictor, names
    return demo
