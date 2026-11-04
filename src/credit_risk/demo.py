"""The Gradio demo. The form is built from data_spec.json and scoring calls the same Predictor as the tests and the command line.

Only made-up applicants (examples/*.json) are used; no real loan rows are ever loaded or shown.
"""
import json
import sys
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
LABELS = {
    "issue_d": "Application date", "loan_amnt": "Loan amount ($)", "term": "Term (months)", "int_rate": "Interest rate (%)",
    "installment": "Monthly installment ($)", "grade": "Platform grade", "sub_grade": "Platform sub-grade", "purpose": "Loan purpose",
    "title": "Loan title", "application_type": "Application type", "emp_title": "Job title", "emp_length": "Employment length",
    "home_ownership": "Home ownership", "annual_inc": "Annual income ($)", "verification_status": "Income verification",
    "dti": "Debt-to-income ratio (%)", "zip_code": "ZIP code (first 3 digits)", "addr_state": "State",
    "earliest_cr_line": "First credit account opened", "fico_range_low": "FICO score, low end", "fico_range_high": "FICO score, high end",
    "inq_last_6mths": "Credit inquiries, last 6 months", "delinq_2yrs": "Delinquencies, last 2 years",
    "mths_since_last_delinq": "Months since last delinquency", "pub_rec": "Public records", "pub_rec_bankruptcies": "Bankruptcies",
    "open_acc": "Open accounts", "total_acc": "Total accounts", "revol_bal": "Revolving balance ($)", "revol_util": "Revolving utilisation (%)",
    "bc_util": "Bank-card utilisation (%)", "tot_cur_bal": "Total current balance ($)", "avg_cur_bal": "Average account balance ($)",
    "total_rev_hi_lim": "Total revolving credit limit ($)", "tot_hi_cred_lim": "Total credit limit ($)", "mort_acc": "Mortgage accounts",
    "acc_open_past_24mths": "Accounts opened, last 24 months", "num_tl_op_past_12m": "Accounts opened, last 12 months",
}
HELP = {"issue_d": "Application date, month and year (e.g. Jun-2015). Credit history length is measured from it.",
        "revol_util": "In percent (55 means 55%).", "bc_util": "In percent (55 means 55%).", "zip_code": "First three digits and 'xx' (e.g. 100xx).",
        "emp_title": "Free text; not used by the model.", "title": "Free text; not used by the model.", "mths_since_last_delinq": "Leave empty if there is no delinquency on record."}


def _component(f, default):
    """One form input from its data_spec.json entry. Ranges are shown, not enforced by the widget: the Predictor checks them, so
    every mistake gets the same plain message and the previous result is cleared."""
    n = f["name"]
    label = f"{LABELS.get(n, n)} · {n}"
    info = HELP.get(n, "")
    if f["type"] in ("float", "int"):
        rng = f"valid {f['min']:,} to {f['max']:,}" + ("" if not f["nullable"] else "; may be left empty")
        return gr.Number(label=label, value=default, precision=0 if f["type"] == "int" else None, info=(info + " " + rng).strip())
    if f["type"] == "category":
        choices = list(f["allowed"]) + ([None] if f["nullable"] else [])
        return gr.Dropdown(label=label, choices=choices, value=default, info=info or ("may be left empty" if f["nullable"] else ""))
    note = {"date": "date as Mon-YYYY, e.g. Jun-2015", "str": "text"}[f["type"]]
    return gr.Textbox(label=label, value=default, info=(info or note) + ("; may be left empty" if f["nullable"] else ""))


def clean_value(v):
    if v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


NOT_SCORED = "<div style='font-size:1.4em;font-weight:700;color:#b3261e'>NOT SCORED</div>"


def _pct(x):
    return f"{100 * x:g}%"


def summary(p, decision, thresholds):
    """The PD in words and why it leads to this decision, using the model's own thresholds."""
    share = "Fewer than 1 in 100" if p < 0.005 else f"About {round(100 * p)} in 100"
    approve, decline = _pct(thresholds["approve_below"]), _pct(thresholds["decline_at_or_above"])
    why = {"approve": f"That is below {approve}, the level up to which a loan is safe enough to approve automatically, so it is approved without a manual check.",
           "review": f"That is above {approve}, the limit for automatic approval, but below {decline}, the point where a loan stops paying for itself. "
                     "The loan is still expected to make money, but only a little, so a person reviews the application.",
           "decline": f"That is above {decline}, the point where a loan stops paying for itself: the expected losses from defaults outweigh the interest earned, "
                      "so the application is declined."}
    rules = (f"*How decisions are made: approve automatically below {approve} · send to a person between {approve} and {decline} · "
             f"decline at {decline} or more.*")
    return f"**{share} applicants with this profile do not repay.** {why[decision]}\n\n{rules}"


def breakdown_table(rows, reference_group):
    if not rows:
        return ""
    lines = ["**What moved the score**, compared with a typical approved applicant", "",
             "| | Factor | This applicant | Typical approved applicant |", "|---|---|---|---|"]
    for r in rows:
        arrow = "▲ raises" if r["direction"] > 0 else "▼ lowers"
        bars = "■" * r["strength"] + "□" * (5 - r["strength"])
        lines.append(f"| {arrow} {bars} | {r['factor']} | {r['value']} | {r['typical'] or '–'} |")
    lines += ["", f"*A typical approved applicant is the average of the {reference_group}. The bars show how strongly each factor moves the score.*"]
    return "\n".join(lines)


def format_result(result, rows=(), warnings=(), thresholds=None, reference_group="training applicants the model approves"):
    p = result["pd"]
    colour = {"approve": "#1b7f3b", "review": "#b26b00", "decline": "#b3261e"}[result["decision"]]
    decision = f"<div style='font-size:1.6em;font-weight:700;color:{colour}'>{result['decision'].upper()}</div>"
    parts = []
    if thresholds:
        parts.append(summary(p, result["decision"], thresholds))
    if warnings:
        parts.append("**Please check these values** (the application was scored, but the PD is less certain):\n\n" + "\n".join(f"- {w}" for w in warnings))
    reasons = "\n".join(f"- {r}" for r in result["reason_codes"]) or "- None: no factor raises the risk noticeably above a typical approved applicant."
    if any(r.startswith("Platform risk sub-grade") for r in result["reason_codes"]):
        reasons += "\n\n*The sub-grade is the platform's own risk grade; the model builds on it and adds the applicant's own details.*"
    parts.append("**Main reasons the risk is higher than for a typical approved applicant**\n\n" + reasons)
    if rows:
        parts.append(breakdown_table(rows, reference_group))
    return f"{p:.1%}", decision, "\n\n".join(parts)


def score_form(predictor, names, *values):
    """Form values -> (pd text, decision html, explanation markdown, model metadata). Bad input is explained, never a stack trace."""
    record = {n: clean_value(v) for n, v in zip(names, values)}
    try:
        result = predictor.predict(record)
        frame = predictor.frame(record, strict=True)
        rows, warnings = predictor.breakdown(frame), predictor.input_warnings(record)
    except ContractViolation as e:
        lines = [l.strip(" -") for l in str(e).splitlines()[1:]] or [str(e)]
        return "-", NOT_SCORED, "**Please correct the input:**\n\n" + "\n".join(f"- {l}" for l in lines), {}
    except Exception as e:  # a safety net: never show a stack trace, and never leave the previous applicant's result on screen
        print(f"unexpected error while scoring: {type(e).__name__}: {e}", file=sys.stderr)
        return "-", NOT_SCORED, f"**This application could not be scored.** Please check the values and try again ({type(e).__name__}).", {}
    pd_text, decision, reasons = format_result(result, rows, warnings, predictor.bundle.thresholds,
                                               predictor.bundle.metadata.get("reference_group", "training applicants the model approves"))
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
        meta_out = gr.JSON(label="Model details")
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
