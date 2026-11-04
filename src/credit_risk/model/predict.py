"""Score one application: PD, decision, model version and reason codes.

    uv run python -m credit_risk.model.predict --json application.json

The input is one raw application with its bureau record (the fields in data_spec.json). It is validated against the
spec before and after cleaning, scored with the contract's transform and the saved model, and the top reasons that push
PD up are returned in plain words. A model built on a different contract version than the transform is refused.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk.io import parse_dates
from credit_risk.model import policy
from credit_risk.model.data import load_contract
from credit_risk.model.inputs import REASONS, reason_text
from credit_risk.model.models import add_grade_rank
from credit_risk.spec import NUMERIC_FIELDS, RAW_FIELDS, ContractViolation, validate_cleaned, validate_raw

MIN_REASON_LOGODDS = 0.10   # a reason must raise the odds of default by roughly 10% or more; smaller contributions are not worth telling the applicant
MAX_REASONS = 4
DATE_FIELDS = [f["name"] for f in RAW_FIELDS if f["type"] == "date"]
LOW_END_WARNINGS = {"fico_mid", "annual_inc"}  # fields where a value below the fitted floor deserves a warning

# Pairs that cannot both be true: the first count or limit can never exceed the second. None of the 30,000 historical loans breaks any of them.
IMPOSSIBLE = [
    ("open_acc", "total_acc", "open_acc ({a:,.0f}) is larger than total_acc ({b:,.0f}): open accounts are part of all accounts"),
    ("num_tl_op_past_12m", "acc_open_past_24mths",
     "num_tl_op_past_12m ({a:,.0f}) is larger than acc_open_past_24mths ({b:,.0f}): accounts opened in 12 months are part of those opened in 24 months"),
    ("pub_rec_bankruptcies", "pub_rec", "pub_rec_bankruptcies ({a:,.0f}) is larger than pub_rec ({b:,.0f}): a bankruptcy is a public record"),
    ("total_rev_hi_lim", "tot_hi_cred_lim", "total_rev_hi_lim ({a:,.0f}) is larger than tot_hi_cred_lim ({b:,.0f}): the revolving limit is part of the total limit"),
]


REPAIR_MESSAGES = {
    "loan_amnt_fixed": "loan_amnt looks 10 times too large for the installment, rate and term",
    "annual_inc_fixed": "annual_inc is below 1,000 and looks like it was entered in thousands",
    "revol_bal_fixed": "revol_bal looks 10 times too large for the revolving limit, utilisation and total balance",
    "earliest_cr_line_invalid": "earliest_cr_line is later than the application date",
    "bc_util_invalid": "bc_util is negative",
    "int_rate_imputed": "int_rate is the placeholder value 6.00",
}


def repairs_needed(raw, cleaned):
    """What the cleaning step had to correct in these rows. Fine for historical data; a live application should be corrected at the source."""
    problems = [msg for flag, msg in REPAIR_MESSAGES.items() if flag in cleaned and bool(cleaned[flag].any())]
    if (pd.to_numeric(raw["fico_range_low"]) > pd.to_numeric(raw["fico_range_high"])).any():
        problems.append("fico_range_low is above fico_range_high")
    if (pd.to_numeric(raw["revol_util"]) < 0).any():
        problems.append("revol_util is negative")
    return problems


def unreadable_dates(raw):
    """Dates that were filled in but cannot be read. The cleaning step would treat them as missing, which hides the mistake."""
    problems = []
    for c in DATE_FIELDS:
        if c in raw.columns:
            v = raw[c]
            given = v.notna() & (v.astype(str).str.strip() != "")
            bad = given & parse_dates(v.where(given)).isna()
            if bad.any():
                problems.append(f"{c} could not be read as a date (got {v[bad].iloc[0]!r}); use month and year, e.g. Jun-2015")
    return problems


def negative_values(raw):
    """Amounts, rates and counts below zero. Said plainly here, because the cleaning rules would otherwise rescale or flip them first."""
    return [f"{c} cannot be negative (got {pd.to_numeric(raw[c])[pd.to_numeric(raw[c]) < 0].iloc[0]:,g})"
            for c in NUMERIC_FIELDS if c in raw.columns and (pd.to_numeric(raw[c]) < 0).any()]


def impossible_combinations(raw):
    """Fields that are each in range but cannot be true together."""
    problems = []
    for a, b, msg in IMPOSSIBLE:
        if a in raw.columns and b in raw.columns:
            x, y = pd.to_numeric(raw[a]), pd.to_numeric(raw[b])
            bad = x > y
            if bad.any():
                i = bad.idxmax()
                problems.append(msg.format(a=x[i], b=y[i]))
    if {"grade", "sub_grade"} <= set(raw.columns):
        g, s = raw["grade"].astype(str).str.strip().str.upper(), raw["sub_grade"].astype(str).str.strip().str.upper()
        bad = raw["grade"].notna() & raw["sub_grade"].notna() & (g != s.str[0])
        if bad.any():
            i = bad.idxmax()
            problems.append(f"grade {g[i]} does not match sub_grade {s[i]}: the sub-grade starts with the grade letter")
    return problems


def _reject(problems):
    raise ContractViolation("the application looks mis-entered, please correct it:\n  - " + "\n  - ".join(problems))


def _strength(logodds):
    """1 to 5 for how strongly one factor moves the score (log-odds, before calibration)."""
    a = abs(logodds)
    return 1 if a < 0.05 else 2 if a < 0.10 else 3 if a < 0.25 else 4 if a < 0.50 else 5


class ContractVersionMismatch(RuntimeError):
    """The saved model was built on a different data contract than the transform it is paired with."""


class Predictor:
    def __init__(self, model_path="models/pd_model.joblib", contract_dir="contract"):
        self.transformer, self.spec = load_contract(contract_dir)
        self.bundle = joblib.load(model_path)
        if self.bundle.contract_version != self.transformer.contract_version or self.bundle.contract_version != self.spec["contract_version"]:
            raise ContractVersionMismatch(f"model built on contract {self.bundle.contract_version}, transform is {self.transformer.contract_version}, "
                                          f"spec is {self.spec['contract_version']}")
        meta = self.bundle.metadata
        self.model_version = f"{meta['model']}-contract{meta['contract_version']}-{meta['training_date']}-{meta['git_commit'][:7]}"

    @staticmethod
    def _frame(records):
        df = pd.DataFrame(records if isinstance(records, list) else [records])
        for f in RAW_FIELDS:  # optional fields may simply be left out of the JSON; required ones may not
            if f["name"] not in df.columns and f["nullable"]:
                df[f["name"]] = np.nan
        return df

    def frame(self, records, strict=False):
        """Raw record(s) -> validated modelling frame. With strict=True an application that needs any repair is rejected."""
        df = self._frame(records)
        if strict and (problems := unreadable_dates(df)):
            _reject(problems)
        validate_raw(df)
        df = df.assign(**{c: pd.to_numeric(df[c]) for c in NUMERIC_FIELDS if c in df.columns})  # numbers sent as text ("8000") become numbers
        if strict and (problems := negative_values(df) + impossible_combinations(df)):
            _reject(problems)
        out = self.transformer.transform(df)
        validate_cleaned(out)
        if strict and (problems := repairs_needed(df, out)):
            _reject(problems)
        return add_grade_rank(out)

    def reasons(self, frame_row):
        model = self.bundle.model
        contrib = model.contributions(frame_row)[0]
        pairs = sorted(zip(model.columns, contrib), key=lambda x: -x[1])
        out = []
        for column, value in pairs:
            if value < MIN_REASON_LOGODDS or len(out) >= MAX_REASONS:
                break
            source = self.bundle.reason_map.get(column, column)
            raw = frame_row[source].iloc[0] if source in frame_row.columns else None
            text = reason_text(source, None if (isinstance(raw, float) and np.isnan(raw)) else raw, self.bundle.reference.get(source))
            if text not in out:
                out.append(text)
        return out

    def predict(self, record, strict=True):
        """record: dict of raw application and bureau fields. Returns a dict with pd, decision, model_version and reason_codes.

        strict=True (the default for live scoring) rejects an application that looks mis-entered; strict=False repairs it the way the
        contract repairs historical data."""
        frame = self.frame(record, strict=strict)
        p = float(self.bundle.score(frame)[0])
        return {"pd": round(p, 4), "decision": str(policy.decide([p], self.bundle.thresholds)[0]), "model_version": self.model_version,
                "reason_codes": self.reasons(frame)}

    def breakdown(self, frame_row, n=6, min_logodds=0.01):
        """The factors that move this applicant's score most, up or down, against a typical approved applicant.

        Same contributions as the reason codes, in both directions. Returns dicts with factor, value, typical, direction
        (+1 raises risk, -1 lowers it) and strength (1 to 5)."""
        model = self.bundle.model
        effect = {}
        for column, v in zip(model.columns, model.contributions(frame_row)[0]):
            source = self.bundle.reason_map.get(column, column)
            effect[source] = effect.get(source, 0.0) + float(v)
        rows = []
        for source, v in sorted(effect.items(), key=lambda x: -abs(x[1])):
            if abs(v) < min_logodds or len(rows) >= n:
                break
            label, fmt = REASONS[source]
            raw = frame_row[source].iloc[0] if source in frame_row.columns else None
            ref = self.bundle.reference.get(source)
            missing = raw is None or (isinstance(raw, float) and np.isnan(raw))
            has_ref = ref is not None and not isinstance(ref, str) and not pd.isna(ref)
            if source.startswith("factor_"):  # a factor score has no natural unit: say how it compares instead
                value, typical = ("not available" if missing else "higher than typical" if has_ref and raw > ref else "lower than typical"), ""
            else:
                value, typical = ("not provided" if missing else fmt(raw)), (fmt(ref) if has_ref else "")
            rows.append({"factor": label, "value": value, "typical": typical, "direction": 1 if v > 0 else -1, "strength": _strength(v)})
        return rows

    def input_warnings(self, record):
        """Values that are scored but deserve a second look: outside what the model learned from, or inconsistent with other fields.

        These do not stop scoring (a few genuine applications look like this); impossible combinations are rejected instead."""
        df = self._frame(record)
        num = {c: pd.to_numeric(df[c], errors="coerce").iloc[0] for c in NUMERIC_FIELDS if c in df.columns}
        treated = self.spec["treated_fields"]
        out = []

        def label_fmt(field):
            for key in (f"{field}_capped", f"{field}_treated", field):
                if key in REASONS:
                    return REASONS[key]
            return field, (lambda v: f"{v:,.0f}")

        def outside(name, value, lo, hi, label, fmt):
            if pd.isna(value):
                return
            if name not in LOW_END_WARNINGS:  # a zero balance or a low ratio is ordinary; only a very low score or income needs a word
                lo = None
            if hi is not None and value > hi:
                out.append(f"{label}: {fmt(value)} is above what the model learned from (99.5% of 2013-2014 applicants had at most {fmt(hi)}); "
                           f"it is scored as {fmt(hi)}.")
            elif lo is not None and value < lo:
                out.append(f"{label}: {fmt(value)} is below what the model learned from (99.5% of 2013-2014 applicants had at least {fmt(lo)}); "
                           f"it is scored as {fmt(lo)}.")

        for group in ("money", "ratios", "counts"):
            for field, rule in treated[group].items():
                if field in num:
                    label, fmt = label_fmt(field)
                    outside(field, num[field], rule.get("cap_low"), rule.get("cap_high"), label, fmt)
        if "fico_range_low" in num and "fico_range_high" in num:
            rule = treated["features"]["fico_mid"]
            outside("fico_mid", (num["fico_range_low"] + num["fico_range_high"]) / 2, rule.get("cap_low"), rule.get("cap_high"),
                    "Credit score (FICO, middle of the range)", lambda v: f"{v:,.0f}")
        util, bal, lim = num.get("revol_util"), num.get("revol_bal"), num.get("total_rev_hi_lim")
        if all(pd.notna(x) for x in (util, bal, lim)) and lim > 0 and 0 <= util < 1.5 and 100 * bal / lim > 10:
            out.append(f"revol_util is {util:g}, but the revolving balance and limit give about {100 * bal / lim:.0f}%: it may have been entered as a "
                       f"fraction instead of a percent (16 of 30,000 historical loans look like this).")
        amt, rate, term, inst = num.get("loan_amnt"), num.get("int_rate"), df["term"].iloc[0] if "term" in df.columns else None, num.get("installment")
        term = pd.to_numeric(pd.Series([term]).astype(str).str.extract(r"(\d+)")[0], errors="coerce").iloc[0]
        if all(pd.notna(x) for x in (amt, rate, term, inst)) and rate > 0:
            r = rate / 1200
            expected = amt * r / (1 - (1 + r) ** -term)
            if abs(inst / expected - 1) > 0.25:
                out.append(f"installment is ${inst:,.2f}, far from the ${expected:,.2f} that the loan amount, rate and term imply; please check it "
                           f"(1 of 30,000 historical loans differs this much).")
        return out

    def predict_frame(self, frame):
        """Batch scoring of an already validated modelling frame."""
        return self.bundle.score(frame)


def main():
    ap = argparse.ArgumentParser(description="Score one application given as JSON.")
    ap.add_argument("--json", required=True, help="path to a JSON file with one application, or '-' for standard input")
    ap.add_argument("--model", default="models/pd_model.joblib")
    ap.add_argument("--contract", default="contract")
    args = ap.parse_args()
    text = Path(args.json).read_text() if args.json != "-" else __import__("sys").stdin.read()
    print(json.dumps(Predictor(args.model, args.contract).predict(json.loads(text)), indent=2))


if __name__ == "__main__":
    main()
