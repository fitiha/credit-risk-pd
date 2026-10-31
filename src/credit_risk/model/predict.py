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

from credit_risk.model import policy
from credit_risk.model.data import load_contract
from credit_risk.model.inputs import reason_text
from credit_risk.model.models import add_grade_rank
from credit_risk.spec import NUMERIC_FIELDS, RAW_FIELDS, ContractViolation, validate_cleaned, validate_raw

MIN_REASON_LOGODDS = 0.10   # a reason must raise the odds of default by roughly 10% or more; smaller contributions are not worth telling the applicant
MAX_REASONS = 4


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
        validate_raw(df)
        df = df.assign(**{c: pd.to_numeric(df[c]) for c in NUMERIC_FIELDS if c in df.columns})  # numbers sent as text ("8000") become numbers
        out = self.transformer.transform(df)
        validate_cleaned(out)
        if strict and (problems := repairs_needed(df, out)):
            raise ContractViolation("the application looks mis-entered, please correct it:\n  - " + "\n  - ".join(problems))
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
