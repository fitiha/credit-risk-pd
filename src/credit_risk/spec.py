"""The data contract's spec (data_spec.json) and input validation built on it.

The spec is generated from the fitted transformer, so it cannot disagree with transform.joblib. Valid ranges describe
the cleaned value of a field (after the repairs in the cleaning step) and come from plausibility (product limits,
physical bounds, the audit), not from the fit window, so later years and the holdout are not rejected.
"""
import re

import numpy as np
import pandas as pd

from credit_risk.categorical import JOB_RULES, JOB_TITLE_ERA_START
from credit_risk.field_catalog import CATALOG, reliability_spec
from credit_risk.io import FIT_CUTOFF

STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD "
          "TN TX UT VT VA WA WV WI WY").split()
PURPOSES = ["car", "credit_card", "debt_consolidation", "educational", "home_improvement", "house", "major_purchase", "medical", "moving",
            "other", "renewable_energy", "small_business", "vacation", "wedding"]
EMP_LENGTH = ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
SUB_GRADES = [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]


def _num(name, kind, lo, hi, nullable=False):
    return {"name": name, "type": kind, "min": lo, "max": hi, "nullable": nullable}


def _cat(name, allowed, nullable=False):
    return {"name": name, "type": "category", "allowed": allowed, "nullable": nullable}


# raw input fields of one application plus its bureau record (what a caller supplies), with valid ranges of the cleaned value
RAW_FIELDS = [
    {"name": "issue_d", "type": "date", "min": "2007-01-01", "max": "2035-12-31", "nullable": False},
    _num("loan_amnt", "float", 1000, 40000), _cat("term", [36, 60]), _num("int_rate", "float", 5, 31), _num("installment", "float", 15, 1500),
    _cat("grade", list("ABCDEFG")), _cat("sub_grade", SUB_GRADES), _cat("purpose", PURPOSES),
    {"name": "title", "type": "str", "max_length": 200, "nullable": True},
    _cat("application_type", ["Individual", "Joint App"]),
    {"name": "emp_title", "type": "str", "max_length": 200, "nullable": True}, _cat("emp_length", EMP_LENGTH, nullable=True),
    _cat("home_ownership", ["MORTGAGE", "OWN", "RENT", "OTHER", "NONE", "ANY"]),
    _num("annual_inc", "float", 1000, 10_000_000), _cat("verification_status", ["Not Verified", "Source Verified", "Verified"]),
    _num("dti", "float", 0, 150), {"name": "zip_code", "type": "str", "pattern": r"^\d{3}xx$", "nullable": False}, _cat("addr_state", STATES),
    {"name": "earliest_cr_line", "type": "date", "min": "1930-01-01", "max": "2035-12-31", "nullable": True},
    _num("fico_range_low", "float", 300, 850), _num("fico_range_high", "float", 300, 850),
    _num("inq_last_6mths", "int", 0, 40), _num("delinq_2yrs", "int", 0, 60), _num("mths_since_last_delinq", "float", 0, 1000, True),
    _num("pub_rec", "int", 0, 100), _num("pub_rec_bankruptcies", "int", 0, 50), _num("open_acc", "int", 0, 150), _num("total_acc", "int", 0, 300),
    _num("revol_bal", "float", 0, 5_000_000), _num("revol_util", "float", 0, 500, True), _num("bc_util", "float", 0, 500, True),
    _num("tot_cur_bal", "float", 0, 20_000_000), _num("avg_cur_bal", "float", 0, 5_000_000), _num("total_rev_hi_lim", "float", 0, 5_000_000),
    _num("tot_hi_cred_lim", "float", 0, 20_000_000), _num("mort_acc", "int", 0, 100), _num("acc_open_past_24mths", "int", 0, 100),
    _num("num_tl_op_past_12m", "int", 0, 100),
]
REQUIRED_COLUMNS = [f["name"] for f in RAW_FIELDS if not f["nullable"]]
ALL_COLUMNS = [f["name"] for f in RAW_FIELDS]

FORMAT_CLEANING = {
    "issue_d": "parsed from Mon-YYYY, YYYY-MM-DD or MM/YYYY; when scoring it is the application date",
    "earliest_cr_line": "parsed from the same three formats",
    "term": "months extracted from text (' 36 months' becomes 36)",
    "addr_state": "stripped and upper-cased, full state names mapped to codes",
    "emp_title": "stripped and lower-cased; grouped into job groups (a restricted field)",
    "installment": "none; used to detect a loan amount 10x too large",
}
MISSING_TEXT = {
    "emp_title": "missing is its own job group", "emp_length": "missing is its own level", "title": "not used",
    "mths_since_last_delinq": "missing means no delinquency on record; kept as its own bin in WoE",
    "earliest_cr_line": "corrupted dates become missing; credit history length is filled with the fit-window median",
}


def build_spec(transformer):
    """data_spec.json content from a fitted ApplicationTransformer."""
    cl, am, ra = transformer.step("cleaner"), transformer.step("amounts"), transformer.step("ratios")
    co, ft, ca = transformer.step("counts"), transformer.step("feature_treater"), transformer.step("categories")
    fm, we = transformer.step("factors"), transformer.step("woe")
    rules, amounts, ratios, counts = cl.spec_rules(), am.spec_entries(), ra.spec_entries(), co.spec_entries()
    rel = reliability_spec()
    fields = []
    for f in RAW_FIELDS:
        n = f["name"]
        pieces = [FORMAT_CLEANING[n]] if n in FORMAT_CLEANING else []
        pieces += [rules[k] for k in (n, "fico_range" if n.startswith("fico_range") else None) if k in rules]
        missing = MISSING_TEXT.get(n, "not allowed: a missing value is rejected" if not f["nullable"] else "kept as missing")
        for src in (amounts, ratios, counts):
            if n in src:
                pieces.append(src[n]["cleaning"])
                missing = src[n]["missing_strategy"] if "missing_strategy" in src[n] else missing
        entry = dict(f)
        entry.update({"cleaning": "; ".join(pieces) if pieces else "none", "missing_strategy": missing,
                      "reliability": rel[n]["reliability"], "reliability_reason": rel[n]["reliability_reason"],
                      "role": rel[n]["role"], "known_at_application": rel[n]["known_at_application"]})
        fields.append(entry)
    pc, fa = fm.spec_entries()["pca"], fm.spec_entries()["factors"]
    return {
        "contract_version": transformer.contract_version,
        "fit_cutoff": str(FIT_CUTOFF.date()),
        "label": {"name": "defaulted", "definition": "1 = charged off or defaulted, 0 = fully paid; loans still in progress were removed"},
        "notes": ["Every fitted value (caps, medians, groups, bins, WoE, scaler, PCA, factor model) comes from loans issued up to fit_cutoff only.",
                  "WoE = ln(share of repaid loans / share of defaulted loans) in the group: positive means safer than average; unseen values get 0.",
                  "Valid ranges describe the cleaned value of a field and come from plausibility, not from the fit window.",
                  "issue_d is the application date when scoring; credit history length is measured from it.",
                  "Percent fields (revol_util, bc_util) must be in percent (91, not 0.91); a whole batch in fractions is rejected."],
        "fields": fields,
        "treated_fields": {"money": amounts, "ratios": ratios, "counts": counts, "features": ft.spec_entries()},
        "category_groupings": {"grouping": ca.spec_entries(), "job_groups": {"rules_in_priority_order": {n: p for n, p in JOB_RULES},
                              "employer_name_era": f"loans issued before {JOB_TITLE_ERA_START.date()} (the field held employer names)",
                              "other": "other_job", "missing": "missing"},
                              "emp_length_order": EMP_LENGTH},
        "cleaning_rules": rules,
        "woe": we.spec_entries(),
        "pca": pc,
        "factors": fa,
        "excluded_fields": [{"name": r["field"], "source": r["source"], "reason": r["reason"]} for r in CATALOG if r["role"] == "excluded"],
        "platform_score_fields": ["int_rate", "grade", "sub_grade"],
        "restricted_fields": [r["field"] for r in CATALOG if r["role"] == "restricted"],
    }


class ContractViolation(ValueError):
    """Input data breaks the contract."""


def _problems(df, fields, require_all):
    out = []
    for f in fields:
        n = f["name"]
        if n not in df.columns:
            if require_all and not f["nullable"]:
                out.append(f"missing required field '{n}'")
            continue
        s = df[n]
        bad_missing = s.isna() & (not f["nullable"])
        if bad_missing.any():
            out.append(f"'{n}': {int(bad_missing.sum())} missing value(s) but the field is required")
        v = s.dropna()
        if f["type"] in ("float", "int"):
            num = pd.to_numeric(v, errors="coerce")
            if num.isna().any():
                out.append(f"'{n}': {int(num.isna().sum())} value(s) are not numbers (e.g. {v[num.isna()].iloc[0]!r})")
            bad = num.notna() & ((num < f["min"]) | (num > f["max"]))
            if bad.any():
                out.append(f"'{n}': {int(bad.sum())} value(s) outside {f['min']} to {f['max']} (e.g. {num[bad].iloc[0]})")
        elif f["type"] == "date":
            d = pd.to_datetime(v, errors="coerce")
            bad = d.notna() & ((d < f["min"]) | (d > f["max"]))
            if d.isna().any() or bad.any():
                out.append(f"'{n}': {int(d.isna().sum() + bad.sum())} date(s) unreadable or outside {f['min']} to {f['max']}")
        elif f["type"] == "category":
            bad = ~v.isin(f["allowed"])
            if bad.any():
                out.append(f"'{n}': {int(bad.sum())} value(s) not allowed (e.g. {v[bad].iloc[0]!r}); allowed: {f['allowed'][:6]}{'...' if len(f['allowed']) > 6 else ''}")
        elif f["type"] == "str":
            sv = v.astype(str)
            if "pattern" in f and (~sv.str.match(f["pattern"])).any():
                out.append(f"'{n}': {int((~sv.str.match(f['pattern'])).sum())} value(s) do not match the pattern {f['pattern']} (e.g. {sv[~sv.str.match(f['pattern'])].iloc[0]!r})")
            if "max_length" in f and (sv.str.len() > f["max_length"]).any():
                out.append(f"'{n}': text longer than {f['max_length']} characters")
    return out


def validate_raw(df):
    """Check that the raw application columns are present and basically readable, before any cleaning."""
    miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if miss:
        raise ContractViolation("missing required field(s): " + ", ".join(miss))
    return df


def validate_cleaned(df, spec=None):
    """Fail fast if any cleaned field is outside the spec's valid range or category list. Returns df unchanged."""
    problems = _problems(df, RAW_FIELDS, require_all=True)
    if problems:
        raise ContractViolation("input breaks the data contract:\n  - " + "\n  - ".join(problems[:12]))
    return df
