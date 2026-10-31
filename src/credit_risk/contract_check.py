"""Parity check: does transform.joblib reproduce reference_transforms.csv?

The reference file holds raw application rows (as a caller would supply them) and the expected output of every
transformed column (prefix `expected_`). Numbers must agree to 1e-6; text, dates and flags must be identical.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk.spec import ALL_COLUMNS, validate_cleaned, validate_raw

TOLERANCE = 1e-6
STRING_COLUMNS = ["id", "issue_d", "term", "grade", "sub_grade", "purpose", "title", "application_type", "emp_title", "emp_length",
                  "home_ownership", "verification_status", "zip_code", "addr_state", "earliest_cr_line"]


def read_reference(path):
    """Read the reference file keeping text columns as text (ids and dates must not be turned into numbers)."""
    header = pd.read_csv(path, nrows=0).columns
    text = [c for c in header if c in STRING_COLUMNS or (c.startswith("expected_") and c[9:] in STRING_COLUMNS)
            or c.startswith("expected_job_group") or c.endswith("_group") or c.startswith("expected_") and c.endswith("_group")]
    return pd.read_csv(path, dtype={c: str for c in text}, keep_default_na=True)


def _same(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    try:
        x, y = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        return bool(np.all(both_nan | (np.abs(np.nan_to_num(x) - np.nan_to_num(y)) <= TOLERANCE))), float(np.nanmax(np.abs(np.where(both_nan, 0, x - y))))
    except (ValueError, TypeError):
        eq = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
        return bool(eq.all()), 0.0 if eq.all() else float("nan")


def verify_contract(contract_dir="contract"):
    """Returns (ok, report lines). Also confirms that all five contract files exist."""
    contract_dir = Path(contract_dir)
    lines, ok = [], True
    for name in ["analytical_dataset.parquet", "transform.joblib", "data_spec.json", "reference_transforms.csv", "findings.md"]:
        exists = (contract_dir / name).exists()
        ok &= exists
        lines.append(f"{'ok  ' if exists else 'MISSING'} {name}")
    if not ok:
        return False, lines
    transformer = joblib.load(contract_dir / "transform.joblib")
    ref = read_reference(contract_dir / "reference_transforms.csv")
    raw = ref[[c for c in ref.columns if not c.startswith("expected_") and c != "defaulted"]]
    validate_raw(raw)
    out = transformer.transform(raw)
    validate_cleaned(out)
    bad, worst = [], 0.0
    expected = [c for c in ref.columns if c.startswith("expected_")]
    for c in expected:
        same, diff = _same(out[c[len("expected_"):]], ref[c])
        if not same:
            bad.append(c)
        worst = max(worst, diff if np.isfinite(diff) else 0.0)
    lines.append(f"{len(ref)} reference rows, {len(expected)} expected columns compared at tolerance {TOLERANCE}; largest numeric difference {worst:.2e}")
    if bad:
        ok = False
        lines.append("MISMATCH in: " + ", ".join(bad[:10]) + (" ..." if len(bad) > 10 else ""))
    else:
        lines.append("parity: transform.joblib reproduces reference_transforms.csv")
    return ok, lines
