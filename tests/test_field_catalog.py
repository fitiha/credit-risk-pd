from pathlib import Path

import pandas as pd
import pytest

from credit_risk.field_catalog import CATALOG, catalog_frame

DATA = Path("data")


def test_no_duplicate_fields_and_valid_values():
    df = catalog_frame()
    assert df.field.is_unique
    assert set(df.reliability) <= {"high", "medium", "low", "n/a"}
    assert set(df.role) <= {"input", "platform_score", "restricted", "time_key", "id", "target", "excluded"}
    assert (df.reason.str.len() > 20).all()


def test_nothing_recorded_after_issue_is_usable():
    df = catalog_frame()
    late = df[~df.known_at_application & (df.role != "target")]
    assert (late.role == "excluded").all()
    assert set(df[df.source == "loan_servicing"].field) == set(late[late.source == "loan_servicing"].field)


def test_every_column_of_the_source_files_is_covered():
    files = ["applications", "credit_bureau", "loan_servicing", "holdout_applications"]
    if not all((DATA / f"{f}.csv").exists() for f in files):
        pytest.skip("data files not present")
    known = {r["field"] for r in CATALOG}
    for f in files:
        cols = set(pd.read_csv(DATA / f"{f}.csv", nrows=0).columns)
        assert cols <= known, f"{f}: unrated {cols - known}"
