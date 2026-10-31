"""The contract folder is self-contained, so these tests run without the training data."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.contract_check import read_reference, verify_contract

CONTRACT = Path("contract")
pytestmark = pytest.mark.skipif(not (CONTRACT / "transform.joblib").exists(), reason="contract not built")


def test_the_committed_contract_passes_the_parity_check():
    ok, lines = verify_contract(CONTRACT)
    assert ok, "\n".join(lines)


def test_the_reference_file_covers_every_kind_of_repair_and_missing_pattern():
    ref = read_reference(CONTRACT / "reference_transforms.csv")
    why = set(ref.why_this_row.str.replace(r", \d{4}", "", regex=True))
    for needed in ["loan amount 10x too large", "income in thousands", "revolving balance 10x too large", "credit date after the loan",
                   "negative bank-card utilisation", "placeholder interest rate", "negative revolving utilisation", "FICO low above high",
                   "no bank-card utilisation in the bureau file", "no employer title", "no employment length", "before the October 2013 title change"]:
        assert needed in why, needed
    assert len(ref) == 50 and ref.expected_loan_amnt.max() <= 40000


@pytest.mark.parametrize("column,value", [("expected_factor_credit_capacity", 0.123), ("expected_loan_amnt", 1.0), ("expected_job_group", "healthcare")])
def test_the_parity_check_catches_a_changed_expected_value(tmp_path, column, value):
    shutil.copytree(CONTRACT, tmp_path / "c")
    ref = read_reference(tmp_path / "c" / "reference_transforms.csv")
    ref.loc[ref.index[3], column] = value
    ref.to_csv(tmp_path / "c" / "reference_transforms.csv", index=False)
    ok, lines = verify_contract(tmp_path / "c")
    assert not ok and column in "\n".join(lines)


def test_a_missing_contract_file_is_reported(tmp_path):
    shutil.copytree(CONTRACT, tmp_path / "c")
    (tmp_path / "c" / "data_spec.json").unlink()
    ok, lines = verify_contract(tmp_path / "c")
    assert not ok and any("MISSING data_spec.json" in l for l in lines)
