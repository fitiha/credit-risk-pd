"""Load the contract and build the modelling frame. The contract's transform is used unchanged and never refitted."""
import hashlib
from pathlib import Path

import joblib
import pandas as pd
import yaml

from credit_risk.io import load_applications, load_holdout
from credit_risk.spec import validate_cleaned, validate_raw

SERVICING_COLUMNS = ["total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries", "last_pymnt_d", "last_pymnt_amnt",
                     "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag", "bureau_recovery_flag"]


def load_config(path="config/train.yaml"):
    return yaml.safe_load(Path(path).read_text())


def load_contract(contract_dir="contract"):
    """Returns (fitted transformer, spec). The transformer is exactly the one in contract/transform.joblib."""
    import json
    contract_dir = Path(contract_dir)
    transformer = joblib.load(contract_dir / "transform.joblib")
    spec = json.loads((contract_dir / "data_spec.json").read_text())
    if spec["contract_version"] != transformer.contract_version:
        raise ValueError(f"contract version mismatch: spec {spec['contract_version']} vs transform {transformer.contract_version}")
    return transformer, spec


def build_frame(transformer, raw):
    """Raw application + bureau rows -> modelling frame, validated against the contract (fails fast on bad data)."""
    validate_raw(raw)
    frame = transformer.transform(raw)
    validate_cleaned(frame)
    leaked = [c for c in SERVICING_COLUMNS if c in frame.columns]
    if leaked:
        raise ValueError(f"servicing columns reached the modelling frame: {leaked}")
    return frame


def training_frames(cfg):
    transformer, spec = load_contract(cfg["contract_dir"])
    raw = load_applications(cfg["data_dir"])
    return transformer, spec, build_frame(transformer, raw)


def holdout_frame(cfg, transformer):
    return build_frame(transformer, load_holdout(cfg["data_dir"]))


def split(frame, start=None, end=None):
    m = pd.Series(True, index=frame.index)
    if start:
        m &= frame.issue_d >= pd.Timestamp(start)
    if end:
        m &= frame.issue_d <= pd.Timestamp(end)
    return frame[m]


def data_hash(frame, columns):
    """A fingerprint of the training data (ids, labels and the values of `columns`), saved with the model.
    Models on WoE columns pass no columns, so their hash covers ids and labels; the values come from the versioned contract."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(frame[["id"] + list(columns) + ["defaulted"]], index=False).to_numpy().tobytes())
    return h.hexdigest()
