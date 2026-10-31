"""Build the data contract (contract/): the fitted transform, the analytical dataset, the spec and the reference rows.

Everything is fitted on loans issued up to 2014-12-31. findings.md is written by hand and left untouched.
Run: uv run python scripts/build_contract.py   then   uv run python scripts/verify_contract.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk.io import load_applications
from credit_risk.spec import ALL_COLUMNS, build_spec
from credit_risk.transform import ApplicationTransformer

OUT = Path("contract")
OUT.mkdir(exist_ok=True)
raw = load_applications()
tr = ApplicationTransformer().fit(raw)
out = tr.transform(raw)

# 1. fitted transform
joblib.dump(tr, OUT / "transform.joblib", compress=3)

# 2. one cleaned row per application: cleaned fields, treated fields, features, groups, PCA, factors, WoE, label
out.to_parquet(OUT / "analytical_dataset.parquet", index=False)

# 3. spec
spec = build_spec(tr)
(OUT / "data_spec.json").write_text(json.dumps(spec, indent=2, default=str) + "\n")

# 4. 50 raw rows with their expected transformed values: every kind of repair, missing pattern and unusual category, plus ordinary rows
rng = np.random.default_rng(0)
flags = {"loan amount 10x too large": "loan_amnt_fixed", "income in thousands": "annual_inc_fixed", "revolving balance 10x too large": "revol_bal_fixed",
         "credit date after the loan": "earliest_cr_line_invalid", "negative bank-card utilisation": "bc_util_invalid", "placeholder interest rate": "int_rate_imputed",
         "zero months since delinquency coded for none on record": "mths_since_last_delinq_recoded"}
pick = {}
for why, col in flags.items():
    pick[why] = out.index[out[col]].to_numpy()
raw_i = raw.reset_index(drop=True)
pick["negative revolving utilisation"] = raw_i.index[raw_i.revol_util < 0].to_numpy()
pick["FICO low above high"] = raw_i.index[raw_i.fico_range_low > raw_i.fico_range_high].to_numpy()
pick["state written in full or lower case"] = raw_i.index[~raw_i.addr_state.str.fullmatch(r"[A-Z]{2}")].to_numpy()
pick["date written as MM/YYYY or YYYY-MM-DD"] = raw_i.index[~raw_i.issue_d.str.fullmatch(r"[A-Za-z]{3}-\d{4}")].to_numpy()
pick["no employer title"] = raw_i.index[raw_i.emp_title.isna()].to_numpy()
pick["no employment length"] = raw_i.index[raw_i.emp_length.isna()].to_numpy()
pick["no bank-card utilisation in the bureau file"] = raw_i.index[raw_i.bc_util.isna()].to_numpy()
pick["no months since last delinquency"] = raw_i.index[raw_i.mths_since_last_delinq.isna()].to_numpy()
pick["rare purpose"] = raw_i.index[raw_i.purpose.isin(["wedding", "renewable_energy"])].to_numpy()
pick["before the October 2013 title change"] = raw_i.index[out.job_group == "employer_name_era"].to_numpy()
chosen, labels = [], []
for why, idx in pick.items():
    for i in rng.choice(idx, size=min(2, len(idx)), replace=False):
        if i not in chosen:
            chosen.append(int(i)); labels.append(why)
for year in (2013, 2014, 2015):  # ordinary rows from each year fill the rest
    pool = out.index[(out.issue_d.dt.year == year) & ~out[list(flags.values())].any(axis=1) & (out.job_group != "employer_name_era")].to_numpy()
    for i in rng.choice(pool, size=20, replace=False):
        if len(chosen) < 50 and int(i) not in chosen:
            chosen.append(int(i)); labels.append(f"ordinary loan, {year}")
ref_raw = raw_i.loc[chosen, ["id"] + ALL_COLUMNS + ["defaulted"]]
skip = {"id", "title", "emp_title", "defaulted"}
expected = out.loc[chosen, [c for c in out.columns if c not in skip]].add_prefix("expected_")
for c in expected.select_dtypes("datetime64").columns:
    expected[c] = expected[c].dt.strftime("%Y-%m-%d")
ref = pd.concat([ref_raw, expected], axis=1)
ref.insert(0, "why_this_row", labels)
ref.to_csv(OUT / "reference_transforms.csv", index=False)
print(f"transform.joblib {(OUT / 'transform.joblib').stat().st_size / 1e6:.2f} MB | analytical_dataset {out.shape} {(OUT / 'analytical_dataset.parquet').stat().st_size / 1e6:.1f} MB | "
      f"spec: {len(spec['fields'])} fields, {len(spec['woe'])} WoE fields, {len(spec['factors'])} factors | reference rows: {len(ref)}")
print("reference rows by reason:", pd.Series(labels).str.replace(r", \d{4}", "", regex=True).value_counts().to_dict())
