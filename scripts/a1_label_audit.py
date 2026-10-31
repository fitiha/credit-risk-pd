"""Audit the `defaulted` label against the loan servicing file.

The servicing file is recorded after the loan is issued, so it is used here only
to check the label and is never an input to the model.
Writes reports/a1_label_audit/tables/{label_checks,early_finishers_60m}.csv.
"""
import pandas as pd

from credit_risk.cleaning import Cleaner
from credit_risk.io import load_applications, load_servicing, parse_dates
from credit_risk.reporting import fresh_dirs

_, TAB = fresh_dirs("a1_label_audit")

raw = load_applications()
clean = Cleaner().fit(raw).transform(raw)
d = clean.merge(load_servicing(), on="id")
flag = raw.set_index("id").bureau_recovery_flag.reindex(d.id).to_numpy()

paid, bad = d.defaulted == 0, d.defaulted == 1
checks = {
    "defaulted loans whose principal was fully repaid": (bad & (d.total_rec_prncp >= 0.999 * d.loan_amnt)).sum(),
    "paid loans with recoveries > 0": (paid & (d.recoveries > 0)).sum(),
    "paid loans with debt_settlement_flag = Y": (paid & (d.debt_settlement_flag == "Y")).sum(),
    "paid loans with total_pymnt = 0": (paid & (d.total_pymnt == 0)).sum(),
    "paid loans that repaid under 99% of principal": (paid & (d.total_rec_prncp < 0.99 * d.loan_amnt)).sum(),
    "loans with debt_settlement_flag = Y": (d.debt_settlement_flag == "Y").sum(),
    "  settlement flag: of which labelled defaulted": ((d.debt_settlement_flag == "Y") & bad).sum(),
    "loans with bureau_recovery_flag = 1": (flag == 1).sum(),
    "  recovery flag: of which labelled defaulted": ((flag == 1) & bad).sum(),
    "  of which recoveries > 0 in servicing": ((flag == 1) & (d.recoveries > 0)).sum(),
    "rows where bureau_recovery_flag differs from (recoveries > 0)": ((flag == 1) != (d.recoveries > 0)).sum(),
}
checks = pd.Series(checks, name="loans").rename_axis("check").reset_index()
checks.to_csv(TAB / "label_checks.csv", index=False)

# 60-month loans issued since July 2014 are in the data only if they ended before Mar 2019.
x = d[(d.term == 60) & (d.issue_d >= "2014-07-01")].copy()
x["last_payment"] = parse_dates(x.last_pymnt_d)
x["months_issue_to_last_payment"] = ((x.last_payment.dt.year - x.issue_d.dt.year) * 12
                                     + (x.last_payment.dt.month - x.issue_d.dt.month))
early = x.groupby("defaulted").agg(loans=("id", "size"),
                                   median_months_to_last_payment=("months_issue_to_last_payment", "median"))
early["share_of_group"] = early.loans / early.loans.sum()
early.rename(index={0: "repaid early", 1: "defaulted"}).to_csv(TAB / "early_finishers_60m.csv", index_label="outcome")
print(checks.to_string(index=False))
print(early.round(3).to_string())
