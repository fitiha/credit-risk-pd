"""Model input sets, monotone constraints and the plain-English text for reason codes.

Inputs are the contract fields marked `input` (reliability high or medium). The platform's own score, the restricted
fields (job group, location) and all servicing fields are never inputs.
"""
import pandas as pd

from credit_risk.spec import SUB_GRADES

CATEGORICAL = ["purpose_group", "home_ownership_group", "verification_status"]
LOAN_STRUCTURE = ["term"]  # 36 or 60 months, numeric

# treated numeric fields from the contract, and the direction in which more of the field should move PD
# (+1 higher means riskier, -1 safer, 0 no constraint). A constraint is used only where credit logic and the data agree.
RAW_NUMERIC = {
    "annual_inc_capped": -1, "loan_amnt_capped": +1, "installment_capped": +1, "revol_bal_capped": 0, "tot_cur_bal_capped": 0,
    "avg_cur_bal_capped": -1, "total_rev_hi_lim_capped": -1, "tot_hi_cred_lim_capped": -1,
    "dti_treated": +1, "revol_util_treated": +1, "bc_util_treated": +1,
    "delinq_2yrs_capped": +1, "pub_rec_capped": +1, "pub_rec_bankruptcies_capped": +1, "inq_last_6mths_capped": +1, "mort_acc_capped": 0,
    "open_acc": 0, "total_acc": 0, "acc_open_past_24mths": +1, "num_tl_op_past_12m": +1,
    "fico_mid_treated": -1, "credit_history_months_treated": -1, "mths_since_last_delinq": 0, "emp_length_years": 0,
    "loan_to_income_treated": +1, "installment_to_income_treated": +1, "revol_bal_to_income_treated": +1,
    "open_acc_share_treated": 0, "recent_accounts_share_treated": +1,
}
FACTOR_COLUMNS = ["factor_credit_capacity", "factor_loan_size", "factor_revolving_credit_size", "factor_account_count",
                  "factor_revolving_utilization", "factor_public_records", "factor_recent_account_opening", "factor_debt_burden"]
FACTOR_CONSTRAINTS = {"factor_credit_capacity": -1, "factor_loan_size": +1, "factor_revolving_credit_size": -1, "factor_account_count": 0,
                      "factor_revolving_utilization": +1, "factor_public_records": 0, "factor_recent_account_opening": +1, "factor_debt_burden": +1}

# WoE columns for the scorecard: every WoE field of the contract that is an input (not platform score, not restricted)
SCORECARD_FIELDS = [
    "annual_inc_log", "loan_amnt_log", "installment_log", "revol_bal_log", "tot_cur_bal_log", "avg_cur_bal_log", "total_rev_hi_lim_log",
    "tot_hi_cred_lim_log", "dti_treated", "revol_util_treated", "bc_util_treated", "delinq_2yrs_capped", "pub_rec_capped",
    "pub_rec_bankruptcies_capped", "inq_last_6mths_capped", "mort_acc_capped", "open_acc", "total_acc", "acc_open_past_24mths",
    "num_tl_op_past_12m", "mths_since_last_delinq", "emp_length_years", "credit_history_months_treated", "loan_to_income_treated",
    "installment_to_income_treated", "fico_mid_treated", "open_acc_share_treated", "recent_accounts_share_treated",
    "revol_bal_to_income_treated", "purpose_group", "home_ownership_group", "verification_status", "term",
]
SCORECARD_COLUMNS = [f"{f}_woe" for f in SCORECARD_FIELDS]

INPUT_SETS = {
    "raw": CATEGORICAL + LOAN_STRUCTURE + list(RAW_NUMERIC),
    "factors": CATEGORICAL + LOAN_STRUCTURE + FACTOR_COLUMNS,
    "both": CATEGORICAL + LOAN_STRUCTURE + list(RAW_NUMERIC) + FACTOR_COLUMNS,
}
# the platform's sub-grade (A1 = 0 ... G5 = 34) is the one platform score allowed as an input (DESIGN.md section 11); a riskier grade must never lower PD
PLATFORM_COLUMNS = ["sub_grade_rank"]
CONSTRAINTS = {**RAW_NUMERIC, **FACTOR_CONSTRAINTS, "term": +1, "sub_grade_rank": +1}


def monotone_vector(columns, enabled=True):
    """Constraint per column. `enabled` is True (all), False (none) or the set of columns whose constraint is switched on."""
    if enabled is True or enabled is False:
        return [CONSTRAINTS.get(c, 0) if enabled else 0 for c in columns]
    return [CONSTRAINTS.get(c, 0) if c in enabled else 0 for c in columns]


# plain-English reasons: a neutral label and how to write the value. A reason shows the applicant's value next to the typical applicant's,
# so it states facts and does not claim "high" or "low" for a value that may be ordinary.
def _money(v):
    return f"${v:,.0f}"


def _pct(v):
    return f"{v:.0f}%"


def _num(v):
    return f"{v:,.0f}"


def _ratio(v):
    return f"{100 * v:.0f}%"


def _score(v):
    return f"{v:+.1f}"


def _subgrade(v):
    return SUB_GRADES[int(round(v))]


REASONS = {
    "annual_inc_capped": ("Annual income", _money), "loan_amnt_capped": ("Loan amount", _money), "installment_capped": ("Monthly installment", _money),
    "revol_bal_capped": ("Revolving balance", _money), "tot_cur_bal_capped": ("Total current balances", _money),
    "avg_cur_bal_capped": ("Average account balance", _money), "total_rev_hi_lim_capped": ("Total revolving credit limit", _money),
    "tot_hi_cred_lim_capped": ("Total credit limit", _money), "dti_treated": ("Debt-to-income ratio", _pct),
    "revol_util_treated": ("Share of revolving credit in use", _pct), "bc_util_treated": ("Share of bank-card limits in use", _pct),
    "delinq_2yrs_capped": ("Delinquencies in the last 2 years", _num), "pub_rec_capped": ("Public records", _num),
    "pub_rec_bankruptcies_capped": ("Bankruptcies", _num), "inq_last_6mths_capped": ("Credit inquiries in the last 6 months", _num),
    "mort_acc_capped": ("Mortgage accounts", _num), "open_acc": ("Open accounts", _num), "total_acc": ("Total accounts", _num),
    "acc_open_past_24mths": ("Accounts opened in the last 24 months", _num), "num_tl_op_past_12m": ("Accounts opened in the last 12 months", _num),
    "fico_mid_treated": ("Credit score (FICO)", _num), "credit_history_months_treated": ("Credit history length (months)", _num),
    "mths_since_last_delinq": ("Months since the last delinquency", _num), "emp_length_years": ("Employment length (years)", _num),
    "loan_to_income_treated": ("Loan amount as a share of annual income", _ratio),
    "installment_to_income_treated": ("Installment as a share of monthly income", _ratio),
    "revol_bal_to_income_treated": ("Revolving balance as a share of annual income", _ratio),
    "open_acc_share_treated": ("Share of accounts that are open", _ratio), "recent_accounts_share_treated": ("Share of accounts opened in the last 24 months", _ratio),
    "term": ("Loan term (months)", _num), "purpose_group": ("Loan purpose", str), "home_ownership_group": ("Home ownership", str),
    "verification_status": ("Income verification", str),
    "sub_grade_rank": ("Platform risk sub-grade", _subgrade),
    "factor_credit_capacity": ("Credit capacity score", _score), "factor_loan_size": ("Loan size score", _score),
    "factor_revolving_credit_size": ("Revolving credit size score", _score), "factor_account_count": ("Account count score", _score),
    "factor_revolving_utilization": ("Revolving utilization score", _score), "factor_public_records": ("Public records score", _score),
    "factor_recent_account_opening": ("Recent account opening score", _score), "factor_debt_burden": ("Debt burden score", _score),
}


def reason_text(column, value, reference=None):
    label, fmt = REASONS[column]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f"{label}: not provided"
    text = f"{label}: {fmt(value)}"
    if reference is not None and not isinstance(reference, str) and not pd.isna(reference):
        text += f" (typical approved applicant: {fmt(reference)})"
    return text
