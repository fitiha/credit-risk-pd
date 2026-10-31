"""Every field of the source files: reliability, whether it is known at application time, and its role.

Reliability rules (evidence in docs/issue_log.md and reports/a3_reliability_evidence/):
  high    recorded by the platform or the credit bureau, internally consistent, no errors or
          only a small share (about 1%) that were found and repaired;
  medium  reported by the borrower but consistent in the tests, or a system field with errors
          that could not be fully repaired, or with large structural missingness;
  low     free text, or a field whose content is duplicated, unreliable or a leak.
Roles: input (usable by the model), platform_score (known at application but produced by the
platform's own risk model), restricted (location, treatment decided in the fairness review),
time_key, id, target, excluded (with the reason).
"""
import pandas as pd

APP, BUR, SRV = "applications", "credit_bureau", "loan_servicing"
POST = "Recorded after the loan was issued; not known when the applicant applies."


def _f(name, source, known, role, reliability, reason):
    return {"field": name, "source": source, "known_at_application": known, "role": role,
            "reliability": reliability, "reason": reason}


CATALOG = [
    _f("id", APP, True, "id", "n/a", "Identifier; joins the three files."),
    _f("defaulted", APP, False, "target", "n/a", "The label: 1 = charged off or defaulted, 0 = fully paid. Verified against payment data (no contradictions)."),
    _f("issue_d", APP, True, "time_key", "high", "Platform-recorded; 1,200 loans (4%) in other date formats, all parsed. Used to split the data in time, not as a model input: the rise in default over issue date is partly an artifact of removed unfinished loans."),
    _f("loan_amnt", APP, True, "input", "high", "Set by the platform. 300 loans (1%) had the amount 10x too large; found with the installment cross-check and repaired."),
    _f("term", APP, True, "input", "high", "Set by the platform; two values (36, 60) after trimming spaces."),
    _f("int_rate", APP, True, "platform_score", "high", "Set by the platform from its own risk model. 8 loans carried a placeholder of exactly 6.00, replaced by the sub-grade median."),
    _f("installment", APP, True, "input", "high", "Platform-computed; matches the loan math except 4 loans; never above 40% of income."),
    _f("grade", APP, True, "platform_score", "high", "Assigned by the platform's own risk model; consistent with sub_grade in every loan."),
    _f("sub_grade", APP, True, "platform_score", "high", "Assigned by the platform's own risk model; interest rates rise with sub-grade."),
    _f("purpose", APP, True, "input", "medium", "Chosen by the borrower from 13 options; clean, but stated intent is not verified."),
    _f("title", APP, True, "excluded", "low", "Borrower-typed text; 1,821 distinct values that mostly restate purpose."),
    _f("application_type", APP, True, "input", "high", "Platform-recorded; only 16 joint applications, too few to learn from."),
    _f("emp_title", APP, True, "input", "low", "Borrower-typed text; 5.7% missing; case and spacing variants (15,657 values fall to 13,484). Missing mostly among verified applicants, not linked to default. Used only after grouping into job groups."),
    _f("emp_length", APP, True, "input", "medium", "Self-reported; 4.8% missing, tied to the verification process, not to default (19.5% vs 17.1%, p = 0.11)."),
    _f("home_ownership", APP, True, "input", "medium", "Self-reported; clean categories."),
    _f("annual_inc", APP, True, "input", "medium", "Self-reported. 899 loans (3%) typed in thousands, repaired. Consistent with credit limits and debt ratios in every verification group (rank correlation 0.55 to 0.57 with total credit limit), so usable; verified incomes are higher (median $66-67k vs $58k)."),
    _f("verification_status", APP, True, "input", "high", "Recorded by the platform. Do not read it as lower risk: verified loans default more (19% vs 13%), likely because the platform chooses whom to verify."),
    _f("dti", APP, True, "input", "medium", "Platform-computed from the stated income and bureau debts, so only as reliable as the income. Rises with revolving balance relative to income (correlation 0.36 to 0.43), as it should."),
    _f("zip_code", APP, True, "restricted", "medium", "First three digits only; disagrees with the state in 33 loans. Location can proxy for protected characteristics; treatment decided in the fairness review."),
    _f("addr_state", APP, True, "restricted", "medium", "Self-reported; 3,000 loans (10%) with formatting variants, all repaired. Treatment decided in the fairness review."),
    _f("earliest_cr_line", BUR, True, "input", "medium", "Bureau-reported. 300 loans (1%) have a date after the loan, corrupted beyond repair: set missing with an indicator. 1,200 in other formats, parsed."),
    _f("fico_range_low", BUR, True, "input", "high", "Bureau score. 150 loans (0.5%) had low and high swapped; repaired."),
    _f("fico_range_high", BUR, True, "input", "high", "Bureau score. Swapped with the low value in 150 loans; repaired."),
    _f("inq_last_6mths", BUR, True, "input", "high", "Bureau count; no errors found."),
    _f("delinq_2yrs", BUR, True, "input", "high", "Bureau count; no errors found; mostly zero, long tail capped."),
    _f("mths_since_last_delinq", BUR, True, "input", "medium", "Missing in 46% of loans, and structurally: 99.7% of the missing have no delinquency in the last two years, so missing means none on record. Not linked to default (16.7% vs 17.7%, p = 0.09). 67 contradictions."),
    _f("pub_rec", BUR, True, "input", "high", "Bureau count; no errors found."),
    _f("pub_rec_bankruptcies", BUR, True, "input", "high", "Bureau count; no errors found."),
    _f("open_acc", BUR, True, "input", "high", "Bureau count; never above total_acc."),
    _f("total_acc", BUR, True, "input", "high", "Bureau count; no errors found; never below open_acc."),
    _f("revol_bal", BUR, True, "input", "high", "Bureau amount; no errors found; heavy tail is genuine and capped; a few real zeros."),
    _f("revol_util", BUR, True, "input", "medium", "295 negative values (1%) were sign flips, proven against balance and limit; 12 missing (5 with no revolving limit, so undefined)."),
    _f("bc_util", BUR, True, "input", "medium", "290 negative values (1%) set to missing because the sign flip cannot be proven; 1% more missing in the bureau file, with a weak link to higher default."),
    _f("tot_cur_bal", BUR, True, "input", "high", "Bureau amount; no errors found; heavy tail capped."),
    _f("avg_cur_bal", BUR, True, "input", "high", "Bureau amount; no errors found; heavy tail capped."),
    _f("total_rev_hi_lim", BUR, True, "input", "high", "Bureau amount; no errors found; heavy tail capped."),
    _f("tot_hi_cred_lim", BUR, True, "input", "high", "Bureau amount; no errors found; heavy tail capped."),
    _f("mort_acc", BUR, True, "input", "high", "Bureau count; no errors found."),
    _f("acc_open_past_24mths", BUR, True, "input", "high", "Bureau count; no errors found."),
    _f("num_tl_op_past_12m", BUR, True, "input", "high", "Bureau count; no errors found."),
    _f("bureau_recovery_flag", BUR, False, "excluded", "low", "Leak: identical to `recoveries > 0` in the servicing file (4,207 of 4,207 loans), and every flagged loan defaulted. Absent from the 2016 holdout in any useful form (always 0)."),
    _f("total_pymnt", SRV, False, "excluded", "n/a", POST + " Payments made so far reveal the outcome."),
    _f("total_rec_prncp", SRV, False, "excluded", "n/a", POST + " Principal repaid reveals the outcome."),
    _f("total_rec_int", SRV, False, "excluded", "n/a", POST + " Interest paid reveals how long the loan ran."),
    _f("recoveries", SRV, False, "excluded", "n/a", POST + " Money recovered after a charge-off: only exists for defaulted loans."),
    _f("last_pymnt_d", SRV, False, "excluded", "n/a", POST),
    _f("last_pymnt_amnt", SRV, False, "excluded", "n/a", POST + " 249 values exceed the total paid."),
    _f("last_credit_pull_d", SRV, False, "excluded", "n/a", POST),
    _f("last_fico_range_high", SRV, False, "excluded", "n/a", POST + " A later score, lower for defaulters (mean 571 vs 701)."),
    _f("last_fico_range_low", SRV, False, "excluded", "n/a", POST + " A later score."),
    _f("debt_settlement_flag", SRV, False, "excluded", "n/a", POST + " Every loan flagged Y (763) defaulted: a leak."),
]


def catalog_frame():
    return pd.DataFrame(CATALOG)


def input_fields():
    return [r["field"] for r in CATALOG if r["role"] == "input"]
