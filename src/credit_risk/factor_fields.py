"""The treated numeric fields that enter correlation, PCA and factor analysis.

Included: treated money fields (log), ratios, capped counts, bureau counts, FICO midpoint, credit history length.
Left out on purpose:
  - int_rate, grade: the platform's own score (decision: not a model input);
  - restricted fields (job group, location);
  - ratio features built from other included fields (loan_to_income, installment_to_income, revol_bal_to_income,
    open_acc_share, recent_accounts_share): they repeat their parts and would only add overlap;
  - mths_since_last_delinq: 46% missing by construction (no delinquency on record), which has no honest numeric fill;
  - tot_cur_bal_log: the most redundant field (VIF 101, rank correlation 0.98 with total credit limit); keeping it
    makes the factor solution improper (almost no unique variance). Dropping it leaves every other factor unchanged;
  - term and other categoricals: handled through WoE.
"""
FACTOR_FIELDS = [
    "annual_inc_log", "loan_amnt_log", "installment_log", "revol_bal_log", "avg_cur_bal_log",
    "total_rev_hi_lim_log", "tot_hi_cred_lim_log",
    "dti_treated", "revol_util_treated", "bc_util_treated",
    "delinq_2yrs_capped", "pub_rec_capped", "pub_rec_bankruptcies_capped", "inq_last_6mths_capped", "mort_acc_capped",
    "open_acc", "total_acc", "acc_open_past_24mths", "num_tl_op_past_12m",
    "fico_mid_treated", "credit_history_months_treated",
]
