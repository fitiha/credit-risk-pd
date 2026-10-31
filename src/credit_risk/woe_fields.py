"""Fields that get Weight of Evidence columns in the contract, and their roles.

All candidate fields are included so the ML side can choose; each field's role (input, platform score, restricted)
comes from the field catalog and is repeated in the spec. `application_type` is left out: all joint applications
are issued after the fit cutoff, so it carries no information to weight.
"""
WOE_NUMERIC = (
    ["annual_inc_log", "loan_amnt_log", "installment_log", "revol_bal_log", "tot_cur_bal_log", "avg_cur_bal_log",
     "total_rev_hi_lim_log", "tot_hi_cred_lim_log", "dti_treated", "revol_util_treated", "bc_util_treated",
     "delinq_2yrs_capped", "pub_rec_capped", "pub_rec_bankruptcies_capped", "inq_last_6mths_capped", "mort_acc_capped",
     "open_acc", "total_acc", "acc_open_past_24mths", "num_tl_op_past_12m", "mths_since_last_delinq", "emp_length_years",
     "int_rate"]
    + [f"{f}_treated" for f in ["credit_history_months", "loan_to_income", "installment_to_income", "fico_mid",
                                "open_acc_share", "recent_accounts_share", "revol_bal_to_income"]]
)
WOE_CATEGORICAL = ["job_group", "purpose_group", "home_ownership_group", "addr_state_group", "verification_status", "term", "grade"]
