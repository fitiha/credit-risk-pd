# Decisions

Each entry records what was tried, what was chosen and why. Experiments live in
`scripts/` and write their evidence to `reports/`.

## Outlier treatment for the money fields

**Fields:** `annual_inc`, `loan_amnt`, `installment`, `revol_bal`, `tot_cur_bal`, `avg_cur_bal`,
`total_rev_hi_lim`, `tot_hi_cred_lim`.

**Why mean ± 3 standard deviations is not a sound rule here.** These fields are heavy-tailed
(skewness up to 63 for `annual_inc`), and the rule assumes a symmetric, bell-shaped field.
- The few huge values inflate the standard deviation that the rule relies on. For `annual_inc`
  one $7.5M income pushes the upper limit to $305,000, above the 99th percentile ($250,000).
- The lower limit is negative for every field (−$156,000 for income), so the rule can never flag
  an unusually low value.
- It disagrees with other rules. On `annual_inc` it flags 57 loans, the IQR fence 616 and the
  robust z-score 654. On mostly-zero counts such as `delinq_2yrs` the spread-based rules flag every
  loan with any delinquency, because the spread is zero.
Evidence: `reports/a2_financial_values_profile/tables/outlier_rule_counts.csv`.

**What was compared** (experiment `scripts/a2_outlier_method_comparison.py`, pre-2015 loans only,
statistics from the training part only): no treatment, clipping at mean ± 3 std, IQR fences,
robust z-score (median and MAD), winsorizing at 1%/99% and at 0.5%/99.5%, log(1 + x), log with a
0.5%/99.5% cap, and log then clipping at 3 std.

| Finding | Result |
|---|---|
| Shape after treatment | Any method using the log gives |skewness| ≤ 0.7 on every field. Clipping or winsorizing without the log leaves 1.4 to 4.5 on several fields. |
| Rows changed | 0.4% to 1% of rows for the 0.5%/99.5% cap; up to 10% for IQR fences and the robust z-score, which cuts genuine values. |
| Predictive power | A logistic model on later, unseen loans scores AUC 0.64 untreated and 0.65 with the log methods. The 95% intervals (±0.02) overlap, so predictive power cannot separate the methods. Using the money fields at all adds about 0.055 AUC (0.586 without them). |

**Choice: cap each field at its 0.5th and 99.5th percentile, then take log(1 + value).**
It gives the most symmetric shape, changes about 1% of rows, deletes nobody, and the cap protects
against freak values at scoring time. The limits are simple to state and store ("capped at
p99.5 = 300,000"). Chosen on shape, rows touched and explainability, since predictive power was a tie.

**Effect.** Default rate by decile is identical before and after the treatment (the cap and log
keep every loan's rank), so no risk ordering is lost. Higher income and higher credit limits go
with lower default (income decile 1: 20.9%, decile 10: 13.8%); larger loans go with higher default
(13.4% to 20.0%). Evidence: `reports/a2_amount_treatment_effect/`.

**Errors versus genuine extremes.** Errors are repaired in the cleaning step (income typed in
thousands, loan amounts 10x too big, negative utilization; see `issue_log.md`). Genuine extremes are
kept and capped, never deleted. A value is judged an error only from information available at
application time (verification status, consistency with other fields), never from how the loan ended.
Example: all five incomes above $1M were verified or source-verified by the platform and have tiny
debt-to-income ratios, so they are treated as genuine and capped.

**Zeros and missing values.** The 0.5% floor lifts a zero to the floor value, so every field with
real zeros (`revol_bal`, `tot_cur_bal`, `avg_cur_bal`, `total_rev_hi_lim`) also gets an
`<field>_is_zero` column. A missing value stays missing and is never turned into a zero.

**Implementation.** `src/credit_risk/outliers.py` (`AmountTreater`). Limits are fitted on loans issued
up to 2014-12-31, and a test checks that 2015 loans do not change them.
