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
(13.4% to 20.0%). Evidence: `reports/a2_treatment_effect/`.

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

## Ratios (`dti`, `revol_util`, `bc_util`) and count fields

**Ratios: cap at the 0.5th and 99.5th percentile, no log.** On a log scale `revol_util` becomes
lopsided (it is already roughly symmetric), so the log used for money fields does not fit. Fitted
limits on pre-2015 loans: `dti` 1.5 to 36.9, `revol_util` 0.9 to 99.2, `bc_util` 0.0 to 102.9.
Utilization above 100% after a limit cut is a genuine extreme, so it is capped, not removed. The
cap also matters at scoring time: `dti` reaches 100 in the 2015 loans but its fitted cap is 36.9.

**Counts (`delinq_2yrs`, `pub_rec`, `pub_rec_bankruptcies`, `inq_last_6mths`, `mort_acc`): cap at
the 99.5th percentile.** Zeros are the most common value and stay zero. Spread-based outlier rules
fail on them (the IQR is zero, so every nonzero value is flagged). The cap touches 0.1% to 0.4% of
loans. Evidence: `reports/a2_treatment_effect/tables/count_caps.csv`.

## Missing ratio values

**Question.** Is a missing `revol_util` or `bc_util` just a gap, or a signal? And which fill rule is best?
(experiment `scripts/a2_missing_amounts_experiment.py`, pre-2015 loans).

| Finding | Result |
|---|---|
| How much is missing | `revol_util`: 4 loans (2 have no revolving limit, so the ratio is undefined). `bc_util`: 151 missing in the bureau file plus 138 set to missing after the sign-flip repair (issue 14). |
| Link to default | Loans with a missing `bc_util` default at 21% (bureau gap) and 23% (sign repair) against 17% for the rest. With 151 and 138 loans the difference is not clearly more than chance (p = 0.19 and 0.07). |
| Fill rule | Median, median plus a missing indicator, and a regression estimate from `revol_util` all score the same AUC (0.6488 to 0.6489). The extra complexity of the regression does not pay. |

**Choice: fill with the median of the capped fit-window values, and add a `<field>_missing`
indicator when at least 1% of fit-window loans are missing.** That applies to `bc_util` only. The
indicator is cheap and keeps the information, because missingness leans toward higher default even
though it cannot be proven. `revol_util` has 4 missing loans, so it is filled without an indicator.
A zero is a real value everywhere and is never treated as missing. Fill values: `dti` 17.2,
`revol_util` 57.0, `bc_util` 69.3.

**Implementation.** `RatioTreater` and `CountTreater` in `src/credit_risk/outliers.py`. Evidence
(before and after charts, default rate by decile): `reports/a2_treatment_effect/`.
