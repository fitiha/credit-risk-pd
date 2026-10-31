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
| Rows changed | 0.4% to 1% of rows for the 0.5%/99.5% cap; up to 6% for IQR fences and 10% for the robust z-score, which cut genuine values. |
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

## Is mean ± 3 standard deviations sound? Field by field

Pre-2015 loans. "Flagged" is the share of loans outside mean ± 3 std. Skewness is 0 for a symmetric field.
Source: `reports/a2_financial_values_profile/tables/`.

| Field | Skew | Limit (upper) vs 99th percentile | Flagged | Sound? | Why |
|---|---|---|---|---|---|
| `annual_inc` | 63.3 | $305,292 vs $250,000 | 0.4% | No | One $7.5M income inflates the std ($77k, against an IQR of $45k), pushing the limit above the 99th percentile. The lower limit is −$156,543 and can never trigger. The IQR and robust rules flag 616 and 654 loans, the 3-std rule 57. |
| `loan_amnt` | 0.7 | $39,620 vs $35,000 | 0% | Not needed | Bounded by the product maximum; no tail. The limit lies above the largest value. |
| `installment` | 0.9 | $1,184 vs $1,196 | 1.2% | Works by accident | Bounded by loan size. The limit lands near the 99th percentile only because the field is bounded; it is a function of amount, rate and term, so it has no independent outliers. |
| `revol_bal` | 12.8 | $100,676 vs $115,303 | 1.2% | No | A $1.18M maximum inflates the std ($27,647). The limit falls below the 99th percentile, so it flags ordinary heavy-tail balances. |
| `tot_cur_bal` | 2.9 | $595,363 vs $631,213 | 1.3% | No | Limit below the 99th percentile; flags real borrowers. Lower limit negative. |
| `avg_cur_bal` | 4.8 | $61,707 vs $70,494 | 1.7% | No | Same pattern. |
| `total_rev_hi_lim` | 5.7 | $120,997 vs $141,140 | 1.6% | No | Same pattern. |
| `tot_hi_cred_lim` | 2.7 | $671,972 vs $724,281 | 1.3% | No | Same pattern. |
| `dti` | 0.2 | 41.5 vs 34.9 | 0% | Sound, unnecessary | Nearly symmetric, nothing flagged. The real risk is values above 100 seen in 2015, which a percentile cap handles more simply. |
| `revol_util` | 0.0 | 125.5 vs 98.0 | 0.01% | Sound, unnecessary | Symmetric. Flags 2 loans (maximum 366%). The meaningful threshold is a business one (100% of the limit), so a cap is used instead. |
| `bc_util` | −0.6 | 144.2 vs 101.2 | 0.01% | Sound, unnecessary | Same as `revol_util`. |

For the six heavy-tailed money fields the rule fails in the same way: the tail inflates the std, the
limit is one-sided in practice (the lower limit is negative), and it flags 1% to 2% of genuine borrowers
while the IQR fences flag 3% to 6% and the robust z-score 5% to 10%, so no single threshold is trustworthy.

## Errors versus genuine extremes, field by field

An error is repaired in the cleaning step (see `issue_log.md`). A genuine extreme is kept and capped.
Judged from application-time information only, never from the outcome. Counts of genuine extremes
are over all 30,000 loans; caps are fitted on pre-2015 loans.

| Field | Errors repaired | Genuine extremes kept and capped |
|---|---|---|
| `annual_inc` | 899 incomes typed in thousands, multiplied by 1,000 (issue 10) | 5 incomes above $1M, all verified or source-verified by the platform with tiny debt-to-income; capped at $300,000 |
| `loan_amnt` | 300 amounts 10x too large, divided by 10 (issue 8) | None left; maximum is the $35,000 product limit |
| `installment` | None | None; 4 loans inconsistent with the loan math are flagged, not changed (issue 9) |
| `revol_bal`, `tot_cur_bal`, `avg_cur_bal`, `total_rev_hi_lim`, `tot_hi_cred_lim` | None found | Values up to $3.5M kept and capped (for example `revol_bal` at $176,561); a few zeros are real and carry an `is_zero` indicator |
| `dti` | None | Zero values are real (8 loans); high values capped at 36.9 |
| `revol_util` | 295 negative values, sign removed (issue 14) | 114 loans above 100% after a limit cut, capped at 99.2. The cap also removes the distinction between 100% and 150%; with 114 of 30,000 loans we accept that. |
| `bc_util` | 290 negative values set to missing with an indicator, because the sign flip cannot be proven (issue 14) | 590 loans above 100%, capped at 102.9 |
| Count fields | None | Capped at the 99.5th percentile (0.1% to 0.4% of loans) |

## Platform's own score (`grade`, `sub_grade`, `int_rate`) as model inputs

**Question.** These are known at application time, but they are outputs of the platform's existing risk
model. Using them can make a new model look good by copying the old one.

**Evidence** (`scripts/a3_platform_score_check.py`, pre-2015 loans, time-ordered folds, AUC on later loans):

| Inputs | AUC |
|---|---|
| sub-grade alone | 0.697 |
| interest rate alone | 0.696 |
| our inputs without the platform score | 0.708 |
| our inputs plus sub-grade | 0.712 |

Our inputs match the platform score on their own, and adding the sub-grade gains 0.004, which is inside the
noise (95% intervals overlap heavily).

**Choice: the platform score is not a default model input.** It stays in the data contract, flagged as a
platform score, so it serves as the benchmark ("does the model beat grade alone?") and as a test variant.
This keeps the model independent of the old risk model and makes it clear what our own fields and factors add.
Cost: almost none, per the table above.

## Application-time features

Built in `src/credit_risk/features.py` from cleaned fields known when the applicant applies. Evidence
(`scripts/a3_feature_evidence.py`, `reports/a3_feature_evidence/`) uses loans up to the fit cutoff.

| Feature | Definition | Default rate, lowest to highest decile |
|---|---|---|
| `credit_history_months` | months from the first credit line to the issue date | 19.7% to 14.4% |
| `loan_to_income` | loan amount / annual income | 10.8% to 23.6% |
| `installment_to_income` | installment / monthly income | 10.5% to 21.3% |
| `revol_bal_to_income` | revolving balance / annual income | 15.5% to 19.5% |
| `fico_mid` | midpoint of the FICO range | 23.2% to 7.9% |
| `open_acc_share` | open accounts / total accounts | 15.4% to 20.2% |
| `recent_accounts_share` | accounts opened in the last 24 months / total accounts | 10.5% to 25.0% |

Each moves in the direction credit logic expects: more debt relative to income, more recent borrowing and a
lower score go with more default; a longer history goes with less. `loan_to_income` and `installment_to_income`
are closely related (both scale loan size by income); the correlation and VIF step will decide whether both stay.

**Treatment: the same rules as the other ratio fields.** Cap at the 0.5th and 99.5th percentile, fill missing
values with the fit-window median, and add a log version only when the capped skewness is above 2. That rule
selects `revol_bal_to_income` alone (skewness 9.8 raw, 3.6 capped). The 133 loans whose credit date was
corrupted get the median history (179 months) and keep the `earliest_cr_line_invalid` indicator from cleaning.
Everything is fitted on loans up to 2014-12-31; a test checks that 2015 loans do not change the fit.

**Dropped:** `credit_limit_to_income` (AUC 0.53, adds little over the credit-limit and income fields already used).

## Job titles (`emp_title`)

**Problem.** After cleaning, the fit window has 13,549 titles with 8,157 distinct values; 7,082 appear once. The
20, 50 and 200 most common titles cover only 13%, 19% and 30% of loans, so a code per title is unusable.

**Compared** (`scripts/a4_emp_title_grouping.py`): keyword groups, the N most common titles (20, 50, 200), and
only a missing flag. Each is judged by IV on the loans it was learned from (up to mid-2014) and on later loans
(second half of 2014) using the WoE learned earlier, because more groups always look better in-sample.

| Option | Groups | Smallest group | IV learned on | IV on later loans |
|---|---|---|---|---|
| Keyword groups | 10 | 439 loans | 0.0145 | 0.0153 |
| Top 20 titles | 22 | 23 | 0.0187 | 0.0208 |
| Top 50 titles | 52 | 11 | 0.0465 | 0.0287 |
| Top 200 titles | 202 | 4 | 0.1195 | 0.0381 |
| Missing flag only | 2 | 575 | 0.0012 | 0.0006 |

**Choice: keyword groups.** Eight occupation groups from keyword rules (first match wins, order in
`src/credit_risk/categorical.py`), plus `other_job` and `missing`. They are stable between training and later
loans, every group is large, and each can be explained to a credit officer. The top-200 option is the textbook
overfit: IV of 0.12 on its own training loans falls to 0.04 on later ones, with groups of 4 loans.

**What job title is worth.** Little. IV is about 0.015, which is below the 0.02 line for a useful field.
Default rates run from 13.9% (professional and technical) to 19.9% (trades, transport and labour); the
spread is real but small. The group stays in the contract because it is cheap and explainable, but we
should not expect it to carry the model.
The 32% in `other_job` is a long tail of company names and one-off titles (Wells Fargo, Walmart, UPS).
Evidence: `reports/a4_emp_title_grouping/`.

**Job group is restricted, not a default model input.** Occupation can stand in for characteristics the data does
not contain (sex, age, ethnicity), which would be indirect discrimination, and we cannot test for it because
those fields are absent. Its predictive value is small (IV about 0.015), so the cost of leaving it out is close to
zero. The grouping stays in the data contract (the brief asks for it) and is available as a test variant; the
fairness review (A7) discusses it together with `zip_code` and `addr_state`. `emp_length` is flagged in the same
review, since length of employment tends to rise with age.

## Rare levels of `purpose`, `home_ownership` and `addr_state`; employment length

**Minimum group size: 1% of fit-window loans (about 144 of 14,373).** Levels below it merge into one
`other_<field>` level, and levels never seen when fitting (a new state, for example) map there too.
Compared at 0%, 0.25%, 0.5%, 1% and 2% (`scripts/a4_rare_level_merging.py`; IV learned on loans to mid-2014 and
checked on later loans):

| Field | Levels | At 1%: groups left, loans in other, smallest group | IV learned on / on later loans |
|---|---|---|---|
| `purpose` | 13 | 7 groups, 3.4%, 96 loans | 0.0253 / 0.0152 (best of the thresholds) |
| `home_ownership` | 3 | 3 groups, nothing merged, 845 loans | 0.0237 / 0.0197 |
| `addr_state` | 47 | 32 groups, 6.0%, 99 loans | 0.0301 / 0.0030 |

Why 1%: at about 100 loans a group's default rate carries an uncertainty of roughly ±4 points, and below that
the WoE values wobble. A threshold of 2% starts merging real categories (medical, small business) and puts 25%
of state loans into `other`; 0.5% leaves groups as small as 50 loans.

**State is mostly noise.** Its IV looks like 0.03 on the loans it was learned from but falls to 0.003 to 0.010
on later loans at every threshold, below the 0.02 line for a useful field. It also carries the location proxy
risk, so it stays restricted (see the fairness review). Merging is applied because the brief asks for it and the
spec needs it.

**Employment length** is ordered 0 (under 1 year) to 10 (10+ years) with missing as its own level.
Default rates are flat at 17% to 19% for 1 to 9 years, lower at 10+ years (15.5%), and higher for under 1 year
(19.7%) and missing (19.5%). It is not a clean staircase, and its IV is 0.012, so it is binned into fewer
steadier groups in the WoE step. Evidence: `reports/a4_rare_level_merging/`.

## Binning numeric fields for WoE

Compared on 30 numeric fields (`scripts/a4_binning_comparison.py`): ten equal-sized bins, equal-sized bins merged until
the default rate moves in one direction with every bin at least 5% of loans ("monotonic merge"), and decision-tree
bins (up to 6 leaves, each at least 5% of loans). Bins were learned on loans to mid-2014 and judged on later loans.

| Method | Average bins | IV learned on | IV on later loans | Share of IV kept on later loans |
|---|---|---|---|---|
| Ten equal-sized bins | 8.7 | 0.047 | 0.049 | 105% |
| Monotonic merge | 5.3 | 0.045 | 0.049 | 109% |
| Tree, 6 leaves | 5.5 | 0.053 | 0.049 | 93% |

**Choice: monotonic merge.** The three tie on later-loan IV, so the choice rests on stability and fit for a scorecard:
fewest bins, no IV lost on later loans, a one-directional pattern by construction (a credit officer can read
"higher means riskier"), and a minimum bin size of 5% of loans. Trees overfit the training window slightly; ten
equal bins are needlessly fine. Missing values get their own bin. Bin edges are learned on the fit window and
reused unchanged at scoring time.
Note that a pattern that is one-directional on training loans is not always one-directional on the smaller later
sample (43% of fields); with about 5,000 later loans, neighbouring bins differ by less than the noise.

## Information value ranking and the "too good to be true" check

`scripts/a4_iv_ranking.py` computes IV on loans to 2014 for 30 numeric and 8 categorical fields (bins and
groups as decided above). Output: `reports/a4_iv_ranking/`.

| Field | IV | Reading |
|---|---|---|
| `int_rate`, `grade` | 0.46, 0.45 | strong, close to the 0.5 line: the platform's own score |
| `term` | 0.20 | medium |
| `fico_mid`, `loan_to_income` | 0.11, 0.11 | medium: the strongest fields the model builds itself |
| `recent_accounts_share`, `acc_open_past_24mths`, `installment_to_income`, `dti`, `num_tl_op_past_12m` | 0.06 to 0.09 | weak |
| most other fields | under 0.05 | weak or useless; 17 of 38 fields are below 0.02 |

**The only IV near the suspicious zone is the platform's own score, and it is not a leak.** `int_rate` has
11 bins with a smooth, one-directional pattern, from 3.6% default in the lowest-rate bin (WoE +1.71) to 37.1% in
the highest (WoE -1.04). It is strong because the platform's existing risk model set it from the same applicant
data, so it is an echo of a model that already knew these fields. It is known at application time and consistent,
so it is not target leakage, but it is the reason it is not a default model input (see the earlier decision).

**What "too good to be true" looks like here.** The same calculation on fields recorded after the loan gives:

| Field (post-issue) | IV |
|---|---|
| `bureau_recovery_flag` | 9.4 |
| `recoveries` | 5.6 |
| `last_fico_range_high` | 3.6 |
| `last_pymnt_amnt` | 3.4 |
| `total_rec_prncp` | 2.2 |
| `total_pymnt` | 0.7 |

Every one is above 0.5 by a wide margin. Any field of ours showing an IV in that range would be a leak; none does.
A model on application-time fields should therefore be expected to reach only a modest AUC (around 0.70), and an
AUC far above that would be a warning sign, not good news.

**Two cautions.** `term` (IV 0.20) is partly affected by the in-progress-loan bias described in the target analysis,
which inflates the default rate of recent 60-month loans. And `grade`/`int_rate` are kept out of the default input
set, so the strongest legitimate fields are `fico_mid`, `loan_to_income` and the recent-borrowing fields.
