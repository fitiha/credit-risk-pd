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
| Shape after treatment | Log with a cap gives |skewness| ≤ 0.8 on every field (log alone leaves `revol_bal` at -2.5). Clipping or winsorizing without the log leaves 1.4 to 3.4 on several fields. |
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
| Fill rule | Median, median plus a missing indicator, and a regression estimate from `revol_util` all score the same AUC (0.6488 to 0.6490). The extra complexity of the regression does not pay. |

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
| `revol_bal` | 8.3 | $77,208 vs $88,339 | 1.3% | No | A $624,356 maximum still inflates the std ($20,230). The limit falls below the 99th percentile, so it flags ordinary heavy-tail balances. |
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
| `annual_inc` | 899 incomes typed in thousands, multiplied by 1,000 (issue 10) | 5 incomes above $1M, all verified or source-verified by the platform with tiny debt-to-income, though their credit limits are small for such incomes (an extra zero is possible for three of them); capped at $300,000 either way |
| `loan_amnt` | 300 amounts 10x too large, divided by 10 (issue 8) | None left; maximum is the $35,000 product limit |
| `installment` | None | None; 4 loans inconsistent with the loan math are flagged, not changed (issue 9) |
| `revol_bal` | 308 balances 10x too large, divided by 10 (issue 38); found by checking balance / limit against the reported utilisation and against the total balance | Balances up to $1.04M kept (every cross-check passes) and capped at $126,556; a few zeros are real and carry an `is_zero` indicator |
| `tot_cur_bal`, `avg_cur_bal`, `total_rev_hi_lim`, `tot_hi_cred_lim` | None found (average balance x account count equals total balance in every loan; no limit is below a balance it contains) | Values up to $3.5M kept and capped; a few zeros are real and carry an `is_zero` indicator |
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
| `revol_bal_to_income` | revolving balance / annual income | 15.6% to 19.9% |
| `fico_mid` | midpoint of the FICO range | 23.2% to 7.9% |
| `open_acc_share` | open accounts / total accounts | 15.4% to 20.2% |
| `recent_accounts_share` | accounts opened in the last 24 months / total accounts | 10.5% to 25.0% |

Each moves in the direction credit logic expects: more debt relative to income, more recent borrowing and a
lower score go with more default; a longer history goes with less. `loan_to_income` and `installment_to_income`
are closely related (both scale loan size by income); the correlation and VIF step will decide whether both stay.

**Treatment: the same rules as the other ratio fields.** Cap at the 0.5th and 99.5th percentile, fill missing
values with the fit-window median, and add a log version only when the capped skewness is above 2. No feature needs
one after the revolving balances were repaired (issue 38): the heaviest, `revol_bal_to_income`, has a capped skewness of 1.7 (4.1 raw);
before the repair it was 3.6, which was caused by the corrupted balances. The 133 loans whose credit date was
corrupted get the median history (179 months) and keep the `earliest_cr_line_invalid` indicator from cleaning.
Everything is fitted on loans up to 2014-12-31; a test checks that 2015 loans do not change the fit.

**Dropped:** `credit_limit_to_income` (AUC 0.53, adds little over the credit-limit and income fields already used).

## Job titles (`emp_title`)

**The field changed meaning in October 2013.** The stability analysis (PSI 2.3 across that date, nothing else comes close)
led to the titles themselves: until September 2013 about 60% of them are employer names ("state farm insurance companies",
"at&t", "holiday inn express"); from October 2013 they are job titles ("teacher", "registered nurse"). The share of titles no
keyword rule can place drops from about 62% to 13% in one month. Loans before October 2013 therefore get their own group,
`employer_name_era`, instead of being mixed into the job groups, and the grouping experiment below uses only loans from October 2013.

**Problem.** Titles are extremely scattered: the 20, 50 and 200 most common titles leave 78%, 70% and 56% of loans in an "other"
level, so a code per title is unusable.

**Compared** (`scripts/a4_emp_title_grouping.py`, loans from October 2013 to 2014): keyword groups, the N most common titles
(20, 50, 200), and only a missing flag. IV is shown on the loans it was learned from (up to mid-2014) and on later loans (second
half of 2014) using the WoE learned earlier, because more groups always look better in-sample.

| Option | Groups | Smallest group | IV learned on | IV on later loans |
|---|---|---|---|---|
| Keyword groups | 10 | 200 loans | 0.0296 | 0.0130 |
| Top 20 titles | 22 | 23 | 0.0306 | 0.0211 |
| Top 50 titles | 52 | 10 | 0.0695 | 0.0252 |
| Top 200 titles | 202 | 3 | 0.1884 | 0.0385 |
| Missing flag only | 2 | 360 | 0.0023 | 0.0009 |

**Choice: keyword groups.** Eight occupation groups from keyword rules (first match wins, order in
`src/credit_risk/categorical.py`) plus `other_job` (13% of loans), `missing` and `employer_name_era`. Honestly stated: individual
common titles carry a little more out-of-time signal (0.039 for the top 200) than the groups (0.013), but at the price of 202
groups with as few as 3 loans, whose WoE is mostly noise (IV 0.19 on the loans it was learned from, 0.04 on later ones). The keyword
groups give up some signal for stability, size (every group has at least 200 loans) and the ability to explain them to a credit
officer. Their IV also halves out of time (0.030 to 0.013), so even that figure is optimistic.

**What job title is worth.** Little: IV is about 0.013 on later loans, below the 0.02 line for a useful field. Default rates run
from 14.1% (professional and technical) to 20.1% (trades, transport and labour). Evidence: `reports/a4_emp_title_grouping/`.

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

`scripts/a4_iv_ranking.py` computes IV on loans to 2014 for 30 numeric and 7 categorical fields (bins and
groups as decided above). Output: `reports/a4_iv_ranking/`.

| Field | IV | Reading |
|---|---|---|
| `int_rate`, `grade` | 0.46, 0.45 | strong, close to the 0.5 line: the platform's own score |
| `term` | 0.20 | medium |
| `fico_mid`, `loan_to_income` | 0.11, 0.11 | medium: the strongest fields the model builds itself |
| `recent_accounts_share`, `acc_open_past_24mths`, `installment_to_income`, `dti`, `num_tl_op_past_12m` | 0.06 to 0.09 | weak |
| most other fields | under 0.05 | weak or useless; 16 of 37 fields are below 0.02 |

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

## Correlation, PCA and factor analysis

**Fields used** (`src/credit_risk/factor_fields.py`): 21 treated numeric fields: seven money fields (log), `dti`, `revol_util`,
`bc_util`, five capped counts, four bureau counts, `fico_mid` and credit history length. Left out on purpose: the platform's
score (not a model input); restricted fields; ratio features built from other included fields (74% to 89% of each is already
explained by the included fields, R-squared in `reports/a5_factor_robustness/`); `mths_since_last_delinq` (46% missing by
construction, with no honest numeric fill); categorical fields (handled through WoE); and `tot_cur_bal` (see below). Everything
is fitted on the 14,373 loans up to the fit cutoff.

**`tot_cur_bal` was dropped after a robustness check.** It is 99% explained by the other fields (VIF 102 among the 22 candidate fields, rank correlation 0.98
with total credit limit) and leaves the factor analysis with almost no unique variance for it (0.002), an improper solution.
Without it the smallest unique variance is 0.02 and the other factors are unchanged (agreement 0.984 or better). The same
check showed the solution does not hinge on `delinq_2yrs` (KMO 0.43; dropping it gives agreement 0.996, so it stays), while
dropping `installment` breaks the loan-size factor (agreement 0.65), which is expected because that factor is the loan
amount and installment pair.

### Why reducing dimensions makes sense
`scripts/a5_correlation_vif.py`, `reports/a5_correlation_vif/`.
- The Spearman matrix shows blocks of fields that say the same thing: loan amount and installment (0.97); average balance and
  total credit limit (0.90); public records and bankruptcies (0.86); revolving and bank-card utilization (0.80); revolving
  balance and revolving limit (0.80); accounts opened in 12 and 24 months (0.71). 6 of 210 pairs are at 0.7 or above, 5% at 0.5
  or above, 20% at 0.3 or above.
- Multicollinearity is serious: 7 of 21 fields have a VIF above 5 and 6 above 10 (`loan_amnt` 23, `tot_hi_cred_lim` 23,
  `installment` 22, `avg_cur_bal` 18, `total_rev_hi_lim` 11, `revol_bal` 11). Linear-model coefficients on these would be unstable, and the fields largely repeat
  one another.
- Not everything overlaps: `dti`, `delinq_2yrs` and `inq_last_6mths` are close to independent of the rest.
A handful of underlying dimensions therefore describes most of the 21 fields, which is the case for reducing them.

### PCA (`scripts/a5_pca.py`, `reports/a5_pca/`)
Fields are standardized first (fitted on the fit window). Without standardizing, the first component explains 77% of the variance
and is 99.8% one field (`credit_history_months`, which is on a larger scale), so raw-scale PCA is not used.
- **Seven components.** Kaiser (eigenvalue above 1) and parallel analysis (eigenvalue beats 95% of pure noise) both give 7, and
  they explain 74.6% of the variance. Reaching 80% would need 9, but components beyond 7 do not beat noise.
- **What they are**, from the loadings: PC1 (22.8%) overall credit size (total credit limit 0.81, revolving limit 0.74, income
  0.68, revolving balance 0.68); PC2 (13.3%) utilization (0.77, 0.69) against recent account opening
  (-0.62, -0.59); PC3 (10.4%) FICO score (0.72) against utilization (-0.59, -0.52); PC4 (9.2%) public records (0.67, 0.65) with
  mortgage accounts and a low `dti`; PC5 (7.3%) loan size (0.53, 0.51) against average balance (-0.47); PC6 (6.3%) history length
  and `dti` against inquiries; PC7 (5.4%) delinquencies (0.80).
- **What the outlier treatment changed.** The structure is almost the same (loading similarity with the untreated PCA: PC1 0.99,
  PC2 0.98, PC3 0.96), but untreated, PC1 explains 20% instead of 23% and its scores are heavy-tailed: kurtosis 6.3 against -0.2,
  and one loan sits 11.5 standard deviations out against 3.6 after treatment. The treatment mostly stops a few extreme loans from
  steering the leading component.

### Factor analysis (`scripts/a5_factor_analysis.py`, `reports/a5_factor_analysis/`)
Maximum-likelihood factor analysis (scikit-learn) with a varimax rotation. KMO, Bartlett, parallel analysis and promax are our own
small functions in `src/credit_risk/factor_tools.py`, checked against the reference library (identical KMO and Bartlett) and
against simulated data with a known factor correlation. The library itself no longer runs with current scikit-learn, so it was
dropped as a dependency.
- **Suitability.** Bartlett's test rejects "no shared structure" (chi-square 225,269, p below 1e-300). KMO is 0.661 overall
  (above the 0.6 minimum). Seven fields are below 0.6 (`delinq_2yrs` 0.43, `avg_cur_bal` 0.56, `dti` 0.56, `revol_util` 0.57,
  `pub_rec` 0.58, `open_acc` 0.58, `pub_rec_bankruptcies` 0.59): they share less with the rest.
- **Eight factors, chosen by a rule fixed before looking at the result.** Parallel analysis alone gives 10, but with 14,000 loans
  nearly any real pattern beats noise, so the count follows a stated rule: the largest number of factors where each has at least two
  loadings of 0.4 or more, at least 3% of the variance, and a 2013-vs-2014 congruence of at least 0.95. Five to eight pass
  (the eighth: 3.5% of variance, congruence 0.989); nine fails (2.4% of variance) and ten fails (one clear loading). Kaiser and the
  PCA give 7, so the factor solution has one more dimension than the PCA. The eighth is `dti` against income, which is why it was
  missing before: with seven factors `dti` was barely represented (communality 0.16).
- **Rotation: varimax.** Under an oblique (promax) rotation the factors are mildly related (average correlation 0.19, largest
  0.50 between loan size and revolving credit size). Varimax keeps one cross-loading (income loads on credit capacity 0.46 and on
  debt burden -0.52) and gives uncorrelated scores (largest correlation between two scores 0.08), which suits a model built beside
  fields we just showed overlap heavily. Limit: some factor pairs are genuinely about half-correlated, which varimax ignores.
- **The factors**, named only from their loadings (68.1% of the variance in total; single-factor AUC in the last column):

| Factor | Variance | Fields and loadings | AUC alone |
|---|---|---|---|
| `credit_capacity` | 12.0% | average balance 0.97, total credit limit 0.91, mortgage accounts 0.54 | 0.559 |
| `loan_size` | 10.5% | installment 0.96, loan amount 0.96 | 0.546 |
| `revolving_credit_size` | 8.8% | revolving limit 0.87, revolving balance 0.83 | 0.535 |
| `account_count` | 7.2% | open accounts 0.87, total accounts 0.63 | 0.517 |
| `revolving_utilization` | 10.5% | revolving utilization 0.94, bank-card utilization 0.85, FICO -0.56 | 0.561 |
| `public_records` | 7.9% | bankruptcies 0.88, public records 0.87 | 0.516 |
| `recent_account_opening` | 7.7% | accounts opened in 12 months 0.82, in 24 months 0.81 | 0.588 |
| `debt_burden` | 3.5% | `dti` 0.60, income -0.52 | 0.569 |

  Names are assigned in code from these loadings (`src/credit_risk/factors.py`): a factor takes a name only if it loads at 0.5 or
  more, on average, on that name's fields; otherwise it stays unnamed.
- **Stable.** Loadings fitted on 2013 and on 2014 loans agree (congruence 0.989 to 0.999).
- **What the factors miss.** Four fields are barely represented (communality: `delinq_2yrs` 0.08, `inq_last_6mths` 0.10,
  credit history 0.14, `mort_acc` 0.37), so they stay as raw fields beside the factors.
- **Factors are not the PCA components.** Each factor's best match with a component is only 0.36 to 0.71 in correlation, so the two
  are different constructs and both go in the contract. On their own the factors predict default modestly (AUC 0.52 to 0.59), and
  the model ablation will show whether they help beyond the raw fields.

### When to prefer factor analysis over PCA here
PCA keeps as much variance as possible in the fewest components, whatever its source, so it is the better choice for compression
when only the model input matters. Factor analysis models only the variance a field *shares* with others and sets each field's own
noise aside. That gives factors that stand for something a credit officer can name (capacity, utilization, public records, debt
burden), and it fits this problem because decisions must be explainable and several fields (`delinq_2yrs`, `inq_last_6mths`) are
mostly their own noise. So the factors serve explanation and the scorecard, and the PCA components are kept as the compact
alternative; the ablation in the ML half reports which of the two, if either, adds anything beyond the raw fields.

## Stability and risk insights

### Stability, 2013 against 2014 (`scripts/a6_stability_psi.py`, `reports/a6_stability_psi/`)
PSI (Population Stability Index) measures how much a field's distribution moved: under 0.1 stable, 0.1 to 0.25 moderate,
0.25 or more major. Bins are set on the earlier year. 44 fields and category sets were checked, including all 8 factor scores.
- **42 of 44 are stable.** All factor scores are stable (0.006 to 0.031), so the factors describe 2014 applicants the way they
  describe 2013 ones. The next period is similar: only the platform's own `int_rate` moves between 2014 and 2015 (PSI 0.19,
  moderate), which means the platform changed how it prices, another reason not to depend on it.
- **`job_group` (6.3): a data change, now repaired.** The title field changed meaning in October 2013 (see "Job titles"). Within
  the valid period (October to December 2013 against 2014) PSI is 0.015, so after the repair the groups are stable. The 6.3 is the
  `employer_name_era` label itself.
- **`verification_status` (0.23, moderate): a real policy change.** "Verified" fell from 55% of loans in early 2013 to 23% by
  mid-2014 and "Source Verified" rose from 14% to 45%. The field stays a candidate input but is flagged for monitoring, since the
  link between verification and default may not hold as the policy moves.

### Default rate by factor score (`scripts/a6_deciles_and_segments.py`, `reports/a6_deciles_and_segments/`)
Deciles of each factor score, 95% Wilson intervals, a Cochran-Armitage trend test and a Mann-Whitney test, with Holm correction for
testing 8 factors. Loans to 2014.

| Factor | Default rate, decile 1 to 10 | Trend z | Holm p | AUC alone |
|---|---|---|---|---|
| `recent_account_opening` | 9.7% to 26.3% | +13.8 | below 0.001 | 0.588 |
| `debt_burden` | 11.1% to 22.2% | +10.6 | below 0.001 | 0.569 |
| `revolving_utilization` | 12.2% to 21.5% | +9.5 | below 0.001 | 0.561 |
| `credit_capacity` | 19.5% to 10.5% | -9.3 | below 0.001 | 0.559 |
| `loan_size` | 12.9% to 22.3% | +7.1 | below 0.001 | 0.546 |
| `revolving_credit_size` | 21.1% to 13.3% | -5.4 | below 0.001 | 0.535 |
| `public_records` | 17.6% to 17.7% | -2.5 | 0.028 | 0.516 |
| `account_count` | 16.5% to 18.5% | +2.4 | 0.028 | 0.516 |

Six factors show a clear, steady trend in the direction credit logic expects (more recent borrowing, more debt relative to income,
higher utilization and larger loans mean more default; more capacity and larger revolving limits mean less). `account_count` and
`public_records` are statistically detectable but practically negligible (a 2.0-point and a 0.1-point range): we do not claim they
separate risk.

### Segments, with a rule against small groups
A level is called a real difference only if it has at least 300 loans and its test against all other loans survives Holm correction
within that segmentation; otherwise the finding is "no clear difference" or "too few loans to say". Notable results:
- **Term:** 36 months 13.2%, 60 months 28.3% (the 60-month figure is inflated by the in-progress-loan bias in the target analysis).
- **Grade:** A 4.5%, B 10.8%, C 19.0%, D 24.6%, E 33.2%, F 38.2%; G (93 loans, 48.4%) is too few to say.
- **Purpose:** credit card 14.6% (safer), debt consolidation 18.2% (riskier).
- **Home ownership:** mortgage 15.4%, rent 19.8%. **Income:** lowest fifth 21.3%, highest fifth 13.2%. **Loan size:** smallest fifth 14.1%, largest 20.1%.
- **Verification:** not verified 12.7%, source verified 19.0%, verified 19.3% (the platform's verification choice, not a safety signal).
- **Job group:** only professional and technical differs (14.1%, safer); 10 other groups show no clear difference. **Year:** 2013 16.0%, 2014 18.0% (affected by the in-progress bias).
- **State:** none of the 17 states with at least 300 loans differs significantly from the rest after correction (overall chi-square p = 0.038);
  15 smaller states are "too few loans to say".

### What the factors add beyond the platform's grade (`scripts/a6_factors_vs_grade.py`)
On later loans (time-ordered folds before 2015), AUC: sub-grade alone 0.697, factors alone 0.651, sub-grade plus factors 0.706. The
paired gain from adding the factors is +0.010 and +0.007 in the two folds, with intervals that touch zero (-0.003 to +0.021), so a
better overall ranking is not proven. Within a grade band, though, the factors separate loans clearly (default rate, lowest against
highest third of factor-based risk, with non-overlapping intervals): grades A-B 7.7% against 13.2%; C 14.1% against 25.8%; D-G 26.2%
against 41.6%. The factors therefore carry risk information the grade does not fully use, but the grade already captures most of the
ranking and the factors correlate with it (utilization 0.33, recent account opening 0.27), so the overall gain is small. The
factors serve as an explanation and a cross-check, not a replacement.

### Could zip code or state act as a proxy for protected characteristics? (`scripts/a6_location_proxy.py`)
The data holds no race, sex or age, so a proxy cannot be measured directly, and that is the problem rather than a reassurance.
Location in the United States is strongly tied to the racial, ethnic and age make-up of neighbourhoods, so using it can reproduce
those differences even when the model never sees them. What the data does show:
- **It is too sparse to use:** 803 distinct 3-digit zip prefixes with a median of 10 loans each; 82% of prefixes (43.5% of loans) have under 30 loans.
- **It carries a little socio-economic signal:** state explains 0.8% of the variation in income (small, but not chance: p below 1e-20); median income runs from $59k to $75k across states.
- **It adds nothing to predictions, and hurts:** adding the state's WoE lowers the AUC on later loans in both folds (-0.009 and -0.007; intervals exclude zero).
- **Default does not differ reliably by state** (see segments above).

**Recommendation.** Do not use `zip_code` at all. Do not use `addr_state` as a model input. Keep state for monitoring only: compare approval,
default and calibration by state and region in the ML half (the brief asks for it), where a gap would be a warning to investigate. With
demographic data (not available here) a proper disparate-impact test would be the next step. Because location is sparse, uninformative and
risky, the model gives up nothing by excluding it.

### How training only on accepted loans limits these findings
Every loan here was approved by the platform and then ran to a final outcome, so the model learns the default rate among applicants the
platform had already chosen to accept, not among everyone who applies. People the platform would have declined, who are likely riskier, never
appear: no borrower in the data has a FICO score below 660, so nothing is known about lower scores, and the model can say nothing reliable
there. The relationships we measure are probably weaker than they would be in the full applicant pool, because the platform's own acceptance
rules already removed the extremes of the fields it used (score, `dti`, income), and a model trained here will look better calibrated and more
stable than it would on the people who actually apply. PD thresholds set on these default rates are therefore valid only for applicants who resemble
accepted ones. Declined applicants' outcomes cannot be observed, so the remedies are partial: restrict decisions to the range the data covers,
monitor the applicant mix, and, if the platform keeps declined applications, apply reject-inference methods or run small controlled approvals
of borderline applicants to learn what happens outside the accepted region. The label has a second selection effect, covered in the target analysis:
unfinished loans were removed, which inflates recent 60-month default rates.

## Review before building the contract and the model

An adversarial review of the work so far, with what each finding changed.
- **The holdout `revol_util` unit error** (issue 36) was missed by the first holdout check, which only looked for the errors found
  in training. It would have corrupted every holdout score silently. Fixed with an explicit conversion at load time and a fail-fast
  guard in the transformer. Lesson applied: a new file is compared with earlier years field by field, not only checked for known errors.
- **The revolving balance error** (issue 38) was found in a second review by testing relationships that must hold between fields, which
  I had not done for the training data. 308 balances were 10 times too large and made up 45% of the largest balances, so the "genuine
  extremes" of that field were mostly errors. Fixed with two independent checks, and every figure that depends on the field was
  regenerated from the scripts (the 3-standard-deviation table, the factor loadings, the VIF, the PCA and the deciles moved slightly;
  no conclusion changed). A regression test now asserts these relationships on the cleaned data.
- **Tuning inside the training years is optimistic.** WoE values are fitted on all pre-2015 labels (as the contract requires), so a
  validation fold inside those years has already been seen by the WoE: AUC 0.719 against 0.711 when the WoE is fitted on the training
  part only. The clean figure is the 2015 validation (0.716 for a quick WoE logistic). Hyperparameter tuning on time folds is therefore
  slightly optimistic and is reported as such; the final metrics come from 2015 only.
- **Thresholds must not be tuned on the data they are reported on.** The approve and decline cut-offs will be chosen on the first half
  of 2015 (or 2014 H2) and reported on the rest, so the policy results are not flattered.
- **60-month loans in 2015 are over-stated** (in-progress bias), so calibration and approval-rate results are reported separately by term.
- **`issue_d` means the application date when scoring.** Credit history length is measured from it. The spec asks for a date, and the demo
  form fills in today's date; the job-title era rule and everything else that reads it behave correctly for dates after 2013.
- **Valid ranges in the spec must come from plausibility, not from the fit window.** The 2016 holdout has `dti` up to 127 and income
  to $7M, which the caps handle; input validation must not reject them, and the contract test checks that the holdout passes.
- **The platform's own `int_rate` keeps drifting** (PSI 0.19 for 2014 to 2015, 0.16 for 2015 to 2016), confirming that it should stay a
  benchmark and not an input.
- **Scoring time:** the transformer takes about 24 to 40 ms for one application (cleaning and amount treatment are the slowest steps),
  which leaves about 60 ms of the 100 ms budget for the model and reason codes; it will be measured end to end.
- **Saving and reloading the fitted transformer** gives identical output (checked), so the handoff file is reliable.

