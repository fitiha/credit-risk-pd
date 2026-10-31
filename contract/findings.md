# Findings: what the data says about default risk, and what we did with it

## Summary for the credit manager (one page)

**The data had serious problems, and we fixed them before using it.** We checked 30,000 accepted loans (2013 to 2015) from three systems and found and logged 38 issues. The ones that mattered most:
- **Two columns revealed the outcome** (a "recovery flag" that was a hidden copy of a post-loan field, and the loan-servicing file itself). A model that used them would look perfect and fail on every new applicant. They are excluded and tested for.
- **About 1,500 loans had money amounts entered wrongly** (incomes typed in thousands, loan amounts and revolving balances ten times too large). They are repaired using cross-checks between fields, and nothing was deleted except 600 duplicate loans.
- **The 2016 file stored one ratio as a fraction instead of a percent.** Unnoticed, every 2016 score would have been wrong. It is converted and a guard now rejects such files.

**What predicts default.** The loan term (60-month loans default 32% of the time, 36-month 14%), the platform's own grade (A: 4.5% to F: 38%), how much and how recently an applicant has borrowed, how much of their credit limits they use, the loan's size against income, and FICO score. Job title, home ownership and place add almost nothing.

**Default is not a steady number.** It rose from 16% (2013) to 18% (2014) to 20% (2015). Part of that is real; part is an artifact: loans still being repaid were removed from the data, which inflates the default rate of recent 60-month loans. We report 36- and 60-month results separately for that reason.

**How good is the model?** On 2015 loans it had never seen, it separates good from bad loans with an AUC of 0.72 (0.5 is a coin flip). The platform's own grade reaches the same level, so our model matches it without using it, and our fields add about +0.01 on top of it. Predicted and observed default rates agree overall.

**Recommendations**
1. **Use the thresholds tied to your profit assumptions:** approve below a 6.25% chance of default, decline at 12.5% or more (where a loan stops making money), review in between. On the second half of 2015 this approves 12.7% of loans at a 4.2% default rate.
2. **Do not use zip code or state in the model.** Place can stand in for protected characteristics; here it also made predictions worse. Use state only to monitor outcomes. Treat job title the same way until a fairness review.
3. **Be careful with 60-month loans:** their risk is under-stated by about 2.5 points in 2015.
4. **Keep the data on declined applicants.** Everyone here was accepted, nobody has a FICO score below 660, and the model cannot speak for lower scores.
5. **Monitor monthly** the shape of the applicant pool (we give alert levels); the 2016 file is stable against 2015.

---

# Technical appendix

Evidence for every figure is in `docs/` (decision log `decisions.md`, issue log `issue_log.md`, target analysis, field reliability) and `reports/` (one folder per analysis, named like the script in `scripts/` that made it). Everything is fitted on loans issued up to 2014-12-31.

## 1. Data audit and cleaning (`docs/issue_log.md`, `scripts/a1_*`, `src/credit_risk/cleaning.py`)
38 issues, each with rows affected, a decision (fix, flag or drop) and a reason. Dropped: 600 duplicate loan ids (the copies differ only by 0.01 in `dti` and in letter case). Repaired: mixed date formats (1,200), state spellings (3,000), padded terms (30,000); unit errors found with cross-checks that do not rely on a range: loan amount x10 (300, via the installment), income in thousands (899), revolving balance x10 (308, via utilisation, limit and total balance); sign flips in utilisation (585); credit dates after the loan (300, set missing); swapped FICO range (150); a placeholder interest rate (8). Flagged and kept: genuine extremes, missing patterns, label oddities. Tested against the servicing data three ways the label has no contradictions. Leaks: `bureau_recovery_flag` is `recoveries > 0` of the servicing file (4,207 of 4,207) and `debt_settlement_flag` is 100% default.

## 2. Target analysis (`docs/target_analysis.md`, `reports/a1_target_analysis`)
Default rate 18.7% (1 in 4.3). By term 14.0% and 32.3%; by year 15.9%, 18.0%, 20.1%. 60-month loans from mid-2014 had not reached their scheduled end by the March 2019 snapshot, so only those that ended early remain (65% repaid early, 35% defaulted); removing the loans still being paid leaves too few paid loans and inflates the rate (24.7% for 60-month loans due by 2019 against 34.9% for the rest). Consequences: no issue-date input, results by term, caution on 2015 calibration.

## 3. Financial values (`docs/decisions.md`, `reports/a2_*`)
Profile (mean, median, std, IQR, skew, percentiles, linear and log charts) for 11 money and ratio fields. Mean +/- 3 std is not sound for six heavy-tailed fields (skew up to 63: the tail inflates the std, the lower limit is negative, it flags real borrowers). Nine treatments were compared on shape, rows changed and predictive power on later loans; chosen: cap at the 0.5th and 99.5th percentile, then log, for money fields (skew at most 0.8, 1% of rows touched); cap only for ratios and counts. Errors were separated from genuine extremes by field. Zeros keep an indicator, missing values are filled with the fit-window median plus an indicator where at least 1% are missing. Default rate by decile is identical before and after treatment.

## 4. Reliability and leakage (`docs/field_reliability.md`)
Every field rated high, medium or low with evidence, and every servicing field rated low as a leak (one field alone predicts default at AUC 0.72 to 0.91, against 0.51 to 0.59 for application-time fields). Income is usable although unverified (it tracks credit limits equally well in every verification group). Missing values are structural and not clearly tied to default. The platform's grade and rate are kept out of the model: our inputs reach 0.708 AUC against 0.697 for the grade, and adding it gains only 0.004 on the tuning folds.

## 5. Categories and Weight of Evidence (`reports/a4_*`)
Job titles grouped by keyword rules (the field held employer names before October 2013, found through a PSI of 2.3 and repaired); rare levels merged below 1% of loans; employment length ordered; numeric fields binned by merging equal-sized bins until the default rate moves in one direction (best IV stability on later loans). IV ranking: the platform's score 0.46, then term 0.20, FICO 0.11, loan-to-income 0.11, with 16 of 37 fields below 0.02. Every post-loan field scores 0.7 to 9.4, the signature of a leak; none of ours does.

## 6. Correlation, PCA and factor analysis (`reports/a5_*`)
Six field pairs correlate at 0.7 or more and six fields have a VIF above 10, which motivates reduction. PCA on standardised treated fields: 7 components (Kaiser and parallel analysis agree), 74.6% of the variance; treatment barely changes their structure but stops single loans from dominating (one loan 11.5 standard deviations out against 3.6). Factor analysis (KMO 0.66, Bartlett significant): 8 factors chosen by a rule fixed beforehand, varimax rotation, named only from loadings (credit capacity, loan size, revolving credit size, account count, revolving utilization, public records, recent account opening, debt burden), 68% of variance, stable across years. Factor analysis is preferred for explanation, PCA for compression. A fit with a near-duplicate field gave an improper solution; the field was dropped after a robustness check.

## 7. Stability and risk insights (`reports/a6_*`)
PSI 2013 to 2014: 42 of 44 fields stable; the exceptions were the job-title field (data change, repaired) and verification status (a real policy change). All factor scores are stable. Default by factor decile and by segment is shown with 95% intervals and Holm-corrected tests; a segment counts only with at least 300 loans. Against the platform's grade the factors separate loans within each grade band but add little to the overall ranking. Zip code is too sparse to use and state hurts predictions; neither is used. The accepted-only limitation is discussed in `decisions.md`.

## 8. The contract (`contract/`)
`transform.joblib` (the whole fitted chain from a raw application to cleaned fields, features, groups, WoE, PCA components and factor scores, with a version), `data_spec.json` (ranges, cleaning rules, reliability, groupings, WoE bins, factors), `analytical_dataset.parquet`, `reference_transforms.csv` (50 raw rows covering every kind of repair, with expected outputs) and this report. `uv run python scripts/verify_contract.py` checks that the transform reproduces the reference rows to 1e-6 (it matches exactly); the same check runs in the test suite.
