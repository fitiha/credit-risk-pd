# Design: probability-of-default model for unsecured personal loans

Written before any model code. The last section records what changed.

## 1. Problem
The credit team needs a probability of default (PD) at application time so each loan can be auto-approved, sent to manual review or declined.
Output per application: `pd`, `decision`, `model_version` and the top 3 to 4 reasons that push PD up. The label is `defaulted` (1 = charged off or
defaulted, 0 = fully paid). All inputs are known when the applicant applies; nothing from the servicing file is used (checked by a test).
The data are accepted loans only, so the model describes applicants the platform already chose to accept (see the findings).

## 2. Data and time split
| Use | Loans | Why |
|---|---|---|
| Train | issued up to 2014-12-31 (14,373) | the contract's fit window; `transform.joblib` is used unchanged and never refitted |
| Tune | two time-ordered folds inside training: 2013 -> 2014 H1, then up to 2014 H1 -> 2014 H2 | no random split; the future never trains the past |
| Calibrate | out-of-fold predictions of those two folds (2014, 9,024 loans) | the calibrator never sees loans the model trained on |
| Policy thresholds | 2015 H1 | chosen on the first half of 2015 and reported on the second half, so the results are not flattered |
| Validate and report | 2015 H2 (and 2015 H1 for the metrics that do not involve thresholds) | the only clean out-of-time test |
| Holdout | 2016 (5,000, no labels) | scored for `holdout_scores.csv`; only drift is measured |

Known limits to be reported with the results: WoE values inside the contract were fitted on all pre-2015 labels, so tuning inside 2013 to 2014 is
slightly optimistic (about 0.008 AUC, measured); 60-month loans from mid-2014 on are over-stated in the label (in-progress bias), so calibration
and policy results are reported separately for 36 and 60 months.

## 3. Inputs and the platform's own score (revised, see section 11)
Inputs are the fields the contract marks as `input` (reliability high or medium) **plus the platform's sub-grade** (as a rank, A1 to G5). The model therefore builds on the existing risk system and adds what it misses,
which is the usual way to improve an incumbent score. Left out: `int_rate` (it adds nothing once the sub-grade is in: tuning-fold AUC 0.705 against 0.705, and it drifts, PSI 0.19 then 0.16), the grade letter (the same information
as the sub-grade), job group and location (restricted pending the fairness review), and every servicing field.
Three comparison points are kept: **grade plus term** (the fair benchmark: term is information the grade does not contain), the **independent scorecard** (our fields only), and the **selected model** (our fields plus the sub-grade).
The ablation of the data scientist's outputs (raw fields, factor scores, both) is run with the sub-grade present in every set.

## 4. Models
1. **Baseline A, the old system:** logistic regression on the sub-grade rank and the loan term. The bar to beat.
2. **Baseline B, independent WoE scorecard:** logistic regression on the contract's WoE columns of our input fields, weights forced to point in the safe direction, L2 penalty tuned on the time folds. Interpretable by construction (points per bin).
3. **Selected, WoE scorecard plus the sub-grade:** the same scorecard with the sub-grade rank as one more input (standardised, and required to raise risk; the WoE weights keep the sign rule).
4. **Challenger, gradient boosting (LightGBM)** with the sub-grade as an input: on the treated fields, features and, in the ablation, factor scores. Monotone constraints only where credit logic and the data agree
   (PD rises with `dti`, utilisation, `loan_to_income`, `installment_to_income`, inquiries, delinquencies, public records, loan size and the 60-month term;
   PD falls with FICO, income, credit limits and credit history length). No constraint on counts of accounts, whose effect is mixed.
   A constraint is kept only if it costs less than 0.002 AUC on the time folds.
5. **Class imbalance (18.7% defaults):** no resampling, because it distorts PD. Class weighting is compared in tuning; the choice is made on Brier score and AUC after calibration.
6. **Calibration:** isotonic or Platt, whichever has the lower Brier score on the out-of-fold predictions, checked on a reliability plot and by term. (Changed during the build, see section 10: calibrated separately for each term.)
7. **Tuning:** a small fixed grid on the two time folds, seeds fixed, every run appended to `reports/experiments.csv`.

## 5. Metrics
Ranking: ROC-AUC, Gini (2 x AUC - 1) and KS. Probability quality: Brier score, reliability plot and calibration by term, by quarter and by decile (a PD is only useful for thresholds if it is
accurate). Stability: all of these by quarter of 2015. Business: profit and default rate by decision band. Everything is compared against grade alone.

## 6. Decision policy
Assumption: a defaulted loan loses 70% of `loan_amnt`, a repaid loan earns 10%. Expected profit per loan = amount x (0.10 (1 - PD) - 0.70 PD), which is zero at PD = 12.5%.
- **Auto-decline** above the break-even PD (12.5%, then adjusted to the profit-maximising value on 2015 H1).
- **Auto-approve** below the PD where the margin falls under 5% of the loan, PD = 6.25%, the minimum margin worth approving without a human look.
- **Manual review** in between.
Reported on 2015 H2: approval, review and decline rates, default rate and profit per band, and the same by state region and income band (monitoring only; state is not a model input).

## 7. Reproducibility and lineage
One command trains from the contract (`uv run python -m credit_risk.model.train --config config/train.yaml`). It loads `transform.joblib`, validates the data against `data_spec.json` and fails on a
violation, takes every setting from a config file, fixes all seeds, appends each experiment to a CSV, and saves with the model: git commit, a hash of the training data, training date,
contract version and the thresholds. Loading a model whose contract version differs from the transform's is refused.

## 8. Constraints
Decisions must be explainable: reason codes come from the model's own contributions (WoE times coefficient for the scorecard, SHAP values for the boosted model), mapped to plain sentences.
Scoring one application must take under 100 ms: the transform takes about 20 ms, leaving room for the model and the reason codes; p50 and p95 are measured and reported.

## 9. Serving, monitoring, retraining (not built, described)
- **Serve:** the same package behind a small stateless HTTP service (the demo app calls the same `predict` function); input validated against the spec before scoring, model and transform versioned together.
- **Monitor:** weekly PSI of the inputs, factor scores and PD (alert at 0.10 investigate, 0.25 act), approval and review rates, and the share of inputs outside the fitted caps. Defaults show up late
  (36 to 60 months), so early-delinquency rates by cohort stand in for outcome monitoring; calibration is checked as cohorts mature. Any input arriving in a different unit is rejected by the spec check.
- **Retrain:** yearly, or when PSI exceeds 0.25 on a key input or calibration drifts; refit the whole contract on the new window, compare with the incumbent on the most recent matured cohort, run in shadow before switching.
- **Fairness:** compare approval and default rates by region and income band every release; test disparate impact when demographic data is available.

## 10. What changed (written at the end)
- **The platform's sub-grade became an input.** The first version kept the platform's score out of the model and only matched it (AUC 0.718 against 0.722 on 2015). After the evidence in section 11 (written before any code change), the sub-grade was allowed in, the interest rate and the grade letter were not. The independent scorecard stays as a fallback, switchable with `platform_score: none` in `config/train.yaml`.
- **Selection was made leakage-free and by a rule fixed in advance.** The contract's WoE values were fitted on all pre-2015 labels, which flatters the scorecard on the tuning folds, so selection and tuning refit the WoE inside each fold (the contract itself is untouched). On the two time-ordered folds (3,997 and 5,027 loans) the mean AUCs are: grade alone 0.697, grade plus term 0.702, independent scorecard 0.699, gradient boosting without the sub-grade 0.700, **scorecard plus sub-grade 0.712**, gradient boosting plus sub-grade 0.712. The boosted model was ahead in one fold (+0.003) and behind in the other (-0.004), so under the rule (ahead by 0.005 on average and in every fold) the scorecard stays. These tuning-fold figures are slightly optimistic, because the regularisation and the number of inputs were chosen on the same folds; 2015 played no part in any choice.
- **Result on 2015, which was never used to choose:** the selected model reaches AUC 0.734 against 0.722 for the grade alone, a gain of +0.012 (95% interval +0.007 to +0.016), so it beats the old system; the independent scorecard does not (-0.004, interval -0.010 to +0.004). The boosted model plus sub-grade reaches 0.735, a difference of 0.0006 from the scorecard, which is noise.
- **The model is small:** the tuned scorecard keeps 12 WoE columns and the sub-grade (strong regularisation, C = 0.03). The sign rule (a WoE weight that points the wrong way is removed; the sub-grade must raise risk) removed nothing in the final fit, but it was needed in the comparison runs, where `mort_acc` otherwise added out-of-time noise once the sub-grade was present.
- **Monotone constraints help; the design had them backwards.** Section 4 said a constraint is kept only if it costs less than 0.002 AUC. In fact the declared constraints raise the time-fold AUC (0.697 without, 0.702 with); an audit one constraint at a time kept 23 of 28.
- **Calibration is per term, not pooled:** slopes on 2015 are 0.97 for 36-month and 0.92 for 60-month loans. The 60-month figure is the one the in-progress-loan bias (target analysis) makes uncertain.
- **The profit policy does not show an advantage.** At the thresholds from the profit assumptions (approve below 6.25%, decline at 12.5% or more) the model approves 16.6% of 2015 H2 loans with a 4.2% default rate. A policy driven by grade alone earned $0.11M more, but the 95% interval of the difference (-$0.41M to +$0.18M) includes zero, so neither policy is measurably more profitable, although the model ranks applicants better.
- **Reason codes compare with a typical approved applicant**, not the average one: against the average, most declined applicants have no reason at all (the average PD, 18.7%, is far above the 12.5% decline line). A reason must raise the odds of default by roughly 10%. The sub-grade can be one of the reasons; it says less than a specific factor, so the other reasons follow it.
- **Live scoring has a strict mode** that rejects an application that looks mis-entered (a loan ten times too large, an income in thousands) instead of repairing it silently; repairs remain for historical data.
- **Thresholds** were kept at the profit-based 6.25% and 12.5%. The profit-maximising decline threshold on 2015 H1 was 11.5%, and break-even earns 93% of that optimum, a small loss; break-even follows from the stated assumptions and not from one half-year.
- **The data contract changed twice during the build** (the binning of count fields, and the recoding of zero delinquency months; `docs/issue_log.md` entries 38 and 39), both found in review before the model was finalised. The contract version stays 1.0.0 because it had not been released to anyone.
- **Not built, as planned or by choice:** the experiment log is a CSV, not MLflow; no production API or Docker (out of scope); no test of disparate impact because the data has no demographic fields.
- **Latency** is 35 ms at the median and 36 ms at the 95th percentile for the whole call (the contract transform is about 21 ms of it), inside the 100 ms budget.

## 11. Revision 2: building on the sub-grade (decided before any code change)
**Why.** The first version kept the platform's grade out of the model. Evaluated fairly, that model only matches the old system: on 2015 loans AUC 0.718 against 0.722 for grade plus term (difference not significant), and a policy driven by the grade earned slightly more. The aim is a model that
improves on the old system, and an independent model cannot beat a model that sees the same facts.
**Evidence, from the two time-ordered tuning folds only (no 2015), against the benchmark grade plus term (mean AUC 0.702):** our independent scorecard 0.699 (-0.000 and -0.007); scorecard plus sub-grade 0.705 (+0.004 and +0.002, same direction in both folds, each
within noise because the folds hold 4,000 and 5,000 loans); adding the interest rate changes nothing (0.705); boosting plus grade and rate is inconsistent between folds (+0.014, then -0.001) and does not beat the scorecard. The later 2015 result for the same model (+0.009 over the grade, interval +0.004 to +0.014) has been seen and is
reported, but the choice rests on the tuning folds.
*Re-run on 31 October after two data-quality fixes in the contract (binning of count fields with many zeros, and the zero coding of delinquency months).* The first run of this comparison gave 0.710 for scorecard plus sub-grade; after the fixes it gives 0.705. The difference comes from `mort_acc`, which became informative after the binning fix and, when no sign rule is applied, adds out-of-time noise once the sub-grade is in (switching off the three newly informative fields restores 0.710). The implementation therefore applies the scorecard's sign rule to the WoE inputs only. The selection rule below is unchanged and still picks the scorecard (the boosted model is ahead by 0.004 on average and in one fold only).
**Decision (taken by the project owner):** use the sub-grade as an input, not the interest rate.
**Selection rule, fixed now:** the selected model is the scorecard plus sub-grade unless a boosted model beats it by at least 0.005 mean AUC on the tuning folds and in both folds; results on 2015 are reported and never used to choose.
**Risks and how they are handled.** (1) The model now depends on the platform's grading: if the platform changes its grading, the sub-grade's meaning shifts. Monitoring: PSI of the sub-grade mix, and observed default rate by grade band against its history, at every release; alert at PSI 0.10 (investigate) and 0.25 (act).
(2) Fallback: the independent scorecard is trained and saved with the same pipeline and can be switched on by changing one setting, so the decision is reversible without retraining. (3) The claim is modest: about +0.01 AUC, and any further claim needs fresh data. (4) The comparison with the old system is always made against grade plus term, never against the grade alone.
**What this changes in the build:** the input sets and the scorecard columns gain the sub-grade rank; the benchmark becomes grade plus term; the model selection, evaluation, reason codes (the sub-grade can be a reason: "platform grade"), the model card and the demo are regenerated; the policy is re-checked against the old system's.

