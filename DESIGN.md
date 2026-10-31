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

## 3. Inputs and the platform's own score
Inputs are the fields the contract marks as `input` (reliability high or medium). Left out: `int_rate`, `grade`, `sub_grade` (the platform's own risk
model; the model must not just copy it and the comparison "beats grade alone" must stay meaningful, and `int_rate` also drifts, PSI 0.19 and 0.16),
job group and location (restricted pending the fairness review), and every servicing field. Grade is used as the benchmark and as a test variant.
Three input sets are compared in the ablation: raw treated fields and features, the eight factor scores, and both.

## 4. Models
1. **Baseline A, grade alone:** logistic regression on the sub-grade rank. The bar to beat.
2. **Baseline B, WoE scorecard:** logistic regression on the contract's WoE columns of the input fields, with an L2 penalty tuned on the time folds. Interpretable by construction (points per bin).
3. **Challenger, gradient boosting (LightGBM):** on the treated fields, features and, in the ablation, factor scores. Monotone constraints only where credit logic and the data agree
   (PD rises with `dti`, utilisation, `loan_to_income`, `installment_to_income`, inquiries, delinquencies, public records, loan size and the 60-month term;
   PD falls with FICO, income, credit limits and credit history length). No constraint on counts of accounts, whose effect is mixed.
   A constraint is kept only if it costs less than 0.002 AUC on the time folds.
4. **Class imbalance (18.7% defaults):** no resampling, because it distorts PD. Class weighting is compared in tuning; the choice is made on Brier score and AUC after calibration.
5. **Calibration:** isotonic or Platt, whichever has the lower Brier score on the out-of-fold predictions, checked on a reliability plot and by term.
6. **Tuning:** a small fixed grid on the two time folds, seeds fixed, every run appended to `reports/experiments.csv`.

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

## 10. What changed
(To be completed at the end.)
