# Model card: probability of default for unsecured personal loans

**Developed by:** Natnael Fisseha.
**Model:** average of the WoE scorecard and gradient boosting, both with the platform sub-grade, version `scorecard_boosted_average-contract1.0.0-2026-11-02-4d83cf8`.
The average, in log-odds, of two models that both use the platform's sub-grade: a logistic regression on the contract's Weight-of-Evidence (WoE) columns (every WoE weight forced to point in the safe direction, the sub-grade required to raise risk) and gradient boosting with monotone constraints on the treated fields. Calibrated separately for 36- and 60-month loans. Reason codes add the two models' contributions, each measured against a typical approved applicant, so they add up exactly to the score.
Trained 2026-11-02 on 14,373 loans issued 2013-01-01 to 2014-12-01 (contract fit window: up to 2014-12-31). Seed 42.

## Intended use
Estimate the probability that a new applicant for a 1,000 to 40,000 USD, 36- or 60-month unsecured loan will default, to auto-approve, send to manual review or decline.
Not for any other product, for applicants unlike those the platform accepted, or as the only basis of an adverse decision without human review.

## Data
Historical consumer loans accepted by a lending platform and issued 2013 to 2015, from application, credit-bureau and loan-servicing extracts. Label: charged off or defaulted (1) against fully paid (0); loans still in progress were removed.
The servicing file is recorded after the loan and is used only to audit the label. The 2016 loans have no labels. The data contract (`contract/`) holds the fitted cleaning, treatment, features, groups, WoE, PCA and factor model, fitted on loans to 2014-12-31.
Inputs: the scorecard uses 13 columns (12 WoE columns and the platform's sub-grade), the boosted model 42 treated fields, features, factor scores and the sub-grade; reasons are reported for 42 fields. Never inputs: the platform's `int_rate` and grade letter, job group and location (restricted pending a fairness review), and every servicing field.

## Performance on 2015 loans (out of time; 15,627 loans, default rate 20.1%)
| Model | AUC | Gini | KS | Brier | AUC, 36 months | AUC, 60 months |
|---|---|---|---|---|---|---|
| Platform sub-grade alone (benchmark) | 0.722 | 0.444 | 0.322 | 0.145 | 0.679 | 0.644 |
| Independent WoE scorecard, no platform score (fallback) | 0.718 | 0.437 | 0.322 | 0.144 | 0.668 | 0.658 |
| WoE scorecard plus the platform sub-grade (variant) | 0.734 | 0.468 | 0.342 | 0.142 | 0.694 | 0.670 |
| Gradient boosting, monotone constraints, no platform score (challenger) | 0.725 | 0.449 | 0.335 | 0.143 | 0.678 | 0.666 |
| Gradient boosting plus the platform sub-grade (challenger) | 0.735 | 0.469 | 0.343 | 0.142 | 0.697 | 0.668 |
| Average of the two sub-grade models (selected) | 0.737 | 0.473 | 0.346 | 0.141 | 0.698 | 0.674 |

Stable by quarter (selected model, AUC): Q1 0.717; Q2 0.744; Q3 0.746; Q4 0.738.
**Against the platform's grade:** the selected model (which uses the sub-grade) has AUC 0.737 against 0.722 for the grade alone, a gain of +0.015 (95% interval +0.011 to +0.019), so it beats the grade measurably. The same scorecard without any platform score is -0.004 (-0.010 to +0.004): it matches the grade but does not beat it, which is why the sub-grade is allowed in. Neither single model beat the other under the rule fixed in `DESIGN.md` (boosting had to be ahead by 0.005 AUC on average and in every fold; its gains over the scorecard were +0.0032, -0.0042). A later screen of about fifteen ideas on the same folds, with its own rule fixed beforehand (a change must add 0.003 AUC on average and in every fold), found one that passed: averaging the two models (mean tuning-fold AUC 0.7158 against 0.7121 for the scorecard). With that many ideas tried, a pass this small can partly be luck. 2015 was never used to choose.
The price of the gain is a dependence on the platform's own grading (see the known limits). The platform's interest rate moved by a PSI of 0.19 and 0.16 in two years, which is why the rate is not an input.
Calibration (slope 1 is perfect): overall 0.98, 36 months 0.99, 60 months 0.94.

## Decision thresholds and what they do (2015 H2, loans the model had not seen)
Assumption: a defaulted loan loses 70% of the amount, a repaid loan earns 10%, so expected profit is zero at PD = 12.5%. **Decline at PD of 12.5% or more;
auto-approve below 6.25%** (where the expected margin drops to 5% of the loan); review in between.

| Band | Share of loans | Predicted PD | Observed default rate | Profit if funded |
|---|---|---|---|---|
| Approve | 17.6% | 4.4% | 4.4% | $1.38M |
| Review | 26.9% | 9.3% | 11.3% | $0.17M |
| Decline | 55.5% | 26.4% | 29.4% | -$11.58M |

Funding every applicant would have lost $10.03M; funding approve plus review earns $1.55M. A policy driven by grade alone, at the same thresholds,
earns $1.69M, against $1.55M for the model. The difference (model minus grade) is -$0.14M, with a 95% interval of -$0.43M to $0.15M, which includes zero: the model's policy is not measurably more or less profitable than the grade's, although it ranks applicants better. Part of that gap is volume, not ranking: at the same PD thresholds the model funds 44% of loans and the grade policy 35% (the two scores spread applicants differently around the 12.5% line). At equal volume, funding the safest 44% of loans by each score (the volume the model's own thresholds fund, fixed by the policy and not chosen from these results), the model earns $0.12M more than the grade on 2015 H2 (95% interval -$0.13M to $0.43M) and $0.52M more on all of 2015 ($0.12M to $0.92M). On all of 2015 the advantage is beyond chance from 44% of applicants funded upward; at smaller volumes the differences are within chance. The tables for every volume are in `reports/b1_model_evaluation/tables/`. The profit-maximising decline threshold on 2015 H1 was 11.0%, against the break-even 12.5%; break-even earns 94% of that optimum, a small loss, and the break-even threshold is kept because it follows from the stated profit assumptions and not from one half-year of data.

## Which data-science outputs helped (ablation, time-ordered tuning folds, AUC)
| Inputs | Gradient boosting | Logistic |
|---|---|---|
| Raw treated fields and features | 0.6970 | 0.6987 (WoE columns, contract bins) |
| Factor scores only | 0.6967 | 0.6975 (plus loan-structure WoE) |
| Both | 0.6997 | 0.6952 |

**Verdict.** The WoE bins helped: a scorecard on them is the best simple model and gives plain reasons. The factor scores helped the boosted model only a little: adding them to the raw fields moved its AUC by +0.0027, just above the 0.002 needed to adopt them (a difference this small is within the noise of two folds of 4,000 and 5,000 loans), so the boosted challenger uses them. They did not help the scorecard (WoE columns alone 0.6987, with factor scores 0.6952), so the selected model does not use them. Eight factor scores alone reach 0.6967 against 0.6970 for the raw inputs: a compact stand-in, not an improvement. The PCA components were not used. What the data scientist would act on next: build factors from fields not yet represented (`dti`, delinquencies, inquiries have the lowest communality) and test fewer, better-defined factors; the monotone constraints raised the boosted model's time-fold AUC (0.697 without, 0.702 with) and are kept.

## Monitoring and drift
The 2016 holdout (5,000 loans, scored in `outputs/holdout_scores.csv`) is stable against 2015: PSI of the PD distribution 0.010, the ten most important inputs at most 0.017
(alert levels: 0.10 investigate, 0.25 act). Decision mix: approve 15.5%, review 26.2%, decline 58.3%.
Outcomes by region and income band are monitored, not modelled: approval rate by income band runs from 6.2% to 30.0% (it follows income, as the credit signals do), and by region from
16.4% to 18.8%; observed default rates by region sit within 18.9% to 21.2%.

## Known limits
- **Accepted loans only.** Applicants the platform declined are missing, nobody in the data has a FICO score below 660, and the model says nothing reliable about lower scores. Relationships are probably weaker here than in the full applicant pool.
- **In-progress loans were removed**, which raises the default rate of recent 60-month loans. 60-month results are therefore reported separately and calibrated separately.
- **Default rates rose** from 2013 to 2015 and the model, trained on 2013 to 2014, under-predicts 2015 risk in places: 16 of 45 segments tested differ significantly from their predicted rate (all under-predict, for example grades B, C, D by 2 to 3 points).
- **The information ceiling is about 0.70 to 0.73 AUC** on the fields of this data set alone: the platform's grade, built from the same kind of data, reaches the same level, and the sub-grade lifts the selected model to 0.74.
- **Dependence on the platform's grading.** The selected model uses the platform's sub-grade, the output of the platform's own risk model. If the platform changes how it grades, the sub-grade's meaning shifts and the model's scores shift with it. The mix of sub-grades is monitored (PSI 2016 against 2015: 0.017) together with the observed default rate by grade band, and the independent scorecard (no platform score) is kept as a fallback, switchable with `platform_score: none` in `config/train.yaml`. A reason code can also cite the sub-grade itself; that explains a decision only by pointing at the platform's own rating, which says less than a specific factor such as the credit score, so the reasons list the other factors after it.
- **WoE inside the contract was fitted on all pre-2015 labels**, so tuning inside 2013 to 2014 is slightly optimistic (about 0.01 AUC); model selection used WoE refitted inside each fold. Settings (regularisation, number of inputs) were also chosen on the tuning folds, so the tuning-fold AUCs (0.7039, 0.7276 for the selected model) are slightly optimistic; 2015 was not used for tuning or selection (it prompted the decision to use the sub-grade, whose evidence came from the tuning folds; `DESIGN.md` section 11).
- **Location and job title are not used.** Occupation and place can stand in for characteristics that are not in the data; the data cannot test for it. State is used only for monitoring.
- **Single-application scoring:** about 46 ms at the median and 57 ms at the 95th percentile on a workstation (limit 100 ms).
- **Application date.** Credit history length is measured from the application date; the demo defaults to mid-2015, inside the period the model was validated on.

## Lineage
Git commit `4d83cf8ef66e5ea71c64fccac89aa5f2d256581c`; training-data hash `4ff160121523e193...` (for models on WoE columns it covers the loan ids and labels; the input values are fixed by the contract version and `transform.joblib`); contract version 1.0.0; calibration by term (36 months: platt, 60 months: platt);
experiments in `reports/experiments.csv`. A model whose contract version differs from the transform's is refused at load time.
