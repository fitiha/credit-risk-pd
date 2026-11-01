# Model card: probability of default for unsecured personal loans

**Model:** WoE scorecard plus the platform sub-grade, version `scorecard_plus_grade-contract1.0.0-2026-11-01-4fc304c`.
A logistic regression on the Weight-of-Evidence (WoE) columns of the data contract plus the platform's sub-grade. Every WoE weight is forced to point in the safe direction and the sub-grade must raise risk. Calibrated separately for 36- and 60-month loans.
Trained 2026-11-01 on 14,373 loans issued 2013-01-01 to 2014-12-01 (contract fit window: up to 2014-12-31). Seed 42.

## Intended use
Estimate the probability that a new applicant for a 1,000 to 40,000 USD, 36- or 60-month unsecured loan will default, to auto-approve, send to manual review or decline.
Not for any other product, for applicants unlike those the platform accepted, or as the only basis of an adverse decision without human review.

## Data
Historical consumer loans accepted by a lending platform and issued 2013 to 2015, from application, credit-bureau and loan-servicing extracts. Label: charged off or defaulted (1) against fully paid (0); loans still in progress were removed.
The servicing file is recorded after the loan and is used only to audit the label. The 2016 loans have no labels. The data contract (`contract/`) holds the fitted cleaning, treatment, features, groups, WoE, PCA and factor model, fitted on loans to 2014-12-31.
Inputs: 13 columns (12 WoE columns and the platform's sub-grade) of 34 candidates. Never inputs: the platform's `int_rate` and grade letter, job group and location (restricted pending a fairness review), and every servicing field.

## Performance on 2015 loans (out of time; 15,627 loans, default rate 20.1%)
| Model | AUC | Gini | KS | Brier | AUC, 36 months | AUC, 60 months |
|---|---|---|---|---|---|---|
| Platform sub-grade alone (benchmark) | 0.722 | 0.444 | 0.322 | 0.145 | 0.679 | 0.644 |
| Independent WoE scorecard, no platform score (fallback) | 0.718 | 0.437 | 0.322 | 0.144 | 0.668 | 0.658 |
| WoE scorecard plus the platform sub-grade (selected) | 0.734 | 0.468 | 0.342 | 0.142 | 0.694 | 0.670 |
| Gradient boosting, monotone constraints, no platform score (challenger) | 0.725 | 0.449 | 0.335 | 0.143 | 0.678 | 0.666 |
| Gradient boosting plus the platform sub-grade (challenger) | 0.735 | 0.469 | 0.343 | 0.142 | 0.697 | 0.668 |

Stable by quarter (selected model, AUC): Q1 0.714; Q2 0.740; Q3 0.744; Q4 0.735.
**Against the platform's grade:** the selected model (which uses the sub-grade) has AUC 0.734 against 0.722 for the grade alone, a gain of +0.012 (95% interval +0.007 to +0.016), so it beats the grade measurably. The same scorecard without any platform score is -0.004 (-0.010 to +0.004): it matches the grade but does not beat it, which is why the sub-grade is allowed in. The boosted model with the sub-grade reaches +0.013; under the rule fixed in `DESIGN.md` (it must be ahead by 0.005 AUC on the tuning folds on average and in every fold; its gains there were +0.0032, -0.0042) it did not replace the scorecard. The choice rested on the tuning folds; 2015 was never used to choose.
The price of the gain is a dependence on the platform's own grading (see the known limits). The platform's interest rate moved by a PSI of 0.19 and 0.16 in two years, which is why the rate is not an input.
Calibration (slope 1 is perfect): overall 0.97, 36 months 0.97, 60 months 0.92.

## Decision thresholds and what they do (2015 H2, loans the model had not seen)
Assumption: a defaulted loan loses 70% of the amount, a repaid loan earns 10%, so expected profit is zero at PD = 12.5%. **Decline at PD of 12.5% or more;
auto-approve below 6.25%** (where the expected margin drops to 5% of the loan); review in between.

| Band | Share of loans | Predicted PD | Observed default rate | Profit if funded |
|---|---|---|---|---|
| Approve | 16.6% | 4.4% | 4.2% | $1.29M |
| Review | 28.4% | 9.3% | 11.1% | $0.28M |
| Decline | 55.0% | 26.3% | 29.6% | -$11.60M |

Funding every applicant would have lost $10.03M; funding approve plus review earns $1.57M. A policy driven by grade alone, at the same thresholds,
earns $1.69M, against $1.57M for the model. The difference (model minus grade) is -$0.11M, with a 95% interval of -$0.41M to $0.18M, which includes zero: the model's policy is not measurably more or less profitable than the grade's, although it ranks applicants better. Part of that gap is volume, not ranking: at the same PD thresholds the model funds 45% of loans and the grade policy 35% (the two scores spread applicants differently around the 12.5% line). At equal volume, funding the safest 45% of loans by each score (the volume the model's own thresholds fund, fixed by the policy and not chosen from these results), the model earns $0.15M more than the grade on 2015 H2 (95% interval -$0.17M to $0.43M) and $0.62M more on all of 2015 ($0.15M to $1.01M). Funding only the safest 20% to 35% it earns about the same; the advantage appears when more than about 40% of applicants are funded. The tables for every volume are in `reports/b1_model_evaluation/tables/`. The profit-maximising decline threshold on 2015 H1 was 11.5%, against the break-even 12.5%; break-even earns 93% of that optimum, a small loss, and the break-even threshold is kept because it follows from the stated profit assumptions and not from one half-year of data.

## Which data-science outputs helped (ablation, time-ordered tuning folds, AUC)
| Inputs | Gradient boosting | Logistic |
|---|---|---|
| Raw treated fields and features | 0.6970 | 0.6987 (WoE columns, contract bins) |
| Factor scores only | 0.6967 | 0.6975 (plus loan-structure WoE) |
| Both | 0.6997 | 0.6952 |

**Verdict.** The WoE bins helped: a scorecard on them is the best simple model and gives plain reasons. The factor scores helped the boosted model only a little: adding them to the raw fields moved its AUC by +0.0027, just above the 0.002 needed to adopt them (a difference this small is within the noise of two folds of 4,000 and 5,000 loans), so the boosted challenger uses them. They did not help the scorecard (WoE columns alone 0.6987, with factor scores 0.6952), so the selected model does not use them. Eight factor scores alone reach 0.6967 against 0.6970 for the raw inputs: a compact stand-in, not an improvement. The PCA components were not used. What the data scientist would act on next: build factors from fields not yet represented (`dti`, delinquencies, inquiries have the lowest communality) and test fewer, better-defined factors; the monotone constraints raised the boosted model's time-fold AUC (0.697 without, 0.702 with) and are kept.

## Monitoring and drift
The 2016 holdout (5,000 loans, scored in `outputs/holdout_scores.csv`) is stable against 2015: PSI of the PD distribution 0.007, the ten most important inputs at most 0.024
(alert levels: 0.10 investigate, 0.25 act). Decision mix: approve 14.7%, review 28.0%, decline 57.3%.
Outcomes by region and income band are monitored, not modelled: approval rate by income band runs from 6.3% to 29.1% (it follows income, as the credit signals do), and by region from
15.4% to 17.6%; observed default rates by region sit within 18.9% to 21.2%.

## Known limits
- **Accepted loans only.** Applicants the platform declined are missing, nobody in the data has a FICO score below 660, and the model says nothing reliable about lower scores. Relationships are probably weaker here than in the full applicant pool.
- **In-progress loans were removed**, which raises the default rate of recent 60-month loans. 60-month results are therefore reported separately and calibrated separately.
- **Default rates rose** from 2013 to 2015 and the model, trained on 2013 to 2014, under-predicts 2015 risk in places: 15 of 45 segments tested differ significantly from their predicted rate (all under-predict, for example grades B, C, D by 2 to 3 points).
- **The information ceiling is about 0.70 to 0.73 AUC** on the fields of this data set alone: the platform's grade, built from the same kind of data, reaches the same level, and the sub-grade lifts the selected model to 0.73.
- **Dependence on the platform's grading.** The selected model uses the platform's sub-grade, the output of the platform's own risk model. If the platform changes how it grades, the sub-grade's meaning shifts and the model's scores shift with it. The mix of sub-grades is monitored (PSI 2016 against 2015: 0.017) together with the observed default rate by grade band, and the independent scorecard (no platform score) is kept as a fallback, switchable with `platform_score: none` in `config/train.yaml`. A reason code can also cite the sub-grade itself; that explains a decision only by pointing at the platform's own rating, which says less than a specific factor such as the credit score, so the reasons list the other factors after it.
- **WoE inside the contract was fitted on all pre-2015 labels**, so tuning inside 2013 to 2014 is slightly optimistic (about 0.01 AUC); model selection used WoE refitted inside each fold. Settings (regularisation, number of inputs) were also chosen on the tuning folds, so the tuning-fold AUCs (0.6983, 0.7259 for the selected model) are slightly optimistic; 2015 played no part in any choice and its results above are clean.
- **Location and job title are not used.** Occupation and place can stand in for characteristics that are not in the data; the data cannot test for it. State is used only for monitoring.
- **Single-application scoring:** about 35 ms at the median and 36 ms at the 95th percentile on a workstation (limit 100 ms).
- **Application date.** Credit history length is measured from the application date; the demo defaults to mid-2015, inside the period the model was validated on.

## Lineage
Git commit `4fc304c1456200562969d268c936e251f7f3676d` (working tree had uncommitted changes); training-data hash `4ff160121523e193...`; contract version 1.0.0; calibration by term (36 months: platt, 60 months: platt);
experiments in `reports/experiments.csv`. A model whose contract version differs from the transform's is refused at load time.
