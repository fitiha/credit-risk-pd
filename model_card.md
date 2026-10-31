# Model card: probability of default for unsecured personal loans

**Model:** woe scorecard, version `woe_scorecard-contract1.0.0-2026-10-31-2826c2d`.
A logistic regression on the Weight-of-Evidence (WoE) columns of the data contract, with every weight forced to point in the safe direction, calibrated separately for 36- and 60-month loans.
Trained 2026-10-31 on 14,373 loans issued 2013-01-01 to 2014-12-01 (contract fit window: up to 2014-12-31). Seed 42.

## Intended use
Estimate the probability that a new applicant for a 1,000 to 40,000 USD, 36- or 60-month unsecured loan will default, to auto-approve, send to manual review or decline.
Not for any other product, for applicants unlike those the platform accepted, or as the only basis of an adverse decision without human review.

## Data
Historical consumer loans accepted by a lending platform and issued 2013 to 2015, from application, credit-bureau and loan-servicing extracts. Label: charged off or defaulted (1) against fully paid (0); loans still in progress were removed.
The servicing file is recorded after the loan and is used only to audit the label. The 2016 loans have no labels. The data contract (`contract/`) holds the fitted cleaning, treatment, features, groups, WoE, PCA and factor model, fitted on loans to 2014-12-31.
Inputs: 29 WoE columns of 33 candidates (revol_bal_log_woe, mort_acc_capped_woe, pub_rec_capped_woe, pub_rec_bankruptcies_capped_woe dropped because their weights pointed the wrong way). Never inputs: the platform's own `grade`, `sub_grade` and `int_rate`, job group and location (restricted pending a fairness review), and every servicing field.

## Performance on 2015 loans (out of time; 15,627 loans, default rate 20.1%)
| Model | AUC | Gini | KS | Brier | AUC, 36 months | AUC, 60 months |
|---|---|---|---|---|---|---|
| Platform sub-grade alone (benchmark) | 0.722 | 0.444 | 0.322 | 0.145 | 0.679 | 0.644 |
| WoE scorecard (selected) | 0.719 | 0.437 | 0.323 | 0.144 | 0.668 | 0.658 |
| Gradient boosting, monotone constraints (challenger) | 0.725 | 0.449 | 0.334 | 0.143 | 0.678 | 0.667 |
| Scorecard plus the platform sub-grade (test variant) | 0.731 | 0.462 | 0.339 | 0.142 | 0.689 | 0.667 |

Stable by quarter (selected model, AUC): Q1 0.700; Q2 0.725; Q3 0.725; Q4 0.721.
**Against the platform's grade:** the selected scorecard's AUC is -0.003 against grade alone (95% interval -0.010 to +0.004), so it does not beat it measurably;
the boosted challenger is +0.003 (-0.003 to +0.010). Letting the scorecard use the sub-grade adds +0.009 over grade alone
(+0.004 to +0.015): our fields carry real information beyond the grade, but the grade is itself a strong model of the same data, so an independent model only matches it.
That independence was a deliberate design choice (the platform's score drifts: its interest rate moved by a PSI of 0.19 and 0.16 in two years); adding it is a business decision, not a technical one.
Calibration (slope 1 is perfect): overall 1.00, 36 months 0.96, 60 months 1.02.

## Decision thresholds and what they do (2015 H2, loans the model had not seen)
Assumption: a defaulted loan loses 70% of the amount, a repaid loan earns 10%, so expected profit is zero at PD = 12.5%. **Decline at PD of 12.5% or more;
auto-approve below 6.25%** (where the expected margin drops to 5% of the loan); review in between.

| Band | Share of loans | Predicted PD | Observed default rate | Profit if funded |
|---|---|---|---|---|
| Approve | 12.5% | 4.4% | 4.2% | $1.06M |
| Review | 27.4% | 9.3% | 11.3% | $0.10M |
| Decline | 60.1% | 25.1% | 27.4% | -$11.19M |

Funding every applicant would have lost $10.03M; funding approve plus review earns $1.16M. A policy driven by grade alone, at the same thresholds,
earns $1.69M, which is more, so the model's business advantage over the grade is not shown. The profit-maximising decline threshold on 2015 H1 was 12.5%, against the break-even 12.5%; the profit curve is flat there (break-even earns 100% of the optimum), so the break-even threshold is kept.

## Which data-science outputs helped (ablation, time-ordered tuning folds, AUC)
| Inputs | Gradient boosting | Logistic |
|---|---|---|
| Raw treated fields and features | 0.6981 | 0.6986 (WoE columns, contract bins) |
| Factor scores only | 0.6967 | 0.6975 (plus loan-structure WoE) |
| Both | 0.6995 | 0.6955 |

**Verdict.** The WoE bins helped: a scorecard on them is the best simple model and gives plain reasons. The factor scores did not add measurable signal: eight factor scores alone reach 0.6967 against 0.6981 for the 33 raw inputs
(a compact stand-in, not an improvement), and adding them to the raw fields moved the AUC by +0.0014, below the 0.002 needed to adopt them. The PCA components were not used. What the data scientist would act on next:
build factors from fields not yet represented (`dti`, delinquencies, inquiries have the lowest communality) and test fewer, better-defined factors; the monotone constraints raised the boosted model's time-fold AUC (0.690 without, 0.699 with) and are kept.

## Monitoring and drift
The 2016 holdout (5,000 loans, scored in `outputs/holdout_scores.csv`) is stable against 2015: PSI of the PD distribution 0.015, the ten most important inputs at most 0.109
(alert levels: 0.10 investigate, 0.25 act). Decision mix: approve 11.7%, review 28.4%, decline 59.9%.
Outcomes by region and income band are monitored, not modelled: approval rate by income band runs from 4.1% to 23.8% (it follows income, as the credit signals do), and by region from
11.6% to 13.5%; observed default rates by region sit within 18.9% to 21.2%.

## Known limits
- **Accepted loans only.** Applicants the platform declined are missing, nobody in the data has a FICO score below 660, and the model says nothing reliable about lower scores. Relationships are probably weaker here than in the full applicant pool.
- **In-progress loans were removed**, which raises the default rate of recent 60-month loans. 60-month results are therefore reported separately and calibrated separately.
- **Default rates rose** from 2013 to 2015 and the model, trained on 2013 to 2014, under-predicts 2015 risk in places: 17 of 45 segments tested differ significantly from their predicted rate (most under-predict, for example grades D and E by 6 to 8 points).
- **The information ceiling is about 0.70 to 0.73 AUC** on these fields. The grade, built by the platform from the same kind of data, reaches the same level.
- **WoE inside the contract was fitted on all pre-2015 labels**, so tuning inside 2013 to 2014 is slightly optimistic (about 0.01 AUC); model selection used WoE refitted inside each fold, and the 2015 results above are clean.
- **Location and job title are not used.** Occupation and place can stand in for characteristics that are not in the data; the data cannot test for it. State is used only for monitoring.
- **Single-application scoring:** about 32 ms at the median and 35 ms at the 95th percentile on a workstation (limit 100 ms).
- **Application date.** Credit history length is measured from the application date; the demo defaults to mid-2015, inside the period the model was validated on.

## Lineage
Git commit `2826c2d37daebec6bc5ed0f38fac1ee9dc8aa195` (working tree had uncommitted changes); training-data hash `4ff160121523e193...`; contract version 1.0.0; calibration by term (36 months: platt, 60 months: platt);
experiments in `reports/experiments.csv`. A model whose contract version differs from the transform's is refused at load time.
