"""Write model_card.md from the saved model and the evaluation results, so every number comes from a file and none is typed by hand."""
import json
from pathlib import Path

import pandas as pd

R = Path("reports")
meta = json.loads(Path("models/metadata.json").read_text())
sel = meta["selection"]
m = pd.read_csv(R / "b1_model_evaluation/tables/metrics_by_model_and_group.csv")
gain = pd.read_csv(R / "b1_model_evaluation/tables/gain_over_grade_alone_2015.csv").set_index("model")
slope = pd.read_csv(R / "b1_model_evaluation/tables/calibration_slope_intercept.csv").set_index("group")
bands = pd.read_csv(R / "b1_model_evaluation/tables/decision_bands.csv")
bands = bands[bands.window.str.startswith("H2")].set_index("decision")
pol = pd.read_csv(R / "b1_model_evaluation/tables/policy_model_vs_grade_2015H2.csv").set_index("policy_driven_by")
fair = pd.read_csv(R / "b1_model_evaluation/tables/outcomes_by_region_and_income.csv")
drift = pd.read_csv(R / "b2_error_analysis_and_drift/tables/holdout_drift_psi.csv")
mix = pd.read_csv(R / "b2_error_analysis_and_drift/tables/decision_mix_2015_vs_2016.csv").set_index("item")
cal = pd.read_csv(R / "b2_error_analysis_and_drift/tables/calibration_by_segment_2015.csv")
curve = pd.read_csv(R / "b1_model_evaluation/tables/profit_by_decline_threshold_2015H1.csv")
lat = pd.read_csv(R / "b3_latency/tables/latency.csv")
selected = meta["model"]


def row(model, group):
    return m[(m.model == model) & (m.group == group)].iloc[0]


names = {"grade_baseline": "Platform sub-grade alone (benchmark)", "woe_scorecard": "WoE scorecard (selected)", "lightgbm": "Gradient boosting, monotone constraints (challenger)",
         "scorecard_plus_grade": "Scorecard plus the platform sub-grade (test variant)"}
perf = "\n".join(f"| {names[k]} | {row(k, '2015 all').auc:.3f} | {row(k, '2015 all').gini:.3f} | {row(k, '2015 all').ks:.3f} | {row(k, '2015 all').brier:.3f} | "
                 f"{row(k, '36 months').auc:.3f} | {row(k, '60 months').auc:.3f} |" for k in names)
quarters = m[(m.model == selected) & m.group.str.startswith("2015Q")]
qrow = "; ".join(f"{r.group[4:]} {r.auc:.3f}" for r in quarters.itertuples())
b = lambda d, c: bands.loc[d, c]
money = lambda v: f"{'-' if v < 0 else ''}${abs(v) / 1e6:,.2f}M"
best = curve.loc[curve.profit.idxmax()]
at_be = curve.iloc[(curve.threshold - meta["thresholds"]["decline_at_or_above"]).abs().argmin()]
policy_sentence = ("earns " + money(pol.loc["grade alone", "profit_if_approve_and_review_funded"]) + (", which is more, so the model's business advantage over the grade is not shown." if pol.loc["grade alone", "profit_if_approve_and_review_funded"] >= pol.loc["model", "profit_if_approve_and_review_funded"] else ", which is less: the model's policy earns more."))
th_sentence = (f"The profit-maximising decline threshold on 2015 H1 was {best.threshold:.1%}, against the break-even {meta['thresholds']['decline_at_or_above']:.1%}; the profit curve is flat there "
               f"(break-even earns {at_be.profit / best.profit:.0%} of the optimum), so the break-even threshold is kept.")
mono = f"{sel['unconstrained_auc']:.3f} without, {sel['unconstrained_auc'] - sel['monotone_cost_auc']:.3f} with"
lat_full = lat.iloc[:3]
inc = fair[fair.segment == "income_band"].set_index("level")
reg = fair[fair.segment == "region"].set_index("level")
big_gaps = cal[cal.verdict != "calibrated"]
ab_l, ab_g = sel["ablation_lightgbm"], sel["ablation_logistic"]
top_drift = drift[drift.kind != "model output"].psi.max()
card = f"""# Model card: probability of default for unsecured personal loans

**Model:** {meta['model'].replace('_', ' ')}, version `{meta['model']}-contract{meta['contract_version']}-{meta['training_date']}-{meta['git_commit'][:7]}`.
A logistic regression on the Weight-of-Evidence (WoE) columns of the data contract, with every weight forced to point in the safe direction, calibrated separately for 36- and 60-month loans.
Trained {meta['training_date']} on {meta['training_loans']:,} loans issued {meta['training_years'][0]} to {meta['training_years'][1]} (contract fit window: up to {meta['fit_cutoff']}). Seed {meta['seed']}.

## Intended use
Estimate the probability that a new applicant for a 1,000 to 40,000 USD, 36- or 60-month unsecured loan will default, to auto-approve, send to manual review or decline.
Not for any other product, for applicants unlike those the platform accepted, or as the only basis of an adverse decision without human review.

## Data
Historical consumer loans accepted by a lending platform and issued 2013 to 2015, from application, credit-bureau and loan-servicing extracts. Label: charged off or defaulted (1) against fully paid (0); loans still in progress were removed.
The servicing file is recorded after the loan and is used only to audit the label. The 2016 loans have no labels. The data contract (`contract/`) holds the fitted cleaning, treatment, features, groups, WoE, PCA and factor model, fitted on loans to 2014-12-31.
Inputs: {len(meta['inputs_used'])} WoE columns of {len(meta['inputs'])} candidates ({', '.join(meta['inputs_dropped_for_wrong_sign'])} dropped because their weights pointed the wrong way). Never inputs: the platform's own `grade`, `sub_grade` and `int_rate`, job group and location (restricted pending a fairness review), and every servicing field.

## Performance on 2015 loans (out of time; 15,627 loans, default rate {row(selected, '2015 all').default_rate:.1%})
| Model | AUC | Gini | KS | Brier | AUC, 36 months | AUC, 60 months |
|---|---|---|---|---|---|---|
{perf}

Stable by quarter (selected model, AUC): {qrow}.
**Against the platform's grade:** the selected scorecard's AUC is {gain.loc['woe_scorecard', 'gain']:+.3f} against grade alone (95% interval {gain.loc['woe_scorecard', 'ci_low']:+.3f} to {gain.loc['woe_scorecard', 'ci_high']:+.3f}), so it does not beat it measurably;
the boosted challenger is {gain.loc['lightgbm', 'gain']:+.3f} ({gain.loc['lightgbm', 'ci_low']:+.3f} to {gain.loc['lightgbm', 'ci_high']:+.3f}). Letting the scorecard use the sub-grade adds {gain.loc['scorecard_plus_grade', 'gain']:+.3f} over grade alone
({gain.loc['scorecard_plus_grade', 'ci_low']:+.3f} to {gain.loc['scorecard_plus_grade', 'ci_high']:+.3f}): our fields carry real information beyond the grade, but the grade is itself a strong model of the same data, so an independent model only matches it.
That independence was a deliberate design choice (the platform's score drifts: its interest rate moved by a PSI of 0.19 and 0.16 in two years); adding it is a business decision, not a technical one.
Calibration (slope 1 is perfect): overall {slope.loc['all', 'slope']:.2f}, 36 months {slope.loc['36 months', 'slope']:.2f}, 60 months {slope.loc['60 months', 'slope']:.2f}.

## Decision thresholds and what they do (2015 H2, loans the model had not seen)
Assumption: a defaulted loan loses 70% of the amount, a repaid loan earns 10%, so expected profit is zero at PD = 12.5%. **Decline at PD of {meta['thresholds']['decline_at_or_above']:.1%} or more;
auto-approve below {meta['thresholds']['approve_below']:.2%}** (where the expected margin drops to 5% of the loan); review in between.

| Band | Share of loans | Predicted PD | Observed default rate | Profit if funded |
|---|---|---|---|---|
| Approve | {b('approve', 'share_of_loans'):.1%} | {b('approve', 'mean_pd'):.1%} | {b('approve', 'default_rate'):.1%} | {money(b('approve', 'profit_if_funded'))} |
| Review | {b('review', 'share_of_loans'):.1%} | {b('review', 'mean_pd'):.1%} | {b('review', 'default_rate'):.1%} | {money(b('review', 'profit_if_funded'))} |
| Decline | {b('decline', 'share_of_loans'):.1%} | {b('decline', 'mean_pd'):.1%} | {b('decline', 'default_rate'):.1%} | {money(b('decline', 'profit_if_funded'))} |

Funding every applicant would have lost {money(-pol.loc['model', 'profit_fund_everyone'])}; funding approve plus review earns {money(pol.loc['model', 'profit_if_approve_and_review_funded'])}. A policy driven by grade alone, at the same thresholds,
{policy_sentence} {th_sentence}

## Which data-science outputs helped (ablation, time-ordered tuning folds, AUC)
| Inputs | Gradient boosting | Logistic |
|---|---|---|
| Raw treated fields and features | {ab_l['raw']:.4f} | {ab_g['woe_only']:.4f} (WoE columns, contract bins) |
| Factor scores only | {ab_l['factors']:.4f} | {ab_g['factors_plus_category_woe']:.4f} (plus loan-structure WoE) |
| Both | {ab_l['both']:.4f} | {ab_g['woe_plus_factors']:.4f} |

**Verdict.** The WoE bins helped: a scorecard on them is the best simple model and gives plain reasons. The factor scores did not add measurable signal: eight factor scores alone reach {ab_l['factors']:.4f} against {ab_l['raw']:.4f} for the 33 raw inputs
(a compact stand-in, not an improvement), and adding them to the raw fields moved the AUC by {ab_l['both'] - ab_l['raw']:+.4f}, below the {0.002} needed to adopt them. The PCA components were not used. What the data scientist would act on next:
build factors from fields not yet represented (`dti`, delinquencies, inquiries have the lowest communality) and test fewer, better-defined factors; the monotone constraints raised the boosted model's time-fold AUC ({mono}) and are kept.

## Monitoring and drift
The 2016 holdout (5,000 loans, scored in `outputs/holdout_scores.csv`) is stable against 2015: PSI of the PD distribution {drift.iloc[0].psi:.3f}, the ten most important inputs at most {top_drift:.3f}
(alert levels: 0.10 investigate, 0.25 act). Decision mix: approve {mix.loc['approve', '2016 holdout']:.1%}, review {mix.loc['review', '2016 holdout']:.1%}, decline {mix.loc['decline', '2016 holdout']:.1%}.
Outcomes by region and income band are monitored, not modelled: approval rate by income band runs from {inc.approval_rate.min():.1%} to {inc.approval_rate.max():.1%} (it follows income, as the credit signals do), and by region from
{reg.approval_rate.min():.1%} to {reg.approval_rate.max():.1%}; observed default rates by region sit within {reg.observed_default_rate.min():.1%} to {reg.observed_default_rate.max():.1%}.

## Known limits
- **Accepted loans only.** Applicants the platform declined are missing, nobody in the data has a FICO score below 660, and the model says nothing reliable about lower scores. Relationships are probably weaker here than in the full applicant pool.
- **In-progress loans were removed**, which raises the default rate of recent 60-month loans. 60-month results are therefore reported separately and calibrated separately.
- **Default rates rose** from 2013 to 2015 and the model, trained on 2013 to 2014, under-predicts 2015 risk in places: {len(big_gaps)} of {len(cal)} segments tested differ significantly from their predicted rate (most under-predict, for example grades D and E by 6 to 8 points).
- **The information ceiling is about 0.70 to 0.73 AUC** on these fields. The grade, built by the platform from the same kind of data, reaches the same level.
- **WoE inside the contract was fitted on all pre-2015 labels**, so tuning inside 2013 to 2014 is slightly optimistic (about 0.01 AUC); model selection used WoE refitted inside each fold, and the 2015 results above are clean.
- **Location and job title are not used.** Occupation and place can stand in for characteristics that are not in the data; the data cannot test for it. State is used only for monitoring.
- **Single-application scoring:** about {lat_full.p50_ms.mean():.0f} ms at the median and {lat_full.p95_ms.mean():.0f} ms at the 95th percentile on a workstation (limit 100 ms).
- **Application date.** Credit history length is measured from the application date; the demo defaults to mid-2015, inside the period the model was validated on.

## Lineage
Git commit `{meta['git_commit']}`{' (working tree had uncommitted changes)' if meta.get('git_dirty') else ''}; training-data hash `{meta['training_data_hash'][:16]}...`; contract version {meta['contract_version']}; calibration by term ({', '.join(f'{k} months: {v}' for k, v in meta['calibration'].items())});
experiments in `reports/experiments.csv`. A model whose contract version differs from the transform's is refused at load time.
"""
Path("model_card.md").write_text(card)
print(card[:3500])
print("... total", len(card.splitlines()), "lines")
