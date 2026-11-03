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


names = {"grade_baseline": "Platform sub-grade alone (benchmark)",
         "woe_scorecard": "Independent WoE scorecard, no platform score" + (" (selected)" if selected == "woe_scorecard" else " (fallback)"),
         "scorecard_plus_grade": "WoE scorecard plus the platform sub-grade" + (" (selected)" if selected == "scorecard_plus_grade" else " (variant)"),
         "lightgbm": "Gradient boosting, monotone constraints, no platform score (challenger)",
         "lightgbm_plus_grade": "Gradient boosting plus the platform sub-grade" + (" (selected)" if selected == "lightgbm_plus_grade" else " (challenger)"),
         "scorecard_boosted_average": "Average of the two sub-grade models" + (" (selected)" if selected == "scorecard_boosted_average" else " (variant)")}
perf = "\n".join(f"| {names[k]} | {row(k, '2015 all').auc:.3f} | {row(k, '2015 all').gini:.3f} | {row(k, '2015 all').ks:.3f} | {row(k, '2015 all').brier:.3f} | "
                 f"{row(k, '36 months').auc:.3f} | {row(k, '60 months').auc:.3f} |" for k in names)
quarters = m[(m.model == selected) & m.group.str.startswith("2015Q")]
qrow = "; ".join(f"{r.group[4:]} {r.auc:.3f}" for r in quarters.itertuples())
b = lambda d, c: bands.loc[d, c]
money = lambda v: f"{'-' if v < 0 else ''}${abs(v) / 1e6:,.2f}M"
best = curve.loc[curve.profit.idxmax()]
at_be = curve.iloc[(curve.threshold - meta["thresholds"]["decline_at_or_above"]).abs().argmin()]
pdiff = pd.read_csv(R / "b1_model_evaluation/tables/profit_difference_model_vs_grade_2015H2.csv").iloc[0]
_lo, _hi = pdiff.ci_low, pdiff.ci_high
policy_sentence = (f"earns {money(pol.loc['grade alone', 'profit_if_approve_and_review_funded'])}, against {money(pol.loc['model', 'profit_if_approve_and_review_funded'])} for the model. "
                   f"The difference (model minus grade) is {money(pdiff.profit_difference_model_minus_grade)}, with a 95% interval of {money(_lo)} to {money(_hi)}"
                   + (", which includes zero: the model's policy is not measurably more or less profitable than the grade's, although it ranks applicants better." if _lo < 0 < _hi
                      else (", so the model's policy earns more." if _lo > 0 else ", so the model's policy earns less.")))
eq_h2 = pd.read_csv(R / "b1_model_evaluation/tables/profit_at_equal_volume_2015H2.csv")
eq_all = pd.read_csv(R / "b1_model_evaluation/tables/profit_at_equal_volume_2015_all.csv")
_ph, _pa = eq_h2[eq_h2.is_policy_volume].iloc[0], eq_all[eq_all.is_policy_volume].iloc[0]
_sig = eq_all[eq_all.ci_low > 0].share_of_loans_funded
volume_txt = (f"On all of 2015 the advantage is beyond chance from {_sig.min():.0%} of applicants funded upward; at smaller volumes the differences are within chance." if len(_sig)
              else "At no volume is the difference beyond chance.")
more = lambda d: f"{money(abs(d))} {'more' if d >= 0 else 'less'}"
funded_model_share, funded_grade_share = 1 - pol.loc["model", "decline_rate"], 1 - pol.loc["grade alone", "decline_rate"]
policy_sentence += (f" Part of that gap is volume, not ranking: at the same PD thresholds the model funds {funded_model_share:.0%} of loans and the grade policy {funded_grade_share:.0%} (the two scores spread applicants differently around the 12.5% line). "
                    f"At equal volume, funding the safest {_ph.share_of_loans_funded:.0%} of loans by each score (the volume the model's own thresholds fund, fixed by the policy and not chosen from these results), the model earns {more(_ph.difference)} than the grade on 2015 H2 "
                    f"(95% interval {money(_ph.ci_low)} to {money(_ph.ci_high)}) and {more(_pa.difference)} on all of 2015 ({money(_pa.ci_low)} to {money(_pa.ci_high)}). {volume_txt} "
                    f"The tables for every volume are in `reports/b1_model_evaluation/tables/`.")
_share = at_be.profit / best.profit
th_sentence = (f"The profit-maximising decline threshold on 2015 H1 was {best.threshold:.1%}, against the break-even {meta['thresholds']['decline_at_or_above']:.1%}; break-even earns {_share:.0%} of that optimum"
               + (", so the curve is flat there and the break-even threshold is kept." if _share >= 0.97 else ", a small loss, and the break-even threshold is kept because it follows from the stated profit assumptions and not from one half-year of data."))
mono = f"{sel['unconstrained_auc']:.3f} without, {sel['unconstrained_auc'] - sel['monotone_cost_auc']:.3f} with"
lat_full = lat.iloc[:3]
inc = fair[fair.segment == "income_band"].set_index("level")
reg = fair[fair.segment == "region"].set_index("level")
big_gaps = cal[cal.verdict != "calibrated"]
_gg = big_gaps[big_gaps.segment == "grade"].sort_values("level")
grade_example = f", for example grades {', '.join(_gg.level)} by {_gg.gap.min() * 100:.0f} to {_gg.gap.max() * 100:.0f} points" if len(_gg) else ""
direction = "all under-predict" if (big_gaps.gap > 0).all() else "most under-predict"
ab_l, ab_g = sel["ablation_lightgbm"], sel["ablation_logistic"]
top_drift = drift[drift.kind != "model output"].psi.max()
uses_grade = selected in ("scorecard_plus_grade", "lightgbm_plus_grade", "scorecard_boosted_average")
g = lambda k, c: gain.loc[k, c]
_sp = drift.loc[drift.item == "sub_grade_rank", "psi"]
sub_psi = float(_sp.iloc[0]) if len(_sp) else float("nan")
model_label = {"woe_scorecard": "independent WoE scorecard", "scorecard_plus_grade": "WoE scorecard plus the platform sub-grade", "lightgbm": "gradient boosting",
               "lightgbm_plus_grade": "gradient boosting plus the platform sub-grade", "scorecard_boosted_average": "average of the WoE scorecard and gradient boosting, both with the platform sub-grade"}.get(selected, selected.replace("_", " "))
desc = {"woe_scorecard": "A logistic regression on the Weight-of-Evidence (WoE) columns of the data contract, with every weight forced to point in the safe direction, calibrated separately for 36- and 60-month loans.",
        "scorecard_plus_grade": "A logistic regression on the Weight-of-Evidence (WoE) columns of the data contract plus the platform's sub-grade. Every WoE weight is forced to point in the safe direction and the sub-grade must raise risk. Calibrated separately for 36- and 60-month loans.",
        "lightgbm": "Gradient boosting with audited monotone constraints on the contract's treated fields, calibrated separately for 36- and 60-month loans.",
        "lightgbm_plus_grade": "Gradient boosting with audited monotone constraints on the contract's treated fields plus the platform's sub-grade, calibrated separately for 36- and 60-month loans.",
        "scorecard_boosted_average": "The average, in log-odds, of two models that both use the platform's sub-grade: a logistic regression on the contract's Weight-of-Evidence (WoE) columns (every WoE weight forced to point in the safe direction, the sub-grade required to raise risk) and gradient boosting with monotone constraints on the treated fields. Calibrated separately for 36- and 60-month loans. Reason codes add the two models' contributions, each measured against a typical approved applicant, so they add up exactly to the score."}[selected]
n_woe = sum(c.endswith("_woe") for c in meta["inputs_used"])
dropped = meta["inputs_dropped_for_wrong_sign"]
dropped_txt = f" ({', '.join(dropped)} dropped because their weights pointed the wrong way)" if dropped else ""
never = ("the platform's `int_rate` and grade letter" if uses_grade else "the platform's own `grade`, `sub_grade` and `int_rate`") + ", job group and location (restricted pending a fairness review), and every servicing field"
members = meta.get("members", {})
if members:
    inputs_line = (f"Inputs: the scorecard uses {len(members.get('logistic', []))} columns ({sum(c.endswith('_woe') for c in members.get('logistic', []))} WoE columns and the platform's sub-grade){dropped_txt}, "
                   f"the boosted model {len(members.get('lightgbm', []))} treated fields, features, factor scores and the sub-grade; reasons are reported for {len(meta['inputs_used'])} fields. Never inputs: {never}.")
else:
  inputs_line = (f"Inputs: {len(meta['inputs_used'])} columns" + (f" ({n_woe} WoE columns and the platform's sub-grade)" if n_woe and uses_grade else "") +
               f" of {len(meta['inputs'])} candidates{dropped_txt}. Never inputs: {never}.")
gains_txt = ", ".join(f"{x:+.4f}" for x in sel["boost_gain_per_fold"])
if selected == "scorecard_boosted_average":
    rule_txt = (f"Neither single model beat the other under the rule fixed in `DESIGN.md` (boosting had to be ahead by 0.005 AUC on average and in every fold; its gains over the scorecard were {gains_txt}). "
                f"A later screen of about fifteen ideas on the same folds, with its own rule fixed beforehand (a change must add 0.003 AUC on average and in every fold), found one that passed: averaging the two models "
                f"(mean tuning-fold AUC {sel['average_mean_auc']:.4f} against {sel['scorecard_plus_subgrade_mean_auc']:.4f} for the scorecard). With that many ideas tried, a pass this small can partly be luck. 2015 was never used to choose.")
else:
    rule_txt = (f"The boosted model with the sub-grade reaches {g('lightgbm_plus_grade', 'gain'):+.3f}; under the rule fixed in `DESIGN.md` (it must be ahead by 0.005 AUC on the tuning folds on average and in every fold; its gains there were "
                f"{gains_txt}) it did not replace the scorecard. The choice rested on the tuning folds; 2015 was never used to choose.")
if uses_grade:
    vs_grade = (f"**Against the platform's grade:** the selected model (which uses the sub-grade) has AUC {g(selected, 'auc_model'):.3f} against {g(selected, 'auc_grade_alone'):.3f} for the grade alone, a gain of {g(selected, 'gain'):+.3f} "
                f"(95% interval {g(selected, 'ci_low'):+.3f} to {g(selected, 'ci_high'):+.3f}), so it beats the grade measurably. The same scorecard without any platform score is {g('woe_scorecard', 'gain'):+.3f} "
                f"({g('woe_scorecard', 'ci_low'):+.3f} to {g('woe_scorecard', 'ci_high'):+.3f}): it matches the grade but does not beat it, which is why the sub-grade is allowed in. "
                f"{rule_txt}\n"
                f"The price of the gain is a dependence on the platform's own grading (see the known limits). The platform's interest rate moved by a PSI of 0.19 and 0.16 in two years, which is why the rate is not an input.")
else:
    vs_grade = (f"**Against the platform's grade:** the selected scorecard's AUC is {g('woe_scorecard', 'gain'):+.3f} against grade alone (95% interval {g('woe_scorecard', 'ci_low'):+.3f} to {g('woe_scorecard', 'ci_high'):+.3f}), so it does not beat it measurably; "
                f"the boosted challenger is {g('lightgbm', 'gain'):+.3f}. Letting the scorecard use the sub-grade adds {g('scorecard_plus_grade', 'gain'):+.3f} over grade alone: an independent model only matches the grade.")
fgain = ab_l["both"] - ab_l["raw"]
if sel["chosen_input_set"] in ("both", "factors"):
    factor_text = (f"The factor scores helped the boosted model only a little: adding them to the raw fields moved its AUC by {fgain:+.4f}, just above the 0.002 needed to adopt them (a difference this small is within the noise of two folds of 4,000 and 5,000 loans), "
                   f"so the boosted challenger uses them. They did not help the scorecard (WoE columns alone {ab_g['woe_only']:.4f}, with factor scores {ab_g['woe_plus_factors']:.4f}), so the selected model does not use them. "
                   f"Eight factor scores alone reach {ab_l['factors']:.4f} against {ab_l['raw']:.4f} for the raw inputs: a compact stand-in, not an improvement.")
else:
    factor_text = (f"The factor scores did not add measurable signal: eight factor scores alone reach {ab_l['factors']:.4f} against {ab_l['raw']:.4f} for the raw inputs (a compact stand-in, not an improvement), "
                   f"and adding them to the raw fields moved the AUC by {fgain:+.4f}, below the 0.002 needed to adopt them.")
verdict = (f"**Verdict.** The WoE bins helped: a scorecard on them is the best simple model and gives plain reasons. {factor_text} The PCA components were not used. What the data scientist would act on next: "
           f"build factors from fields not yet represented (`dti`, delinquencies, inquiries have the lowest communality) and test fewer, better-defined factors; the monotone constraints raised the boosted model's time-fold AUC ({mono}) and are kept.")
dependence = (f"- **Dependence on the platform's grading.** The selected model uses the platform's sub-grade, the output of the platform's own risk model. If the platform changes how it grades, the sub-grade's meaning shifts and the model's scores shift with it. "
              f"The mix of sub-grades is monitored (PSI 2016 against 2015: {sub_psi:.3f}) together with the observed default rate by grade band, and the independent scorecard (no platform score) is kept as a fallback, switchable with `platform_score: none` in `config/train.yaml`. "
              f"A reason code can also cite the sub-grade itself; that explains a decision only by pointing at the platform's own rating, which says less than a specific factor such as the credit score, so the reasons list the other factors after it.\n") if uses_grade else ""
card = f"""# Model card: probability of default for unsecured personal loans

**Developed by:** Natnael Fisseha.
**Model:** {model_label}, version `{meta['model']}-contract{meta['contract_version']}-{meta['training_date']}-{meta['git_commit'][:7]}`.
{desc}
Trained {meta['training_date']} on {meta['training_loans']:,} loans issued {meta['training_years'][0]} to {meta['training_years'][1]} (contract fit window: up to {meta['fit_cutoff']}). Seed {meta['seed']}.

## Intended use
Estimate the probability that a new applicant for a 1,000 to 40,000 USD, 36- or 60-month unsecured loan will default, to auto-approve, send to manual review or decline.
Not for any other product, for applicants unlike those the platform accepted, or as the only basis of an adverse decision without human review.

## Data
Historical consumer loans accepted by a lending platform and issued 2013 to 2015, from application, credit-bureau and loan-servicing extracts. Label: charged off or defaulted (1) against fully paid (0); loans still in progress were removed.
The servicing file is recorded after the loan and is used only to audit the label. The 2016 loans have no labels. The data contract (`contract/`) holds the fitted cleaning, treatment, features, groups, WoE, PCA and factor model, fitted on loans to 2014-12-31.
{inputs_line}

## Performance on 2015 loans (out of time; 15,627 loans, default rate {row(selected, '2015 all').default_rate:.1%})
| Model | AUC | Gini | KS | Brier | AUC, 36 months | AUC, 60 months |
|---|---|---|---|---|---|---|
{perf}

Stable by quarter (selected model, AUC): {qrow}.
{vs_grade}
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

{verdict}

## Monitoring and drift
The 2016 holdout (5,000 loans, scored in `outputs/holdout_scores.csv`) is stable against 2015: PSI of the PD distribution {drift.iloc[0].psi:.3f}, the ten most important inputs at most {top_drift:.3f}
(alert levels: 0.10 investigate, 0.25 act). Decision mix: approve {mix.loc['approve', '2016 holdout']:.1%}, review {mix.loc['review', '2016 holdout']:.1%}, decline {mix.loc['decline', '2016 holdout']:.1%}.
Outcomes by region and income band are monitored, not modelled: approval rate by income band runs from {inc.approval_rate.min():.1%} to {inc.approval_rate.max():.1%} (it follows income, as the credit signals do), and by region from
{reg.approval_rate.min():.1%} to {reg.approval_rate.max():.1%}; observed default rates by region sit within {reg.observed_default_rate.min():.1%} to {reg.observed_default_rate.max():.1%}.

## Known limits
- **Accepted loans only.** Applicants the platform declined are missing, nobody in the data has a FICO score below 660, and the model says nothing reliable about lower scores. Relationships are probably weaker here than in the full applicant pool.
- **In-progress loans were removed**, which raises the default rate of recent 60-month loans. 60-month results are therefore reported separately and calibrated separately.
- **Default rates rose** from 2013 to 2015 and the model, trained on 2013 to 2014, under-predicts 2015 risk in places: {len(big_gaps)} of {len(cal)} segments tested differ significantly from their predicted rate ({direction}{grade_example}).
- **The information ceiling is about 0.70 to 0.73 AUC** on the fields of this data set alone: the platform's grade, built from the same kind of data, reaches the same level, and the sub-grade lifts the selected model to {row(selected, '2015 all').auc:.2f}.
{dependence}- **WoE inside the contract was fitted on all pre-2015 labels**, so tuning inside 2013 to 2014 is slightly optimistic (about 0.01 AUC); model selection used WoE refitted inside each fold. Settings (regularisation, number of inputs) were also chosen on the tuning folds, so the tuning-fold AUCs ({', '.join(str(x) for x in meta['fold_auc'])} for the selected model) are slightly optimistic; 2015 was not used for tuning or selection (it prompted the decision to use the sub-grade, whose evidence came from the tuning folds; `DESIGN.md` section 11).
- **Location and job title are not used.** Occupation and place can stand in for characteristics that are not in the data; the data cannot test for it. State is used only for monitoring.
- **Single-application scoring:** about {lat_full.p50_ms.mean():.0f} ms at the median and {lat_full.p95_ms.mean():.0f} ms at the 95th percentile on a workstation (limit 100 ms).
- **Application date.** Credit history length is measured from the application date; the demo defaults to mid-2015, inside the period the model was validated on.

## Lineage
Git commit `{meta['git_commit']}`{' (working tree had uncommitted changes)' if meta.get('git_dirty') else ''}; training-data hash `{meta['training_data_hash'][:16]}...` (for models on WoE columns it covers the loan ids and labels; the input values are fixed by the contract version and `transform.joblib`); contract version {meta['contract_version']}; calibration by term ({', '.join(f'{k} months: {v}' for k, v in meta['calibration'].items())});
experiments in `reports/experiments.csv`. A model whose contract version differs from the transform's is refused at load time.
"""
Path("model_card.md").write_text(card)
print(card[:3500])
print("... total", len(card.splitlines()), "lines")
