# A2. Target analysis: `defaulted`

Data: 30,000 unique loans issued Jan 2013 - Dec 2015, after cleaning (see `issue_log.md`).
Reproduce: `uv run python scripts/target_analysis.py` (output in `reports/target_analysis/`).

## Class balance
5,617 defaults out of 30,000 loans: **18.7%**, about 1 default for every 4.3 repaid loans.
The classes are imbalanced but not extreme, so no resampling is planned (see DESIGN.md).

## By loan term and by year (95% confidence intervals)
Monthly rates are in `reports/target_analysis/tables/default_by_month.csv` (36 months, 263-1,794 loans each, so monthly intervals are wide: up to 8 points). The conclusions are drawn from half-years, where intervals are narrow.

| Group | Loans | Default rate | 95% CI |
|---|---|---|---|
| 36 months | 22,328 | 14.0% | 13.6 - 14.5 |
| 60 months | 7,672 | 32.3% | 31.3 - 33.4 |
| 2013 | 5,349 | 15.9% | 15.0 - 17.0 |
| 2014 | 9,024 | 18.0% | 17.2 - 18.8 |
| 2015 | 15,627 | 20.1% | 19.5 - 20.7 |

60-month loans default more than twice as often as 36-month loans. The share of 60-month loans is stable over time (26%, 27%, 25%), so the rise by year is not a change in loan mix.

## The in-progress bias
The label was built in March 2019 from loans that had ended. Loans still being repaid were removed.

| Issue half-year | 36m: share finished by Mar 2019 | 36m default | 60m: share finished by Mar 2019 | 60m default |
|---|---|---|---|---|
| 2013 H1 | 100% | 13.2% | 100% | 24.6% |
| 2013 H2 | 100% | 12.4% | 100% | 25.5% |
| 2014 H1 | 100% | 12.8% | 48% | 26.8% |
| 2014 H2 | 100% | 14.1% | 0% | 32.6% |
| 2015 H1 | 100% | 15.1% | 0% | 34.5% |
| 2015 H2 | 100% | 14.5% | 0% | 37.9% |

- **36-month loans are safe.** Every one had reached its scheduled end, so each is a full outcome. Their default rate moves only from 12-13% to 14-15%.
- **60-month loans from mid-2014 on are likely biased upward.** None had reached their scheduled end. Only loans that *ended before March 2019* were kept; loans still being paid were removed. Of the 5,183 60-month loans issued since July 2014, 65% were kept because they were **repaid early** and 35% because they **defaulted**. Defaults tend to come sooner (median 18 months after issue vs 23 for early repayments). The loans removed are the ones still being paid in March 2019, and most of those would eventually have been repaid in full. Taking them out leaves a sample that has too few "paid" loans, which pushes the default rate up. We cannot measure how much, because we do not see how those removed loans ended.
- Compare the 60-month loans that had finished (24.7% default) with the unfinished ones (34.9%). The two groups also differ in issue date, so part of the gap is real deterioration, but the jump appears exactly where loans stop being finished.
- A 36-month default rate rising by about 2 points suggests part of the 60-month rise is real, but we cannot separate the two with this data.

## What this means for the later work
1. **Do not read the 2013 -> 2015 rise as "the market got worse".** Issue date is not used as a model input.
2. **The 2015 validation set is inflated for 60-month loans** (3,844 loans, all unfinished). Observed default there overstates true risk, so calibration (predicted vs observed PD) will look too optimistic. Report all validation metrics **separately for 36 and 60 months**, and treat 60-month calibration with caution.
3. **The fit window (through 2014) is slightly affected** (2014 H2 60-month loans, 1,339 loans). We keep them, flag them, and let `term` be a model input so the model can learn the two terms separately.
4. **Ranking metrics (AUC, KS) are less affected** than calibration, because the bias raises default rates broadly rather than reshuffling who is riskier. The decision thresholds in Part B (break-even PD 12.5%) must be read with this bias in mind.
5. **The 2016 holdout has no labels**, so there is nothing to bias there; its PD distribution is only compared for drift.

## Decision log
| Decision | Choice | Reason |
|---|---|---|
| Remove or reweight unfinished 60-month loans | **No; keep and flag** | They are the only 2014H2-2015 60-month data we have; dropping them would remove most recent 60-month history and the label would stay biased anyway. |
| Report by term | **Yes, always** | The bias is almost entirely in 60-month loans. |
