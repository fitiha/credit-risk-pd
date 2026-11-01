# Credit-risk take-home: data contract and probability-of-default model

A probability-of-default (PD) model for unsecured personal loans, built in two parts that share a data contract.
**Part A (data scientist):** audit and clean three data sources, treat financial values, rate field reliability, group categories and compute WoE, run PCA and factor analysis, check stability, and package everything as a versioned contract (`contract/`).
**Part B (ML engineer):** build the model on that contract (grade baseline, WoE scorecard, monotone gradient boosting), evaluate it out of time, set a profit-based decision policy, and serve it with reason codes, tests and a demo.

## Results in five lines
- 39 data problems found and fixed or flagged (`docs/issue_log.md`), including two columns that reveal the outcome and about 1,500 mis-scaled amounts; one unit error in the 2016 file would have corrupted every holdout score and a guard now rejects it.
- Out-of-time AUC on 2015 loans (never used to choose anything): **0.737 for the selected model, the average of a WoE scorecard and gradient boosting that both use the platform's sub-grade, against 0.722 for the platform's grade alone** (gain +0.015, 95% interval +0.011 to +0.019). Without any platform score the scorecard reaches 0.718 and only matches the grade. A screen of further ideas (new features, per-term models, other algorithms) found nothing else that helps: the limit is the information in an application. Funding the same share of loans as the model's thresholds (44%), the model earns $0.52M more than the grade on 2015 (interval +$0.12M to +$0.92M).
- Predictions are calibrated per term (slopes 0.99 and 0.94); thresholds from the profit assumptions: approve below 6.25%, decline at 12.5% or more; the approved band is 17.6% of 2015 H2 loans with a 4.4% default rate.
- The 2016 holdout is stable against 2015 (PSI 0.010 for the PD); scores are in `outputs/holdout_scores.csv`. The price of the sub-grade is a dependence on the platform's grading, monitored and with an independent fallback (`platform_score: none`).
- A scored application takes about 43 ms (limit 100 ms), reasons included. The factor scores helped the boosted model by +0.003 AUC and the scorecard not at all (`model_card.md`).

## Setup
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it). Dependencies are pinned in `uv.lock`.
```bash
uv sync
uv run hf download metasebiya21/credit-risk-take-home --repo-type dataset --local-dir data   # the four CSV files; not stored in this repository
```

## Run
```bash
bash scripts/run_all.sh                                   # everything, from the raw data, in order (a few minutes)
uv run pytest -q                                          # 141 tests (some need data/, the rest run without it)
uv run python scripts/verify_contract.py                  # the transform reproduces the 50 reference rows to 1e-6 (needs no data)
uv run python -m credit_risk.model.train --config config/train.yaml                  # train from the contract, one command
uv run python -m credit_risk.model.predict --json examples/borderline.json           # score one application: pd, decision, version, reasons
uv run python space/app.py                                # the demo, locally
scripts/deploy_space.sh <username>/<space-name>           # deploy to a private Hugging Face Space (after `uv run hf auth login`)
```

## What is where
| Path | Contents |
|---|---|
| `contract/` | The five contract files: `analytical_dataset.parquet`, `transform.joblib`, `data_spec.json`, `reference_transforms.csv`, `findings.md` (manager summary + technical appendix) |
| `DESIGN.md` | Committed before any model code; section 10 records what changed |
| `model_card.md` | Data window, metrics, thresholds, ablation result, known limits (generated from result files) |
| `docs/` | `decisions.md` (every choice with its evidence), `issue_log.md`, `target_analysis.md`, `field_reliability.md` |
| `reports/` | One folder per analysis, named like its script in `scripts/` (`a1_...` to `a6_...` for Part A, `b1_...` to `b3_...` for Part B); `experiments.csv` (every trial), `latency.md`, error analysis and drift in `b2_error_analysis_and_drift/` |
| `outputs/holdout_scores.csv` | `id, pd` for the 2016 holdout |
| `src/credit_risk/` | The package: cleaning, treatment, features, WoE, PCA and factors, the transformer, spec and validation (Part A); `model/` (training, evaluation, policy, prediction) and `demo.py` (Part B) |
| `models/` | The selected model (`pd_model.joblib`), baselines and `metadata.json` (git commit, training-data hash, date, contract version, thresholds) |
| `tests/`, `config/`, `examples/`, `space/` | Tests; training settings; made-up applicants; the Space entry point |

## Choices worth knowing
- Everything fitted (caps, medians, groups, bins, WoE, scaler, PCA, factors) uses only loans issued up to 2014-12-31; 2015 is the out-of-time test and 2016 is scored only. A test checks that 2015 loans change nothing that was fitted.
- The platform's own grade and rate are not model inputs (benchmark and test variant only); job title and location are restricted pending a fairness review; nothing from the servicing file reaches the model (tested).
- The contract holds real loan rows (`analytical_dataset.parquet`, `reference_transforms.csv`) because the brief requires them: keep this repository private. The Space contains none of them.
- Scripts are used instead of notebooks; each writes a labelled chart and table folder so the walkthrough can open them directly.

## Hours spent and tools used
**Hours spent:** *to be filled in by the author with the real figure.*
**AI tools:** Claude Code (Anthropic) was used throughout, for planning, writing and reviewing code and tests, running the analyses, and drafting the documents; the Hugging Face, testing and statistics skills it offers were consulted for the demo and tests. Every analytical choice and its evidence is written down in `docs/decisions.md` so it can be explained and challenged.

## Hugging Face Space
Link: *to be added after deployment.* Check that it is awake the day before the interview (`scripts/check_space.sh <username>/<space-name>`): free Spaces sleep when idle.
