# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 33.7 | 35.7 | 65.1 | 71.7 |
| full predict(), borderline | 33.6 | 35.0 | 64.1 | 65.3 |
| full predict(), high_risk | 33.7 | 36.0 | 66.4 | 80.0 |
| step: input validation (raw) | 1.7 | 1.9 | 1.9 | 2.2 |
| step: contract transform (cleaning ... WoE) | 22.1 | 23.1 | 53.5 | 56.4 |
| step: validation of the cleaned values | 7.6 | 8.1 | 8.3 | 8.7 |
| step: score (model + calibration) | 0.6 | 0.7 | 0.8 | 0.9 |
| step: reason codes | 0.2 | 0.3 | 0.3 | 0.5 |

**Result:** a full `predict()` takes about 34 ms at the median and 36 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
