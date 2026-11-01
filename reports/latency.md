# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 34.7 | 35.9 | 65.2 | 69.1 |
| full predict(), borderline | 34.6 | 36.1 | 65.3 | 67.1 |
| full predict(), high_risk | 34.6 | 36.6 | 65.6 | 72.3 |
| step: input validation (raw) | 1.6 | 1.8 | 1.8 | 1.9 |
| step: contract transform (cleaning ... WoE) | 21.3 | 22.4 | 53.2 | 54.8 |
| step: validation of the cleaned values | 7.3 | 7.8 | 8.0 | 40.8 |
| step: score (model + calibration) | 0.6 | 0.6 | 0.7 | 0.9 |
| step: reason codes | 0.3 | 0.3 | 0.4 | 0.5 |

**Result:** a full `predict()` takes about 35 ms at the median and 36 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
