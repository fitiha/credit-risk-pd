# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 46.5 | 56.7 | 78.9 | 84.5 |
| full predict(), borderline | 46.3 | 54.2 | 78.4 | 80.0 |
| full predict(), high_risk | 46.6 | 59.5 | 79.7 | 83.8 |
| step: input validation (raw) | 1.6 | 1.8 | 1.9 | 2.0 |
| step: contract transform (cleaning ... WoE) | 21.3 | 22.4 | 53.4 | 54.9 |
| step: validation of the cleaned values | 7.3 | 7.7 | 7.9 | 8.0 |
| step: score (model + calibration) | 3.7 | 4.2 | 4.5 | 38.0 |
| step: reason codes | 3.3 | 4.0 | 4.5 | 6.0 |

**Result:** a full `predict()` takes about 46 ms at the median and 57 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
