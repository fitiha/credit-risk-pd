# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 42.6 | 47.6 | 73.6 | 79.0 |
| full predict(), borderline | 42.6 | 47.6 | 74.5 | 75.2 |
| full predict(), high_risk | 42.7 | 47.6 | 72.9 | 75.1 |
| step: input validation (raw) | 1.6 | 1.8 | 1.9 | 2.5 |
| step: contract transform (cleaning ... WoE) | 21.2 | 21.9 | 53.3 | 55.5 |
| step: validation of the cleaned values | 7.3 | 7.6 | 7.8 | 38.4 |
| step: score (model + calibration) | 3.6 | 4.0 | 4.0 | 4.1 |
| step: reason codes | 3.3 | 3.8 | 4.2 | 37.0 |

**Result:** a full `predict()` takes about 43 ms at the median and 48 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
