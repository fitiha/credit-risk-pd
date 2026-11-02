# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 42.9 | 48.9 | 74.5 | 80.6 |
| full predict(), borderline | 43.1 | 48.7 | 72.9 | 74.2 |
| full predict(), high_risk | 43.6 | 48.8 | 74.4 | 78.8 |
| step: input validation (raw) | 1.7 | 1.9 | 3.6 | 3.7 |
| step: contract transform (cleaning ... WoE) | 21.4 | 22.8 | 51.5 | 52.4 |
| step: validation of the cleaned values | 7.6 | 8.0 | 8.3 | 40.3 |
| step: score (model + calibration) | 3.7 | 4.1 | 4.6 | 5.3 |
| step: reason codes | 3.4 | 4.0 | 5.4 | 35.8 |

**Result:** a full `predict()` takes about 43 ms at the median and 49 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
