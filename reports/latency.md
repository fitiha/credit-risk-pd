# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 32.1 | 33.9 | 59.8 | 65.8 |
| full predict(), borderline | 31.8 | 34.1 | 60.1 | 62.3 |
| full predict(), high_risk | 32.4 | 35.7 | 62.0 | 63.4 |
| step: input validation (raw) | 1.6 | 1.7 | 1.7 | 1.9 |
| step: contract transform (cleaning ... WoE) | 19.8 | 22.3 | 46.8 | 49.2 |
| step: validation of the cleaned values | 6.9 | 7.4 | 7.9 | 34.8 |
| step: score (model + calibration) | 0.5 | 0.6 | 0.6 | 0.6 |
| step: reason codes | 0.3 | 0.3 | 0.5 | 0.5 |

**Result:** a full `predict()` takes about 32 ms at the median and 35 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
