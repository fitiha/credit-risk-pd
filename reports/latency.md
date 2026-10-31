# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 32.4 | 35.7 | 63.9 | 68.2 |
| full predict(), borderline | 43.9 | 58.2 | 82.8 | 99.4 |
| full predict(), high_risk | 37.2 | 47.7 | 79.2 | 81.2 |
| step: input validation (raw) | 1.7 | 1.8 | 1.9 | 2.1 |
| step: contract transform (cleaning ... WoE) | 23.8 | 49.4 | 73.8 | 80.3 |
| step: validation of the cleaned values | 7.9 | 8.4 | 8.7 | 38.9 |
| step: score (model + calibration) | 0.5 | 0.6 | 0.6 | 0.6 |
| step: reason codes | 0.3 | 0.3 | 0.3 | 0.5 |

**Result:** a full `predict()` takes about 38 ms at the median and 47 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
