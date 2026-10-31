# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. 300 repetitions per case after a warm-up, single process, no batching.
Machine: x86_64, Linux 7.2.9-1-MANJARO, Python 3.12.15.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
| full predict(), low_risk | 33.9 | 65.7 | 105.1 | 113.5 |
| full predict(), borderline | 33.6 | 60.4 | 82.5 | 104.1 |
| full predict(), high_risk | 32.7 | 39.6 | 68.0 | 75.8 |
| step: input validation (raw) | 1.6 | 1.7 | 1.8 | 2.0 |
| step: contract transform (cleaning ... WoE) | 20.6 | 42.9 | 53.7 | 66.4 |
| step: validation of the cleaned values | 7.0 | 8.2 | 15.8 | 34.3 |
| step: score (model + calibration) | 0.5 | 0.6 | 0.7 | 1.0 |
| step: reason codes | 0.7 | 0.7 | 0.7 | 0.8 |

**Result:** a full `predict()` takes about 33 ms at the median and 55 ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
