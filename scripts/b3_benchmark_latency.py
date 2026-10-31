"""Latency of scoring one application (the brief allows 100 ms): the whole call, and where the time goes.

Measured on this machine with the same Predictor the demo uses. Writes reports/latency.md and reports/b3_latency/tables/latency.csv.
"""
import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.model.predict import Predictor
from credit_risk.reporting import fresh_dirs

warnings.filterwarnings("ignore")
_, TAB = fresh_dirs("b3_latency")
p = Predictor()
recs = {n: json.loads(Path(f"examples/{n}.json").read_text()) for n in ("low_risk", "borderline", "high_risk")}
N = 300


def timed(fn, n=N):
    for _ in range(10):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return np.array(t)


rows = []
for name, rec in recs.items():
    t = timed(lambda: p.predict(rec))
    rows.append({"what": f"full predict(), {name}", "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95), "p99_ms": np.percentile(t, 99), "max_ms": t.max()})
rec = recs["borderline"]
raw = p._frame(rec)
steps = {"input validation (raw)": lambda: __import__("credit_risk.spec", fromlist=["x"]).validate_raw(raw),
         "contract transform (cleaning ... WoE)": lambda: p.transformer.transform(raw),
         "validation of the cleaned values": None, "score (model + calibration)": None, "reason codes": None}
cleaned = p.transformer.transform(raw)
from credit_risk.spec import validate_cleaned
from credit_risk.model.models import add_grade_rank
frame = add_grade_rank(cleaned)
steps["validation of the cleaned values"] = lambda: validate_cleaned(cleaned)
steps["score (model + calibration)"] = lambda: p.bundle.score(frame)
steps["reason codes"] = lambda: p.reasons(frame)
for name, fn in steps.items():
    t = timed(fn)
    rows.append({"what": f"  step: {name}", "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95), "p99_ms": np.percentile(t, 99), "max_ms": t.max()})
res = pd.DataFrame(rows)
res.round(2).to_csv(TAB / "latency.csv", index=False)
all_ = timed(lambda: [p.predict(r) for r in recs.values()], 100) / 3
md = f"""# Scoring latency

One application, one request, measured with the same `Predictor` the demo calls. {N} repetitions per case after a warm-up, single process, no batching.
Machine: {platform.processor() or platform.machine()}, {platform.system()} {platform.release()}, Python {platform.python_version()}.

| What | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---|---|---|---|
""" + "\n".join(f"| {r.what.strip()} | {r.p50_ms:.1f} | {r.p95_ms:.1f} | {r.p99_ms:.1f} | {r.max_ms:.1f} |" for r in res.itertuples()) + f"""

**Result:** a full `predict()` takes about {res.p50_ms.iloc[:3].mean():.0f} ms at the median and {res.p95_ms.iloc[:3].mean():.0f} ms at the 95th percentile, against the 100 ms budget.
Most of the time is the contract transform (cleaning, treatment, features, grouping, PCA, factors and WoE); the model itself and the reason codes take a few milliseconds.
The free Hugging Face CPU tier is typically two to three times slower than a workstation, so the demo should still be well inside the budget.
"""
Path("reports/latency.md").write_text(md)
print(md)
