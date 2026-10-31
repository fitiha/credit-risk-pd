"""Assemble the Hugging Face Space in build/space/: the app, the package, the fitted transform and spec, the model, made-up examples and the model card.

Never included: the training data, the holdout, the analytical dataset, the reference rows (they are real rows), or any CSV/parquet file.
Run: uv run python scripts/build_space.py     Deploy: see scripts/deploy_space.sh
"""
import shutil
import sys
from pathlib import Path

OUT = Path("build/space")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
shutil.copy("space/app.py", OUT / "app.py")
shutil.copytree("src/credit_risk", OUT / "src" / "credit_risk", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
(OUT / "contract").mkdir()
for f in ("transform.joblib", "data_spec.json"):  # the contract's other files hold real loan rows and stay out
    shutil.copy(Path("contract") / f, OUT / "contract" / f)
(OUT / "models").mkdir()
shutil.copy("models/pd_model.joblib", OUT / "models" / "pd_model.joblib")
shutil.copytree("examples", OUT / "examples")
shutil.copy("model_card.md", OUT / "model_card.md")
(OUT / "requirements.txt").write_text("""numpy==2.5.4
pandas==3.1.0
scipy==1.18.1
scikit-learn==1.9.1
lightgbm==4.7.0
joblib==1.6.0
pyyaml==6.0.3
pyarrow>=14
""")
front = """---
title: Personal-loan default risk
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Probability of default, decision and reasons for a personal-loan application
---

"""
(OUT / "README.md").write_text(front + Path("model_card.md").read_text())

# safety checks: no data of any kind
bad = [p for p in OUT.rglob("*") if p.is_file() and (p.suffix in {".csv", ".parquet", ".xlsx", ".pkl"} or p.stat().st_size > 5_000_000)]
assert not bad, f"unexpected files in the Space folder: {bad}"
names = {p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file()}
assert not any("reference" in n or "analytical" in n or n.startswith("data/") for n in names), names
print("Space folder:", OUT, "|", len(names), "files |", sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) // 1000, "KB")
print(sorted(n for n in names if not n.startswith("src/"))); sys.exit(0)
