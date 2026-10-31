"""Hugging Face Space entry point. The same code runs locally: `uv run python space/app.py`."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))          # the Space ships the package next to this file
sys.path.insert(0, str(root.parent / "src"))   # in the repository it lives one level up

from credit_risk.demo import build_demo  # noqa: E402

demo = build_demo(model_path=str(next(p for p in (root / "models" / "pd_model.joblib", root.parent / "models" / "pd_model.joblib") if p.exists())),
                  contract_dir=str(next(p for p in (root / "contract", root.parent / "contract") if (p / "transform.joblib").exists())),
                  examples_dir=str(next(p for p in (root / "examples", root.parent / "examples") if p.exists())))

if __name__ == "__main__":
    demo.launch()
