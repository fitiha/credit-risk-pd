"""Check that transform.joblib reproduces reference_transforms.csv and that the contract is complete.

Run: uv run python scripts/verify_contract.py [contract_dir]   (exit code 0 = pass)
"""
import sys

from credit_risk.contract_check import verify_contract

ok, lines = verify_contract(sys.argv[1] if len(sys.argv) > 1 else "contract")
print("\n".join(lines))
sys.exit(0 if ok else 1)
