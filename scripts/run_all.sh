#!/usr/bin/env bash
# Rebuild everything from the raw data, end to end. Needs data/ (see README). Takes a few minutes.
set -euo pipefail
run() { echo; echo "== $*"; uv run python "$@"; }
# Part A: the data scientist's analyses, in order (each writes reports/<same name>/)
for s in a1_label_audit a1_target_analysis a2_financial_values_profile a2_outlier_method_comparison a2_missing_amounts_experiment a2_treatment_effect \
         a3_reliability_evidence a3_platform_score_check a3_field_catalog a3_feature_evidence a4_emp_title_grouping a4_rare_level_merging a4_binning_comparison \
         a4_iv_ranking a5_correlation_vif a5_pca a5_factor_analysis a5_factor_robustness a6_stability_psi a6_deciles_and_segments a6_factors_vs_grade a6_location_proxy; do
  run scripts/$s.py
done
# the data contract and its parity check
run scripts/build_contract.py
run scripts/verify_contract.py
# Part B: train from the contract (one command), evaluate, analyse, benchmark, write the model card, build the Space folder
echo; echo "== train"; uv run python -m credit_risk.model.train --config config/train.yaml
for s in b1_evaluate_model b2_error_analysis_and_drift b3_benchmark_latency b4_write_model_card build_space; do
  run scripts/$s.py
done
echo; echo "== tests"; uv run pytest -q
