"""Experiment E4: how to cut numeric fields into bins for WoE.

Methods: equal-sized bins, equal-sized bins merged until the pattern is one-directional (monotonic),
and decision-tree bins. Bins are learned on loans up to mid-2014 and judged on later loans (second half of 2014):
IV there, and whether the default-rate pattern is still one-directional.
Writes reports/a4_binning_comparison/{figures,tables}.
"""
import numpy as np
import pandas as pd

from credit_risk.binning import assign_bins, fit_edges
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.reporting import fresh_dirs
from credit_risk.transform import ApplicationTransformer
from credit_risk.woe import apply_woe, information_value, iv_on_new_data, woe_table

FIG, TAB = fresh_dirs("a4_binning_comparison")
raw = load_applications()
d = ApplicationTransformer().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
train, test = fit[fit.issue_d <= "2014-06-30"], fit[fit.issue_d > "2014-06-30"]

FIELDS = (["annual_inc_log", "loan_amnt_log", "installment_log", "revol_bal_log", "tot_cur_bal_log", "avg_cur_bal_log",
           "total_rev_hi_lim_log", "tot_hi_cred_lim_log", "dti_treated", "revol_util_treated", "bc_util_treated",
           "delinq_2yrs_capped", "pub_rec_capped", "pub_rec_bankruptcies_capped", "inq_last_6mths_capped", "mort_acc_capped",
           "open_acc", "total_acc", "acc_open_past_24mths", "num_tl_op_past_12m", "mths_since_last_delinq", "emp_length_years", "int_rate"]
          + [f"{f}_treated" for f in ["credit_history_months", "loan_to_income", "installment_to_income", "fico_mid",
                                      "open_acc_share", "recent_accounts_share"]] + ["revol_bal_to_income_log"])
METHODS = {"quantile_10": ("quantile", {"n_bins": 10}), "monotonic_merge": ("monotonic", {}), "tree_6": ("tree", {})}


def one_directional(table):
    r = table[table.group != "missing"].sort_values("group").default_rate.to_numpy()
    return bool((np.diff(r) >= -1e-12).all() or (np.diff(r) <= 1e-12).all())


rows = []
for f in FIELDS:
    for name, (method, kw) in METHODS.items():
        edges = fit_edges(train[f], train.defaulted, method, **kw)
        gtr, gte = assign_bins(train[f], edges), assign_bins(test[f], edges)
        t = woe_table(gtr, train.defaulted)
        t_later = woe_table(gte, test.defaulted)
        rows.append({"field": f, "method": name, "bins": len(edges) + 1, "smallest_bin_pct": 100 * t.loans.min() / len(train),
                     "iv_on_training_loans": information_value(t), "iv_on_later_loans": iv_on_new_data(t, gte, test.defaulted),
                     "pattern_one_directional_on_training": one_directional(t),
                     "pattern_one_directional_on_later_loans": one_directional(t_later)})
res = pd.DataFrame(rows)
res.round(4).to_csv(TAB / "binning_by_field_and_method.csv", index=False)
summary = res.groupby("method").agg(avg_bins=("bins", "mean"), avg_iv_training=("iv_on_training_loans", "mean"),
                                    avg_iv_later=("iv_on_later_loans", "mean"),
                                    pct_fields_one_directional_on_later=("pattern_one_directional_on_later_loans", lambda s: 100 * s.mean()),
                                    smallest_bin_pct=("smallest_bin_pct", "min"))
summary["iv_kept_pct"] = 100 * summary.avg_iv_later / summary.avg_iv_training
summary.round(3).to_csv(TAB / "method_summary.csv")
pd.set_option("display.width", 250, "display.max_columns", 20)
print(summary.round(3).to_string())
w = res.pivot(index="field", columns="method", values="iv_on_later_loans").round(4)
print("\nIV on later loans by field:\n", w.assign(best=w.idxmax(axis=1)).to_string())
