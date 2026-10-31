"""How strong is the platform's own score (grade, sub_grade, int_rate) compared with our inputs?

Time-ordered folds on pre-2015 loans; statistics from the training part only.
Writes reports/a3_platform_score_check/tables/auc_by_input_set.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_risk.cleaning import Cleaner
from credit_risk.io import FIT_CUTOFF, load_applications
from credit_risk.outliers import AMOUNT_FIELDS, RATIO_FIELDS, AmountTreater, RatioTreater
from credit_risk.reporting import fresh_dirs

_, TAB = fresh_dirs("a3_platform_score_check")
raw = load_applications()
d = Cleaner().fit(raw).transform(raw)
fit = d[d.issue_d <= FIT_CUTOFF].reset_index(drop=True)
fit["sub_grade_rank"] = fit.sub_grade.rank(method="dense").astype(int)
fit["sub_grade_rank"] = fit.sub_grade.map({g: i for i, g in enumerate(sorted(fit.sub_grade.unique()))})
folds = [(fit.issue_d < "2014-01-01", fit.issue_d.between("2014-01-01", "2014-06-30")),
         (fit.issue_d <= "2014-06-30", fit.issue_d >= "2014-07-01")]
bureau = ["fico_range_low", "inq_last_6mths", "delinq_2yrs", "pub_rec", "open_acc", "total_acc", "mort_acc", "acc_open_past_24mths"]


def features(kind, tr, te):
    a = AmountTreater().fit(tr)
    r = RatioTreater().fit(tr)
    xtr, xte = r.transform(a.transform(tr)), r.transform(a.transform(te))
    cols = {"sub_grade_only": ["sub_grade_rank"], "int_rate_only": ["int_rate"]}
    ours = [f"{f}_log" for f in AMOUNT_FIELDS] + [f"{f}_treated" for f in RATIO_FIELDS] + bureau + ["term"]
    cols["our_inputs"] = ours
    cols["our_inputs_plus_sub_grade"] = ours + ["sub_grade_rank"]
    c = cols[kind]
    mu, sd = xtr[c].mean(), xtr[c].std().replace(0, 1)
    return (xtr[c] - mu) / sd, (xte[c] - mu) / sd


rng = np.random.default_rng(0)
rows = []
for kind in ["sub_grade_only", "int_rate_only", "our_inputs", "our_inputs_plus_sub_grade"]:
    aucs, los, his = [], [], []
    for mtr, mte in folds:
        tr, te = fit[mtr], fit[mte]
        xtr, xte = features(kind, tr, te)
        p = LogisticRegression(max_iter=3000).fit(xtr, tr.defaulted).predict_proba(xte)[:, 1]
        y = te.defaulted.to_numpy()
        boot = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(200))]
        aucs.append(roc_auc_score(y, p)); los.append(np.percentile(boot, 2.5)); his.append(np.percentile(boot, 97.5))
    rows.append({"input_set": kind, "auc": np.mean(aucs), "auc_ci_low": np.mean(los), "auc_ci_high": np.mean(his)})
out = pd.DataFrame(rows).round(4)
out.to_csv(TAB / "auc_by_input_set.csv", index=False)
print(out.to_string(index=False))
