"""Ranking and probability-quality metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def auc(y, p):
    return float(roc_auc_score(y, p))


def gini(y, p):
    return 2 * auc(y, p) - 1


def ks(y, p):
    """Largest gap between the cumulative share of defaults and of repaid loans when ranked by PD."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def brier(y, p):
    return float(brier_score_loss(y, p))


def summary(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"loans": len(y), "default_rate": float(y.mean()), "mean_pd": float(p.mean()), "auc": auc(y, p), "gini": gini(y, p), "ks": ks(y, p), "brier": brier(y, p)}


def calibration_table(y, p, bins=10):
    """Predicted against observed default rate in groups of equal size ordered by PD."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), bins, labels=False)
    t = df.groupby("bin").agg(loans=("y", "size"), mean_pd=("p", "mean"), default_rate=("y", "mean")).reset_index()
    t["gap"] = t.default_rate - t.mean_pd
    return t


def calibration_slope_intercept(y, p):
    """Slope 1 and intercept 0 mean perfectly calibrated probabilities (logistic fit of the outcome on the model's log-odds)."""
    from sklearn.linear_model import LogisticRegression
    y, p = np.asarray(y), np.asarray(p, float)
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])
