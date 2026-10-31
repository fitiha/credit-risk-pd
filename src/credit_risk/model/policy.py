"""Decision policy: expected profit and PD thresholds.

A repaid loan earns `margin_if_repaid` of the amount, a defaulted loan loses `loss_given_default` of it:
expected profit per loan = amount x (margin x (1 - PD) - loss x PD), which is zero at PD = margin / (margin + loss).
"""
import numpy as np
import pandas as pd


def breakeven_pd(cfg):
    return cfg["margin_if_repaid"] / (cfg["margin_if_repaid"] + cfg["loss_given_default"])


def margin_per_dollar(pd_, cfg):
    return cfg["margin_if_repaid"] * (1 - np.asarray(pd_)) - cfg["loss_given_default"] * np.asarray(pd_)


def realised_profit(amount, defaulted, cfg):
    """Profit if the loan is funded, from what actually happened."""
    amount, defaulted = np.asarray(amount, float), np.asarray(defaulted)
    return amount * np.where(defaulted == 1, -cfg["loss_given_default"], cfg["margin_if_repaid"])


def thresholds(cfg):
    """Auto-approve below the PD where the margin falls to the minimum worth approving unseen; auto-decline at break-even."""
    approve = (cfg["margin_if_repaid"] - cfg["min_approval_margin"]) / (cfg["margin_if_repaid"] + cfg["loss_given_default"])
    return {"approve_below": round(float(approve), 10), "decline_at_or_above": round(float(breakeven_pd(cfg)), 10)}  # rounded: no floating-point noise at the boundary


def decide(pd_, th):
    p = np.asarray(pd_, float)
    return np.where(p < th["approve_below"], "approve", np.where(p >= th["decline_at_or_above"], "decline", "review"))


def best_decline_threshold(df, cfg, grid=None):
    """Empirical check: the PD cut-off that maximises realised profit when every loan below it is funded."""
    grid = np.arange(0.04, 0.40, 0.005) if grid is None else grid
    rows = []
    for t in grid:
        funded = df[df.pd < t]
        rows.append({"threshold": float(t), "approval_rate": len(funded) / len(df), "profit": float(realised_profit(funded.loan_amnt, funded.defaulted, cfg).sum())})
    return pd.DataFrame(rows)


def band_table(df, th, cfg):
    """Loans, default rate and profit (if funded) for each decision band."""
    d = df.assign(decision=decide(df.pd, th))
    d["profit_if_funded"] = realised_profit(d.loan_amnt, d.defaulted, cfg)
    t = d.groupby("decision").agg(loans=("defaulted", "size"), default_rate=("defaulted", "mean"), mean_pd=("pd", "mean"),
                                  amount=("loan_amnt", "sum"), profit_if_funded=("profit_if_funded", "sum")).reindex(["approve", "review", "decline"]).fillna(0)
    t["share_of_loans"] = t.loans / t.loans.sum()
    t["profit_per_dollar"] = t.profit_if_funded / t.amount.replace(0, np.nan)
    return t.reset_index()
