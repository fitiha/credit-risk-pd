"""Train the PD models from the contract. One command:

    uv run python -m credit_risk.model.train --config config/train.yaml

The contract's transform is used unchanged. Every trial is appended to reports/experiments.csv. The run saves the selected model
(models/pd_model.joblib) and the baselines used for comparison, each with the metadata needed to trace it.
"""
import argparse
import datetime as dt
import itertools
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk.model import metrics, policy
from credit_risk.model.data import data_hash, load_config, split, training_frames
from credit_risk.model.inputs import CONSTRAINTS, INPUT_SETS, REASONS, SCORECARD_COLUMNS, SCORECARD_FIELDS
from credit_risk.model.log import ExperimentLog
from credit_risk.model.models import Calibrator, LgbmPD, LogitPD, PDModel, PerTermCalibrator, add_grade_rank
from credit_risk.woe import WoEEncoder
from credit_risk.woe_fields import WOE_CATEGORICAL, WOE_NUMERIC

CATEGORY_WOE = ["purpose_group_woe", "home_ownership_group_woe", "verification_status_woe", "term_woe"]
MIN_GAIN_TO_ADD_FACTORS = 0.002   # a bigger input set must beat the raw set by this much to be chosen
MIN_GAIN_FOR_BOOSTING = 0.005     # boosting must beat the scorecard by this much to replace it
AUC_TIE = 0.001                   # configurations within this AUC are tied and the lower Brier score wins (no class-weight distortion for nothing)


def git_state():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def scorecard_reason_column(field):
    """The field a scorecard WoE column describes, in the vocabulary of the reason texts."""
    col = field.replace("_log", "_capped") if field.endswith("_log") else field
    assert col in REASONS, col
    return col


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg["seed"]
        self.run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log = ExperimentLog("reports/experiments.csv", self.run_id)
        self.transformer, self.spec, frame = training_frames(cfg)
        self.frame = add_grade_rank(frame)
        self.train = split(self.frame, end=cfg["fit_cutoff"])
        self.folds = [(split(self.train, end=f["train_end"]), split(self.train, f["valid_start"], f["valid_end"])) for f in cfg["folds"]]
        # WoE values are fitted on labels, and the contract's were fitted on all pre-2015 loans, so a validation fold would already have been seen.
        # For tuning and model selection only, WoE is refitted inside each fold on that fold's training part (the contract itself is untouched).
        self.woe_folds = []
        for tr, va in self.folds:
            enc = WoEEncoder(numeric=WOE_NUMERIC, categorical=WOE_CATEGORICAL).fit(tr)
            self.woe_folds.append((enc.transform(tr), enc.transform(va)))

    # ---- fitting on the time folds ------------------------------------------------------
    def evaluate(self, factory, woe_fold=False):
        aucs, briers, preds = [], [], []
        for tr, va in (self.woe_folds if woe_fold else self.folds):
            p = factory().fit(tr, tr.defaulted).predict(va)
            aucs.append(metrics.auc(va.defaulted, p)); briers.append(metrics.brier(va.defaulted, p)); preds.append(pd.Series(p, index=va.index))
        return aucs, briers, preds

    def lgbm_factory(self, columns, params, class_weight, monotone):
        full = {**params, "learning_rate": self.cfg["lightgbm"]["learning_rate"], "colsample_bytree": self.cfg["lightgbm"]["colsample_bytree"], "subsample": self.cfg["lightgbm"]["subsample"]}
        return lambda: LgbmPD(columns, full, class_weight, monotone, self.seed)

    def tune_lightgbm(self, set_name, monotone=True):
        grid = self.cfg["lightgbm"]["grid"]
        results = []
        for values, cw in itertools.product(itertools.product(*grid.values()), self.cfg["lightgbm"]["class_weight"]):
            params = dict(zip(grid.keys(), values))
            aucs, briers, _ = self.evaluate(self.lgbm_factory(INPUT_SETS[set_name], params, cw, monotone))
            self.log.add("lightgbm", set_name, params, cw, monotone, aucs, briers)
            results.append({"auc": float(np.mean(aucs)), "brier": float(np.mean(briers)), "params": params, "class_weight": cw})
        top = max(r["auc"] for r in results)
        return min((r for r in results if top - r["auc"] <= AUC_TIE), key=lambda r: r["brier"])

    def tune_scorecard(self, columns, name, scale=False):
        results = []
        for C, k in itertools.product(self.cfg["scorecard"]["C"], self.cfg["scorecard"]["top_k"] if not scale else [None]):
            aucs, briers, _ = self.evaluate(lambda C=C, k=k: LogitPD(columns, C, scale, self.seed, nonpositive=not scale, top_k=k), woe_fold=True)
            self.log.add("logistic", name, {"C": C, "top_k": k}, None, None, aucs, briers, "WoE refitted inside each fold")
            results.append({"auc": float(np.mean(aucs)), "brier": float(np.mean(briers)), "C": C, "top_k": k})
        top = max(r["auc"] for r in results)
        return min((r for r in results if top - r["auc"] <= AUC_TIE), key=lambda r: (r["top_k"] or 99, r["brier"]))  # ties go to the simpler scorecard

    # ---- final fits ----------------------------------------------------------------------
    def finalize(self, name, factory, columns, reason_map, note, woe_fold=False):
        """OOF predictions on the time folds -> calibrator -> final model on all training loans -> saved bundle."""
        aucs, briers, preds = self.evaluate(factory, woe_fold)
        oof = pd.concat(preds)
        y = self.train.defaulted.loc[oof.index]
        half = len(preds[0])
        terms = self.train.term.loc[oof.index].to_numpy()
        half_mask = np.arange(len(oof)) < half
        # each calibrator is fitted on one fold's predictions and scored on the other; the better method is chosen separately for each term
        chosen, scores = {}, {}
        for tm in sorted(set(terms)):
            s = {}
            for method in ("platt", "isotonic"):
                b = []
                for fit_part, test_part in ((half_mask, ~half_mask), (~half_mask, half_mask)):
                    fm, tm_ = fit_part & (terms == tm), test_part & (terms == tm)
                    cal = Calibrator(method).fit(oof.to_numpy()[fm], y.to_numpy()[fm])
                    b.append(metrics.brier(y.to_numpy()[tm_], cal.predict(oof.to_numpy()[tm_])))
                s[method] = float(np.mean(b))
            chosen[int(tm)], scores[int(tm)] = min(s, key=s.get), s
        method = chosen
        calibrator = PerTermCalibrator(chosen).fit(oof.to_numpy(), y.to_numpy(), terms)
        model = factory().fit(self.train, self.train.defaulted)
        sha, dirty = git_state()
        meta = {"model": name, "note": note, "contract_version": self.transformer.contract_version, "fit_cutoff": self.cfg["fit_cutoff"],
                "training_loans": int(len(self.train)), "training_years": [str(self.train.issue_d.min().date()), str(self.train.issue_d.max().date())],
                "git_commit": sha, "git_dirty": dirty, "training_data_hash": data_hash(self.train, columns if not any(c.endswith("_woe") for c in columns) else []),
                "training_date": dt.date.today().isoformat(), "seed": self.seed, "calibration": method, "calibration_brier_cross_check": scores,
                "fold_auc": [round(a, 4) for a in aucs], "fold_brier": [round(b, 4) for b in briers], "inputs": list(columns), "run_id": self.run_id}
        th = policy.thresholds(self.cfg)
        meta["thresholds"] = th
        # reasons compare an applicant with a typical APPROVED applicant: against the average one, most declined applicants would have no reason at all
        bundle = PDModel(model, calibrator, th, meta, reason_map)
        approved = pd.Series(policy.decide(bundle.score(self.train), th), index=self.train.index) == "approve"
        if getattr(model, "kind", "") == "logistic":
            model.reference_ = self.train.loc[approved, model.columns].to_numpy(float).mean(axis=0)
        meta["reference_group"] = f"{int(approved.sum()):,} training loans the model approves"
        meta["inputs_used"] = list(getattr(model, "columns", columns))
        meta["inputs_dropped_for_wrong_sign"] = list(getattr(model, "dropped_", []))
        reference = {}
        for c in set(reason_map.values()):
            if c in self.train.columns and pd.api.types.is_numeric_dtype(self.train[c]):
                reference[c] = float(self.train.loc[approved, c].median())
        return PDModel(model, calibrator, th, meta, reason_map, reference), oof

    def run(self):
        cfg, out = self.cfg, Path(self.cfg["model_dir"])
        out.mkdir(exist_ok=True)
        print(f"training loans {len(self.train):,}; folds {[len(v) for _, v in self.folds]}")
        # baselines
        g_aucs, g_briers, _ = self.evaluate(lambda: LogitPD(["sub_grade_rank"], 1.0, True, self.seed))
        self.log.add("logistic", "grade_alone", {"C": 1.0}, None, None, g_aucs, g_briers, "benchmark: the platform's sub-grade")
        sc = self.tune_scorecard(SCORECARD_COLUMNS, "woe_scorecard")
        print("scorecard", sc)
        # ablation: raw only vs factor scores only vs both, for boosting and for the logistic family
        sets = {s: self.tune_lightgbm(s) for s in INPUT_SETS}
        for s, r in sets.items():
            print("lightgbm", s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        lin = {"woe_only": sc, "factors_plus_category_woe": self.tune_scorecard(CATEGORY_WOE + [c for c in INPUT_SETS["factors"] if c.startswith("factor_")], "factors_plus_category_woe", scale=True),
               "woe_plus_factors": self.tune_scorecard(SCORECARD_COLUMNS + [c for c in INPUT_SETS["factors"] if c.startswith("factor_")], "woe_plus_factors", scale=True)}
        # input set rule: a larger set must beat the raw set by MIN_GAIN_TO_ADD_FACTORS
        chosen_set = "raw"
        if sets["both"]["auc"] - sets["raw"]["auc"] >= MIN_GAIN_TO_ADD_FACTORS:
            chosen_set = "both"
        if sets["factors"]["auc"] - sets[chosen_set]["auc"] >= MIN_GAIN_TO_ADD_FACTORS:
            chosen_set = "factors"
        best = sets[chosen_set]
        # Monotone constraints encode credit logic (more debt, more utilisation, a lower score should never lower PD). They are kept only
        # where the data agree: audit them one at a time and switch off each constraint whose removal raises the time-fold AUC.
        cols = INPUT_SETS[chosen_set]
        constrained = [c for c in cols if CONSTRAINTS.get(c, 0) != 0]
        limit = cfg["lightgbm"]["max_cost_of_a_monotone_constraint_auc"]
        lg = lambda mono: self.evaluate(self.lgbm_factory(cols, best["params"], best["class_weight"], mono))
        aucs0, briers0, _ = lg(False)
        self.log.add("lightgbm", chosen_set, best["params"], best["class_weight"], False, aucs0, briers0, "no monotone constraints")
        aucs_all, briers_all, _ = lg(set(constrained))
        self.log.add("lightgbm", chosen_set, best["params"], best["class_weight"], "all", aucs_all, briers_all, "all declared monotone constraints")
        keep, audit = set(constrained), []
        for c in constrained:
            a_, b_, _ = lg(set(constrained) - {c})
            gain = float(np.mean(a_) - np.mean(aucs_all))
            audit.append({"field": c, "direction": CONSTRAINTS[c], "auc_gain_if_constraint_removed": round(gain, 5)})
            self.log.add("lightgbm", chosen_set, best["params"], best["class_weight"], f"all except {c}", a_, b_, "constraint audit")
            if gain > 0.0005:
                keep.discard(c)
        aucs_k, briers_k, _ = lg(keep)
        self.log.add("lightgbm", chosen_set, best["params"], best["class_weight"], "audited", aucs_k, briers_k, "constraints that survived the audit")
        cost = float(np.mean(aucs0) - np.mean(aucs_k))
        keep_monotone = keep if cost <= limit else False
        print(f"input set {chosen_set}; constraints declared {len(constrained)}, kept {len(keep)}; cost of the audited set {cost:+.4f} AUC (limit {limit}) -> {'kept' if keep_monotone else 'dropped'}")
        Path("reports").mkdir(exist_ok=True)
        pd.DataFrame(audit).to_csv("reports/monotone_constraint_audit.csv", index=False)
        use_lgbm = best["auc"] - sc["auc"] >= MIN_GAIN_FOR_BOOSTING
        decision = {"lightgbm_mean_auc": best["auc"], "scorecard_mean_auc": sc["auc"], "chosen_input_set": chosen_set, "monotone_cost_auc": cost, "monotone_declared": len(constrained), "monotone_kept": sorted(keep) if keep_monotone else [], "unconstrained_auc": float(np.mean(aucs0)), "selected": "lightgbm" if use_lgbm else "woe_scorecard", "ablation_lightgbm": {k: round(v["auc"], 4) for k, v in sets.items()},
                    "ablation_logistic": {k: round(v["auc"], 4) for k, v in lin.items()}, "grade_alone_auc": float(np.mean(g_aucs))}
        # final bundles
        bundles = {}
        wmap = {f"{f}_woe": scorecard_reason_column(f) for f in SCORECARD_FIELDS}
        bundles["grade_baseline"], _ = self.finalize("grade_baseline", lambda: LogitPD(["sub_grade_rank"], 1.0, True, self.seed), ["sub_grade_rank"], {}, "platform sub-grade only")
        bundles["woe_scorecard"], _ = self.finalize("woe_scorecard", lambda: LogitPD(SCORECARD_COLUMNS, sc["C"], False, self.seed, nonpositive=True, top_k=sc["top_k"]), SCORECARD_COLUMNS, wmap, "logistic regression on the contract's WoE columns", woe_fold=True)
        # test variant promised in DESIGN.md: what the platform's own sub-grade adds if it were allowed as an input (never selected)
        bundles["scorecard_plus_grade"], _ = self.finalize("scorecard_plus_grade", lambda: LogitPD(SCORECARD_COLUMNS + ["sub_grade_rank"], 0.1, True, self.seed),
                                                           SCORECARD_COLUMNS + ["sub_grade_rank"], {**wmap, "sub_grade_rank": "sub_grade_rank"},
                                                           "test variant: the scorecard plus the platform's sub-grade", woe_fold=True)
        bundles["lightgbm"], _ = self.finalize("lightgbm", self.lgbm_factory(INPUT_SETS[chosen_set], best["params"], best["class_weight"], keep_monotone), INPUT_SETS[chosen_set], {c: c for c in INPUT_SETS[chosen_set]}, f"gradient boosting on the '{chosen_set}' inputs")
        for name, b in bundles.items():
            b.metadata["selection"] = decision
            joblib.dump(b, out / f"{name}.joblib", compress=3)
        selected = bundles[decision["selected"]]
        joblib.dump(selected, out / "pd_model.joblib", compress=3)
        (out / "metadata.json").write_text(json.dumps({**selected.metadata, "selection": decision}, indent=2, default=str) + "\n")
        print("selected:", decision["selected"], "| thresholds", selected.thresholds, "| calibration", selected.metadata["calibration"])
        print(json.dumps(decision, indent=2, default=str))
        return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train.yaml")
    args = ap.parse_args()
    Trainer(load_config(args.config)).run()


if __name__ == "__main__":
    main()
