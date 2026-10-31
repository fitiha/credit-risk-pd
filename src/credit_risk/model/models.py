"""The models: a logistic model (grade baseline, WoE scorecard, factor models), gradient boosting, a calibrator and the saved bundle."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from credit_risk.model.inputs import CATEGORICAL, monotone_vector
from credit_risk.spec import SUB_GRADES

GRADE_RANK = {g: i for i, g in enumerate(SUB_GRADES)}


def add_grade_rank(frame):
    return frame.assign(sub_grade_rank=frame.sub_grade.map(GRADE_RANK).astype(float))


class LogitPD:
    """Logistic regression on numeric columns. With scale=False the columns are used as they are (WoE values: a scorecard).

    With nonpositive=True (standard scorecard practice) every WoE weight must point in the safe direction: a column whose weight comes out
    positive is removed, the most positive first, and the model is refitted until none remains. This keeps every reason code meaningful.
    """

    kind = "logistic"

    def __init__(self, columns, C=0.1, scale=True, seed=42, nonpositive=False, top_k=None):
        self.columns, self.C, self.scale, self.seed, self.nonpositive, self.top_k = list(columns), C, scale, seed, nonpositive, top_k

    def _fit_once(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) if self.scale else np.ones(X.shape[1])
        self.scale_[self.scale_ == 0] = 1
        self.model_ = LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed).fit((X - self.mean_) / self.scale_, y)

    def fit(self, df, y):
        self.candidate_columns = list(self.columns)
        self.dropped_ = []
        if self.top_k and self.top_k < len(self.columns):  # keep the k columns most related to the label in the data this model is fitted on
            strength = pd.Series({c: abs(np.corrcoef(df[c].to_numpy(float), np.asarray(y))[0, 1]) for c in self.columns}).sort_values(ascending=False)
            self.columns = [c for c in self.columns if c in set(strength.index[: self.top_k])]
        while True:
            self._fit_once(df[self.columns].to_numpy(float), y)
            coef = self.model_.coef_[0]
            if not self.nonpositive or (coef <= 0).all() or len(self.columns) == 1:
                break
            worst = int(np.argmax(coef))
            self.dropped_.append(self.columns.pop(worst))
        return self

    def _z(self, df):
        return (df[self.columns].to_numpy(float) - self.mean_) / self.scale_

    def predict(self, df):
        return self.model_.predict_proba(self._z(df))[:, 1]

    def contributions(self, df):
        """Log-odds each input adds to this applicant's default risk compared with the average applicant (positive pushes PD up)."""
        return self._z(df) * self.model_.coef_[0]


class LgbmPD:
    """Gradient boosting with optional monotone constraints. Categorical columns keep the training categories."""

    kind = "lightgbm"

    def __init__(self, columns, params, class_weight=None, monotone=True, seed=42):
        self.columns, self.params, self.class_weight, self.monotone, self.seed = list(columns), dict(params), class_weight, monotone, seed

    def _matrix(self, df):
        X = df[self.columns].copy()
        for c in self.categories_:
            X[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return X

    def fit(self, df, y):
        self.categories_ = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.columns if c in CATEGORICAL}
        df = df.assign(**{c: df[c].astype(str) for c in self.categories_})
        self.model_ = LGBMClassifier(objective="binary", monotone_constraints=monotone_vector(self.columns, self.monotone), random_state=self.seed,
                                     n_jobs=2, deterministic=True, force_row_wise=True, subsample_freq=1, verbosity=-1,
                                     class_weight=self.class_weight, **self.params).fit(self._matrix(df), y)
        return self

    def _prep(self, df):
        return self._matrix(df.assign(**{c: df[c].astype(str) for c in self.categories_}))

    def predict(self, df):
        return self.model_.predict_proba(self._prep(df))[:, 1]

    def contributions(self, df):
        """Exact tree SHAP values in log-odds, one column per input (the last column of the booster output is the base rate and is dropped)."""
        return self.model_.booster_.predict(self._prep(df), pred_contrib=True)[:, :-1]


class Calibrator:
    """Maps a model's score to a probability. Platt scaling (logistic on the log-odds) or isotonic regression."""

    def __init__(self, method="platt"):
        self.method = method

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y)
        else:
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(np.asarray(p, float), y)
        return self

    def predict(self, p):
        if self.method == "platt":
            return self.model_.predict_proba(self._logit(p))[:, 1]
        return self.model_.predict(np.asarray(p, float))


class PerTermCalibrator:
    """One calibrator per loan term (36 and 60 months behave like different products). Terms never seen use the pooled calibrator."""

    per_term = True

    def __init__(self, methods, pooled_method="platt"):
        self.methods, self.pooled_method = dict(methods), pooled_method

    def fit(self, p, y, term):
        p, y, term = np.asarray(p, float), np.asarray(y), np.asarray(term)
        self.pooled_ = Calibrator(self.pooled_method).fit(p, y)
        self.by_term_ = {t: Calibrator(m).fit(p[term == t], y[term == t]) for t, m in self.methods.items()}
        return self

    def predict(self, p, term):
        p, term = np.asarray(p, float), np.asarray(term)
        out = self.pooled_.predict(p)
        for t, cal in self.by_term_.items():
            m = term == t
            if m.any():
                out[m] = cal.predict(p[m])
        return out


class PDModel:
    """What is saved: the model, its calibrator, the decision thresholds and the metadata needed to trace it."""

    def __init__(self, model, calibrator, thresholds, metadata, reason_map=None, reference=None):
        self.model, self.calibrator, self.thresholds, self.metadata, self.reason_map = model, calibrator, thresholds, metadata, reason_map or {}
        self.reference = reference or {}  # the typical (median) applicant, shown next to the applicant's own value in a reason

    @property
    def contract_version(self):
        return self.metadata["contract_version"]

    def score(self, frame):
        raw = self.model.predict(frame)
        if getattr(self.calibrator, "per_term", False):
            return self.calibrator.predict(raw, frame["term"].to_numpy())
        return self.calibrator.predict(raw)
