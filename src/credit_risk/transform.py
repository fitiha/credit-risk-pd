"""The data transform handed to the ML Engineer: raw application + bureau record -> cleaned, treated fields.

One object chains the fitted steps in order. Every step learns only from loans issued up to the fit cutoff,
so applying it to later loans (or to a single new application) never changes what was learned.
"""
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.categorical import CategoryGrouper
from credit_risk.cleaning import Cleaner
from credit_risk.factors import FactorModel
from credit_risk.features import FeatureBuilder, FeatureTreater
from credit_risk.outliers import AmountTreater, CountTreater, RatioTreater


class ApplicationTransformer(BaseEstimator, TransformerMixin):
    """Cleaning, outlier treatment, application-time features, category grouping, PCA components and factor scores."""

    def _new_steps(self):
        return [("cleaner", Cleaner()), ("amounts", AmountTreater()), ("ratios", RatioTreater()),
                ("counts", CountTreater()), ("features", FeatureBuilder()), ("feature_treater", FeatureTreater()),
                ("categories", CategoryGrouper()), ("factors", FactorModel())]

    def fit(self, X, y=None):
        self.steps_ = self._new_steps()
        d = X
        for _, step in self.steps_:
            d = step.fit(d).transform(d)
        return self

    def transform(self, X):
        d = X
        for _, step in self.steps_:
            d = step.transform(d)
        return d

    def step(self, name):
        return dict(self.steps_)[name]
