"""Grouping of categorical fields before WoE."""
import re

import numpy as np

import pandas as pd

# Job-title groups. A title gets the first group whose keywords it contains, so the order matters:
# "sales manager" is a manager, "medical assistant" is healthcare, "security officer" is public safety.
JOB_RULES = [
    ("executive_owner", r"owner|president|ceo|cfo|coo|cto|chief|vice president|\bvp\b|partner|founder|proprietor"),
    ("healthcare", r"nurse|nursing|\brn\b|\blpn\b|\bcna\b|medical|health|hospital|physician|doctor|dental|dentist|pharmac|therapist|clinic|\bcare\b|paramedic|\bemt\b|surgeon|caregiver|social worker"),
    ("education", r"teacher|teaching|professor|school|instructor|university|college|educat|counselor|principal"),
    ("public_safety_government", r"police|sheriff|deputy|correction|fire ?fighter|trooper|detective|military|army|navy|air force|marine|usps|postal|government|federal|county|city of|security|department|\bdept\b|sergeant|carrier|investigator"),
    ("professional_technical", r"engineer|analyst|accountant|attorney|lawyer|developer|programmer|software|\bit\b|consultant|scientist|architect|designer|auditor|controller|actuary|economist|researcher|accounting|bookkeeper|underwrit|paralegal|planner|adjuster"),
    ("trades_transport_labor", r"driver|truck|operator|technician|\btech\b|mechanic|maintenance|machinist|welder|electrician|plumber|warehouse|production|construction|installer|laborer|delivery|pilot|carpenter|assembler|\bcook\b|\bchef\b|janitor|custodian|manufactur|forklift|handler|loader"),
    ("manager_supervisor", r"manager|\bmgr\b|manage|director|supervisor|\blead\b|foreman|superintendent|\bhead\b|coordinator|administrator"),
    ("sales_service_office", r"sales|representative|\brep\b|agent|clerk|customer|service|assistant|secretary|receptionist|teller|cashier|associate|support|advisor|broker|\badmin|office|banker|\bbank\b|specialist|server|clerical|retail|bartender|attendant"),
]
_COMPILED = [(name, re.compile(pattern)) for name, pattern in JOB_RULES]
JOB_GROUPS = [name for name, _ in JOB_RULES] + ["other_job", "missing", "employer_name_era"]
# Until September 2013 the field mostly holds employer names ("state farm insurance", "at&t"), not job titles;
# from October 2013 it holds job titles. Earlier loans get their own group instead of being mixed in.
JOB_TITLE_ERA_START = pd.Timestamp("2013-10-01")


def job_group(title):
    """Map one cleaned (lower-case) job title to a job group."""
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return "missing"
    for name, pattern in _COMPILED:
        if pattern.search(title):
            return name
    return "other_job"


def group_job_titles(titles):
    return pd.Series(titles).map(job_group)


class RareLevelMerger:
    """Merge levels with fewer than `min_share` of the fit-window loans into one `other_<field>` level.

    Levels that were never seen when fitting (a new state, for example) also map to `other_<field>`
    when scoring. Missing values stay missing.
    """

    def __init__(self, field, min_share=0.01):
        self.field = field
        self.min_share = min_share

    @property
    def other_label(self):
        return f"other_{self.field}"

    def fit(self, values):
        values = pd.Series(values).dropna()
        counts = values.value_counts()
        self.min_loans_ = int(np.ceil(self.min_share * len(pd.Series(values))))
        self.kept_ = sorted(counts[counts >= self.min_loans_].index)
        self.merged_ = sorted(set(counts.index) - set(self.kept_))
        return self

    def transform(self, values):
        values = pd.Series(values)
        out = values.where(values.isin(self.kept_), self.other_label)
        return out.where(values.notna())

    def spec_entry(self):
        return {"min_share": self.min_share, "min_loans_in_fit_window": self.min_loans_, "kept_levels": self.kept_,
                "merged_levels": self.merged_, "other_label": self.other_label}


EMP_LENGTH_ORDER = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5, "6 years": 6,
                    "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}


def order_emp_length(values):
    """Employment length as a number from 0 (under 1 year) to 10 (10+ years); missing stays missing."""
    return pd.Series(values).map(EMP_LENGTH_ORDER)


from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.frames import add_columns
from credit_risk.io import FIT_CUTOFF

MERGED_FIELDS = ["purpose", "home_ownership", "addr_state", "application_type"]


class CategoryGrouper(BaseEstimator, TransformerMixin):
    """Adds the grouped categorical columns that WoE is computed on.

    `job_group` (keyword rules), `emp_length_years` (ordered), and `<field>_group` for purpose, home_ownership,
    addr_state and application_type, where levels under `min_share` of the fit-window loans are merged.
    """

    def __init__(self, min_share=0.01):
        self.min_share = min_share

    def fit(self, X, y=None):
        pre = X[X["issue_d"] <= FIT_CUTOFF]
        self.mergers_ = {f: RareLevelMerger(f, self.min_share).fit(pre[f]) for f in MERGED_FIELDS}
        return self

    def transform(self, X):
        job = group_job_titles(X["emp_title"])
        job[(X["issue_d"] < JOB_TITLE_ERA_START) & X["emp_title"].notna()] = "employer_name_era"
        new = {"job_group": job.to_numpy(), "emp_length_years": order_emp_length(X["emp_length"]).to_numpy()}
        for f, m in self.mergers_.items():
            new[f"{f}_group"] = m.transform(X[f]).to_numpy()
        return add_columns(X, new)

    def spec_entries(self):
        return {f: m.spec_entry() for f, m in self.mergers_.items()}
