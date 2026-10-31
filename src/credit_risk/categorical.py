"""Grouping of categorical fields before WoE."""
import re

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
JOB_GROUPS = [name for name, _ in JOB_RULES] + ["other_job", "missing"]


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
