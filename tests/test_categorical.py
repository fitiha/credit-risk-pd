import numpy as np
import pytest

from credit_risk.categorical import JOB_GROUPS, group_job_titles, job_group


@pytest.mark.parametrize("title,group", [
    ("registered nurse", "healthcare"), ("rn", "healthcare"), ("teacher", "education"),
    ("sales manager", "manager_supervisor"), ("truck driver", "trades_transport_labor"),
    ("software engineer", "professional_technical"), ("police officer", "public_safety_government"),
    ("owner", "executive_owner"), ("vice president of sales", "executive_owner"),
    ("medical assistant", "healthcare"), ("executive assistant", "sales_service_office"),
    ("gardener", "other_job"),
])
def test_job_groups(title, group):
    assert job_group(title) == group


def test_missing_title_is_its_own_group():
    out = group_job_titles(["teacher", None, np.nan]).tolist()
    assert out == ["education", "missing", "missing"]


def test_every_result_is_a_known_group():
    assert set(group_job_titles(["x", "manager", None])) <= set(JOB_GROUPS)


import pandas as pd

from credit_risk.categorical import RareLevelMerger, order_emp_length


def test_rare_levels_merge_and_unseen_levels_fall_into_other():
    vals = ["car"] * 60 + ["debt"] * 100 + ["wedding"] * 2 + ["moving"] * 3
    m = RareLevelMerger("purpose", min_share=0.05).fit(vals)  # 5% of 165 loans = 9
    assert m.kept_ == ["car", "debt"] and m.merged_ == ["moving", "wedding"]
    out = m.transform(["car", "wedding", "brand_new", None])
    assert out.iloc[:3].tolist() == ["car", "other_purpose", "other_purpose"]
    assert pd.isna(out.iloc[3])  # missing stays missing


def test_min_group_size_is_reported():
    m = RareLevelMerger("x", min_share=0.01).fit(["a"] * 150 + ["b"] * 50)
    assert m.min_loans_ == 2 and m.spec_entry()["kept_levels"] == ["a", "b"]


def test_emp_length_is_ordered_and_missing_stays_missing():
    out = order_emp_length(["< 1 year", "10+ years", "3 years", None])
    assert out.iloc[:3].tolist() == [0, 10, 3] and pd.isna(out.iloc[3])
