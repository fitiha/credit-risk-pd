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


from credit_risk.categorical import CategoryGrouper


def _loans():
    n = 600
    return pd.DataFrame({
        "issue_d": pd.Timestamp("2014-03-01"),
        "emp_title": ["teacher"] * 300 + [None] * 300, "emp_length": ["10+ years"] * 300 + ["< 1 year"] * 300,
        "purpose": ["debt"] * 500 + ["car"] * 95 + ["wedding"] * 5, "home_ownership": ["RENT"] * 600,
        "addr_state": ["CA"] * 590 + ["VT"] * 10, "application_type": ["Individual"] * 600,
    })


def test_category_grouper_adds_group_columns_and_merges_rare_levels():
    df = _loans()
    g = CategoryGrouper(min_share=0.02).fit(df)  # 2% of 600 = 12 loans
    out = g.transform(df)
    assert set(out.job_group) == {"education", "missing"}
    assert out.emp_length_years.iloc[0] == 10 and out.emp_length_years.iloc[-1] == 0
    assert set(out.purpose_group) == {"debt", "car", "other_purpose"}
    assert out.addr_state_group.iloc[-1] == "other_addr_state"
    assert out.purpose_group.iloc[-1] == "other_purpose"


def test_category_grouper_ignores_2015_loans_and_maps_unseen_levels_to_other():
    df = _loans()
    later = df.assign(issue_d=pd.Timestamp("2015-06-01"), purpose="wedding")  # a 2015 surge must not keep 'wedding'
    g = CategoryGrouper(min_share=0.02).fit(pd.concat([df, later], ignore_index=True))
    assert "wedding" not in g.mergers_["purpose"].kept_
    new = df.head(2).assign(purpose="brand_new")
    assert g.transform(new).purpose_group.tolist() == ["other_purpose"] * 2


def test_loans_before_october_2013_get_their_own_job_group():
    df = _loans().assign(issue_d=pd.to_datetime(["2013-05-01"] * 300 + ["2013-11-01"] * 300),
                         emp_title=["teacher"] * 450 + [None] * 150)
    out = CategoryGrouper(min_share=0.02).fit(df).transform(df)
    assert (out.job_group.iloc[:300] == "employer_name_era").all()  # before October 2013, even though the text says "teacher"
    assert (out.job_group.iloc[300:450] == "education").all()
    assert (out.job_group.iloc[450:] == "missing").all()
