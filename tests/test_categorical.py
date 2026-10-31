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
