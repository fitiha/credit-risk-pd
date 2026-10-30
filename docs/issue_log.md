# Data issue log

One row per issue found in the audit. Decision is one of **fix**, **flag** or **drop**.
Row counts refer to the raw files in `data/` (applications 30,600; bureau 30,000; servicing 30,000).

| # | File | Issue | Rows affected | Decision | Reason |
|---|------|-------|---------------|----------|--------|
| 1 | applications | Duplicate `id`: 600 ids appear twice (1,200 rows, 600 surplus). Copies agree on every field and on `defaulted`, except `dti` (differs by 0.01 in all 600 pairs, rounding noise) and `emp_title` (528 pairs differ only in letter case). | 600 surplus rows | **drop** (keep first occurrence per `id`) | A loan is one row per id; the label is identical across copies, so no information is lost. The `dti` gap is below rounding precision. Fix applied before any other step, so every later count is on 30,000 unique loans. |
| 2 | applications | `emp_title` inconsistent casing (e.g. `SERGEANT` vs `Sergeant`) | to be counted | **fix** (normalise case and whitespace, then group) | Same job title must map to one value before grouping. Counted in the audit. |
| 3 | applications | `issue_d` in mixed formats (e.g. `2013-02-01` and `Feb-2015`) | to be counted | **fix** (parse both to a date) | The time split at 2014-12-31 depends on a correct date. Counted in the audit. |
