# Data issue log

One row per issue found in the audit. Decision is one of **fix**, **flag** or **drop**.
Row counts refer to the raw files in `data/` (applications 30,600; bureau 30,000; servicing 30,000).

| # | File | Issue | Rows affected | Decision | Reason |
|---|------|-------|---------------|----------|--------|
| 1 | applications | Duplicate `id`: 600 ids appear twice (1,200 rows, 600 surplus). Copies agree on every field and on `defaulted`, except `dti` (differs by 0.01 in all 600 pairs, rounding noise) and `emp_title` (528 pairs differ only in letter case). | 600 surplus rows | **drop** (keep first occurrence per `id`) | A loan is one row per id; the label is identical across copies, so no information is lost. The `dti` gap is below rounding precision. Fix applied before any other step, so every later count is on 30,000 unique loans. |
| 2 | applications | `emp_title` inconsistent case and whitespace (e.g. `SERGEANT` vs `Sergeant`); 1,379 rows have leading or trailing spaces. 15,657 distinct values fall to 13,484 after lowercase and strip. | ~2,173 distinct values merged; 1,379 padded rows | **fix** (strip, lowercase, then group into job groups in A5) | The same job must map to one value before grouping. Free text is never used raw. |
| 3 | applications | `issue_d` in three formats: `Mon-YYYY` (28,800), `YYYY-MM-DD` (615), `MM/YYYY` (585). | 1,200 rows | **fix** (parse all three to a date) | The time split at 2014-12-31 depends on a correct date. Holdout is all `Mon-YYYY`. |
| 4 | credit_bureau | `earliest_cr_line` in the same three formats: `Mon-YYYY` (28,800), `YYYY-MM-DD` (607), `MM/YYYY` (593). | 1,200 rows | **fix** (parse to a date) | Needed for credit-history length. |
| 5 | applications | `addr_state` non-canonical: 154 distinct values for 49 real codes. 652 full names (`California`, `New York`, 12 states), 2,348 lower-case or padded codes (`ca`, ` TX`). | 3,000 rows | **fix** (strip, uppercase, map full names to codes) | Same state split across several levels would distort WoE and rare-level merging. |
| 6 | applications | `term` stored with a leading space (` 36 months`). | 30,000 rows | **fix** (parse to integer 36 / 60) | Cosmetic but breaks category matching and the spec's valid values. |
| 7 | credit_bureau, holdout | Unlisted column `bureau_recovery_flag`: when 1 (4,207 loans, 14%), `defaulted` is 1 in 100% of cases. In the 2016 holdout it is 0 for all 5,000 rows. | 4,207 rows (flag = 1) | **drop** from every model input | Post-outcome information labelled as a bureau field (target leakage). It separates the classes perfectly in training and carries no signal at scoring time, so a model using it would fail on the holdout. Rated low reliability in the spec. |
