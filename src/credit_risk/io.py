"""Load the raw CSV files and parse formats. No cleaning decisions happen here."""
import warnings
from pathlib import Path

import pandas as pd

FIT_CUTOFF = pd.Timestamp("2014-12-31")

# The raw files mix three date styles: "Feb-2015", "2015-02-01" and "02/2015".
_DATE_FORMATS = ("%b-%Y", "%Y-%m-%d", "%m/%Y")


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse a column of mixed-format dates into datetimes (NaT if unreadable)."""
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for fmt in _DATE_FORMATS:
        todo = out.isna() & values.notna()
        if not todo.any():
            break
        out[todo] = pd.to_datetime(values[todo], format=fmt, errors="coerce")
    return out


def load_applications(data_dir: str | Path = "data") -> pd.DataFrame:
    """Applications joined with bureau data, one row per loan id.

    Duplicate ids in applications.csv are dropped, keeping the first row
    (issue 1 in docs/issue_log.md). loan_servicing.csv is never loaded here:
    it is recorded after the loan was issued and must not reach the model.
    """
    data_dir = Path(data_dir)
    apps = pd.read_csv(data_dir / "applications.csv", low_memory=False)
    apps = apps.drop_duplicates("id", keep="first")
    bureau = pd.read_csv(data_dir / "credit_bureau.csv", low_memory=False)
    return apps.merge(bureau, on="id", how="left", validate="one_to_one")


PERCENT_COLUMNS = ("revol_util", "bc_util")
MIN_ROWS_FOR_UNIT_CHECK = 100
FRACTION_MEDIAN_BELOW = 1.5  # a percent column in training has a median near 55-65; a fraction column has one below 1


def percent_columns_stored_as_fractions(df: pd.DataFrame) -> list[str]:
    """Columns that hold percentages in training but look like fractions (0.91 for 91%) in this batch.

    Decided on the whole batch, never per row: a single 0.9 could be a real 0.9% utilisation.
    """
    if len(df) < MIN_ROWS_FOR_UNIT_CHECK:
        return []
    return [c for c in PERCENT_COLUMNS if c in df and df[c].notna().any() and df[c].median() < FRACTION_MEDIAN_BELOW]


def load_holdout(data_dir: str | Path = "data") -> pd.DataFrame:
    """The 2016 holdout. Its `revol_util` is stored as a fraction (all rows), so it is converted to percent here."""
    df = pd.read_csv(Path(data_dir) / "holdout_applications.csv", low_memory=False)
    fractions = percent_columns_stored_as_fractions(df)
    for c in fractions:
        df[c] = df[c] * 100
    if fractions:
        warnings.warn(f"holdout columns stored as fractions were multiplied by 100: {fractions}", stacklevel=2)
    return df


def load_servicing(data_dir: str | Path = "data") -> pd.DataFrame:
    """For auditing the label only. Never feed this to the model."""
    return pd.read_csv(Path(data_dir) / "loan_servicing.csv", low_memory=False)
