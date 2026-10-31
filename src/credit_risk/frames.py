"""Small DataFrame helper: add many columns in one operation (much faster than one assignment per column)."""
import pandas as pd


def add_columns(df, new):
    """Return df with the columns in `new` (dict name -> array or Series) added; existing columns of the same name are replaced."""
    if not new:
        return df
    base = df.drop(columns=[c for c in new if c in df.columns])
    return pd.concat([base, pd.DataFrame(new, index=df.index)], axis=1)
