"""Output folders for analysis scripts: reports/<script_name>/{figures,tables}."""
import shutil
from pathlib import Path


def fresh_dirs(name, root="reports"):
    """Empty and recreate reports/<name>/figures and /tables; return both paths.

    A script owns its folder entirely, so files from an earlier run that the
    script no longer writes cannot be left behind.
    """
    base = Path(root) / name
    if base.exists():
        shutil.rmtree(base)
    figures, tables = base / "figures", base / "tables"
    figures.mkdir(parents=True)
    tables.mkdir()
    return figures, tables
