from credit_risk.reporting import fresh_dirs


def test_fresh_dirs_creates_layout_and_removes_stale_files(tmp_path):
    figures, tables = fresh_dirs("demo", root=tmp_path)
    (figures / "old_chart.png").write_text("x")
    figures, tables = fresh_dirs("demo", root=tmp_path)
    assert figures == tmp_path / "demo" / "figures" and tables.is_dir()
    assert list(figures.iterdir()) == []
