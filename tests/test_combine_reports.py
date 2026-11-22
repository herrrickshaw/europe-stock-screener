"""combine_reports.main(): merge, dedupe, sort and summarise scan CSVs."""
import runpy
import sys

import pandas as pd
import pytest

import combine_reports


def _write(d, fname, rows):
    d.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(d / fname, index=False)


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["combine_reports.py", *argv])
    combine_reports.main()


@pytest.fixture
def dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "darvas_buffett_scan.csv", [
        {"ticker": "AAA", "score": 5, "signal": "BUY", "exchange": "LSE"},
        {"ticker": "BBB", "score": 9, "signal": "BUY", "exchange": "LSE"},
        {"ticker": "CCC", "score": 1, "signal": "AVOID", "exchange": "Xetra"}])
    _write(b, "darvas_buffett_scan.csv", [
        {"ticker": "AAA", "score": 99, "signal": "AVOID", "exchange": "Other"},   # duplicate, loses
        {"ticker": "DDD", "score": 3, "signal": "WATCH", "exchange": "Vienna"}])
    return a, b


def test_merge_dedupes_keeps_first_sorts_desc(dirs, tmp_path, monkeypatch, capsys):
    a, b = dirs
    out = tmp_path / "out"
    _run(monkeypatch, str(a), str(b), "--out", str(out))
    m = pd.read_csv(out / "darvas_buffett_scan.csv")
    assert list(m["ticker"]) == ["BBB", "AAA", "DDD", "CCC"]
    assert list(m["score"]) == [9, 5, 3, 1]
    assert m.set_index("ticker").loc["AAA", "exchange"] == "LSE"     # first dir wins
    assert list(m.index) == [0, 1, 2, 3]
    assert list(m.columns) == ["ticker", "score", "signal", "exchange"]    # no stray index column
    text = capsys.readouterr().out
    assert "darvas_buffett_scan.csv: dropped 1 duplicate ticker(s)" in text
    assert "darvas_buffett_scan.csv: 4 rows ->" in text
    assert "BUY=2 WATCH=1 AVOID=1" in text
    assert "exchanges: 3" in text
    # piotroski file exists in neither dir
    assert f"skip: {a / 'piotroski_scan.csv'} not found" in text
    assert f"skip: {b / 'piotroski_scan.csv'} not found" in text
    assert not (out / "piotroski_scan.csv").exists()


def test_order_of_dirs_decides_which_duplicate_survives(dirs, tmp_path, monkeypatch):
    a, b = dirs
    out = tmp_path / "out"
    _run(monkeypatch, str(b), str(a), "--out", str(out))
    m = pd.read_csv(out / "darvas_buffett_scan.csv").set_index("ticker")
    assert m.loc["AAA", "score"] == 99 and m.loc["AAA", "exchange"] == "Other"
    assert list(pd.read_csv(out / "darvas_buffett_scan.csv")["ticker"])[0] == "AAA"


def test_no_duplicates_prints_no_drop_message(tmp_path, monkeypatch, capsys):
    a = tmp_path / "a"
    _write(a, "piotroski_scan.csv", [{"ticker": "X", "score": 1, "signal": "WATCH", "exchange": "E"},
                                     {"ticker": "Y", "score": 2, "signal": "BUY", "exchange": "E"}])
    _run(monkeypatch, str(a), "--out", str(tmp_path / "o"))
    text = capsys.readouterr().out
    assert "dropped" not in text
    assert "piotroski_scan.csv: 2 rows" in text
    assert "BUY=1 WATCH=1 AVOID=0" in text and "exchanges: 1" in text
    assert list(pd.read_csv(tmp_path / "o" / "piotroski_scan.csv")["ticker"]) == ["Y", "X"]


def test_both_files_and_single_dir(tmp_path, monkeypatch):
    a = tmp_path / "a"
    row = [{"ticker": "X", "score": 1, "signal": "BUY", "exchange": "E"}]
    _write(a, "darvas_buffett_scan.csv", row)
    _write(a, "piotroski_scan.csv", row)
    _run(monkeypatch, str(a), "--out", str(tmp_path / "o"))
    assert (tmp_path / "o" / "darvas_buffett_scan.csv").exists()
    assert (tmp_path / "o" / "piotroski_scan.csv").exists()


def test_no_inputs_writes_nothing_but_creates_out_dir(tmp_path, monkeypatch, capsys):
    out = tmp_path / "o"
    _run(monkeypatch, str(tmp_path / "missing"), "--out", str(out))
    assert out.is_dir() and list(out.iterdir()) == []
    assert capsys.readouterr().out.count("not found") == 2


def test_existing_out_dir_is_fine_and_default_out_name(tmp_path, monkeypatch):
    a = tmp_path / "a"
    _write(a, "darvas_buffett_scan.csv", [{"ticker": "X", "score": 1, "signal": "BUY", "exchange": "E"}])
    (tmp_path / "reports_combined").mkdir()
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, str(a))                                     # default --out
    assert (tmp_path / "reports_combined" / "darvas_buffett_scan.csv").exists()


def test_files_constant():
    assert combine_reports.FILES == ["darvas_buffett_scan.csv", "piotroski_scan.csv"]


def test_script_entrypoint(tmp_path, monkeypatch):
    a = tmp_path / "a"
    _write(a, "piotroski_scan.csv", [{"ticker": "X", "score": 1, "signal": "BUY", "exchange": "E"}])
    monkeypatch.setattr(sys, "argv", ["combine_reports.py", str(a), "--out", str(tmp_path / "o")])
    runpy.run_path(combine_reports.__file__, run_name="__main__")
    assert (tmp_path / "o" / "piotroski_scan.csv").exists()
