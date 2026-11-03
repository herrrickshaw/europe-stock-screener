"""End-to-end: run_scan.main on a fake downloader; CSV outputs are pinned to golden files
(numeric columns compared to the CSV rounding step, as float last digits differ across CPUs)."""
import sys
from pathlib import Path

import pandas as pd
from synthetic import make_download_any

import run_scan
from screener import fetch


def _run(monkeypatch, tmp_path, capsys, *argv):
    monkeypatch.setattr(fetch.yf, 'download', make_download_any)
    monkeypatch.setattr(sys, 'argv', ['run_scan.py', '--out', str(tmp_path), *argv])
    run_scan.main()
    return capsys.readouterr().out


def test_cli_writes_reports_for_one_exchange(monkeypatch, tmp_path, capsys):
    out = _run(monkeypatch, tmp_path, capsys, '--exchange', 'SIX Swiss')
    assert 'Universe: 20 stocks across 1 exchange(s)' in out
    assert 'Fetched 20/20 tickers' in out
    for name in ('darvas_buffett_scan.csv', 'piotroski_scan.csv'):
        lines = (tmp_path / name).read_text().splitlines()
        assert len(lines) == 21
        assert lines[0].startswith('ticker,name,exchange,cmp')
    for name in ('darvas_buffett_scan.csv', 'piotroski_scan.csv'):
        got = pd.read_csv(tmp_path / name)
        want = pd.read_csv(GOLDEN / name)
        pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=0, atol=0.011)


def test_load_universe_filters_and_limits():
    df = run_scan.load_universe('SIX Swiss,Vienna (ATX)', 5)
    assert len(df) == 5 and set(df['exchange']) <= {'SIX Swiss', 'Vienna (ATX)'}


GOLDEN = Path(__file__).parent / 'golden'
