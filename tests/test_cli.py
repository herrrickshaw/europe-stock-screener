"""End-to-end: run_scan.main on a fake downloader; CSV outputs are pinned."""
import hashlib
import sys

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
    digests = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()[:16]
               for n in ('darvas_buffett_scan.csv', 'piotroski_scan.csv')}
    assert digests == EXPECTED_DIGESTS


def test_load_universe_filters_and_limits():
    df = run_scan.load_universe('SIX Swiss,Vienna (ATX)', 5)
    assert len(df) == 5 and set(df['exchange']) <= {'SIX Swiss', 'Vienna (ATX)'}


EXPECTED_DIGESTS = {'darvas_buffett_scan.csv': '6ee2d7326e3e9c9f',
                    'piotroski_scan.csv': '4660614f7571f00a'}
