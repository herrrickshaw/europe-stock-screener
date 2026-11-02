"""Characterisation tests: scanner output on a synthetic panel must not drift."""
import json
from pathlib import Path

from capture import build_rows
from screener.scanner import scan_darvas, scan_piotroski

GOLDEN = Path(__file__).parent / 'golden' / 'scan_outputs.json'


def _roundtrip(rows):
    return json.loads(json.dumps(rows, sort_keys=True, default=str))


def test_darvas_matches_golden():
    assert _roundtrip(scan_darvas(build_rows())) == json.loads(GOLDEN.read_text())['darvas']


def test_piotroski_matches_golden():
    assert _roundtrip(scan_piotroski(build_rows())) == json.loads(GOLDEN.read_text())['piotroski']


def test_empty_frame_returns_empty_list():
    import pandas as pd
    assert scan_darvas(pd.DataFrame()) == []
    assert scan_piotroski(pd.DataFrame()) == []


def test_signal_thresholds():
    rows = {r['ticker']: r for r in scan_darvas(build_rows())}
    assert rows['BBB.DE']['signal'] == 'AVOID'
    assert rows['HHH.MC']['signal'] == 'BUY'
    assert all(r['score'] <= r['max_score'] for r in rows.values())
