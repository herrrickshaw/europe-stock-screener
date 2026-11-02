"""Regenerate tests/golden/scan_outputs.json (run only when behaviour intentionally changes)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import pandas as pd  # noqa: E402

from screener.scanner import scan_darvas, scan_piotroski  # noqa: E402
from screener.technicals import QuoteData, compute_technicals  # noqa: E402
from synthetic import SPECS, make_ohlcv  # noqa: E402

FUNDAMENTALS = {'AAA.L': dict(pe=18.0, pb=2.1, roe=15.0, opm=18.0, debt_to_equity=0.4,
                              current_ratio=1.8, revenue_growth=5.0)}


def build_rows() -> pd.DataFrame:
    rows = []
    for t, ex, n, d, s in SPECS:
        o = make_ohlcv(n, d, s)
        tech = compute_technicals(o['Close'], o['Volume'], o['High'], o['Low'])
        q = QuoteData(ticker=t, source='yahoo', name=f'Name {t}', exchange=ex, **tech)
        for k, v in FUNDAMENTALS.get(t, {}).items():
            setattr(q, k, v)
        rows.append(q.to_dict() | {'ticker': t})
    return pd.DataFrame(rows)


def main() -> None:
    df = build_rows()
    out = {'darvas': scan_darvas(df), 'piotroski': scan_piotroski(df)}
    (ROOT / 'tests' / 'golden' / 'scan_outputs.json').write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str) + '\n')


if __name__ == '__main__':
    main()
