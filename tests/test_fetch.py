"""fetch_bulk against a fake yfinance download (no network)."""
from synthetic import SPECS, make_download, make_ohlcv, make_universe

from screener import fetch
from screener.technicals import compute_technicals


def test_fetch_bulk_matches_direct_technicals(monkeypatch):
    monkeypatch.setattr(fetch.yf, 'download', make_download)
    quotes = fetch.fetch_bulk(make_universe())
    assert set(quotes) == {s[0] for s in SPECS}
    t, _ex, n, d, s = SPECS[0]
    o = make_ohlcv(n, d, s)
    expect = compute_technicals(o['Close'], o['Volume'], o['High'], o['Low'])
    assert quotes[t].cmp == expect['cmp'] and quotes[t].rsi == expect['rsi']


def test_fetch_bulk_empty_download_gives_no_quotes(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(fetch.yf, 'download', lambda *a, **k: pd.DataFrame())
    assert fetch.fetch_bulk(make_universe()) == {}
