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


def test_fetch_bulk_accepts_injected_downloader():
    quotes = fetch.fetch_bulk(make_universe(), download=make_download, sleep=lambda s: None)
    assert set(quotes) == {s[0] for s in SPECS}


def test_fetch_bulk_rate_limit_backs_off_then_gives_up():
    sleeps, calls = [], []

    def limited(batch, **kw):
        calls.append(1)
        raise RuntimeError('429 Too Many Requests')

    assert fetch.fetch_bulk(make_universe(), download=limited, sleep=sleeps.append) == {}
    assert sleeps == [90, 180, 300] and len(calls) == 4


def test_fetch_bulk_non_rate_error_does_not_retry():
    sleeps = []

    def broken(batch, **kw):
        raise ValueError('boom')

    assert fetch.fetch_bulk(make_universe(), download=broken, sleep=sleeps.append) == {}
    assert sleeps == []
