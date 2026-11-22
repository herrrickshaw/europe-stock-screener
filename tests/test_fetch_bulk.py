"""fetch_bulk() batching, retry/back-off and per-ticker isolation with yfinance stubbed."""
import numpy as np
import pandas as pd
import pytest

import screener.fetch as fetch
from screener.technicals import QuoteData

N = 30
IDX = pd.RangeIndex(N)


def _ohlcv(base=100.0, step=1.0):
    close = pd.Series([base + step * i for i in range(N)], index=IDX)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": [1000.0 + i for i in range(N)]})


def _multi(tickers, frames=None):
    frames = frames or {}
    return pd.concat({t: frames.get(t, _ohlcv()) for t in tickers}, axis=1)


def _universe(tickers):
    return pd.DataFrame({"yf_ticker": tickers,
                         "name": [f"{t} Corp" for t in tickers],
                         "exchange": [f"EX-{t}" for t in tickers]})


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch.time, "sleep", calls.append)
    return calls


@pytest.fixture
def download(monkeypatch):
    """Install a scripted yf.download; returns the recorded calls."""
    calls = []

    def install(responder):
        def fake(batch, **kw):
            calls.append((list(batch), kw))
            return responder(list(batch), len(calls))
        monkeypatch.setattr(fetch.yf, "download", fake)
        return calls
    return install


# ---------------------------------------------------------------- helpers

def test_chunks_splits_and_keeps_remainder():
    assert list(fetch._chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(fetch._chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    assert list(fetch._chunks([], 3)) == []
    assert list(fetch._chunks([1], 5)) == [[1]]


def test_safe_float_and_pct():
    assert fetch._safe_float("1.5") == 1.5
    assert fetch._safe_float(0) == 0.0
    assert fetch._safe_float(None) is None
    assert fetch._safe_float("abc") is None
    assert fetch._safe_float(float("nan")) is None
    assert fetch._pct(0.12345) == 12.35          # round to 2 dp after x100
    assert fetch._pct(0) == 0.0                  # zero is a value, not "missing"
    assert fetch._pct(None) is None
    assert fetch._pct("x") is None


def test_defaults():
    assert fetch._BATCH_SIZE == 75
    assert fetch._RATE_BACKOFFS == [90, 180, 300]


# ---------------------------------------------------------------- happy path

def test_multiindex_download_builds_quotes_with_metadata(download, sleeps):
    calls = download(lambda b, n: _multi(b))
    out = fetch.fetch_bulk(_universe(["AAA.DE", "BBB.L"]))
    assert set(out) == {"AAA.DE", "BBB.L"}
    q = out["AAA.DE"]
    assert isinstance(q, QuoteData)
    assert (q.ticker, q.source, q.name, q.exchange) == ("AAA.DE", "yahoo", "AAA.DE Corp", "EX-AAA.DE")
    assert q.cmp == 129.0 and q.volume == 1029 and q.atr_14 is not None
    assert sleeps == []                                  # no back-off on success
    assert len(calls) == 1
    batch, kw = calls[0]
    assert batch == ["AAA.DE", "BBB.L"]
    assert kw == {"period": "1y", "auto_adjust": True, "progress": False,
                  "threads": False, "group_by": "ticker"}


def test_batches_by_batch_size(download, monkeypatch):
    monkeypatch.setattr(fetch, "_BATCH_SIZE", 2)
    calls = download(lambda b, n: _multi(b))
    out = fetch.fetch_bulk(_universe(["A", "B", "C", "D", "E"]))
    assert [c[0] for c in calls] == [["A", "B"], ["C", "D"], ["E"]]
    assert list(out) == ["A", "B", "C", "D", "E"]


def test_ticker_missing_from_download_is_skipped(download):
    download(lambda b, n: _multi(["A"]))
    out = fetch.fetch_bulk(_universe(["A", "GONE"]))
    assert list(out) == ["A"]


def test_short_or_nan_history_is_skipped(download):
    frames = {"ONE": _ohlcv().iloc[:1], "NAN": _ohlcv().assign(Close=np.nan)}
    download(lambda b, n: _multi(b, frames))
    out = fetch.fetch_bulk(_universe(["OK", "ONE", "NAN"]))
    assert list(out) == ["OK"]


def test_two_closes_is_enough(download):
    """boundary: len(closes) < 2 is rejected, == 2 is accepted"""
    frames = {"TWO": _ohlcv().iloc[:2]}
    download(lambda b, n: _multi(b, frames))
    assert "TWO" in fetch.fetch_bulk(_universe(["TWO"]))


def test_zero_price_is_skipped(download):
    frames = {"ZERO": _ohlcv().assign(Close=0.0)}
    download(lambda b, n: _multi(b, frames))
    assert list(fetch.fetch_bulk(_universe(["ZERO", "OK"]))) == ["OK"]


def test_missing_close_column_is_skipped(download):
    frames = {"NOCLOSE": _ohlcv().drop(columns="Close")}
    download(lambda b, n: _multi(b, frames))
    assert list(fetch.fetch_bulk(_universe(["NOCLOSE", "OK"]))) == ["OK"]


def test_missing_volume_high_low_still_yield_a_quote(download):
    frames = {"BARE": _ohlcv()[["Close"]]}
    download(lambda b, n: _multi(b, frames))
    q = fetch.fetch_bulk(_universe(["BARE"]))["BARE"]
    assert q.cmp == 129.0
    assert q.volume is None and q.atr_14 is None and q.stoch_k is None


def test_with_highs_and_lows_atr_is_computed(download):
    download(lambda b, n: _multi(b))
    q = fetch.fetch_bulk(_universe(["A"]))["A"]
    assert q.atr_14 == pytest.approx(2.0) and q.stoch_k is not None


def test_flat_frame_without_multiindex(download):
    download(lambda b, n: _ohlcv())
    q = fetch.fetch_bulk(_universe(["SOLO"]))["SOLO"]
    assert q.cmp == 129.0 and q.volume == 1029 and q.atr_14 == pytest.approx(2.0)
    assert (q.name, q.exchange) == ("SOLO Corp", "EX-SOLO")


def test_flat_frame_without_optional_columns(download):
    download(lambda b, n: _ohlcv()[["Close"]])
    q = fetch.fetch_bulk(_universe(["SOLO"]))["SOLO"]
    assert q.cmp == 129.0 and q.volume is None and q.atr_14 is None


def test_flat_frame_with_too_few_rows(download):
    download(lambda b, n: _ohlcv().iloc[:1])
    assert fetch.fetch_bulk(_universe(["SOLO"])) == {}


def test_per_ticker_error_does_not_abort_batch(download, monkeypatch):
    real = fetch.compute_technicals
    def flaky(closes, *a, **k):
        if closes.iloc[0] == 500.0:
            raise ValueError("boom")
        return real(closes, *a, **k)
    monkeypatch.setattr(fetch, "compute_technicals", flaky)
    download(lambda b, n: _multi(b, {"BAD": _ohlcv(base=500.0)}))
    out = fetch.fetch_bulk(_universe(["BAD", "GOOD"]))
    assert list(out) == ["GOOD"]


def test_empty_universe_makes_no_calls(download):
    calls = download(lambda b, n: _multi(b))
    assert fetch.fetch_bulk(_universe([])) == {}
    assert calls == []


# ---------------------------------------------------------------- failure paths

def test_none_and_empty_downloads_skip_the_batch(download, monkeypatch):
    monkeypatch.setattr(fetch, "_BATCH_SIZE", 1)
    responses = {"A": None, "B": pd.DataFrame(), "C": _multi(["C"])}
    download(lambda b, n: responses[b[0]])
    assert list(fetch.fetch_bulk(_universe(["A", "B", "C"]))) == ["C"]


@pytest.mark.parametrize("msg", ["Too Many Requests: rate limited", "HTTP 429", "RATE LIMIT"])
def test_rate_limit_retries_with_backoff_then_succeeds(download, sleeps, msg):
    def responder(b, n):
        if n < 3:
            raise RuntimeError(msg)
        return _multi(b)
    calls = download(responder)
    out = fetch.fetch_bulk(_universe(["A"]))
    assert list(out) == ["A"]
    assert len(calls) == 3
    assert sleeps == [90, 180]


def test_rate_limit_gives_up_after_all_backoffs(download, sleeps):
    def responder(b, n):
        raise RuntimeError("429 Too Many Requests")
    calls = download(responder)
    assert fetch.fetch_bulk(_universe(["A"])) == {}
    assert len(calls) == 4                      # first try + 3 back-offs
    assert sleeps == [90, 180, 300]


def test_other_errors_do_not_retry_and_next_batch_still_runs(download, sleeps, monkeypatch):
    monkeypatch.setattr(fetch, "_BATCH_SIZE", 1)
    def responder(b, n):
        if b == ["A"]:
            raise ConnectionError("dns failure")
        return _multi(b)
    calls = download(responder)
    out = fetch.fetch_bulk(_universe(["A", "B"]))
    assert list(out) == ["B"]
    assert [c[0] for c in calls] == [["A"], ["B"]]
    assert sleeps == []


def test_success_after_non_final_retry_stops_retrying(download, sleeps):
    def responder(b, n):
        if n == 1:
            raise RuntimeError("rate limit")
        return _multi(b)
    calls = download(responder)
    fetch.fetch_bulk(_universe(["A"]))
    assert len(calls) == 2 and sleeps == [90]


# ---------------------------------------------------------------- fetch_fundamentals extras

@pytest.fixture
def info(monkeypatch):
    import types
    def _set(d):
        monkeypatch.setattr(fetch.yf, "Ticker", lambda t: types.SimpleNamespace(info=d))
        return fetch.fetch_fundamentals("X")
    return _set


def test_fundamentals_full_mapping(info):
    out = info({"trailingPE": 12.5, "priceToBook": 1.8, "returnOnEquity": 0.1234,
                "operatingMargins": 0.2, "marketCap": 1_234_567_890, "debtToEquity": 150.0,
                "beta": 1.1, "currentRatio": 1.7, "revenueGrowth": 0.05, "trailingEps": 3.2,
                "dividendYield": 0.031, "sector": "Tech", "industry": "Software",
                "longName": "Long", "shortName": "Short"})
    assert out == {"pe": 12.5, "pb": 1.8, "roe": 12.34, "opm": 20.0, "market_cap": 1234.57,
                   "debt_to_equity": 1.5, "beta": 1.1, "current_ratio": 1.7,
                   "revenue_growth": 5.0, "eps": 3.2, "dividend_yield": 3.1,
                   "sector": "Tech", "industry": "Software", "name": "Long"}


def test_fundamentals_missing_values_become_none(info):
    out = info({"marketCap": 0, "sector": "", "industry": "", "shortName": "Short",
                "debtToEquity": 0})
    assert out["market_cap"] is None            # falsy cap -> None, not 0.0
    assert out["sector"] is None and out["industry"] is None
    assert out["name"] == "Short"               # falls back to shortName
    assert out["debt_to_equity"] == 0.0         # zero leverage is a value
    assert out["pe"] is None and out["roe"] is None and out["eps"] is None
    assert info({"x": 1})["name"] is None
