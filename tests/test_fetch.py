"""fetch_fundamentals() field mapping with yfinance mocked (no network)."""
import types

import pytest

import screener.fetch as fetch


@pytest.fixture
def info(monkeypatch):
    def _set(d):
        monkeypatch.setattr(fetch.yf, "Ticker", lambda t: types.SimpleNamespace(info=d))
        return fetch.fetch_fundamentals("SAP.DE")
    return _set


def test_fraction_fields_become_percent(info):
    out = info({"returnOnEquity": 0.183, "operatingMargins": 0.25, "revenueGrowth": -0.02,
                "trailingPE": 21.5, "marketCap": 2.5e11, "longName": "SAP SE"})
    assert out["roe"] == 18.3 and out["opm"] == 25.0 and out["revenue_growth"] == -2.0
    assert out["pe"] == 21.5 and out["market_cap"] == 250000.0 and out["name"] == "SAP SE"


def test_empty_info_and_errors_return_empty(info, monkeypatch):
    assert info({}) == {}
    def boom(t):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(fetch.yf, "Ticker", boom)
    assert fetch.fetch_fundamentals("X") == {}


def test_debt_to_equity_is_a_ratio_not_yahoo_percent(info):
    """Prove-it: Yahoo's debtToEquity is a percentage (45.2 means 0.452x; the sibling
    platform's stock_utils documents the same). The scanner tests it as a ratio
    (D7 'D/E < 1', F5 'D/E < 0.5'), so passing 45.2 through made both criteria
    fail for any company with more than 1% debt/equity."""
    out = info({"debtToEquity": 45.2})
    assert out["debt_to_equity"] == pytest.approx(0.452)
