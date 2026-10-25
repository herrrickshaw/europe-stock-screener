"""compute_technicals(): hand-computed indicator values on small synthetic series."""
import math

import pandas as pd
import pytest

from screener.technicals import QuoteData, compute_technicals


def _ramp(n):
    return pd.Series([float(i) for i in range(1, n + 1)])


def test_too_short_or_empty_returns_empty():
    assert compute_technicals(pd.Series([], dtype=float)) == {}
    assert compute_technicals(pd.Series([5.0, float("nan")])) == {}


def test_ramp_30_bars_hand_computed():
    t = compute_technicals(_ramp(30))
    assert t["cmp"] == 30
    assert t["rsi"] == 100.0                       # only gains -> avg loss 0
    assert t["high_52w"] == 30 and t["low_52w"] == 1
    # Bollinger on closes 11..30: mean 20.5, sample std sqrt(35)
    sd = math.sqrt(35)
    assert t["bb_upper"] == pytest.approx(round(20.5 + 2 * sd, 2))
    assert t["bb_lower"] == pytest.approx(round(20.5 - 2 * sd, 2))
    assert t["bb_pct"] == pytest.approx((30 - t["bb_lower"]) / (t["bb_upper"] - t["bb_lower"]), abs=1e-3)
    assert t["ret_1d"] == pytest.approx(round(1 / 29 * 100, 2))
    assert t["ret_1w"] == pytest.approx(20.0)      # 25 -> 30
    assert t["ret_1m"] == pytest.approx(round(21 / 9 * 100, 2))   # 9 -> 30
    assert t["ret_3m"] is None and t["ret_6m"] is None
    assert t["ret_1y"] == pytest.approx(2900.0)    # 1 -> 30
    assert t["macd"] is not None and t["macd_signal"] is None      # 26 <= n < 35


def test_constant_series_emas_equal_price_and_no_bands_pct():
    t = compute_technicals(pd.Series([50.0] * 40))
    assert t["ema_20"] == t["ema_50"] == 50.0
    assert t["bb_upper"] == t["bb_lower"] == 50.0 and t["bb_pct"] is None   # zero-width band
    assert t["macd"] == 0.0


def test_rsi_all_losses_is_zero():
    t = compute_technicals(_ramp(30)[::-1].reset_index(drop=True))
    assert t["rsi"] == 0.0


def test_volume_ratio_uses_last_20():
    v = pd.Series([100.0] * 19 + [300.0])
    t = compute_technicals(_ramp(20), volumes=v)
    assert t["volume"] == 300 and t["volume_20d_avg"] == 110
    assert t["volume_ratio"] == pytest.approx(2.73)


def test_atr_and_stochastic_on_steady_ramp():
    c = _ramp(40)
    t = compute_technicals(c, highs=c + 1, lows=c - 1)
    # true range = max(2, |h - prev c| = 2, |l - prev c| = 0) = 2 every bar
    assert t["atr_14"] == pytest.approx(2.0)
    # %K = (c - lowest low of 14) / (highest high - lowest low) = 14 / 15
    assert t["stoch_k"] == pytest.approx(93.33) and t["stoch_d"] == pytest.approx(93.33)


def test_ema200_not_reported_without_200_bars():
    """Prove-it: ema_200 was reported from 50 bars on. An adjust=False EMA seeded on
    the first close with 120 bars still puts (1-2/201)^119 = 30% of its weight on
    that first close, so 'CMP > EMA-200' (D8) compared price to a stale seed."""
    t = compute_technicals(_ramp(120))
    assert t["ema_200"] is None
    assert compute_technicals(_ramp(200))["ema_200"] is not None


def test_quote_to_dict_drops_none():
    q = QuoteData(ticker="SAP.DE", source="yf", cmp=100.0)
    assert q.to_dict() == {"ticker": "SAP.DE", "source": "yf", "cmp": 100.0, "rsi_signal": "HOLD"}
