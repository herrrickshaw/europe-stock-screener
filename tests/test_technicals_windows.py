"""Minimum-history thresholds and window lengths in compute_technicals()."""
import pytest

import pandas as pd

from screener.technicals import compute_technicals


def _ramp(n):
    return pd.Series([float(i) for i in range(1, n + 1)])


def test_atr_needs_15_aligned_bars():
    c = _ramp(15)
    assert compute_technicals(c, highs=c + 1, lows=c - 1)["atr_14"] == pytest.approx(2.0)
    c14 = _ramp(14)
    assert compute_technicals(c14, highs=c14 + 1, lows=c14 - 1)["atr_14"] is None


def test_atr_uses_wilder_smoothing_1_over_14():
    c = _ramp(40)
    h = c + 1
    h.iloc[1] = c.iloc[1] + 3                    # first true range is 4, all others 2
    t = compute_technicals(c, highs=h, lows=c - 1)
    assert t["atr_14"] == pytest.approx(round(2 + 2 * (13 / 14) ** 38, 4))


def test_stochastic_needs_17_bars():
    c = _ramp(17)
    assert compute_technicals(c, highs=c, lows=c)["stoch_k"] is not None
    c16 = _ramp(16)
    assert compute_technicals(c16, highs=c16, lows=c16)["stoch_k"] is None


def test_stochastic_d_is_mean_of_last_three_k_values():
    # flat window -> zero range -> NaN %K is dropped; three valid %K remain: 100, 100, 66.67
    c = pd.Series([10.0] * 14 + [11.0, 13.0, 12.0])
    t = compute_technicals(c, highs=c, lows=c)
    assert t["stoch_k"] == pytest.approx(66.67)
    assert t["stoch_d"] == pytest.approx(88.89)


def test_stochastic_needs_three_valid_k_values():
    c = pd.Series([10.0] * 15 + [11.0, 12.0])      # only two non-NaN %K
    t = compute_technicals(c, highs=c, lows=c)
    assert t["stoch_k"] is None and t["stoch_d"] is None


def test_volume_average_needs_five_samples():
    t4 = compute_technicals(_ramp(10), volumes=pd.Series([100.0, 200.0, 300.0, 400.0]))
    assert t4["volume"] == 400 and t4["volume_20d_avg"] is None and t4["volume_ratio"] is None
    t5 = compute_technicals(_ramp(10), volumes=pd.Series([100.0, 200.0, 300.0, 400.0, 500.0]))
    assert t5["volume_20d_avg"] == 300 and t5["volume_ratio"] == pytest.approx(1.67)


def test_zero_average_volume_gives_no_ratio():
    t = compute_technicals(_ramp(10), volumes=pd.Series([0.0] * 5))
    assert t["volume"] == 0 and t["volume_20d_avg"] == 0 and t["volume_ratio"] is None


def test_volume_average_window_is_exactly_20():
    v = pd.Series([1000.0] * 5 + [100.0] * 20)
    t = compute_technicals(_ramp(25), volumes=v)
    assert t["volume_20d_avg"] == 100 and t["volume_ratio"] == pytest.approx(1.0)


def test_empty_volume_series_is_ignored():
    t = compute_technicals(_ramp(10), volumes=pd.Series([], dtype=float))
    assert t["volume"] is None


def test_ema200_boundary():
    assert compute_technicals(_ramp(199))["ema_200"] is None
    assert compute_technicals(_ramp(200))["ema_200"] is not None


def test_macd_signal_needs_35_bars():
    assert compute_technicals(_ramp(34))["macd_signal"] is None
    assert compute_technicals(_ramp(35))["macd_signal"] is not None
    assert compute_technicals(_ramp(25))["macd"] is None
    assert compute_technicals(_ramp(26))["macd"] is not None


def test_52_week_window_is_252_bars():
    c = pd.Series([1000.0, 500.0] + [100.0] * 250 + [101.0])      # 253 bars
    t = compute_technicals(c)
    assert t["high_52w"] == 500.0                                 # bar 0 fell out, bar 1 is still in
    c2 = pd.Series([1.0, 2.0] + [100.0] * 250 + [101.0])
    assert compute_technicals(c2)["low_52w"] == 2.0


def test_period_returns_need_more_bars_than_lookback():
    assert compute_technicals(_ramp(5))["ret_1w"] is None
    assert compute_technicals(_ramp(6))["ret_1w"] == pytest.approx(round(5 / 1 * 100, 2))
    assert compute_technicals(_ramp(2))["ret_1d"] == pytest.approx(100.0)
    assert compute_technicals(_ramp(21))["ret_1m"] is None
    assert compute_technicals(_ramp(22))["ret_1m"] is not None


def test_returns_none_when_reference_price_not_positive():
    c = pd.Series([0.0] + [10.0] * 5)
    t = compute_technicals(c)
    assert t["ret_1y"] is None and t["ret_1w"] is None
