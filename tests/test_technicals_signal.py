"""rsi_signal classification (BUY / SELL / HOLD) from compute_technicals()."""
import pandas as pd

from screener.technicals import compute_technicals


def _step_then_move(low, high, hold, per_bar, bars, up_first):
    start, level = (low, high) if up_first else (high, low)
    sign = -1 if up_first else 1
    return pd.Series([start] * 100 + [level] * hold + [level + sign * per_bar * j for j in range(1, bars + 1)])


def test_oversold_above_ema50_is_buy():
    t = compute_technicals(_step_then_move(100.0, 300.0, 40, 3.0, 8, up_first=True))
    assert t["rsi"] < 30 and t["cmp"] > t["ema_50"]
    assert t["rsi_signal"] == "BUY"


def test_overbought_below_ema50_is_sell():
    t = compute_technicals(_step_then_move(100.0, 300.0, 40, 3.0, 8, up_first=False))
    assert t["rsi"] > 70 and t["cmp"] < t["ema_50"]
    assert t["rsi_signal"] == "SELL"


def test_oversold_but_below_ema50_is_hold():
    t = compute_technicals(pd.Series([float(x) for x in range(60, 0, -1)]))
    assert t["rsi"] < 30 and t["cmp"] < t["ema_50"]
    assert t["rsi_signal"] == "HOLD"


def test_overbought_but_above_ema50_is_hold():
    t = compute_technicals(pd.Series([float(x) for x in range(1, 61)]))
    assert t["rsi"] > 70 and t["cmp"] > t["ema_50"]
    assert t["rsi_signal"] == "HOLD"


def test_no_rsi_under_15_bars_is_hold():
    t = compute_technicals(pd.Series([float(x) for x in range(1, 15)]))
    assert t["rsi"] is None and t["rsi_signal"] == "HOLD"
    assert compute_technicals(pd.Series([float(x) for x in range(1, 16)]))["rsi"] == 100.0


def _boundary_series(up, last):
    lo, hi = (100.0, 300.0) if up else (300.0, 100.0)
    sign = -1 if up else 1
    return pd.Series([lo] * 100 + [hi] * 40 + [hi + sign * 3 * j for j in range(1, 8)] + [last])


def test_rsi_exactly_30_is_not_oversold():
    # last close tuned so the rounded RSI is exactly 30.0 while price is still above EMA-50
    t = compute_technicals(_boundary_series(True, 279.6234130859375))
    assert t["rsi"] == 30.0 and t["cmp"] > t["ema_50"]
    assert t["rsi_signal"] == "HOLD"


def test_rsi_exactly_70_is_not_overbought():
    t = compute_technicals(_boundary_series(False, 120.3765869140625))
    assert t["rsi"] == 70.0 and t["cmp"] < t["ema_50"]
    assert t["rsi_signal"] == "HOLD"
