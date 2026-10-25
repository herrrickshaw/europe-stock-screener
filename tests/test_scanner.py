"""scan_darvas / scan_piotroski point scores on hand-built quote rows."""
import pandas as pd

from screener.scanner import scan_darvas, scan_piotroski

STRONG = {
    "ticker": "AAA.DE", "cmp": 100.0, "high_52w": 105.0, "low_52w": 60.0,   # D1 within 4.8%, D3 0.89
    "ema_20": 98.0, "ema_50": 95.0, "ema_200": 85.0, "rsi": 55.0,          # D2, D4, D8, D11
    "macd": 1.0, "macd_signal": 0.5, "volume_ratio": 2.0,                  # D9, D10
    "bb_pct": 0.8, "stoch_k": 60.0, "stoch_d": 50.0,                      # D12, D13
    "roe": 20.0, "pe": 15.0, "debt_to_equity": 0.4,                        # D5-D7
    "opm": 20.0, "revenue_growth": 5.0, "current_ratio": 2.0, "pb": 2.0, "atr_14": 2.0,
}


def test_empty_frame():
    assert scan_darvas(pd.DataFrame()) == [] and scan_piotroski(pd.DataFrame()) == []


def test_darvas_full_marks():
    # D1 2 + D2 2 + D3..D13 1 each = 15 (the module docstring's "0-16" is off by one)
    r = scan_darvas(pd.DataFrame([STRONG]))[0]
    assert r["score"] == 15 and r["max_score"] == 15 and r["signal"] == "BUY"
    assert r["completeness"] == 100 and all(r["criteria"].values())


def test_darvas_technical_only_row():
    row = {k: STRONG[k] for k in ("ticker", "cmp", "high_52w", "low_52w", "ema_50", "rsi")}
    r = scan_darvas(pd.DataFrame([row]))[0]
    # D1 2 + D2 2 + D3 1 + D4 1 = 6 of a possible 6
    assert (r["score"], r["max_score"], r["signal"]) == (6, 6, "BUY")


def test_unavailable_criteria_are_reported_unknown_not_passed():
    """Prove-it: _vcrit leaves unavailable rows as NaN and _b() did bool(NaN) == True,
    so a row with no fundamentals showed buffett_roe/pe/de etc. as PASSED in the
    report's criteria (the score itself was right)."""
    row = {k: STRONG[k] for k in ("ticker", "cmp", "high_52w", "low_52w", "ema_50", "rsi")}
    d = scan_darvas(pd.DataFrame([row]))[0]["criteria"]
    assert d["buffett_roe"] is None and d["stoch_bullish"] is None and d["above_ema200"] is None
    p = scan_piotroski(pd.DataFrame([row]))[0]["criteria"]
    assert all(v is None for k, v in p.items() if k not in
               ("FX1_above_ema50", "FX2_rsi_range", "FX3_near_high")), p


def test_darvas_weak_row_is_avoid():
    row = dict(STRONG, cmp=62.0, ema_20=70.0, rsi=30.0, macd=-1.0, volume_ratio=0.5,
               bb_pct=0.1, stoch_k=10.0, roe=5.0, pe=50.0, debt_to_equity=3.0, ema_200=80.0)
    r = scan_darvas(pd.DataFrame([row]))[0]
    assert r["score"] == 0 and r["signal"] == "AVOID"


def test_piotroski_dynamic_thresholds():
    full = scan_piotroski(pd.DataFrame([STRONG]))[0]
    assert full["score"] == 11 and full["max_score"] == 11 and full["signal"] == "BUY"
    # Technical-only: FX1-FX3 + FX6 available -> max 4; BUY >= ceil(2.6)=3, WATCH >= ceil(1.6)=2
    row = {k: STRONG[k] for k in ("ticker", "cmp", "high_52w", "ema_20", "ema_50", "rsi")}
    r = scan_piotroski(pd.DataFrame([row]))[0]
    assert (r["score"], r["max_score"], r["signal"]) == (4, 4, "BUY")
    weak = dict(row, cmp=80.0, rsi=80.0, ema_20=90.0)       # only FX3 fails too? (105-80)/105=24% -> 0
    r = scan_piotroski(pd.DataFrame([weak]))[0]
    assert (r["score"], r["signal"]) == (0, "AVOID")
