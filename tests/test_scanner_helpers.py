"""Private helpers of screener.scanner and default-column behaviour."""

import numpy as np
import pandas as pd

from screener import scanner
from screener.scanner import scan_darvas, scan_piotroski


def test_r_rounds_and_rejects_non_finite():
    assert scanner._r(1.23456) == 1.23
    assert scanner._r("2.5") == 2.5
    assert scanner._r(0) == 0.0
    for bad in (None, float("nan"), float("inf"), float("-inf"), "x", [1]):
        assert scanner._r(bad) is None


def test_b_maps_nan_none_and_unbool_able_to_none():
    class Weird:
        def __bool__(self):
            raise ValueError("ambiguous")
    assert scanner._b(True) is True and scanner._b(False) is False
    assert scanner._b(np.True_) is True and scanner._b(0) is False
    assert scanner._b(None) is None and scanner._b(float("nan")) is None
    assert scanner._b(Weird()) is None


def test_strs_defaults_and_nan_fill():
    df = pd.DataFrame({"a": ["x", None, "z"]})
    assert scanner._strs(df, "a") == ["x", "", "z"]
    assert scanner._strs(df, "a", "?") == ["x", "?", "z"]
    assert scanner._strs(df, "missing") == ["", "", ""]
    assert scanner._strs(df, "missing", "HOLD") == ["HOLD"] * 3


def test_col_coerces_and_missing_is_nan():
    df = pd.DataFrame({"a": ["1", "bad", None]})
    out = scanner._col(df, "a")
    assert out.iloc[0] == 1.0 and out.iloc[1:].isna().all()
    assert scanner._col(df, "nope").isna().all() and len(scanner._col(df, "nope")) == 3


def test_vcrit_and_pts():
    passes = pd.Series([True, False, True])
    valid = pd.Series([True, True, False])
    c = scanner._vcrit(passes, valid)
    assert bool(c.iloc[0]) is True and bool(c.iloc[1]) is False
    assert pd.isna(c.iloc[2])                       # invalid rows stay unknown
    assert list(scanner._pts(c)) == [1, 0, 0]
    assert list(scanner._pts(c, 2)) == [2, 0, 0]


MIN_ROW = {"ticker": "T1", "cmp": 100.0, "high_52w": 105.0, "low_52w": 60.0,
           "ema_50": 95.0, "rsi": 55.0}


def test_missing_text_columns_default_to_empty_and_hold():
    for fn in (scan_darvas, scan_piotroski):
        r = fn(pd.DataFrame([MIN_ROW]))[0]
        assert (r["name"], r["exchange"], r["sector"], r["industry"]) == ("", "", "", "")
        assert r["rsi_signal"] == "HOLD"


def test_text_and_signal_columns_pass_through():
    row = dict(MIN_ROW, name="Alpha", exchange="LSE", sector="Tech", industry="SW", rsi_signal="BUY")
    r = scan_darvas(pd.DataFrame([row]))[0]
    assert (r["name"], r["exchange"], r["sector"], r["industry"], r["rsi_signal"]) == (
        "Alpha", "LSE", "Tech", "SW", "BUY")


def test_volume_is_int_or_none_and_values_rounded():
    rows = [dict(MIN_ROW, volume=1234.9, beta=1.23456), dict(MIN_ROW, ticker="T2"),
            dict(MIN_ROW, ticker="T3", volume=float("nan"))]
    out = scan_darvas(pd.DataFrame(rows))
    assert out[0]["volume"] == 1234 and isinstance(out[0]["volume"], int)
    assert out[0]["beta"] == 1.23
    assert out[1]["volume"] is None and out[2]["volume"] is None
    assert out[1]["beta"] is None


def test_zero_or_negative_52w_high_disables_proximity_criteria():
    for h in (0.0, -5.0):
        row = dict(MIN_ROW, high_52w=h)
        c = scan_darvas(pd.DataFrame([row]))[0]["criteria"]
        assert c["near_52w_high"] is None and c["range_strength"] is None
        assert scan_piotroski(pd.DataFrame([row]))[0]["criteria"]["FX3_near_high"] is None


def test_results_keep_one_row_per_input_in_order():
    rows = [dict(MIN_ROW, ticker=f"T{i}") for i in range(4)]
    assert [r["ticker"] for r in scan_darvas(pd.DataFrame(rows, index=[7, 3, 9, 1]))] == ["T0", "T1", "T2", "T3"]


def test_completeness_counts_available_inputs():
    full = {"ticker": "F", "cmp": 100.0, "high_52w": 105.0, "ema_50": 95.0, "rsi": 55.0,
            "roe": 10.0, "debt_to_equity": 0.3, "opm": 10.0, "revenue_growth": 1.0,
            "ema_20": 96.0, "current_ratio": 2.0, "pb": 1.0, "atr_14": 1.0}
    assert scan_piotroski(pd.DataFrame([full]))[0]["completeness"] == 100
    assert scan_piotroski(pd.DataFrame([MIN_ROW]))[0]["completeness"] == round(3 / 11 * 100)
