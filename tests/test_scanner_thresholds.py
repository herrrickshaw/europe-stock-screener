"""Exact boundaries of every scoring criterion and the BUY/WATCH/AVOID cut-offs."""
import pandas as pd
import pytest

from screener.scanner import scan_darvas, scan_piotroski

BASE = {"ticker": "T", "cmp": 100.0, "high_52w": 100.0, "low_52w": 0.0, "ema_20": 95.0,
        "ema_50": 90.0, "ema_200": 80.0, "rsi": 50.0, "macd": 1.0, "macd_signal": 0.5,
        "volume_ratio": 2.0, "bb_pct": 0.8, "stoch_k": 60.0, "stoch_d": 50.0,
        "roe": 20.0, "pe": 15.0, "debt_to_equity": 0.4, "opm": 20.0, "revenue_growth": 5.0,
        "current_ratio": 2.0, "pb": 2.0, "atr_14": 2.0}


def crit(fn, name, **over):
    return fn(pd.DataFrame([dict(BASE, **over)]))[0]["criteria"][name]


D = scan_darvas
P = scan_piotroski

CASES = [
    # (scanner, criterion, overrides, expected)
    (D, "near_52w_high", dict(cmp=85.0), True),        # exactly 15% below -> pass
    (D, "near_52w_high", dict(cmp=84.9), False),
    (D, "above_ema50", dict(cmp=90.0), False),         # equal is not above
    (D, "above_ema50", dict(cmp=90.01), True),
    (D, "range_strength", dict(cmp=60.0), True),       # exactly 60% of range
    (D, "range_strength", dict(cmp=59.9), False),
    (D, "rsi_healthy", dict(rsi=40.0), True),
    (D, "rsi_healthy", dict(rsi=39.99), False),
    (D, "rsi_healthy", dict(rsi=68.0), True),
    (D, "rsi_healthy", dict(rsi=68.01), False),
    (D, "buffett_roe", dict(roe=12.0), False),
    (D, "buffett_roe", dict(roe=12.01), True),
    (D, "buffett_pe", dict(pe=5.0), True),
    (D, "buffett_pe", dict(pe=4.99), False),
    (D, "buffett_pe", dict(pe=30.0), True),
    (D, "buffett_pe", dict(pe=30.01), False),
    (D, "buffett_de", dict(debt_to_equity=1.0), False),
    (D, "buffett_de", dict(debt_to_equity=0.99), True),
    (D, "above_ema200", dict(ema_200=100.0), False),
    (D, "above_ema200", dict(ema_200=99.99), True),
    (D, "macd_bull", dict(macd=0.5), False),
    (D, "macd_bull", dict(macd=0.51), True),
    (D, "volume_surge", dict(volume_ratio=1.5), False),
    (D, "volume_surge", dict(volume_ratio=1.51), True),
    (D, "ema_cross", dict(ema_20=90.0), False),
    (D, "ema_cross", dict(ema_20=90.01), True),
    (D, "bb_upper_half", dict(bb_pct=0.5), False),
    (D, "bb_upper_half", dict(bb_pct=0.51), True),
    (D, "stoch_bullish", dict(stoch_k=20.0, stoch_d=10.0), True),
    (D, "stoch_bullish", dict(stoch_k=19.9, stoch_d=10.0), False),
    (D, "stoch_bullish", dict(stoch_k=80.0, stoch_d=70.0), True),
    (D, "stoch_bullish", dict(stoch_k=80.1, stoch_d=70.0), False),
    (D, "stoch_bullish", dict(stoch_k=50.0, stoch_d=50.0), False),   # K must exceed D
    (P, "F1_roe_positive", dict(roe=0.0), False),
    (P, "F1_roe_positive", dict(roe=0.01), True),
    (P, "F5_low_leverage", dict(debt_to_equity=0.5), False),
    (P, "F5_low_leverage", dict(debt_to_equity=0.49), True),
    (P, "F8_op_margin", dict(opm=15.0), False),
    (P, "F8_op_margin", dict(opm=15.01), True),
    (P, "F9_rev_growth", dict(revenue_growth=0.0), False),
    (P, "F9_rev_growth", dict(revenue_growth=0.01), True),
    (P, "FX1_above_ema50", dict(cmp=90.0), False),
    (P, "FX1_above_ema50", dict(cmp=90.01), True),
    (P, "FX2_rsi_range", dict(rsi=40.0), True),
    (P, "FX2_rsi_range", dict(rsi=39.99), False),
    (P, "FX2_rsi_range", dict(rsi=70.0), True),
    (P, "FX2_rsi_range", dict(rsi=70.01), False),
    (P, "FX3_near_high", dict(cmp=80.0), True),          # exactly 20% below -> pass
    (P, "FX3_near_high", dict(cmp=79.9), False),
    (P, "FX4_current_ratio", dict(current_ratio=1.5), False),
    (P, "FX4_current_ratio", dict(current_ratio=1.51), True),
    (P, "FX5_pb_value", dict(pb=3.0), False),
    (P, "FX5_pb_value", dict(pb=2.99), True),
    (P, "FX6_ema_cross", dict(ema_20=90.0), False),
    (P, "FX6_ema_cross", dict(ema_20=90.01), True),
    (P, "FX7_low_atr", dict(atr_14=3.0), False),         # exactly 3% of price -> fail
    (P, "FX7_low_atr", dict(atr_14=2.99), True),
]


@pytest.mark.parametrize("fn,name,over,expected", CASES,
                         ids=[f"{c[1]}-{sorted(c[2].items())}" for c in CASES])
def test_criterion_boundary(fn, name, over, expected):
    assert crit(fn, name, **over) is expected


def test_each_criterion_is_none_when_its_input_is_missing():
    full = {k: v for k, v in BASE.items()}
    drop = {"roe": ("D", "buffett_roe"), "pe": ("D", "buffett_pe"),
            "debt_to_equity": ("D", "buffett_de"), "ema_200": ("D", "above_ema200"),
            "volume_ratio": ("D", "volume_surge"), "bb_pct": ("D", "bb_upper_half"),
            "rsi": ("D", "rsi_healthy")}
    for col, (_, name) in drop.items():
        row = {k: v for k, v in full.items() if k != col}
        assert scan_darvas(pd.DataFrame([row]))[0]["criteria"][name] is None, col
    for col, name in {"opm": "F8_op_margin", "revenue_growth": "F9_rev_growth",
                      "current_ratio": "FX4_current_ratio", "pb": "FX5_pb_value",
                      "atr_14": "FX7_low_atr", "ema_20": "FX6_ema_cross"}.items():
        row = {k: v for k, v in full.items() if k != col}
        assert scan_piotroski(pd.DataFrame([row]))[0]["criteria"][name] is None, col


def test_macd_and_stochastic_need_both_inputs():
    no_sig = {k: v for k, v in BASE.items() if k != "macd_signal"}
    no_d = {k: v for k, v in BASE.items() if k != "stoch_d"}
    assert scan_darvas(pd.DataFrame([no_sig]))[0]["criteria"]["macd_bull"] is None
    assert scan_darvas(pd.DataFrame([no_d]))[0]["criteria"]["stoch_bullish"] is None


def test_atr_criterion_needs_positive_price():
    assert crit(P, "FX7_low_atr", cmp=0.0) is None


def test_range_strength_needs_a_nonzero_range():
    assert crit(D, "range_strength", high_52w=60.0, low_52w=60.0, cmp=60.0) is None


# ------------------------------------------------------------------ weights and cut-offs

TECH = {"ticker": "T", "cmp": 100.0, "high_52w": 100.0, "low_52w": 0.0, "ema_50": 90.0, "rsi": 50.0}


def _darvas(**over):
    r = scan_darvas(pd.DataFrame([dict(TECH, **over)]))[0]
    return r["score"], r["max_score"], r["signal"]


def test_darvas_weights_and_cutoffs():
    assert _darvas() == (6, 6, "BUY")
    assert _darvas(rsi=30.0) == (5, 6, "BUY")                       # D1 2 + D2 2 + D3 1 -> BUY at 5
    assert _darvas(rsi=30.0, ema_50=110.0) == (3, 6, "WATCH")       # D1 2 + D3 1 -> WATCH at 3
    assert _darvas(ema_50=110.0) == (4, 6, "WATCH")                 # D1 2 + D3 1 + D4 1
    # D2 only (2 pts): price mid-range, above EMA-50, RSI unhealthy -> AVOID
    assert _darvas(cmp=50.0, ema_50=40.0, rsi=30.0) == (2, 6, "AVOID")
    assert _darvas(cmp=50.0, ema_50=40.0) == (3, 6, "WATCH")        # + D4


def test_darvas_max_score_grows_with_available_fundamentals():
    assert scan_darvas(pd.DataFrame([dict(TECH, roe=20.0)]))[0]["max_score"] == 7
    assert scan_darvas(pd.DataFrame([dict(TECH, roe=20.0, pe=10.0, debt_to_equity=0.1)]))[0]["max_score"] == 9
    assert scan_darvas(pd.DataFrame([BASE]))[0]["max_score"] == 15


def test_darvas_empty_row_scores_zero():
    r = scan_darvas(pd.DataFrame([{"ticker": "X"}]))[0]
    assert (r["score"], r["max_score"], r["signal"], r["completeness"]) == (0, 6, "AVOID", 0)


def _pio(**over):
    row = {"ticker": "T", "cmp": 100.0, "high_52w": 100.0, "ema_50": 90.0, "ema_20": 95.0, "rsi": 50.0}
    r = scan_piotroski(pd.DataFrame([dict(row, **over)]))[0]
    return r["score"], r["max_score"], r["signal"]


def test_piotroski_cutoffs_for_max_4():
    assert _pio() == (4, 4, "BUY")
    assert _pio(ema_20=85.0) == (3, 4, "BUY")                       # ceil(4*0.65) = 3
    assert _pio(ema_20=85.0, rsi=80.0) == (2, 4, "WATCH")           # ceil(4*0.40) = 2
    assert _pio(ema_20=85.0, rsi=80.0, cmp=89.0) == (1, 4, "AVOID")


def test_piotroski_cutoffs_for_max_2():
    row = {"ticker": "T", "cmp": 100.0, "ema_50": 90.0, "rsi": 50.0}
    def run(**o):
        r = scan_piotroski(pd.DataFrame([dict(row, **o)]))[0]
        return r["score"], r["max_score"], r["signal"]
    assert run() == (2, 2, "BUY")                                   # ceil(1.3) = 2
    assert run(cmp=80.0) == (1, 2, "WATCH")                         # ceil(0.8) = 1
    assert run(cmp=80.0, rsi=80.0) == (0, 2, "AVOID")


def test_piotroski_cutoffs_for_larger_max():
    # max 10: BUY >= 7 (6.5 rounds up), WATCH >= 4
    base = dict(BASE)
    base.pop("atr_14")                                              # 10 criteria available
    def run(fails):
        row = dict(base)
        for k in fails:
            row[k] = {"roe": -1.0, "debt_to_equity": 2.0, "opm": 1.0, "revenue_growth": -1.0,
                      "current_ratio": 1.0, "pb": 9.0}[k]
        r = scan_piotroski(pd.DataFrame([row]))[0]
        return r["score"], r["max_score"], r["signal"]
    assert run([]) == (10, 10, "BUY")
    assert run(["roe", "debt_to_equity", "opm"]) == (7, 10, "BUY")
    assert run(["roe", "debt_to_equity", "opm", "revenue_growth"]) == (6, 10, "WATCH")
    assert run(["roe", "debt_to_equity", "opm", "revenue_growth", "current_ratio", "pb"]) == (4, 10, "WATCH")


def test_piotroski_no_data_row_is_avoid_with_floor_of_one():
    r = scan_piotroski(pd.DataFrame([{"ticker": "X"}]))[0]
    assert (r["score"], r["max_score"], r["signal"], r["completeness"]) == (0, 1, "AVOID", 0)


def test_piotroski_completeness_rounds():
    full = scan_piotroski(pd.DataFrame([BASE]))[0]
    assert (full["score"], full["max_score"], full["completeness"]) == (11, 11, 100)
    row = {k: BASE[k] for k in ("ticker", "cmp", "high_52w", "ema_50", "rsi", "ema_20", "roe", "pb")}
    assert scan_piotroski(pd.DataFrame([row]))[0]["completeness"] == round(6 / 11 * 100)
