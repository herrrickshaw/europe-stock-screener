"""
Technical-indicator computation and the canonical per-ticker quote record.

Adapted from the multi-market screening platform's `fetchers/providers/base.py`
(Phase-1 OHLCV+RSI path — no fundamentals required).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd


@dataclass
class QuoteData:
    """Canonical quote + fundamentals record for one ticker."""
    ticker:          str
    source:          str

    # Price / technicals
    cmp:             Optional[float] = None
    rsi:             Optional[float] = None
    ema_20:          Optional[float] = None
    ema_50:          Optional[float] = None
    ema_200:         Optional[float] = None
    macd:            Optional[float] = None
    macd_signal:     Optional[float] = None
    rsi_signal:      str             = 'HOLD'
    high_52w:        Optional[float] = None
    low_52w:         Optional[float] = None
    volume:          Optional[int]   = None
    volume_20d_avg:  Optional[int]   = None
    volume_ratio:    Optional[float] = None

    # Bollinger Bands (20-day SMA +/- 2 sigma)
    bb_upper:        Optional[float] = None
    bb_lower:        Optional[float] = None
    bb_pct:          Optional[float] = None   # 0 = at lower band, 1 = at upper band

    # Volatility / momentum oscillators
    atr_14:          Optional[float] = None   # Average True Range (14-day Wilder's)
    stoch_k:         Optional[float] = None   # Stochastic %K (14-day)
    stoch_d:         Optional[float] = None   # Stochastic %D (3-day SMA of %K)

    # Period returns (%)
    ret_1d:  Optional[float] = None
    ret_1w:  Optional[float] = None
    ret_1m:  Optional[float] = None
    ret_3m:  Optional[float] = None
    ret_6m:  Optional[float] = None
    ret_1y:  Optional[float] = None

    # Fundamentals (optional — filled only with --with-fundamentals)
    pe:             Optional[float] = None
    pb:             Optional[float] = None
    roe:            Optional[float] = None   # %
    opm:            Optional[float] = None   # %
    market_cap:     Optional[float] = None   # EUR/GBP/... M, currency native to exchange
    debt_to_equity: Optional[float] = None
    beta:           Optional[float] = None
    current_ratio:  Optional[float] = None
    revenue_growth: Optional[float] = None   # %
    eps:            Optional[float] = None
    dividend_yield: Optional[float] = None   # %

    # Meta
    sector:   Optional[str] = None
    industry: Optional[str] = None
    name:     Optional[str] = None
    exchange: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def compute_technicals(
    closes: pd.Series,
    volumes: Optional[pd.Series] = None,
    highs:   Optional[pd.Series] = None,
    lows:    Optional[pd.Series] = None,
) -> dict:
    """
    Compute technical indicators from a daily OHLCV series.

    Requires only `closes` (+ optional `volumes`, `highs`, `lows`).
    Returns a dict compatible with QuoteData field names.

    Indicators computed:
      Always (from closes): RSI-14, EMA-20/50/200, MACD+signal, 52W hi/lo,
                            Bollinger Bands (20-day), period returns, volume ratio
      With highs+lows:      ATR-14 (Wilder's), Stochastic %K/%D
    """
    if closes is None or len(closes) < 2:
        return {}

    closes = closes.dropna()
    n = len(closes)
    if n < 2:
        return {}

    cmp = float(closes.iloc[-1])

    # RSI-14 (Wilder's smoothed)
    rsi = None
    if n >= 15:
        delta = closes.diff().dropna()
        gains  = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)
        ag = float(gains.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
        al = float(losses.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
        rsi = 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 2)

    # EMAs
    ema20  = round(float(closes.ewm(span=20,  adjust=False).mean().iloc[-1]), 2)
    ema50  = round(float(closes.ewm(span=50,  adjust=False).mean().iloc[-1]), 2)
    ema200 = round(float(closes.ewm(span=200, adjust=False).mean().iloc[-1]), 2) if n >= 50 else None

    # MACD (12/26/9)
    macd_val = macd_sig = None
    if n >= 26:
        e12 = closes.ewm(span=12, adjust=False).mean()
        e26 = closes.ewm(span=26, adjust=False).mean()
        ml  = e12 - e26
        macd_val = round(float(ml.iloc[-1]), 4)
        if n >= 35:
            macd_sig = round(float(ml.ewm(span=9, adjust=False).mean().iloc[-1]), 4)

    # 52-Week high / low
    w   = min(252, n)
    h52 = round(float(closes.iloc[-w:].max()), 2)
    l52 = round(float(closes.iloc[-w:].min()), 2)

    # Bollinger Bands (20-day SMA +/- 2 sigma)
    bb_upper = bb_lower = bb_pct = None
    if n >= 20:
        tail20  = closes.iloc[-20:]
        sma20   = float(tail20.mean())
        std20   = float(tail20.std(ddof=1))
        bb_upper = round(sma20 + 2 * std20, 2)
        bb_lower = round(sma20 - 2 * std20, 2)
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_pct = round((cmp - bb_lower) / bb_range, 3)

    # ATR-14 (Wilder's) — needs highs/lows
    atr_14 = None
    if highs is not None and lows is not None:
        h_s = highs.dropna()
        l_s = lows.dropna()
        min_len = min(len(h_s), len(l_s), n)
        if min_len >= 15:
            h_a = h_s.iloc[-min_len:].values
            l_a = l_s.iloc[-min_len:].values
            c_a = closes.iloc[-min_len:].values
            tr_hl   = h_a[1:] - l_a[1:]
            tr_hpc  = abs(h_a[1:] - c_a[:-1])
            tr_lpc  = abs(l_a[1:] - c_a[:-1])
            tr = pd.Series(
                [max(a, b, c) for a, b, c in zip(tr_hl, tr_hpc, tr_lpc)]
            )
            atr_14 = round(float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]), 4)

    # Stochastic %K/%D (14/3) — needs highs/lows
    stoch_k = stoch_d = None
    if highs is not None and lows is not None:
        h_s = highs.dropna()
        l_s = lows.dropna()
        min_len = min(len(h_s), len(l_s), n)
        if min_len >= 17:  # 14 for %K + 3 for %D
            c_s  = closes.iloc[-min_len:]
            h_s2 = h_s.iloc[-min_len:]
            l_s2 = l_s.iloc[-min_len:]
            h_14 = h_s2.rolling(14).max()
            l_14 = l_s2.rolling(14).min()
            denom = (h_14 - l_14).replace(0, float('nan'))
            k_series = 100 * (c_s.values - l_14.values) / denom.values
            k_series = pd.Series(k_series).dropna()
            if len(k_series) >= 3:
                stoch_k = round(float(k_series.iloc[-1]), 2)
                stoch_d = round(float(k_series.iloc[-3:].mean()), 2)

    # Period returns
    def _ret(nb: int) -> Optional[float]:
        if n <= nb:
            return None
        prev = float(closes.iloc[-(nb + 1)])
        return round((cmp - prev) / prev * 100, 2) if prev > 0 and not math.isnan(prev) else None

    ret_1y = None
    if n >= 2:
        first = float(closes.iloc[0])
        ret_1y = round((cmp - first) / first * 100, 2) if first > 0 and not math.isnan(first) else None

    # Volume metrics
    vol = vol_avg = vol_ratio = None
    if volumes is not None:
        vs = volumes.dropna()
        if not vs.empty:
            vol = int(vs.iloc[-1])
            if len(vs) >= 5:
                vw      = min(20, len(vs))
                vol_avg = int(vs.iloc[-vw:].mean())
                vol_ratio = round(vol / vol_avg, 2) if vol_avg > 0 else None

    # RSI signal
    if rsi is None:
        sig = 'HOLD'
    elif rsi < 30 and cmp > ema50:
        sig = 'BUY'
    elif rsi > 70 and cmp < ema50:
        sig = 'SELL'
    else:
        sig = 'HOLD'

    return {
        'cmp': round(cmp, 2),
        'rsi': rsi,
        'ema_20': ema20, 'ema_50': ema50, 'ema_200': ema200,
        'macd': macd_val, 'macd_signal': macd_sig, 'rsi_signal': sig,
        'high_52w': h52, 'low_52w': l52,
        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_pct': bb_pct,
        'atr_14': atr_14, 'stoch_k': stoch_k, 'stoch_d': stoch_d,
        'volume': vol, 'volume_20d_avg': vol_avg, 'volume_ratio': vol_ratio,
        'ret_1d': _ret(1), 'ret_1w': _ret(5), 'ret_1m': _ret(21),
        'ret_3m': _ret(63), 'ret_6m': _ret(126), 'ret_1y': ret_1y,
    }
