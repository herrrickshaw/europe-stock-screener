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


# -- indicator parameters -----------------------------------------------------
RSI_PERIOD = 14
ATR_PERIOD = 14
STOCH_PERIOD = 14
STOCH_SMOOTH = 3
BB_WINDOW = 20
TRADING_DAYS_1Y = 252
PERIOD_BARS = {'ret_1d': 1, 'ret_1w': 5, 'ret_1m': 21, 'ret_3m': 63, 'ret_6m': 126}
RSI_OVERSOLD, RSI_OVERBOUGHT = 30, 70


def _rsi(closes: pd.Series) -> Optional[float]:
    """Wilder-smoothed RSI-14 of a close series; None below 15 bars."""
    if len(closes) < RSI_PERIOD + 1:
        return None
    delta = closes.diff().dropna()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    ag = float(gains.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean().iloc[-1])
    al = float(losses.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean().iloc[-1])
    return 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 2)


def _macd(closes: pd.Series) -> tuple[Optional[float], Optional[float]]:
    """(MACD line, signal line) for 12/26/9; each None until enough bars (26 / 35)."""
    n = len(closes)
    if n < 26:
        return None, None
    e12 = closes.ewm(span=12, adjust=False).mean()
    e26 = closes.ewm(span=26, adjust=False).mean()
    ml = e12 - e26
    sig = round(float(ml.ewm(span=9, adjust=False).mean().iloc[-1]), 4) if n >= 35 else None
    return round(float(ml.iloc[-1]), 4), sig


def _bollinger(closes: pd.Series, cmp: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(upper, lower, %B) for the 20-day SMA +/- 2 sigma; %B is 0 at the lower band, 1 at the upper."""
    if len(closes) < BB_WINDOW:
        return None, None, None
    tail = closes.iloc[-BB_WINDOW:]
    sma = float(tail.mean())
    std = float(tail.std(ddof=1))
    upper = round(sma + 2 * std, 2)
    lower = round(sma - 2 * std, 2)
    width = upper - lower
    return upper, lower, (round((cmp - lower) / width, 3) if width > 0 else None)


def _atr(closes: pd.Series, highs: pd.Series, lows: pd.Series) -> Optional[float]:
    """Wilder ATR-14 from aligned-from-the-end high/low/close series; None below 15 bars."""
    h_s, l_s = highs.dropna(), lows.dropna()
    min_len = min(len(h_s), len(l_s), len(closes))
    if min_len < ATR_PERIOD + 1:
        return None
    h_a = h_s.iloc[-min_len:].values
    l_a = l_s.iloc[-min_len:].values
    c_a = closes.iloc[-min_len:].values
    tr = pd.Series([max(a, b, c) for a, b, c in
                    zip(h_a[1:] - l_a[1:], abs(h_a[1:] - c_a[:-1]), abs(l_a[1:] - c_a[:-1]))])
    return round(float(tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean().iloc[-1]), 4)


def _stochastic(closes: pd.Series, highs: pd.Series, lows: pd.Series) -> tuple[Optional[float], Optional[float]]:
    """Stochastic (%K, %D) with 14/3 periods; None below 17 bars."""
    h_s, l_s = highs.dropna(), lows.dropna()
    min_len = min(len(h_s), len(l_s), len(closes))
    if min_len < STOCH_PERIOD + STOCH_SMOOTH:
        return None, None
    c_s = closes.iloc[-min_len:]
    h_14 = h_s.iloc[-min_len:].rolling(STOCH_PERIOD).max()
    l_14 = l_s.iloc[-min_len:].rolling(STOCH_PERIOD).min()
    denom = (h_14 - l_14).replace(0, float('nan'))
    k_series = pd.Series(100 * (c_s.values - l_14.values) / denom.values).dropna()
    if len(k_series) < STOCH_SMOOTH:
        return None, None
    return (round(float(k_series.iloc[-1]), 2),
            round(float(k_series.iloc[-STOCH_SMOOTH:].mean()), 2))


def _volume_metrics(volumes: Optional[pd.Series]) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """(last volume, 20-day average, ratio) - each None when unavailable."""
    if volumes is None:
        return None, None, None
    vs = volumes.dropna()
    if vs.empty:
        return None, None, None
    vol = int(vs.iloc[-1])
    if len(vs) < 5:
        return vol, None, None
    vol_avg = int(vs.iloc[-min(20, len(vs)):].mean())
    return vol, vol_avg, (round(vol / vol_avg, 2) if vol_avg > 0 else None)


def _period_return(closes: pd.Series, cmp: float, bars: int) -> Optional[float]:
    """Percent change of `cmp` versus the close `bars` bars earlier."""
    if len(closes) <= bars:
        return None
    prev = float(closes.iloc[-(bars + 1)])
    return round((cmp - prev) / prev * 100, 2) if prev > 0 and not math.isnan(prev) else None


def _rsi_signal(rsi: Optional[float], cmp: float, ema50: float) -> str:
    """Contrarian-in-uptrend RSI signal: BUY / SELL / HOLD."""
    if rsi is None:
        return 'HOLD'
    if rsi < RSI_OVERSOLD and cmp > ema50:
        return 'BUY'
    if rsi > RSI_OVERBOUGHT and cmp < ema50:
        return 'SELL'
    return 'HOLD'


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

    Note: the 52-week high/low window includes the latest bar (it is a screening
    snapshot, not a Darvas box formation - see scanner.py).
    """
    if closes is None or len(closes) < 2:
        return {}

    closes = closes.dropna()
    n = len(closes)
    if n < 2:
        return {}

    cmp = float(closes.iloc[-1])
    rsi = _rsi(closes)

    ema20  = round(float(closes.ewm(span=20,  adjust=False).mean().iloc[-1]), 2)
    ema50  = round(float(closes.ewm(span=50,  adjust=False).mean().iloc[-1]), 2)
    ema200 = round(float(closes.ewm(span=200, adjust=False).mean().iloc[-1]), 2) if n >= 50 else None

    macd_val, macd_sig = _macd(closes)

    w = min(TRADING_DAYS_1Y, n)
    h52 = round(float(closes.iloc[-w:].max()), 2)
    l52 = round(float(closes.iloc[-w:].min()), 2)

    bb_upper, bb_lower, bb_pct = _bollinger(closes, cmp)

    atr_14 = stoch_k = stoch_d = None
    if highs is not None and lows is not None:
        atr_14 = _atr(closes, highs, lows)
        stoch_k, stoch_d = _stochastic(closes, highs, lows)

    first = float(closes.iloc[0])
    ret_1y = round((cmp - first) / first * 100, 2) if first > 0 and not math.isnan(first) else None

    vol, vol_avg, vol_ratio = _volume_metrics(volumes)

    return {
        'cmp': round(cmp, 2),
        'rsi': rsi,
        'ema_20': ema20, 'ema_50': ema50, 'ema_200': ema200,
        'macd': macd_val, 'macd_signal': macd_sig, 'rsi_signal': _rsi_signal(rsi, cmp, ema50),
        'high_52w': h52, 'low_52w': l52,
        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_pct': bb_pct,
        'atr_14': atr_14, 'stoch_k': stoch_k, 'stoch_d': stoch_d,
        'volume': vol, 'volume_20d_avg': vol_avg, 'volume_ratio': vol_ratio,
        **{k: _period_return(closes, cmp, bars) for k, bars in PERIOD_BARS.items()},
        'ret_1y': ret_1y,
    }
