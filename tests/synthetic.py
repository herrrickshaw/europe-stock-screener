"""Deterministic synthetic OHLCV panel (no network, no yfinance)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# (ticker, exchange, n_days, drift, seed) - exchange-suffixed tickers only.
SPECS = [
    ('AAA.L', 'London Stock Exchange', 260, 0.0009, 1),
    ('BBB.DE', 'Deutsche Boerse Frankfurt', 260, -0.0007, 2),
    ('CCC.PA', 'Euronext Paris', 120, 0.0004, 3),
    ('DDD.MI', 'Borsa Italiana (Milan)', 40, 0.0002, 4),
    ('EEE.ST', 'Nasdaq Stockholm', 20, 0.0, 5),
    ('FFF.SW', 'SIX Swiss', 260, 0.0, 6),
    ('GGG.AS', 'Euronext Amsterdam', 10, 0.001, 7),
    ('HHH.MC', 'BME Madrid', 200, 0.0012, 8),
]


def make_ohlcv(n: int, drift: float, seed: int) -> pd.DataFrame:
    """Return a business-day OHLCV frame of n rows from a seeded random walk."""
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    high = close * (1 + rng.uniform(0.0, 0.02, n))
    low = close * (1 - rng.uniform(0.0, 0.02, n))
    vol = rng.integers(1_000, 50_000, n).astype(float)
    idx = pd.bdate_range('2024-01-01', periods=n)
    return pd.DataFrame({'Open': close, 'High': high, 'Low': low,
                         'Close': close, 'Volume': vol}, index=idx)


def make_universe() -> pd.DataFrame:
    """Universe frame with yf_ticker, name, exchange columns."""
    return pd.DataFrame([{'yf_ticker': t, 'name': f'Name {t}', 'exchange': ex}
                         for t, ex, *_ in SPECS])


def make_download(batch, **_kwargs) -> pd.DataFrame:
    """Stand-in for yf.download: multi-ticker frame, group_by='ticker' layout."""
    frames = {t: make_ohlcv(n, d, s) for t, _ex, n, d, s in SPECS if t in batch}
    return pd.concat(frames, axis=1)


def make_download_any(batch, **_kwargs) -> pd.DataFrame:
    """Like make_download but works for any ticker (seed derived from its name)."""
    import zlib
    frames = {t: make_ohlcv(250, 0.0005, zlib.crc32(t.encode()) % 10_000) for t in batch}
    return pd.concat(frames, axis=1)
