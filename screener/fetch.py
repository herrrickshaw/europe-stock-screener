"""
Batch OHLCV fetch + technical-indicator computation for the European universe.

Adapted from the multi-market screening platform's `fetchers/providers/yahoo.py`
(Phase-1 bulk path). No API key required — yfinance only.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Optional, Protocol

import pandas as pd
import yfinance as yf

from screener.technicals import QuoteData, compute_technicals

log = logging.getLogger(__name__)

_RATE_BACKOFFS = [90, 180, 300]
_BATCH_SIZE = 75


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _pct(v) -> Optional[float]:
    f = _safe_float(v)
    return round(f * 100, 2) if f is not None else None


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class Downloader(Protocol):
    """Anything shaped like ``yfinance.download`` (injected so tests never hit the network)."""

    def __call__(self, tickers: list, **kwargs: Any) -> pd.DataFrame: ...


def _yf_download(tickers: list, **kwargs: Any) -> pd.DataFrame:
    return yf.download(tickers, **kwargs)


def _download_batch(batch: list[str], batch_num: int, download: Downloader,
                    sleep: Callable[[float], None]) -> Optional[pd.DataFrame]:
    """Download one batch, backing off on rate-limit errors; None if the batch failed."""
    for backoff in [0] + _RATE_BACKOFFS:
        if backoff:
            sleep(backoff)
        try:
            return download(batch, period='1y', auto_adjust=True,
                            progress=False, threads=False, group_by='ticker')
        except Exception as exc:  # yfinance raises assorted network/HTTP errors
            if 'rate' in str(exc).lower() or '429' in str(exc):
                continue
            log.warning('batch %d failed: %s', batch_num, exc)
            return None
    return None


def _ohlcv_for(raw: pd.DataFrame, ticker: str, multi: bool):
    """Return (closes, volumes, highs, lows) for `ticker`, or None if it is absent from `raw`."""
    if multi:
        if ticker not in raw.columns.get_level_values(0):
            return None
        sub = raw[ticker]
        closes  = sub['Close'].dropna()  if 'Close'  in sub.columns else pd.Series(dtype=float)
        volumes = sub['Volume'].dropna() if 'Volume' in sub.columns else None
        highs   = sub['High'].dropna()   if 'High'   in sub.columns else None
        lows    = sub['Low'].dropna()    if 'Low'    in sub.columns else None
    else:
        closes  = raw['Close'].dropna()
        volumes = raw.get('Volume', pd.Series(dtype=float)).dropna()
        highs   = raw['High'].dropna() if 'High' in raw.columns else None
        lows    = raw['Low'].dropna()  if 'Low'  in raw.columns else None
    return closes, volumes, highs, lows


def fetch_bulk(
    universe: pd.DataFrame,
    download: Optional[Downloader] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, QuoteData]:
    """
    universe: DataFrame with columns yf_ticker, name, exchange.
    Downloads 1y OHLCV in batches of `_BATCH_SIZE`, computes technicals per ticker.
    `download` defaults to yfinance.download; inject a stub for offline use.
    Returns {yf_ticker: QuoteData}.
    """
    download = download or _yf_download
    result: dict[str, QuoteData] = {}
    tickers = universe['yf_ticker'].tolist()
    meta = universe.set_index('yf_ticker')[['name', 'exchange']].to_dict('index')

    for batch_num, batch in enumerate(_chunks(tickers, _BATCH_SIZE), start=1):
        log.info('batch %d: %d tickers', batch_num, len(batch))
        raw = _download_batch(batch, batch_num, download, sleep)
        if raw is None or raw.empty:
            continue

        multi = isinstance(raw.columns, pd.MultiIndex)
        for ticker in batch:
            try:
                series = _ohlcv_for(raw, ticker, multi)
                if series is None:
                    continue
                closes, volumes, highs, lows = series
                if len(closes) < 2:
                    continue

                tech = compute_technicals(closes, volumes, highs, lows)
                if not tech.get('cmp'):
                    continue

                m = meta.get(ticker, {})
                result[ticker] = QuoteData(
                    ticker=ticker, source='yahoo',
                    name=m.get('name'), exchange=m.get('exchange'),
                    **tech,
                )
            except Exception as exc:  # one bad ticker must not abort the batch
                log.warning('bulk[%s]: %s', ticker, exc)

    return result


def fetch_fundamentals(ticker: str) -> dict:
    """Single-ticker .info call. Slow (~1 req/sec) — only used with --with-fundamentals."""
    try:
        info = yf.Ticker(ticker).info
        if not info:
            return {}
        mc = info.get('marketCap')
        return {
            'pe':             _safe_float(info.get('trailingPE')),
            'pb':             _safe_float(info.get('priceToBook')),
            'roe':            _pct(info.get('returnOnEquity')),
            'opm':            _pct(info.get('operatingMargins')),
            'market_cap':     round(mc / 1e6, 2) if mc else None,
            'debt_to_equity': _safe_float(info.get('debtToEquity')),
            'beta':           _safe_float(info.get('beta')),
            'current_ratio':  _safe_float(info.get('currentRatio')),
            'revenue_growth': _pct(info.get('revenueGrowth')),
            'eps':            _safe_float(info.get('trailingEps')),
            'dividend_yield': _pct(info.get('dividendYield')),
            'sector':         info.get('sector') or None,
            'industry':       info.get('industry') or None,
            'name':           info.get('longName') or info.get('shortName') or None,
        }
    except Exception as exc:
        log.debug('fundamentals[%s]: %s', ticker, exc)
        return {}
