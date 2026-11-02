"""Universe loading and the exchange-suffix allowlist.

Tickers always carry a Yahoo Finance exchange suffix taken from
``EXCHANGE_SUFFIXES``; a suffix is never inferred from the shape of a ticker.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
UNIVERSE_CSV = DATA_DIR / 'europe_all_list.csv'

# Suffix -> exchange. Allowlist of every European Yahoo suffix this repo accepts.
# '.F' and '.DE' are both Deutsche Boerse (Frankfurt vs XETRA); the bundled list uses '.F'.
EXCHANGE_SUFFIXES: dict[str, str] = {
    '.L': 'London Stock Exchange',
    '.F': 'Deutsche Boerse Frankfurt',
    '.DE': 'Deutsche Boerse (XETRA)',
    '.PA': 'Euronext Paris',
    '.MI': 'Borsa Italiana (Milan)',
    '.MC': 'BME Madrid',
    '.ST': 'Nasdaq Stockholm',
    '.OL': 'Oslo Bors',
    '.HE': 'Nasdaq Helsinki',
    '.CO': 'Nasdaq Copenhagen',
    '.AT': 'Athens Stock Exchange',
    '.AS': 'Euronext Amsterdam',
    '.VS': 'Nasdaq Vilnius',
    '.IR': 'Euronext Dublin',
    '.BR': 'Euronext Brussels',
    '.TL': 'Nasdaq Tallinn',
    '.RO': 'Bucharest Stock Exchange (BET)',
    '.SW': 'SIX Swiss',
    '.VI': 'Vienna (ATX)',
    '.WA': 'Warsaw GPW',
    '.LS': 'Euronext Lisbon',
    '.BD': 'Budapest Stock Exchange (BUX)',
    '.PR': 'Prague Stock Exchange (PX)',
    '.RG': 'Nasdaq Riga',
}


def ticker_suffix(ticker: str) -> Optional[str]:
    """Return the allowlisted exchange suffix of `ticker` (e.g. '.L'), or None if absent/unknown."""
    dot = ticker.rfind('.')
    if dot <= 0:
        return None
    suffix = ticker[dot:].upper()
    return suffix if suffix in EXCHANGE_SUFFIXES else None


def unknown_suffix_tickers(tickers: list[str]) -> list[str]:
    """Return the tickers whose suffix is not on the allowlist."""
    return [t for t in tickers if ticker_suffix(t) is None]


def load_universe(exchange: Optional[str] = None, limit: Optional[int] = None,
                  path: Path = UNIVERSE_CSV) -> pd.DataFrame:
    """Read the universe CSV (cols yf_ticker, name, index, exchange); optionally filter by
    comma-separated exchange names and cap the row count. Warns on non-allowlisted suffixes."""
    df = pd.read_csv(path)
    if exchange:
        names = [e.strip() for e in exchange.split(',')]
        df = df[df['exchange'].isin(names)]
    if limit:
        df = df.head(limit)
    bad = unknown_suffix_tickers(df['yf_ticker'].tolist())
    if bad:
        log.warning('%d ticker(s) have a suffix outside EXCHANGE_SUFFIXES: %s', len(bad), bad[:5])
    return df.reset_index(drop=True)
