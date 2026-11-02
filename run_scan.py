#!/usr/bin/env python3
"""
Run the Darvas/Buffett + Piotroski-simplified scan across the full European
equity universe (23 exchanges, 1,063 stocks — data/europe_all_list.csv).

Usage:
    python run_scan.py                          # OHLCV+RSI only (fast, ~5-10 min)
    python run_scan.py --with-fundamentals       # + PE/ROE/D-E via yfinance .info (slow, ~1-3h)
    python run_scan.py --exchange "London Stock Exchange"
    python run_scan.py --limit 100               # smoke-test on a subset
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from screener.fetch import fetch_bulk, fetch_fundamentals  # noqa: E402
from screener.scanner import scan_darvas, scan_piotroski  # noqa: E402
from screener.technicals import QuoteData  # noqa: E402
from screener.universe import DATA_DIR, UNIVERSE_CSV, load_universe  # noqa: E402,F401  (re-exported)

log = logging.getLogger('run_scan')

DARVAS_CSV = 'darvas_buffett_scan.csv'
PIOTROSKI_CSV = 'piotroski_scan.csv'
FUNDAMENTALS_SLEEP_S = 1      # throttle between per-ticker .info calls
FUNDAMENTALS_PROGRESS_EVERY = 50
TOP_N = 10


def enrich_with_fundamentals(
    quotes: dict[str, QuoteData],
    fetcher: Callable[[str], dict] = fetch_fundamentals,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Fill still-empty QuoteData fields in place from a per-ticker fundamentals fetcher."""
    for i, ticker in enumerate(quotes.keys()):
        for k, v in fetcher(ticker).items():
            if v is not None and getattr(quotes[ticker], k, None) is None:
                setattr(quotes[ticker], k, v)
        if (i + 1) % FUNDAMENTALS_PROGRESS_EVERY == 0:
            print(f'  fundamentals: {i + 1}/{len(quotes)}')
        sleep(FUNDAMENTALS_SLEEP_S)


def quotes_to_frame(quotes: dict[str, QuoteData]) -> pd.DataFrame:
    """One row per ticker (QuoteData fields + 'ticker'), the input shape the scanners expect."""
    return pd.DataFrame([q.to_dict() | {'ticker': t} for t, q in quotes.items()])


def run_scans(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score `df` with both scanners; return (darvas, piotroski) frames without the criteria dicts."""
    darvas = pd.DataFrame(scan_darvas(df)).drop(columns=['criteria'], errors='ignore')
    piotroski = pd.DataFrame(scan_piotroski(df)).drop(columns=['criteria'], errors='ignore')
    return darvas, piotroski


def write_reports(darvas: pd.DataFrame, piotroski: pd.DataFrame, out_dir: Path) -> None:
    """Write both scans to `out_dir`, sorted by score descending."""
    out_dir.mkdir(exist_ok=True)
    darvas.sort_values('score', ascending=False).to_csv(out_dir / DARVAS_CSV, index=False)
    piotroski.sort_values('score', ascending=False).to_csv(out_dir / PIOTROSKI_CSV, index=False)


def print_summary(darvas: pd.DataFrame, piotroski: pd.DataFrame) -> None:
    """Print BUY/WATCH/AVOID counts and the top BUY rows for each scan."""
    for name, d in [('Darvas/Buffett', darvas), ('Piotroski', piotroski)]:
        counts = d['signal'].value_counts()
        print(f'\n{name}: BUY={counts.get("BUY", 0)} WATCH={counts.get("WATCH", 0)} AVOID={counts.get("AVOID", 0)}')
        top = d[d['signal'] == 'BUY'].sort_values('score', ascending=False).head(TOP_N)
        if not top.empty:
            print(top[['ticker', 'name', 'exchange', 'cmp', 'score', 'max_score']].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exchange', default=None, help='Filter to one or more exchanges, comma-separated (see data/europe_all_list.csv for names)')
    ap.add_argument('--limit', type=int, default=None, help='Cap universe size (for smoke tests)')
    ap.add_argument('--with-fundamentals', action='store_true',
                    help='Also fetch PE/ROE/D-E/etc via per-ticker yfinance .info (slow: ~1-3h for full universe)')
    ap.add_argument('--out', default='reports', help='Output directory for CSV reports')
    ap.add_argument('-v', '--verbose', action='store_true')
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry: fetch -> (optional fundamentals) -> score -> write reports."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(asctime)s %(levelname)s %(message)s')

    universe = load_universe(args.exchange, args.limit)
    print(f'Universe: {len(universe)} stocks across {universe["exchange"].nunique()} exchange(s)')

    t0 = time.time()
    quotes = fetch_bulk(universe)
    print(f'Fetched {len(quotes)}/{len(universe)} tickers in {time.time() - t0:.0f}s')

    if args.with_fundamentals:
        print('Fetching fundamentals (throttled, ~1/sec)...')
        enrich_with_fundamentals(quotes)

    df = quotes_to_frame(quotes)
    if df.empty:
        print('No data fetched — aborting.')
        return

    darvas_df, piotroski_df = run_scans(df)
    out_dir = Path(args.out)
    write_reports(darvas_df, piotroski_df, out_dir)
    print_summary(darvas_df, piotroski_df)

    print(f'\nFull results written to {out_dir}/{DARVAS_CSV} and {out_dir}/{PIOTROSKI_CSV}')


if __name__ == '__main__':
    main()
