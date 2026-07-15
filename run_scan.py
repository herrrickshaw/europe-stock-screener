#!/usr/bin/env python3
"""
Run the Darvas/Buffett + Piotroski-simplified scan across the full European
equity universe (17 exchanges, 966 stocks — data/europe_all_list.csv).

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

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from screener.fetch import fetch_bulk, fetch_fundamentals
from screener.scanner import scan_darvas, scan_piotroski

log = logging.getLogger('run_scan')

DATA_DIR = Path(__file__).parent / 'data'
UNIVERSE_CSV = DATA_DIR / 'europe_all_list.csv'


def load_universe(exchange: str | None, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    if exchange:
        df = df[df['exchange'] == exchange]
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exchange', default=None, help='Filter to one exchange (see data/europe_all_list.csv for names)')
    ap.add_argument('--limit', type=int, default=None, help='Cap universe size (for smoke tests)')
    ap.add_argument('--with-fundamentals', action='store_true',
                     help='Also fetch PE/ROE/D-E/etc via per-ticker yfinance .info (slow: ~1-3h for full universe)')
    ap.add_argument('--out', default='reports', help='Output directory for CSV reports')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format='%(asctime)s %(levelname)s %(message)s')

    universe = load_universe(args.exchange, args.limit)
    print(f'Universe: {len(universe)} stocks across {universe["exchange"].nunique()} exchange(s)')

    t0 = time.time()
    quotes = fetch_bulk(universe)
    print(f'Fetched {len(quotes)}/{len(universe)} tickers in {time.time() - t0:.0f}s')

    if args.with_fundamentals:
        print('Fetching fundamentals (throttled, ~1/sec)...')
        for i, ticker in enumerate(quotes.keys()):
            fund = fetch_fundamentals(ticker)
            for k, v in fund.items():
                if v is not None and getattr(quotes[ticker], k, None) is None:
                    setattr(quotes[ticker], k, v)
            if (i + 1) % 50 == 0:
                print(f'  fundamentals: {i + 1}/{len(quotes)}')
            time.sleep(1)

    rows = [q.to_dict() | {'ticker': t} for t, q in quotes.items()]
    df = pd.DataFrame(rows)
    if df.empty:
        print('No data fetched — aborting.')
        return

    darvas_results = scan_darvas(df)
    piotroski_results = scan_piotroski(df)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    darvas_df = pd.DataFrame(darvas_results).drop(columns=['criteria'], errors='ignore')
    piotroski_df = pd.DataFrame(piotroski_results).drop(columns=['criteria'], errors='ignore')

    darvas_df.sort_values('score', ascending=False).to_csv(out_dir / 'darvas_buffett_scan.csv', index=False)
    piotroski_df.sort_values('score', ascending=False).to_csv(out_dir / 'piotroski_scan.csv', index=False)

    for name, d in [('Darvas/Buffett', darvas_df), ('Piotroski', piotroski_df)]:
        counts = d['signal'].value_counts()
        print(f'\n{name}: BUY={counts.get("BUY", 0)} WATCH={counts.get("WATCH", 0)} AVOID={counts.get("AVOID", 0)}')
        top = d[d['signal'] == 'BUY'].sort_values('score', ascending=False).head(10)
        if not top.empty:
            print(top[['ticker', 'name', 'exchange', 'cmp', 'score', 'max_score']].to_string(index=False))

    print(f'\nFull results written to {out_dir}/darvas_buffett_scan.csv and {out_dir}/piotroski_scan.csv')


if __name__ == '__main__':
    main()
