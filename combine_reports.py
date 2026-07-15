#!/usr/bin/env python3
"""
Merge the main-universe scan output with a supplementary exchange-scoped run
(e.g. reports/ + reports_new_exchanges/) into one combined report covering the
full universe.

Usage:
    python combine_reports.py reports reports_new_exchanges --out reports_combined
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FILES = ['darvas_buffett_scan.csv', 'piotroski_scan.csv']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('report_dirs', nargs='+', help='Report directories to merge, in order')
    ap.add_argument('--out', default='reports_combined')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    for fname in FILES:
        frames = []
        for d in args.report_dirs:
            fp = Path(d) / fname
            if fp.exists():
                frames.append(pd.read_csv(fp))
            else:
                print(f'  skip: {fp} not found')
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset='ticker', keep='first')
        if before != len(merged):
            print(f'{fname}: dropped {before - len(merged)} duplicate ticker(s)')
        merged = merged.sort_values('score', ascending=False).reset_index(drop=True)
        merged.to_csv(out_dir / fname, index=False)
        print(f'{fname}: {len(merged)} rows -> {out_dir / fname}')

        counts = merged['signal'].value_counts()
        print(f'  BUY={counts.get("BUY", 0)} WATCH={counts.get("WATCH", 0)} AVOID={counts.get("AVOID", 0)}')
        print(f'  exchanges: {merged["exchange"].nunique()}')


if __name__ == '__main__':
    main()
