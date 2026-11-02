# Europe Stock Screener

Darvas/Buffett and Piotroski-simplified momentum-and-quality screener for the European equity universe:
**1,063 stocks across 23 exchanges** (`data/europe_all_list.csv`). It downloads one year of daily prices
from Yahoo Finance, computes technical indicators, scores every stock, and writes two ranked CSV reports
with BUY / WATCH / AVOID signals.

> Status: maintained. Research and education only; not investment advice. Prices come from Yahoo Finance
> via `yfinance` and inherit its terms and data-quality limits.

New here? See [How to use this repo with AI](HOW_TO_USE_WITH_AI.md).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/herrrickshaw/europe-stock-screener/blob/main/notebooks/colab_test.ipynb)

## What it does
- Bulk-downloads 1y OHLCV in batches of 75 tickers with 90/180/300 s back-off on rate limits.
- Computes RSI-14, EMA-20/50/200, MACD, Bollinger, ATR-14, Stochastic, volume ratio and period returns.
- Scores each stock on Darvas/Buffett (0-16 max) and Piotroski-simplified (0-11 max).
- Optionally adds PE / ROE / D-E / margins via per-ticker `yfinance .info` (`--with-fundamentals`).
- Merges a main run with a supplementary exchange-scoped run (`combine_reports.py`).

## Quick start
```bash
git clone https://github.com/herrrickshaw/europe-stock-screener.git
cd europe-stock-screener
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_scan.py --limit 50 -v      # smoke test; drop --limit for the full universe
```
More runs:
```bash
python run_scan.py                                  # OHLCV + RSI only, ~5-10 min for the full universe
python run_scan.py --with-fundamentals              # adds PE/ROE/D-E, ~1-3 h
python run_scan.py --exchange "London Stock Exchange"   # comma-separate several exchanges
python combine_reports.py reports reports_new_exchanges --out reports_combined
```
Output: `reports/darvas_buffett_scan.csv` and `reports/piotroski_scan.csv` (sorted by score, descending),
plus a console summary of BUY/WATCH/AVOID counts and the top 10 BUY signals per scan.

## How it works
| Stage | Module | Output |
|---|---|---|
| Universe + suffix allowlist | `screener/universe.py` | DataFrame of `yf_ticker, name, index, exchange` |
| Fetch | `screener/fetch.py` (`fetch_bulk`, injectable downloader) | `{ticker: QuoteData}` |
| Indicators | `screener/technicals.py` (`compute_technicals`, pure) | indicator dict per ticker |
| Fundamentals (optional) | `screener/fetch.py::fetch_fundamentals`, `run_scan.enrich_with_fundamentals` | filled `QuoteData` fields |
| Score | `screener/scanner.py` (`scan_darvas`, `scan_piotroski`, vectorised pandas) | result rows |
| Report | `run_scan.py` (`write_reports`, `print_summary`) | two CSVs + console summary |

**Darvas/Buffett:** near-52W-high, above-EMA50/200, range-strength, RSI-healthy, MACD-bullish, volume-surge,
EMA-cross, Bollinger, Stochastic, plus optional ROE/PE/D-E overlay. `BUY >= 5 / WATCH >= 3 / AVOID < 3`
(the displayed max shrinks to the fields actually available).

**Piotroski-simplified:** momentum proxies (above-EMA50, RSI-range, near-52W-high, EMA-cross, low-ATR) plus
fundamentals overlay (ROE, D/E, OPM, revenue growth, current ratio, PB) when fetched. Thresholds scale with
available fields: `BUY >= ceil(max*0.65)`, `WATCH >= ceil(max*0.40)`.

The per-criterion weights are in the `screener/scanner.py` docstring. Tickers always carry an exchange suffix
from the `EXCHANGE_SUFFIXES` allowlist in `screener/universe.py`; a suffix is never inferred from ticker shape.
This is a snapshot screener, not a Darvas box backtester: the 52-week high/low includes the latest bar.

## Repository layout
| Path | Purpose |
|---|---|
| `run_scan.py` | CLI entry point (fetch, score, write reports) |
| `combine_reports.py` | Merge scan CSVs from several report directories (first row per ticker wins) |
| `screener/` | `universe.py`, `fetch.py`, `technicals.py`, `scanner.py` |
| `data/europe_all_list.csv` | Tracked universe: `yf_ticker, name, index, exchange` |
| `tests/` | pytest suite on synthetic OHLCV (no network); `golden/` pins scanner output |
| `notebooks/colab_test.ipynb` | Colab smoke test |
| `reports/` | Generated output, gitignored |

## Data sources
| Source | What is taken | Access route | Licence / caveat |
|---|---|---|---|
| Yahoo Finance | 1y daily OHLCV; optional `.info` fundamentals | `yfinance` (no API key) | Unofficial API; rate-limited; personal/research use |
| Exchange index pages, Nasdaq Baltic share list | Constituent lists in `data/europe_all_list.csv` | Compiled by hand/scrape, validated against yfinance | Main + Secondary tiers only for Baltics; First North excluded |

Universe by exchange (suffix): London 436 (`.L`), Frankfurt 142 (`.F`/`.DE`), Paris 40 (`.PA`), Milan 40 (`.MI`),
Madrid 35 (`.MC`), Stockholm 30 (`.ST`), Oslo 25 (`.OL`), Helsinki 25 (`.HE`), Copenhagen 25 (`.CO`),
Athens 25 (`.AT`), Amsterdam 25 (`.AS`), Vilnius 22 (`.VS`), Dublin 20 (`.IR`), Brussels 20 (`.BR`),
Tallinn 20 (`.TL`), Bucharest 20 (`.RO`), SIX Swiss 20 (`.SW`), Vienna 20 (`.VI`), Warsaw 20 (`.WA`),
Lisbon 18 (`.LS`), Budapest 17 (`.BD`), Prague 10 (`.PR`), Riga 8 (`.RG`).
Vilnius, Tallinn, Riga, Prague, Budapest and Bucharest were added because the parent platform already
recognised those suffixes; all 97 of their tickers were validated against yfinance. Prague (10 of 12) is
slightly incomplete because the TradingView source paywalled the two smallest constituents.

## Configuration
No environment variables or secrets. Tunables are module constants: `_BATCH_SIZE` and `_RATE_BACKOFFS`
in `screener/fetch.py`, signal thresholds in `screener/scanner.py`, indicator periods in `screener/technicals.py`.

## Testing
```bash
pip install pytest ruff
pytest -q && ruff check .
```
Tests use synthetic OHLCV and an injected downloader; yfinance is never called. CI runs the same on Python 3.9 and 3.12.

## Limitations and known issues
- yfinance rate-limits large batch downloads; if a run stalls repeatedly, lower `_BATCH_SIZE` in `screener/fetch.py`.
- `--with-fundamentals` makes one `.info` call per ticker (~1/sec); budget the time.
- No survivorship-bias handling: this is a live snapshot scan, not a backtest.

## Related repositories
Extracted from a larger private multi-market research platform's Phase-1 (OHLCV + RSI) scanner.

## Licence
No licence file yet; all rights reserved.
