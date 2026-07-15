# Europe Stock Screener

Darvas/Buffett + Piotroski-simplified momentum-and-quality screener, scoped to the
full European equity universe: **1,063 stocks across 23 exchanges**.

Extracted and adapted from a larger multi-market research platform's Phase-1
(OHLCV+RSI, no fundamentals required) scanner, so it runs standalone with only
`yfinance` — no database, no broker API keys.

Research/education only. Not investment advice.

## Universe (`data/europe_all_list.csv`)

| Exchange group | Stocks | Suffix |
|---|---|---|
| London Stock Exchange | 436 | `.L` |
| Deutsche Boerse Frankfurt | 142 | `.F` / `.DE` |
| Euronext Paris | 40 | `.PA` |
| Borsa Italiana (Milan) | 40 | `.MI` |
| BME Madrid | 35 | `.MC` |
| Nasdaq Stockholm | 30 | `.ST` |
| Oslo Bors | 25 | `.OL` |
| Nasdaq Helsinki | 25 | `.HE` |
| Nasdaq Copenhagen | 25 | `.CO` |
| Athens Stock Exchange | 25 | `.AT` |
| Euronext Amsterdam | 25 | `.AS` |
| Nasdaq Vilnius | 22 | `.VS` |
| Euronext Dublin | 20 | `.IR` |
| Euronext Brussels | 20 | `.BR` |
| Nasdaq Tallinn | 20 | `.TL` |
| Bucharest Stock Exchange (BET) | 20 | `.RO` |
| SIX Swiss | 20 | `.SW` |
| Vienna (ATX) | 20 | `.VI` |
| Warsaw GPW | 20 | `.WA` |
| Euronext Lisbon | 18 | `.LS` |
| Budapest Stock Exchange (BUX) | 17 | `.BD` |
| Prague Stock Exchange (PX) | 10 | `.PR` |
| Nasdaq Riga | 8 | `.RG` |

The 6 newest exchanges (Vilnius/Tallinn/Riga/Prague/Budapest/Bucharest) were added
because the parent platform's `_market_from_ticker()` already recognized these
suffixes as European, but no instrument list had been built for them yet. Sourced
from each exchange's official index page + Nasdaq Baltic's share list (Main +
Secondary tiers; First North/growth-board names excluded for liquidity quality),
all 97 tickers validated against yfinance before being added. Prague and Budapest
lists are slightly incomplete (10/12 and 17/17 respectively — Prague's TradingView
source paywalled 2 of the smallest constituents) — see `data/europe_all_list.csv`
for the exact roster.

## Scoring

**Darvas/Buffett** (0-16 max): near-52W-high, above-EMA50/200, range-strength,
RSI-healthy, MACD-bullish, volume-surge, EMA-cross, Bollinger, Stochastic —
plus optional ROE/PE/D-E overlay when fundamentals are fetched.
`BUY >= 5 / WATCH >= 3 / AVOID < 3` (out of a max that shrinks to the fields
actually available).

**Piotroski-simplified** (0-11 max): momentum proxies (above-EMA50, RSI-range,
near-52W-high, EMA-cross, low-ATR) plus fundamentals overlay (ROE, D/E, OPM,
revenue growth, current ratio, PB) when fetched. Thresholds scale dynamically
with how many fields are available: `BUY >= ceil(max*0.65)`, `WATCH >= ceil(max*0.40)`.

See `screener/scanner.py` docstring for the exact per-criterion weights.

## Usage

```bash
pip install -r requirements.txt

# Fast pass — OHLCV + RSI only, ~5-10 min for all 966 stocks
python run_scan.py

# Slower pass — adds PE/ROE/D-E via per-ticker yfinance .info (~1-3h)
python run_scan.py --with-fundamentals

# Scope to one exchange
python run_scan.py --exchange "London Stock Exchange"

# Smoke test
python run_scan.py --limit 50 -v
```

Output: `reports/darvas_buffett_scan.csv` and `reports/piotroski_scan.csv`,
sorted by score descending, plus a console summary of BUY/WATCH/AVOID counts
and the top 10 BUY signals per scan.

## Structure

```
screener/
  technicals.py   RSI-14, EMA-20/50/200, MACD, Bollinger, ATR, Stochastic, period returns
  fetch.py        Batched yfinance OHLCV download (75 tickers/batch) + technicals
  scanner.py      scan_darvas() / scan_piotroski() — vectorized pandas scoring
run_scan.py       CLI entry point
data/europe_all_list.csv   966-stock universe (yf_ticker, name, index, exchange)
```

## Known limitations

- yfinance rate-limits on large batch downloads; `fetch.py` backs off at 90s/180s/300s
  on 429s. If a run stalls repeatedly, lower `_BATCH_SIZE` in `screener/fetch.py`.
- `--with-fundamentals` makes one `.info` call per ticker (~1/sec) — budget the time.
- No survivorship-bias handling — this is a live snapshot scan, not a backtest.
