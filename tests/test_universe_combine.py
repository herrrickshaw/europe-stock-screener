import pandas as pd

import combine_reports
from screener.universe import EXCHANGE_SUFFIXES, load_universe, ticker_suffix, unknown_suffix_tickers


def test_bundled_universe_uses_only_allowlisted_suffixes():
    df = load_universe()
    assert len(df) == 1063 and df['exchange'].nunique() == 23
    assert unknown_suffix_tickers(df['yf_ticker'].tolist()) == []


def test_suffix_is_looked_up_never_inferred():
    assert ticker_suffix('BP.L') == '.L'
    assert ticker_suffix('BP') is None          # no suffix -> not guessed
    assert ticker_suffix('AAPL.US') is None     # not on the allowlist
    assert '.NS' not in EXCHANGE_SUFFIXES


def test_unknown_suffix_warns_but_keeps_row(tmp_path, caplog):
    p = tmp_path / 'u.csv'
    p.write_text('yf_ticker,name,index,exchange\nXYZ.ZZ,X,I,E\nBP.L,BP,I,London Stock Exchange\n')
    with caplog.at_level('WARNING'):
        df = load_universe(path=p)
    assert len(df) == 2 and 'XYZ.ZZ' in caplog.text


def test_merge_frames_dedupes_first_wins_and_sorts():
    a = pd.DataFrame({'ticker': ['A', 'B'], 'score': [1, 5]})
    b = pd.DataFrame({'ticker': ['A', 'C'], 'score': [9, 3]})
    out = combine_reports.merge_frames([a, b])
    assert out['ticker'].tolist() == ['B', 'C', 'A'] and out.loc[out.ticker == 'A', 'score'].item() == 1
