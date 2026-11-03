from synthetic import make_ohlcv

from screener.technicals import QuoteData, compute_technicals


def test_too_short_series_returns_empty():
    assert compute_technicals(make_ohlcv(1, 0, 1)['Close']) == {}


def test_full_history_populates_all_indicators():
    o = make_ohlcv(260, 0.0009, 1)
    t = compute_technicals(o['Close'], o['Volume'], o['High'], o['Low'])
    for k in ('rsi', 'ema_200', 'macd_signal', 'bb_pct', 'atr_14', 'stoch_k', 'ret_1y', 'volume_ratio'):
        assert t[k] is not None, k
    assert t['high_52w'] >= t['cmp'] >= t['low_52w']


def test_close_only_has_no_atr_or_stochastic():
    o = make_ohlcv(100, 0.0, 3)
    t = compute_technicals(o['Close'])
    assert t['atr_14'] is None and t['stoch_k'] is None and t['volume'] is None


def test_quote_to_dict_drops_none():
    d = QuoteData(ticker='X.L', source='yahoo', cmp=1.0).to_dict()
    assert d['cmp'] == 1.0 and 'rsi' not in d
