"""run_scan.load_universe() and main() end to end with fetching stubbed."""
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

import run_scan
import screener.fetch as fetch
from screener.technicals import QuoteData

UNIVERSE = pd.DataFrame({
    "yf_ticker": ["A.DE", "B.DE", "C.L", "D.PA", "E.L"],
    "name": ["A", "B", "C", "D", "E"],
    "index": ["X"] * 5,
    "exchange": ["Deutsche Boerse", "Deutsche Boerse", "London Stock Exchange",
                 "Euronext Paris", "London Stock Exchange"],
})

STRONG = dict(cmp=100.0, high_52w=105.0, low_52w=60.0, ema_20=98.0, ema_50=95.0, rsi=55.0)
WEAK = dict(cmp=62.0, high_52w=105.0, low_52w=60.0, ema_20=70.0, ema_50=95.0, rsi=30.0)


@pytest.fixture
def universe_csv(tmp_path, monkeypatch):
    p = tmp_path / "u.csv"
    UNIVERSE.to_csv(p, index=False)
    monkeypatch.setattr(run_scan, "UNIVERSE_CSV", p)
    return p


# ---------------------------------------------------------------- load_universe

def test_load_universe_all(universe_csv):
    df = run_scan.load_universe(None, None)
    assert list(df["yf_ticker"]) == list(UNIVERSE["yf_ticker"])


def test_load_universe_single_and_multiple_exchanges(universe_csv):
    assert list(run_scan.load_universe("London Stock Exchange", None)["yf_ticker"]) == ["C.L", "E.L"]
    both = run_scan.load_universe(" London Stock Exchange , Euronext Paris", None)
    assert list(both["yf_ticker"]) == ["C.L", "D.PA", "E.L"]
    assert list(both.index) == [0, 1, 2]                 # index is reset


def test_load_universe_unknown_exchange_is_empty(universe_csv):
    assert run_scan.load_universe("Nowhere", None).empty


def test_load_universe_limit_applies_after_filter_and_resets_index(universe_csv):
    df = run_scan.load_universe("London Stock Exchange", 1)
    assert list(df["yf_ticker"]) == ["C.L"] and list(df.index) == [0]
    assert len(run_scan.load_universe(None, 2)) == 2
    assert len(run_scan.load_universe(None, 99)) == 5
    assert len(run_scan.load_universe(None, 0)) == 5     # 0 means "no limit"


def test_universe_path_points_at_bundled_csv():
    assert run_scan.UNIVERSE_CSV == Path(run_scan.__file__).parent / "data" / "europe_all_list.csv"
    assert run_scan.UNIVERSE_CSV.exists()
    assert {"yf_ticker", "name", "exchange"} <= set(pd.read_csv(run_scan.UNIVERSE_CSV, nrows=1).columns)


# ---------------------------------------------------------------- main

def _quotes(**extra):
    q = {"A.DE": QuoteData("A.DE", "yahoo", name="A", exchange="Deutsche Boerse", **STRONG),
         "C.L": QuoteData("C.L", "yahoo", name="C", exchange="London Stock Exchange", **WEAK),
         "D.PA": QuoteData("D.PA", "yahoo", name="D", exchange="Euronext Paris", **STRONG)}
    q.update(extra)
    return q


@pytest.fixture
def env(tmp_path, monkeypatch, universe_csv):
    monkeypatch.chdir(tmp_path)
    sleeps = []
    monkeypatch.setattr(run_scan.time, "sleep", sleeps.append)
    state = {"sleeps": sleeps, "seen": None}

    def run(argv, quotes=None, funds=None):
        quotes = _quotes() if quotes is None else quotes

        def fake_bulk(universe):
            state["seen"] = universe
            return quotes
        monkeypatch.setattr(run_scan, "fetch_bulk", fake_bulk)
        monkeypatch.setattr(run_scan, "fetch_fundamentals", lambda t: (funds or {}).get(t, {}))
        monkeypatch.setattr(sys, "argv", ["run_scan.py", *argv])
        run_scan.main()
    state["run"] = run
    return state


def test_main_writes_sorted_reports_without_criteria(env, tmp_path, capsys):
    env["run"]([])
    out = capsys.readouterr().out
    assert "Universe: 5 stocks across 3 exchange(s)" in out
    assert "Fetched 3/5 tickers" in out
    d = pd.read_csv(tmp_path / "reports" / "darvas_buffett_scan.csv")
    p = pd.read_csv(tmp_path / "reports" / "piotroski_scan.csv")
    assert "criteria" not in d.columns and "criteria" not in p.columns
    assert len(d) == len(p) == 3
    assert d["score"].is_monotonic_decreasing and p["score"].is_monotonic_decreasing
    assert d["ticker"].iloc[-1] == "C.L" and d.set_index("ticker").loc["C.L", "signal"] == "AVOID"
    assert "Darvas/Buffett: BUY=2 WATCH=0 AVOID=1" in out
    assert "Piotroski: BUY=2 WATCH=0 AVOID=1" in out
    assert "A.DE" in out and "D.PA" in out and "C.L" not in out.split("Darvas/Buffett")[1].split("Full results")[0]
    assert "Full results written to reports/darvas_buffett_scan.csv and reports/piotroski_scan.csv" in out
    assert env["sleeps"] == []


def test_main_exchange_limit_out_args(env, tmp_path, capsys):
    env["run"](["--exchange", "London Stock Exchange,Euronext Paris", "--limit", "2", "--out", "custom"])
    assert list(env["seen"]["yf_ticker"]) == ["C.L", "D.PA"]
    assert (tmp_path / "custom" / "darvas_buffett_scan.csv").exists()
    assert not (tmp_path / "reports").exists()
    assert "Universe: 2 stocks across 2 exchange(s)" in capsys.readouterr().out


def test_main_no_buy_rows_omits_top_table(env, capsys):
    env["run"]([], quotes={"C.L": _quotes()["C.L"]})
    out = capsys.readouterr().out
    assert "BUY=0 WATCH=0 AVOID=1" in out
    assert "ticker  name" not in out                 # no top-picks table header printed


def test_main_empty_fetch_aborts_before_writing(env, tmp_path, capsys):
    env["run"]([], quotes={})
    out = capsys.readouterr().out
    assert "No data fetched" in out
    assert "Full results" not in out
    assert not (tmp_path / "reports").exists()


def test_fundamentals_flag_off_never_calls_fetch_fundamentals(env, monkeypatch, capsys):
    env["run"]([])
    assert "Fetching fundamentals" not in capsys.readouterr().out


def test_fundamentals_fill_only_missing_attributes(env, tmp_path, capsys):
    q = _quotes()
    q["A.DE"].roe = 99.0                              # already known: must not be overwritten
    funds = {"A.DE": {"roe": 5.0, "pe": 20.0, "pb": None},
             "C.L": {"pe": 8.0, "sector": "Energy"}}
    env["run"](["--with-fundamentals"], quotes=q, funds=funds)
    assert q["A.DE"].roe == 99.0 and q["A.DE"].pe == 20.0 and q["A.DE"].pb is None
    assert q["C.L"].pe == 8.0 and q["C.L"].sector == "Energy"
    assert env["sleeps"] == [1, 1, 1]                 # throttled once per ticker
    out = capsys.readouterr().out
    assert "Fetching fundamentals" in out
    assert "fundamentals: " not in out                # no progress line under 50 tickers
    d = pd.read_csv(tmp_path / "reports" / "darvas_buffett_scan.csv").set_index("ticker")
    assert d.loc["C.L", "sector"] == "Energy"


@pytest.mark.parametrize("n,lines", [(49, []), (50, ["50/50"]), (100, ["50/100", "100/100"])])
def test_fundamentals_progress_every_50(env, capsys, n, lines):
    quotes = {f"T{i}.DE": QuoteData(f"T{i}.DE", "yahoo", **STRONG) for i in range(n)}
    env["run"](["--with-fundamentals"], quotes=quotes)
    out = capsys.readouterr().out
    got = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("fundamentals:")]
    assert got == [f"fundamentals: {x}" for x in lines]
    assert len(env["sleeps"]) == n


def test_script_entrypoint(tmp_path, monkeypatch, universe_csv, capsys):
    """`python run_scan.py` runs main() via the __main__ guard."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fetch, "fetch_bulk", lambda u: {})
    monkeypatch.setattr(sys, "argv", ["run_scan.py"])
    # fresh module namespace: reads its own UNIVERSE_CSV, so point the real csv at tmp via chdir-free patch
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: UNIVERSE.copy())
    runpy.run_path(run_scan.__file__, run_name="__main__")
    assert "No data fetched" in capsys.readouterr().out


def test_top_table_is_capped_at_ten_rows(env, capsys):
    import re
    quotes = {f"T{i:02d}.DE": QuoteData(f"T{i:02d}.DE", "yahoo", **STRONG) for i in range(12)}
    env["run"]([], quotes=quotes)
    out = capsys.readouterr().out
    rows = re.findall(r"^\s*T\d\d\.DE", out, flags=re.M)
    assert len(rows) == 20                          # 10 per scan, two scans
    assert "Darvas/Buffett: BUY=12" in out and "Piotroski: BUY=12" in out
