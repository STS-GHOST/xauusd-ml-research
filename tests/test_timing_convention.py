"""
Entry-timing regression tests.

The convention is fixed and non-negotiable:

    signal   = bar t   (decided on bar t's CLOSE)
    entry    = bar t+1 OPEN
    SL check = bars t+1 .. t+h
    exit     = SL price, or bar t+h CLOSE

An earlier revision of wrapper_decomp.py entered at bar t's OPEN and scanned
SL from bar t, i.e. it used one hour of data from before the signal existed.
It produced two confident, entirely false conclusions (that the 3-ATR stop was
destroying 0.075 ATR/trade, and that the single-position constraint was the
largest source of loss at -0.116 ATR/trade). Both reversed after the fix.

These tests exist so that failure mode fails CI instead of a report.
"""
import numpy as np
import pandas as pd
import pytest

from wrapper_decomp import simulate

H = 6


def _series(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    c = 1800 + np.cumsum(rng.normal(0, 1.0, n))
    o = np.concatenate([[c[0]], c[:-1]])
    lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.5, n))
    ts = pd.date_range("2021-01-01", periods=n, freq="1h").values.astype("datetime64[ns]")
    atr = np.full(n, 2.0)
    pos = np.arange(300, n - 20, 7)
    return c, o, lo, ts, atr, pos


def _run(o, c, lo, atr, ts, pos):
    return simulate(pos, o, c, lo, atr, ts, lo, c, None, ts, 0.01,
                    cost_mode="flat", sl_source="h1")[0]


def test_signal_bar_data_cannot_affect_pnl():
    """Bar t's own open and low are in the past relative to entry at t+1."""
    c, o, lo, ts, atr, pos = _series()
    base = _run(o, c, lo, atr, ts, pos)

    o2, lo2 = o.copy(), lo.copy()
    o2[pos] -= 500.0
    lo2[pos] -= 500.0          # would trip a look-ahead stop if the window is wrong
    pert = _run(o2, c, lo2, atr, ts, pos)

    m = np.isfinite(base)
    assert np.allclose(base[m], pert[m])


def test_entry_bar_open_does_affect_pnl():
    """Guards against the above passing because nothing is used at all."""
    c, o, lo, ts, atr, pos = _series()
    base = _run(o, c, lo, atr, ts, pos)

    o3 = o.copy()
    o3[pos + 1] -= 500.0
    pert = _run(o3, c, lo, atr, ts, pos)

    m = np.isfinite(base)
    assert not np.allclose(base[m], pert[m])


def test_timeout_exit_uses_bar_t_plus_h_close():
    """With no stop reachable, PnL must equal (close[t+h] - open[t+1]) / atr."""
    c, o, lo, ts, atr, pos = _series()
    lo_safe = np.full_like(c, 1e9)          # no bar can ever touch the stop
    pnl = _run(o, c, lo_safe, atr, ts, pos)
    want = (c[pos + H] - o[pos + 1]) / atr[pos]
    assert np.allclose(pnl, want)
