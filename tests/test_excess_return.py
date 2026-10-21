"""
Regression tests for the measurement core.

Each test corresponds to a specific mistake that was actually made during this
project. They are kept as tests, not comments, because the only reliable way to
stop a look-ahead bug from coming back is to let CI fail when it does.
"""
import numpy as np
import pandas as pd
import pytest

from excess_return import (
    wilder_atr, build_targets, information_coefficient, partial_ic, _synth,
)

H, N = 6, 500


@pytest.fixture(scope="module")
def flat():
    """Driftless synthetic series."""
    return build_targets(_synth(40000, mu=0.0, seed=11), H, N)


def test_no_lookahead_in_excess_return():
    """Corrupting the future must not change any already-computed target."""
    df = _synth(6000, mu=0.01, seed=1)
    a = build_targets(df, H, N)
    df2 = df.copy()
    cut = 4000
    df2.iloc[cut:, :] = df2.iloc[cut:, :] + 500.0
    b = build_targets(df2, H, N)
    lhs = a["excess_causal"].iloc[: cut - H].dropna()
    rhs = b["excess_causal"].iloc[: cut - H].dropna()
    assert np.allclose(lhs, rhs)


def test_drift_is_actually_removed():
    df = _synth(40000, mu=0.02, seed=2)
    a = build_targets(df, H, N)
    raw, exc = a["fwd_h"].mean(), a["excess_causal"].mean()
    assert abs(exc) < abs(raw) * 0.25


def test_variance_inflation_matches_theory():
    """Subtracting a trailing-mean drift estimate inflates variance by 1 + h/N."""
    df = _synth(80000, mu=0.0, seed=3)
    a = build_targets(df, H, N).dropna(subset=["fwd_h", "excess_causal"])
    ratio = a["excess_causal"].var() / a["fwd_h"].var()
    assert ratio == pytest.approx(1 + H / N, abs=0.03)


def test_atr_is_wilder_not_sma():
    """A past bug: Python used SMA while MQL5 used Wilder (mean abs diff 10.17%)."""
    df = _synth(3000, seed=4)
    atr_w = wilder_atr(df.high, df.low, df.close, 14)
    tr = np.maximum(
        df.high - df.low,
        np.maximum((df.high - df.close.shift()).abs(),
                   (df.low - df.close.shift()).abs()),
    )
    atr_sma = tr.rolling(14).mean().to_numpy()
    assert np.nanmean(np.abs(atr_w - atr_sma) / atr_sma) > 0.005


def test_null_score_gives_ci_containing_zero(flat):
    a = flat.dropna(subset=["excess_causal"])
    score = np.random.default_rng(9).normal(size=len(a))
    r = information_coefficient(score, a["excess_causal"], H, n_boot=400)
    assert r["lo"] < 0 < r["hi"]


def test_excess_target_is_biased_but_partial_ic_is_not():
    """
    Positive control. A pure drift-tracking score must score ~0 on a measure of
    'signal beyond drift'.

    Using excess return as the target gives a large spurious NEGATIVE IC
    (mechanical term -rho * h/sqrt(N*h)). The partial IC, which removes drift
    from the score side as well, is unbiased. This defect is invisible to a
    null test: a meaningless score scores 0 either way.
    """
    exc_ics, par_ics = [], []
    for sd in range(4):
        a = build_targets(_synth(40000, mu=0.0, seed=200 + sd), H, N)
        sc = a["r_bar"].rolling(200, min_periods=200).mean()
        j = a.join(sc.rename("sc")).dropna(
            subset=["fwd_h", "excess_causal", "drift_causal", "sc"])
        exc_ics.append(information_coefficient(
            j["sc"], j["excess_causal"], H, n_boot=150, block=404)["ic"])
        par_ics.append(partial_ic(
            j["sc"], j["fwd_h"], [j["drift_causal"]], H,
            n_boot=150, block=404)["ic"])

    assert np.mean(exc_ics) < -0.03, "expected the known bias to reproduce"
    assert abs(np.mean(par_ics)) < 0.015, "partial IC should be unbiased"


def test_ic_rejects_length_mismatch():
    """A silent failure once seen: misaligned indices produced an all-NaN IC."""
    with pytest.raises(ValueError):
        information_coefficient(np.zeros(100), np.zeros(99), H, n_boot=10)
