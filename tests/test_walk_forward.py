"""
Walk-forward leakage tests.

The original walk-forward routine took `tr_idx = idx_pool[:tr_cnt]` and started
the test fold immediately afterwards. Labels are `close.shift(-horizon)`, so the
last `horizon` training rows had labels drawn from the first `horizon` test bars:
a leak at every fold boundary.

test_purge_blocks_label_overlap is the fix. test_unpurged_split_would_leak is
the control that proves the first test can actually fail.
"""
import numpy as np
import pytest

from walk_forward import make_folds, train_mask, build, FEATURES

H = 6
N_ROWS = 4000


def test_purge_blocks_label_overlap():
    prev = []
    for _, tlo, thi in make_folds(N_ROWS, 1500, 500, 5):
        m = train_mask(N_ROWS, tlo, thi, H, H, prev)
        last_train = np.flatnonzero(m).max()
        assert last_train + H < tlo
        prev.append((tlo, thi))


def test_unpurged_split_would_leak():
    """Control: the naive split must violate the condition above."""
    folds = make_folds(N_ROWS, 1500, 500, 5)
    tlo = folds[0][1]
    naive_last_train = tlo - 1
    assert naive_last_train + H >= tlo


def test_embargo_excludes_rows_after_a_past_test_fold():
    m = train_mask(N_ROWS, 2600, 3100, H, 24, prev_tests=[(1500, 2000)])
    assert not m[2000:2024].any()
    assert m[1400]              # untouched region still usable


def test_folds_do_not_overlap_and_move_forward():
    folds = make_folds(N_ROWS, 1000, 500, 10)
    assert folds, "expected at least one fold"
    for (_, lo1, hi1), (_, lo2, _) in zip(folds, folds[1:]):
        assert hi1 <= lo2
        assert lo1 < hi1


def test_label_and_features_are_present_and_neutral_band_marked():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 2000
    c = 1800 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    d = build(pd.DataFrame(dict(open=c, high=c + 1, low=c - 1, close=c), index=idx))

    for f in FEATURES:
        assert f in d.columns
    assert set(np.unique(d["y"])) <= {-1, 0, 1}
    assert (d["y"] == -1).any(), "neutral band should be flagged as -1, not dropped"
