"""
walk_forward.py — purge/embargo 付きウォークフォワードで全期間の OOS スコアを作る

既存 walk_forward_p1 との違い
-----------------------------
1. PURGE: 訓練行のうち、ラベル窓（close.shift(-h)）がテスト区間に掛かるものを落とす。
   既存実装は tr_idx = idx_pool[:tr_cnt] / テストは直後から、なので各フォールド境界で
   h 本ぶんラベルが漏れていた。

2. EMBARGO: テスト区間の直後 e 本を、以降のフォールドの訓練から除外する。
   拡大窓では前フォールドのテストが次フォールドの訓練になるため、逆方向にも必要。

3. シードアンサンブル: 複数シードの p1 を平均する。CatBoost 内部乱数由来の分散だけを
   落とす操作で、データ由来の分散は落ちない（そちらはブロックブートストラップで測る）。

自己テストで漏れがないことを毎回検査する（表明であって print ではない）。

使い方
------
  python3 walk_forward.py --m1 XAUUSD_M1_...csv --out model_scores_wf.csv
  python3 walk_forward.py --selftest
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd

HORIZON = 6
THRESHOLD = 0.5
FEATURES = ["ret_1", "ret_3", "ret_6", "dist_ma50", "dist_ma200",
            "rsi", "atr_ratio", "hour", "dayofweek"]


# ------------------------------------------------------------------ 特徴量

def wilder_atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    v = tr.to_numpy(float)
    out = np.full(len(v), np.nan)
    if len(v) > period:
        out[period - 1] = v[:period].mean()
        for i in range(period, len(v)):
            out[i] = (out[i - 1] * (period - 1) + v[i]) / period
    return pd.Series(out, index=close.index)


def wilder_rsi(close, period=14):
    delta = close.diff()
    g = delta.clip(lower=0).to_numpy()
    l = (-delta.clip(upper=0)).to_numpy()
    n = len(close)
    ag, al = np.full(n, np.nan), np.full(n, np.nan)
    if n > period:
        ag[period] = np.nanmean(g[1:period + 1])
        al[period] = np.nanmean(l[1:period + 1])
        for i in range(period + 1, n):
            ag[i] = (ag[i - 1] * (period - 1) + g[i]) / period
            al[i] = (al[i - 1] * (period - 1) + l[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(100 - (100 / (1 + ag / al)), index=close.index)


def build(h1: pd.DataFrame, horizon=HORIZON, thr=THRESHOLD) -> pd.DataFrame:
    """既存 build() と同一定義（比較可能性を保つため意図的に揃えている）。"""
    d = h1.copy()
    hi, lo, c = d["high"], d["low"], d["close"]
    d["atr"] = wilder_atr(hi, lo, c, 14)
    d["rsi"] = wilder_rsi(c, 14)
    for k in (1, 3, 6):
        d[f"ret_{k}"] = (c - c.shift(k)) / d["atr"]
    d["dist_ma50"] = (c - c.rolling(50).mean()) / d["atr"]
    d["dist_ma200"] = (c - c.rolling(200).mean()) / d["atr"]
    d["atr_ratio"] = d["atr"] / d["atr"].rolling(100).mean()
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    fut = (c.shift(-horizon) - c) / d["atr"]
    d["y"] = np.select([fut > thr, fut < -thr], [1, 0], default=-1)
    return d


# ------------------------------------------------------- ウォークフォワード

def make_folds(n_rows: int, min_train: int, test_size: int, max_splits: int):
    """(train_end, test_lo, test_hi) を行位置で返す。拡大窓。"""
    folds = []
    lo = min_train
    while lo < n_rows and len(folds) < max_splits:
        hi = min(lo + test_size, n_rows)
        if hi - lo < test_size // 4:
            break
        folds.append((lo, lo, hi))
        lo = hi
    return folds


def train_mask(n_rows, test_lo, test_hi, horizon, embargo, prev_tests):
    """
    訓練に使ってよい行。
      ・test_lo より前
      ・PURGE: ラベル窓 [i, i+horizon] が test_lo 以降に掛かる行を除く → i + horizon < test_lo
      ・EMBARGO: 過去のテスト区間の直後 embargo 本を除く
    """
    m = np.zeros(n_rows, dtype=bool)
    m[:max(0, test_lo - horizon)] = True          # PURGE
    for plo, phi in prev_tests:
        m[phi:min(n_rows, phi + embargo)] = False  # EMBARGO
    return m


def walk_forward(d: pd.DataFrame, seeds, min_train, test_size, max_splits,
                 horizon, embargo, iterations, depth, l2, lr, verbose=True):
    from catboost import CatBoostClassifier

    feat_ok = d[FEATURES].notna().all(axis=1).to_numpy()
    lbl_ok = d["y"].to_numpy() >= 0
    X = d[FEATURES].to_numpy(float)
    y = d["y"].to_numpy(int)
    n = len(d)

    folds = make_folds(n, min_train, test_size, max_splits)
    acc = np.zeros((len(seeds), n))
    acc[:] = np.nan

    prev_tests = []
    for fi, (tr_end, tlo, thi) in enumerate(folds):
        tm = train_mask(n, tlo, thi, horizon, embargo, prev_tests) & feat_ok & lbl_ok
        te = np.zeros(n, dtype=bool)
        te[tlo:thi] = True
        te &= feat_ok

        tr_idx = np.flatnonzero(tm)
        if len(tr_idx) < 500 or te.sum() == 0:
            prev_tests.append((tlo, thi))
            continue

        # --- 漏れの表明（print ではなく assert）
        assert tr_idx.max() + horizon < tlo, (
            f"PURGE 違反: fold {fi} 訓練最終行 {tr_idx.max()} +{horizon} >= テスト先頭 {tlo}")

        for si, sd in enumerate(seeds):
            m = CatBoostClassifier(iterations=iterations, depth=depth,
                                   learning_rate=lr, l2_leaf_reg=l2,
                                   loss_function="Logloss", random_seed=sd,
                                   verbose=False)
            m.fit(X[tr_idx], y[tr_idx])
            acc[si, te] = m.predict_proba(X[te])[:, 1]

        if verbose:
            print(f"  fold {fi:2d}  train {len(tr_idx):6d} 行 (〜{d.index[tr_idx.max()]})"
                  f"  test {int(te.sum()):5d} 行 "
                  f"({d.index[tlo]} 〜 {d.index[thi-1]})", flush=True)
        prev_tests.append((tlo, thi))

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p1 = np.nanmean(acc, axis=0)
            spread = (np.nanmax(acc, axis=0) - np.nanmin(acc, axis=0)
                      if len(seeds) > 1 else None)
    return p1, spread, folds


# ------------------------------------------------------------------ selftest

def selftest():
    ok = True
    n = 4000
    rng = np.random.default_rng(0)
    c = 1800 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    h1 = pd.DataFrame(dict(open=c, high=c + 1, low=c - 1, close=c), index=idx)
    d = build(h1)

    # T1 PURGE がラベル窓の重なりを実際に消していること
    folds = make_folds(n, 1500, 500, 5)
    prev = []
    for fi, (_, tlo, thi) in enumerate(folds):
        tm = train_mask(n, tlo, thi, HORIZON, HORIZON, prev)
        last = np.flatnonzero(tm).max()
        if last + HORIZON >= tlo:
            ok = False
        prev.append((tlo, thi))
    print(f"T1 PURGE: 訓練最終行+h < テスト先頭   : {'PASS' if ok else 'FAIL'}")

    # T2 EMBARGO が過去テスト直後を訓練から外していること
    prev = [(1500, 2000)]
    tm = train_mask(n, 2600, 3100, HORIZON, 24, prev)
    t2 = not tm[2000:2024].any()
    print(f"T2 EMBARGO: テスト直後 24 本を除外    : {'PASS' if t2 else 'FAIL'}")

    # T3 既存実装（purge なし）なら T1 が落ちること = 検査が実際に効いている
    naive = np.zeros(n, dtype=bool)
    naive[:folds[0][1]] = True
    t3 = (np.flatnonzero(naive).max() + HORIZON) >= folds[0][1]
    print(f"T3 purge なしなら T1 は落ちる（検査の有効性）: {'PASS' if t3 else 'FAIL'}")

    ok = ok and t2 and t3
    print("ALL PASS" if ok else "*** FAILURE ***")
    return ok


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1")
    ap.add_argument("--out", default="model_scores_wf.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2026, 31337])
    ap.add_argument("--min_train", type=int, default=6000)
    ap.add_argument("--test_size", type=int, default=1500)
    ap.add_argument("--max_splits", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--embargo", type=int, default=24)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--l2", type=float, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.m1:
        sys.exit("--m1 か --selftest を指定してください")

    print("1. M1 読み込み → H1 → 特徴量・ラベル")
    m1 = pd.read_csv(a.m1, sep="\t")
    m1.columns = [c.strip("<>").lower() for c in m1.columns]
    m1.index = pd.to_datetime(m1["date"].astype(str) + " " + m1["time"].astype(str))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    h1 = m1.resample("1h").agg(agg).dropna(subset=["close"])
    d = build(h1, a.horizon)
    print(f"   H1 {len(d)} 本  {d.index[0]} 〜 {d.index[-1]}")
    print(f"   ラベル有効 {int((d['y']>=0).sum())} 行 "
          f"(中立帯除外 {int((d['y']<0).sum())} 行)")

    print(f"\n2. ウォークフォワード（purge={a.horizon} / embargo={a.embargo} / "
          f"seeds={len(a.seeds)}）")
    p1, spread, folds = walk_forward(
        d, a.seeds, a.min_train, a.test_size, a.max_splits,
        a.horizon, a.embargo, a.iterations, a.depth, a.l2, a.lr)

    res = pd.DataFrame({"datetime": d.index, "score": p1}).dropna(subset=["score"])
    print(f"\n3. OOS スコア {len(res)} 行  {res.datetime.min()} 〜 {res.datetime.max()}")
    print(f"   全 H1 に対する被覆率 {len(res)/len(d):.1%}")
    q = np.percentile(res["score"], [1, 25, 50, 75, 99])
    print(f"   p1={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} "
          f"p75={q[3]:.4f} p99={q[4]:.4f}")
    print(f"   conf>=0.60 割合 {np.mean(res['score']>=0.60):.1%}")
    if len(np.unique(np.round(res['score'], 4))) < 20:
        print("   *** 警告: スコアがほぼ定数。IC は測れません ***")
    if spread is not None:
        sp = spread[~np.isnan(spread)]
        print(f"   シード間の幅 中央値 {np.median(sp):.4f} "
              f"(アンサンブルが消したのはこの内部乱数ぶんだけ)")

    res.to_csv(a.out, index=False)
    print(f"\n4. 出力 {a.out}")


if __name__ == "__main__":
    main()
